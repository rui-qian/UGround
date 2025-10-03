import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .refer import REFER
from .qa_template import SHORT_QUESTION_TEMPLATE, SHORT_ANSWER_TEMPLATE, NEG_ANSWER_TEMPLATE, CORRECT_ANSWER_TEMPLATE


def create_zero_mask(height, width):
    return np.zeros((height, width), dtype=np.uint8)


def decode_segmentation(ann_segmentation, height, width):
    if type(ann_segmentation[0]) == list:  # polygon
        rle = mask.frPyObjects(ann_segmentation, height, width)
    else:
        rle = ann_segmentation
        for seg in rle:
            if not isinstance(seg["counts"], bytes):
                seg["counts"] = seg["counts"].encode()
    masks = mask.decode(rle)
    return np.sum(masks, axis=2).astype(np.uint8)  # Convert to np.uint8


def process_annotation(ann, image_info):
    if len(ann["segmentation"]) == 0:
        return create_zero_mask(image_info["height"], image_info["width"])
    else:
        return decode_segmentation(ann["segmentation"], image_info["height"], image_info["width"])

class fpReferSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 1024,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
        train_val_split="train"
    ):
        self.pad_train_clip_images = pad_train_clip_images
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.seg_token_num = seg_token_num
        self.use_expand_question_list = use_expand_question_list

        # Image transformations
        self.transform = ResizeLongestSide(image_size)  # for SAM
        self.masks_process_with_clip = masks_process_with_clip
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])  # for CLIP

        # Load templates
        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE
        self.correct_answer_list = CORRECT_ANSWER_TEMPLATE

        # Load data
        self.refer_seg_data = self.load_refer_seg_data(refer_seg_data, train_val_split)

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor, decoder_image_size) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = decoder_image_size - h
        padw = decoder_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def load_refer_seg_data(self, refer_seg_data, train_val_split):
        """Loads the refer segmentation data."""
        data_dir = os.path.join(self.base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split("||")
        refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            split_by = self.determine_split_by(ds)
            refer_api = REFER(data_dir, ds, split_by)
            ref_ids_train = refer_api.getRefIds(split=train_val_split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = self.prepare_dataset(ds, refer_api, images_ids_train, refs_train, data_dir)
            refer_seg_data[ds] = refer_seg_ds
        return refer_seg_data

    def determine_split_by(self, ds):
        """Determines the split type based on the dataset."""
        if ds == "refclef":
            return "unc"
        elif ds in ["refcocog", "R-refcocog"]:
            return "umd_exclude_unified"
        elif ds in ["fprefcocog", "fprefcoco", "fprefcoco+"]:
            return "berkeley_exclude_unified"
        return "unc_exclude_unified"

    def prepare_dataset(self, ds, refer_api, image_ids, refs, data_dir):
        """Prepares the dataset for a given segmentation data source."""
        refer_seg_ds = {"images": [], "annotations": refer_api.Anns}
        for item in refer_api.loadImgs(image_ids):
            item = item.copy()
            item["file_name"] = self.get_image_path(ds, item, data_dir)
            refer_seg_ds["images"].append(item)
        img2refs = {}
        for ref in refs:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref]
        refer_seg_ds["img2refs"] = img2refs

        print(f"Dataset {ds} (refs {self.determine_split_by(ds)}) (train split) has {len(refer_seg_ds['images'])} images and {len(refer_seg_ds['annotations'])} annotations.")
        return refer_seg_ds

    def get_image_path(self, ds, item, data_dir):
        """Returns the correct image path based on the dataset."""
        if ds == "refclef":
            return os.path.join(data_dir, "images/saiapr_tc-12", item["file_name"])
        return os.path.join(data_dir, "images/mscoco/images/train2014", item["file_name"])

    def select_dataset_and_image(self):
        """Selects a random dataset and an image from it."""
        ds = random.choice(self.refer_seg_ds_list)
        refer_seg_ds = self.refer_seg_data[ds]
        images, annotations, img2refs = refer_seg_ds["images"], refer_seg_ds["annotations"], refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        return ds, image_info, refs, annotations

    def process_referring_expressions(self, refs):
        # Load referring expression info.
        sents = []
        gt_sents = []
        ann_ids = []
        exists = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
                gt_sents.append(sent.get("gt_sent", ""))
                if "is_false_premise" in sent:
                    exists.append(not sent["is_false_premise"])
                elif "exist" in sent:
                    exists.append(sent["exist"])
                else:
                    exists.append(True)
        
        sample_size = min(len(sents), self.num_classes_per_sample)
        sampled_inds = random.sample(range(len(sents)), sample_size) if len(sents) >= self.num_classes_per_sample else range(len(sents))
        
        # Sampling process
        sampled_Q_sents = [sents[ind] for ind in sampled_inds]
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_exists = [exists[ind] for ind in sampled_inds]
        sampled_A_sents = [gt_sents[ind] for ind in sampled_inds]
        return sampled_Q_sents, sampled_A_sents, sampled_ann_ids, sampled_exists

    def create_conversations(self, ds, Q_sents, A_sents, exists):
        # Load conversations and Q/A
        conversations = []
        questions = []
        answers = []
        
        # Create seg token string
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token) 
        seg_token = '[SEG]' if self.seg_token_num == 1 else seg_token
        
        for idx, text in enumerate(Q_sents):
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            corrected_sentence = A_sents[idx].strip()
            text = text.strip()
            assert len(text.split("||")) == 1

            if exists[idx] is True:
                answer_template = random.choice(self.answer_list)
                answer = answer_template.format(class_name=text.lower())
                # Replace [SEG] with proper seg token
                if self.seg_token_num == 1:
                    answer = answer.replace('[SEG]', seg_token)
                else:
                    answer = answer.replace('[SEG]', seg_token)
                answers.append(answer)
            else:
                # false premise correction
                if ds in ["fprefcocog", "fprefcoco", "fprefcoco+"]:
                    answer_template = random.choice(self.correct_answer_list)
                    answer = answer_template.format(
                        class_name=text.lower(), gt_name=corrected_sentence.lower()
                    )
                else:
                    answer_template = random.choice(self.neg_answer_list)
                    answer = answer_template.format(class_name=text.lower())
                # Note: Don't replace with REJ tokens for false premise cases
                answers.append(answer)
        
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], questions[idx])
            conv.append_message(conv.roles[1], answers[idx])
            conversations.append(conv.get_prompt())
        return conversations

    def load_segmentation_masks(self, image_info, annotations, ann_ids, exists):
        # Load segmentation masks
        masks = []
        for i, ann_id in enumerate(ann_ids):
            if isinstance(ann_id, list):
                combined_mask = create_zero_mask(image_info["height"], image_info["width"]) 
                if -1 not in ann_id: # valid annotations
                    for ann_id_i in ann_id:
                        combined_mask |= process_annotation(annotations[ann_id_i], image_info)
                m = combined_mask
            else:
                m = process_annotation(annotations[ann_id], image_info)
            
            # If object doesn't exist, create empty mask
            if exists[i] is False:
                m = np.zeros_like(m)
            masks.append(m)
        
        if len(masks) == 0:
            masks = np.zeros((0, image_info["height"], image_info["width"]))
        else:
            masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks.astype(np.uint8))
        return masks

    def __getitem__(self, idx):
        # get one sample
        ds, image_info, refs, annotations = self.select_dataset_and_image()
        
        # Load and preprocess image
        image_path = image_info["file_name"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # preprocess image for clip
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        
        # load referring expression
        Q_sents, A_sents, ann_ids, exists = self.process_referring_expressions(refs)
        
        # create conversation Q/A (convert it to LLaVA type)
        conversations = self.create_conversations(ds, Q_sents, A_sents, exists)
        
        # load segmentation masks
        masks = self.load_segmentation_masks(image_info, annotations, ann_ids, exists)
        
        seg_count = ' '.join(conversations).count('[SEG') / self.seg_token_num
        valid_masks = sum(1 for j in range(masks.shape[0]) if masks[j].sum() > 0)
        if valid_masks != seg_count: return self.__getitem__(0)
        
        # preprocess image for SAM
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)
        
        # Create label tensor
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label if len(masks) > 0 else torch.ones(self.img_size, self.img_size) * self.ignore_label
        
        # Process masks with CLIP if needed
        if self.masks_process_with_clip:
            mask_shape = image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)
        
        return (
            image_path,                 # filename
            image,                      # raw image (for SAM)
            image_clip,                 # image clip feature (for LMMs)
            conversations,              # QA
            masks,                      # segmentation GT
            label,                      # label tensor
            resize,                     # sam input shape
            clip_resize,                # clip input shape  
            Q_sents,                    # questions
            [Q_sents],                  # sampled_classes (keeping original structure)
            False,                      # use_assign_list
            False                       # inference
        )


def transform_mask(masks, size):
    height, width = masks.shape[-2:]
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_shape = (new_long, new_short) if width <= height else (new_short, new_long)
    masks = F.interpolate(masks[None].float(), size=new_shape, mode="nearest")[0].bool()

    orig_height, orig_width = new_shape
    crop_height, crop_width = size, size
    crop_height, crop_width = int(crop_height), int(crop_width)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
    # if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
    masks = masks[..., top:bottom, left:right]

    return masks


if __name__ == "__main__":
    import random, numpy as np, torch
    import transformers
    random.seed(42)
    np.random.seed(42) 
    torch.manual_seed(42)
    import torch.utils.data
    from torch.utils.data import DataLoader
    
    def custom_collate_fn(batch):
        image_paths = [item[0] for item in batch]
        images = torch.stack([item[1] for item in batch])
        image_clips = torch.stack([item[2] for item in batch])
        conversations = [item[3] for item in batch]
        masks = [item[4] for item in batch]  
        labels = [item[5] for item in batch]  
        resizes = [item[6] for item in batch]
        clip_resizes = [item[7] for item in batch]
        questions = [item[8] for item in batch]
        sampled_sents = [item[9] for item in batch]
        use_assign_lists = [item[10] for item in batch]
        inferences = [item[11] for item in batch]
    
        return {
            'image_paths': image_paths,
            'images': images,
            'image_clips': image_clips,
            'conversations': conversations,
            'masks': masks,
            'labels': labels,
            'resizes': resizes,
            'clip_resizes': clip_resizes,
            'questions': questions,
            'sampled_sents': sampled_sents,
            'use_assign_lists': use_assign_lists,
            'inferences': inferences
        }

    version = '../dataset_sesame/LLaVA-7B-Lightening-v1-1'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    ret_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids
    
    seg_token_num = 6
    dataset = fpReferSegDataset(
        base_image_dir='../dataset_sesame', 
        tokenizer=tokenizer, 
        vision_tower='../dataset_sesame/clip-vit-large-patch14-336',
        # refer_seg_data="fprefcoco||fprefcoco+||fprefcocog",
        refer_seg_data="R-refcoco||R-refcoco+||R-refcocog",
        seg_token_num=seg_token_num,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  
        shuffle=False, 
        num_workers=0,  
        collate_fn=custom_collate_fn 
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"DataLoader size: {len(dataloader)}")

    for batch_idx, batch_data in enumerate(dataloader):
        
        for i, (conversation, masks) in enumerate(zip(batch_data['conversations'], batch_data['masks'])):
            print('==='*60)
            conv_text = ' '.join(conversation)
            print(conv_text)
            seg_count = conv_text.count('[SEG') / seg_token_num
            rej_count = conv_text.count('[REJ') / seg_token_num
            valid_masks = sum(1 for j in range(masks.shape[0]) if masks[j].sum() > 0)
            empty_masks = (masks.shape[0] - valid_masks)
            
            check1 = valid_masks == seg_count 
            check2 = masks.shape[0] == seg_count + rej_count 
            check3 = empty_masks == rej_count 
            
            if not (check1 and check2 and check3):
                print(batch_data['image_paths'][i])
                print(f"Sample {i}: SEG={seg_count}, REJ={rej_count}, ValidMasks={valid_masks}, EmptyMasks={empty_masks}, TotalMasks={masks.shape[0]}")
                print(f"  Check: valid==SEG({check1}), total==SEG+REJ({check2}), empty==REJ({check3})")
        
        if batch_idx >= 10:  # Only test first few batches
            break