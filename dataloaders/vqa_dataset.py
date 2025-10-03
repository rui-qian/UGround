import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide, ResizeShortestSide

# from .utils import DEFAULT_IMAGE_TOKEN
from .utils import DEFAULT_IMAGE_TOKEN

def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class VQADataset(torch.utils.data.Dataset):
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
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava_instruct_150k",
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
    ):
        self.pad_train_clip_images = pad_train_clip_images
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size) 
        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])
        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

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

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            # print("self.clip_image_processor.size['shortest_edge']:", self.clip_image_processor.size['shortest_edge'])
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv =  conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

    
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        if self.masks_process_with_clip:
            mask_shape =  image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            clip_resize,
            questions,
            sampled_classes,
            False,  # use_assign_list
            False   # inference
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


def center_crop_image(image, size):
    orig_height, orig_width = image.shape[:2]
    crop_height, crop_width = size, size
    crop_height, crop_width = int(crop_height), int(crop_width)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
    # if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
    image = image[top:bottom, left:right]

    return image


if __name__ == "__main__":
    import torch.utils.data
    import transformers
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
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    ret_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids
    seg_token_num = 6
    dataset = VQADataset(
        base_image_dir='../dataset_sesame', 
        tokenizer=tokenizer, 
        vision_tower='../dataset_sesame/clip-vit-large-patch14-336',
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
            conv_text = ' '.join(conversation)
            seg_count = conv_text.count('[SEG') / seg_token_num
            rej_count = conv_text.count('[REJ') / seg_token_num
            valid_masks = sum(1 for j in range(masks.shape[0]) if masks[j].sum() > 0)
            empty_masks = masks.shape[0] - valid_masks
            
            check1 = valid_masks == seg_count 
            check2 = masks.shape[0] == seg_count + rej_count 
            check3 = empty_masks == rej_count 
            
            # if not (check1 and check2 and check3):
            if not check1:
                print(batch_data['image_paths'][i])
                print(f"Sample {i}: SEG={seg_count}, REJ={rej_count}, ValidMasks={valid_masks}, EmptyMasks={empty_masks}, TotalMasks={masks.shape[0]}")
                print(f"  Check: valid==SEG({check1}), total==SEG+REJ({check2}), empty==REJ({check3})")

        pass
