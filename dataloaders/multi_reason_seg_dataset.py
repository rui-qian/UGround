import glob
import json
import os
import random
from unicodedata import category
from pycocotools import mask
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, PretrainedConfig
import transformers

from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
# from .conversation import get_default_conv_template

from .utils import (
    MR_SINGLE_ANSWER_LIST,
    MR_MULTI_ANSWER_LIST,
    ANSWER_LIST,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    EXPLANATORY_QUESTION_LIST,
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    EXPAND_LONG_QUESTION_LIST,
)
from .qa_template import LONG_QUESTION_TEMPLATE

class MultiReasonSegDataset(torch.utils.data.Dataset):
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
        multi_reason_seg_data="MultiReasonSeg|train",
        explanatory=0.1,
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False
    ):
        self.exclude_val = exclude_val
        self.multi_reason_seg_data = multi_reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.single_answer_list = MR_SINGLE_ANSWER_LIST
        self.multi_answer_list = MR_MULTI_ANSWER_LIST   
        self.seg_token_num = seg_token_num
        self.num_classes_per_question = num_classes_per_question
        
        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge']) 
        
        if use_expand_question_list:
            self.long_question_list.extend(LONG_QUESTION_TEMPLATE)
            self.long_question_list.extend(EXPAND_LONG_QUESTION_LIST)
        
        # print("___________self.single_answer_list:", self.single_answer_list)
        # print("___________self.multi_answer_list:", self.multi_answer_list)
                
        multi_reason_seg_data, split = multi_reason_seg_data.split("|")
        json_file_name = os.path.join(base_image_dir, "multi_reason_seg", multi_reason_seg_data, f'MUSE_{split}.json')
        with open(json_file_name, 'r') as f:
            reason_file = json.load(f)
        
        self.multi_reason_seg_data = reason_file
        print("number of multi_reason_seg samples(train split): ", len(reason_file))
    
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
        
        idx = random.randint(0, len(self.multi_reason_seg_data) - 1)
        image_info = self.multi_reason_seg_data[idx]
        if 'file_name' in image_info:
            image_root = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2014")
            image_path = os.path.join(image_root, image_info['file_name'])
        else:
            if 'train2017' in image_info['coco_url']:
                image_root = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2017")
                image_path = os.path.join(image_root, image_info['coco_url'].split('/')[-1])
            else:
                image_root = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/val2017")
                image_path = os.path.join(image_root, image_info['coco_url'].split('/')[-1])

        anns = image_info['ann_list']
        question = image_info['questions'] if 'questions' in image_info else None
        gt_answer = image_info['answers'] if 'answers' in image_info else None
        if question is not None:
            text_answers = image_info['text_answers'] if 'text_answers' in image_info else [None] * len(gt_answer)
        else:
            text_answers = None
        
        img = cv2.imread(image_path)
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_size = images.shape[:2]
        # preprocess images for clip
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(images)
            clip_resize = image_clip.shape[:2]
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
        else:
            image_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        
        images = self.transform.apply_image(images)  # preprocess images for sam
        resize = images.shape[:2]
        masks = []
        if len(anns) == 0:
            return self[0]

        category_ids = [ann['category_id'] for ann in anns]
        category_ids = list(set(category_ids))
        sampled_num = min(self.num_classes_per_sample, len(category_ids))
        sampled_category_ids = np.random.choice(category_ids, size=sampled_num, replace=False)

        sampled_sents = question
        sampled_answers = gt_answer
        sampled_masks = masks
        sample_text_answers = text_answers

        image_name = image_path.split("/")[-1]
        questions = []
        answers = []
        use_assign_list = []
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)

        if question is not None:
            for text, answer_list, text_answer in zip(sampled_sents, sampled_answers, sample_text_answers):
                # if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
                
                for answer in answer_list:
                    rle = mask.frPyObjects(answer["segmentation"], image_info["height"], image_info["width"])
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        # assert m.shape[-1] == 1, m.shape
                        m = np.sum(m, axis=2)  # so
                    m = m.astype(np.uint8)
                    masks.append(m)

                use_assign = False
                if text_answer is not None:
                    if text_answer.count('{seg}') != len(answer_list):
                        return self[0]
                    # Fix: Use replace instead of format to avoid IndexError from coordinate noise  
                    _text_answer = text_answer.replace('{seg}', '[SEG]') if self.seg_token_num == 1 else text_answer.replace('{seg}', seg_token)
                    answers.append(_text_answer)
                    use_assign_list.append(False)
                else:
                    target_list = [a['rephrased_name'] if (random.random() > 0.1 and 'rephrased_name' in a) else a['category_name'] for a in answer_list ]
                    target_answer = []
                    separate_answer = random.randint(0, 1)
                    _seg = ['[SEG]'] * len(target_list)
                    if len(target_list) > 1:
                        part1 = ', '.join(_seg[:-1])
                        part2 = ' and ' + _seg[-1]
                        _seg = part1 + part2 
                    else:
                        _seg = _seg[0]
                    
                    if separate_answer:
                        choice_list = self.single_answer_list
                        answer_temp = random.choice(choice_list) if self.seg_token_num == 1 else random.choice(choice_list).replace('[SEG]', seg_token)
                        use_assign = False if "{class_name}" in answer_temp else True
                        for i, sampled_cls in enumerate(target_list):
                            _answer_temp = answer_temp.format(class_name=sampled_cls) if "{class_name}" in answer_temp else answer_temp
                            target_answer.append(_answer_temp[:-1])
                        if len(target_answer) > 1:
                            part1 = ', '.join(target_answer[:-1])
                            part2 = ' and ' + target_answer[-1]
                            target_answer = part1 + part2 + '.'
                        else:
                            target_answer = target_answer[0] + '.'
                    else:
                        answer_temp = random.choice(self.multi_answer_list)
                        _answer_temp = answer_temp.format(class_name=', '.join(target_list).lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                        use_assign = False if "{class_name}" in answer_temp else True
                        _answer_temp = _answer_temp if self.seg_token_num == 1 else _answer_temp.replace('[SEG]', seg_token)
                        target_answer = _answer_temp

                    answers.append(target_answer)
                    use_assign_list.append(use_assign)
            
        else:
            for sampled_category_id in sampled_category_ids:
                question_template = random.choice(self.instance_question_list)
                category_names = self.lvis_name_dict[str(sampled_category_id)]
                category_name = random.choice(category_names)
                questions.append(question_template.format(class_name=category_name))
                answer_list = [ann for ann in anns if ann['category_id'] == sampled_category_id]
                for answer in answer_list:
                    rle = mask.frPyObjects(answer["segmentation"], image_info["height"], image_info["width"])
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        # assert m.shape[-1] == 1, m.shape
                        m = np.sum(m, axis=2)  # so
                    m = m.astype(np.uint8)
                    masks.append(m)

                target_list = [a['rephrased_name'] if random.random() > 0.1 else a['category_name'] for a in answer_list ]
                target_answer = []
                separate_answer = random.randint(0, 1)
                
                target_list = [a['rephrased_name'] if random.random() > 0.1 else a['category_name'] for a in answer_list ]
                target_answer = []
                separate_answer = random.randint(0, 1)
                _seg = ['[SEG]'] * len(target_list)
                if len(target_list) > 1:
                    part1 = ', '.join(_seg[:-1])
                    part2 = ' and ' + _seg[-1]
                    _seg = part1 + part2 
                else:
                    _seg = _seg[0]

                separate_answer = random.randint(0, 1)
                # if len(answer_list) == 1 or separate_answer:
                choice_list = self.single_answer_list
                answer_temp = random.choice(choice_list) if self.seg_token_num == 1 else random.choice(choice_list).replace('[SEG]', seg_token)
                use_assign = False if "{class_name}" in answer_temp else True
                for i, sampled_cls in enumerate(target_list):
                    _answer_temp = answer_temp.format(class_name=sampled_cls) if "{class_name}" in answer_temp else answer_temp
                    target_answer.append(_answer_temp[:-1])
                if len(target_answer) > 1:
                    part1 = ', '.join(target_answer[:-1])
                    part2 = ' and ' + target_answer[-1]
                    target_answer = part1 + part2 + '.'
                else:
                    target_answer = target_answer[0] + '.'
                

                answers.append(target_answer)
                use_assign_list.append(use_assign)
    
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1


        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous(), self.img_size)
        
        if len(sampled_masks) == 0:
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks.astype(np.uint8))
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        seg_count = ' '.join(conversations).count('[SEG') / self.seg_token_num
        valid_masks = sum(1 for j in range(masks.shape[0]) if masks[j].sum() > 0)
        if valid_masks != seg_count: return self.__getitem__(0) 

        if self.masks_process_with_clip:
            mask_shape =  image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)
        return (
            image_path,
            images,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            clip_resize,
            questions,
            sampled_sents,
            use_assign_list,
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
    dataset = MultiReasonSegDataset(
        base_image_dir='../dataset_sesame', 
        tokenizer=tokenizer, 
        vision_tower='../dataset_sesame/clip-vit-large-patch14-336',
        seg_token_num=seg_token_num,
        num_classes_per_question=3,
        num_classes_per_sample=3,
        multi_reason_seg_data="MultiReasonSeg|train",
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

