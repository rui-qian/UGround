import glob
import json
import os
from queue import Empty
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide, ResizeShortestSide

from .data_processing import get_mask_from_json, get_mask_from_json_v2
from .reason_seg_dataset import ReasonSegDataset
from .reason_seg_plus_dataset import ReasonSegPlusDataset
from .refer import REFER
from .grefer import G_REFER
from .refzom import REFZOM_REFER
from .refer_seg_dataset import ReferSegDataset
from .fp_refer_seg_dataset import fpReferSegDataset
from .sem_seg_dataset import SemSegDataset

from .vqa_dataset import VQADataset
from .multi_reason_seg_dataset import MultiReasonSegDataset

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

from model.llava import conversation as conversation_lib
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

def filter_reason_seg_images_by_query_type(base_image_dir, ds, split):

    all_images = glob.glob(
        os.path.join(base_image_dir, "reason_seg", ds, split.split("_")[0], "*.jpg")
    )
    filter_strategy = {
        "test": lambda: all_images,
        "test_overall": lambda: all_images,
        "test_longquery": lambda: _filter_images_by_sentence_type(all_images, True),
        "test_shortquery": lambda: _filter_images_by_sentence_type(all_images, False),
    }

    return filter_strategy.get(split, lambda: all_images)()

def _filter_images_by_sentence_type(image_paths, target_is_sentence):
    def check_image(image_path):
        json_path = image_path.replace(".jpg", ".json")
        try:
            with open(json_path, "r") as f:
                return json.load(f).get("is_sentence", False) == target_is_sentence
        except:
            return False

    return [img for img in image_paths if os.path.exists(img.replace(".jpg", ".json")) and check_image(img)]

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    clip_resize_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    multi_reasons = []
    sam_mask_shape_list = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        clip_resize,
        questions,
        sampled_classes,
        multi_reason,
        inference,

    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        clip_resize_list.append(clip_resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        multi_reasons.append(multi_reason)
        sam_mask_shape_list.append([resize, (masks.shape[1], masks.shape[2])])
    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.conv_templates['chatml'].copy() if conv_type == "chatml" else conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1" or "chatml":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    # print(conv)
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if conv.sep2 not in conversation:
            break
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if conv_type == "chatml":
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(rou+sep, tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(rou+sep).input_ids) - 2

                if i == 0:
                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                # cur_len += round_len
                    
            else:
                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
        if conv_type == "chatml":
            cur_len = total_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )
        # print(tokenizer.model_max_length, cur_len, total_len)
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "clip_resize_list": clip_resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "multi_reason_list": multi_reasons,
        "sam_mask_shape_list": sam_mask_shape_list,
    }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        neg_refer_seg_data="R-refcoco||R-refcoco+||R-refcocog",
        correct_refer_seg_data="fprefcoco||fprefcoco+||fprefcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        reason_seg_plus_data="instance_seg||cot||conversations||caption",
        multi_reason_seg_data="MultiReasonSeg|train",
        explanatory=0.1,
        seg_token_num=1,
        num_classes_per_question=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
        negative_sampling_weight=-1
    ):
        self.pad_train_clip_images = pad_train_clip_images
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        self.seg_token_num = seg_token_num
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
                        negative_sampling_weight,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
                    )
                )
            elif dataset == "neg_refer_seg":
                self.all_datasets.append(
                    fpReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        neg_refer_seg_data,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
                    )
                )
            elif dataset == "correct_refer_seg":
                self.all_datasets.append(
                    fpReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        correct_refer_seg_data,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
                    )
                )            
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,

                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
                    )
                )
            elif dataset == "reason_seg_plus":
                self.all_datasets.append(
                    ReasonSegPlusDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_plus_data,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
                    )
                )
            elif dataset == "multi_reason_seg":
                self.all_datasets.append(
                    MultiReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        multi_reason_seg_data,
                        explanatory,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        return data[idx] #data[0]


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        seg_token_num=1,
        pad_val_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',

    ):
       
        self.seg_token_num=seg_token_num
        self.base_image_dir = base_image_dir
        self.pad_val_clip_images = pad_val_clip_images
        self.masks_process_with_clip = masks_process_with_clip
        self.multiseg_inference = False
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            self.images = filter_reason_seg_images_by_query_type(self.base_image_dir, ds, split)
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
                        
            if 'multi' in ds:
                self.multiseg_inference = True
                ds = ds.split('multi')[-1]

            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"
            
            if ds == "grefcoco":
                refer_api = G_REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
            elif ds == 'refzom':
                refer_api = REFZOM_REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds)
            else:
                refer_api = REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)

            # refer_api = REFER(self.base_image_dir+'/refer_seg/', ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "refer_seg/images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "refer_seg/images/mscoco/images/train2014",
                        item["file_name"],
                    )
                elif ds == "refzom":
                    # Use REFZOM_REFER's smart path resolution
                    item["file_name"] = refer_api.get_image_path(item["file_name"])
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg" if not self.multiseg_inference else "multi_refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size) 
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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
       
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]
    
        # preprocess image for clip
        if self.pad_val_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            # print("self.clip_image_processor.size['shortest_edge']:", self.clip_image_processor.size['shortest_edge'])
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                # grefcoco multiple annid start
                if self.ds in ['grefcoco', 'refzom']:
                    no_target = ann_id == [-1] if self.ds == 'grefcoco' else ann_id == []
                    if no_target: # no target
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    elif len(ann_id) > 1: # multi target / already merged ?
                        m = []
                        for sub_ann_id in ann_id:
                            sub_mask_info = annotations[sub_ann_id]['segmentation']
                            if len(sub_mask_info) == 0:
                                sub_m = np.zeros((image_info["height"], image_info["width"], 1))
                            else:
                                if isinstance(sub_mask_info, dict):
                                    if isinstance(sub_mask_info["counts"], list):
                                        # convert to compressed RLE
                                        rle = mask.frPyObjects(sub_mask_info, image_info["height"], image_info["width"])
                                else:
                                    # filter out invalid polygons (< 3 points)
                                    polygons = [poly for poly in sub_mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                    if len(polygons) == 0:
                                        continue  # ignore this instance
                                    rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                                sub_m = mask.decode(rle)
                                if sub_m.ndim < 3:
                                    assert sub_m.ndim == 2
                                    sub_m = sub_m[..., np.newaxis]
                            sub_m = np.sum(sub_m, axis=2)
                            m.append(sub_m)
                        m = np.sum(m, axis=0)[..., np.newaxis]
                    else:
                        assert len(ann_id) == 1 and ann_id[0] != -1
                        mask_info = annotations[ann_id[0]]['segmentation']
                        if len(mask_info) == 0:
                            m = np.zeros((image_info["height"], image_info["width"], 1))
                        else:
                            if isinstance(mask_info, dict):
                                if isinstance(mask_info["counts"], list):
                                    # convert to compressed RLE
                                    rle = mask.frPyObjects(mask_info, image_info["height"], image_info["width"])
                            else:
                                # filter out invalid polygons (< 3 points)
                                polygons = [poly for poly in mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                if len(polygons) == 0:
                                    continue  # ignore this instance
                                rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                            m = mask.decode(rle)
                            if m.ndim < 3:
                                assert m.ndim == 2
                                m = m[..., np.newaxis]
                    m = np.sum(m, axis=2)
                    masks.append(m)
                else:
                    ann = annotations[ann_id]
                    if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    else:
                        if type(ann["segmentation"][0]) == list:  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"],
                                image_info["height"],
                                image_info["width"],
                            )
                        else:
                            rle = ann["segmentation"]
                            for i in range(len(rle)):
                                if not isinstance(rle[i]["counts"], bytes):
                                    rle[i]["counts"] = rle[i]["counts"].encode()
                        m = mask.decode(rle)
                    m = np.sum(
                        m, axis=2
                    )  # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = m.astype(np.uint8)  # convert to np.uint8
                    masks.append(m)
        else:
            masks = [mask_json]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        
        _seg = "[SEG]" if self.seg_token_num == 1 else ' '.join(["[SEG{}]".format(i) for i in range(self.seg_token_num)])
        _rej = "[REJ]" if self.seg_token_num == 1 else ' '.join(["[REJ{}]".format(i) for i in range(self.seg_token_num)])
        multi_sample_num = [6, 5, 4]
        multi_sample_index = 0
        if self.data_type == 'refer_seg':
            conv.messages = []
            stripped_refers = [s.strip().strip('.') for s in sampled_sents]
            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN + "\n What are " +
                ", ".join(stripped_refers) + 
                "in this image? Please output segmentation masks."
            )
            ref_w_segs = []
            assert len(stripped_refers) == len(masks)
            for t_idx, t in enumerate(stripped_refers):
                formatted_text = f"{t}:{_rej}" if masks[t_idx].sum() < 1.0 else f"{t}:{_seg}"
                ref_w_segs.append(formatted_text)
            conv.append_message(
                conv.roles[1],
                "Sure," + ", ".join(ref_w_segs) + "."
            )
            conversations.append(conv.get_prompt())
        else:
            i = 0
            while i < len(sampled_sents):
                conv.messages = []
                if self.multiseg_inference:
                    sample_num = multi_sample_num[multi_sample_index]
                    texts = [sampled_sents[k].strip() for k in range(i, i+sample_num)] if len(sampled_sents) - i >= sample_num else [sampled_sents[k].strip() for k in range(i, len(sampled_sents))]
                    text = ', '.join(texts[:-1]) + ' and {}'.format(texts[-1]) if len(texts) > 1 else texts[0]
                else:
                    text = sampled_sents[i].strip()
                if is_sentence:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n {} Please output segmentation mask.".format(text),
                    )
                    
                    conv.append_message(conv.roles[1], "{}.".format(_seg))
                else:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n What is {} in this image? Please output segmentation mask.".format(
                            text
                        ),
                    )
                    if self.multiseg_inference:
                        answer = [_seg] * len(texts)
                        answer = ', '.join(answer[:-1]) + ' and ' + answer[-1] + '.' if len(answer) > 1 else answer[0]
                        conv.append_message(conv.roles[1], answer)
                    else:
                        conv.append_message(conv.roles[1], "{}.".format(_seg))
                conversations.append(conv.get_prompt())
                if self.multiseg_inference:
                    i += sample_num
                    multi_sample_index = (multi_sample_index + 1) % len(multi_sample_num)
                else:
                    i += 1

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks.astype(np.uint8))
        # masks = torch.from_numpy(masks.astype(np.uint8)).bool().byte()  # align with GSVA 
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

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
            labels,
            resize,
            clip_resize,
            None,
            None,
            False,
            inference,
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



from .qa_template import SHORT_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE, NEG_ANSWER_TEMPLATE, CORRECT_ANSWER_TEMPLATE, LONG_QUESTION_TEMPLATE, LONG_ANSWER_TEMPLATE
from .utils import replace_image_tokens, tokenize_and_pad, handle_conversation_specifics

def collate_fn_val(batch, tokenizer=None, use_mm_start_end=True, padding="right"):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    exists_list = []
    ref_id_list = []
    sent_id_list = []
    sam_mask_shape_list = []
    clip_resize_list = []
    offset_list = [0]
    cnt = 0
    for (image_path, images, images_clip, conversations,
            masks, sam_mask_shape, clip_resize, exists, ref_id, sent_id) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        sam_mask_shape_list.append(sam_mask_shape)
        clip_resize_list.append(clip_resize)
        cnt += len(conversations)
        offset_list.append(cnt)
        exists_list.append(exists)
        ref_id_list.append(ref_id)
        sent_id_list.append(sent_id)

    # Replace <image> token if use_mm_start_end is True
    if use_mm_start_end:
        conversation_list = replace_image_tokens(conversation_list)

    # Tokenization and padding of input IDs
    input_ids, attention_masks = tokenize_and_pad(conversation_list, tokenizer, padding=padding)

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": None,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "sam_mask_shape_list": sam_mask_shape_list,
        "clip_resize_list": clip_resize_list,
        "offset": torch.LongTensor(offset_list),
        "inference": True,
        "conversation_list": conversation_list,
        "exists": exists_list,
        "ref_ids": ref_id_list,
        "sent_ids": sent_id_list,
    }

collate_fn_test = collate_fn_val

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

class TestReferDataset(ReferSegDataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        train_test_split="val",
        datasetname="fprefcoco",
        pad_val_clip_images=False,
        use_val_mode=True,
        use_test_mode=False,
        preprocessor_config='',
    ):
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.pad_val_clip_images = pad_val_clip_images
        self.use_val_mode = use_val_mode
        self.use_test_mode = use_test_mode

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE
        self.correct_answer_list = CORRECT_ANSWER_TEMPLATE
        # Load dataset
        self.ds = ds = datasetname
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size) 
        self.clip_image_processor = vision_tower if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])


        data_dir = os.path.join(self.base_image_dir, "refer_seg")
        split_by = self.determine_split_by(ds)
        refer_api = REFER(data_dir, ds, split_by)
        ref_ids_test = refer_api.getRefIds(split=train_test_split)
        images_ids_test = refer_api.getImgIds(ref_ids=ref_ids_test)
        refs_test = refer_api.loadRefs(ref_ids=ref_ids_test)
        self.test_dataset = self.prepare_dataset(ds, refer_api, images_ids_test, refs_test, data_dir)
        print("data length = ", len(self.test_dataset["images"]))

    def __len__(self):
        return len(self.test_dataset["images"])

    def select_dataset_and_image(self, idx):
        """Selects a random dataset and an image from it."""
        refer_seg_ds = self.test_dataset
        images, annotations, img2refs = refer_seg_ds["images"], refer_seg_ds["annotations"], refer_seg_ds["img2refs"]
        
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        return self.ds, image_info, refs, annotations
    
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

    def process_referring_expressions(self, refs):
        # Load referring expression info.
        Q_sents = []
        gt_sents = []
        ann_ids = []
        ref_ids = []
        sent_ids = []
        exists = []
        for ref in refs:
            for idx, sent in enumerate(ref["sentences"]):
                text = sent["sent"]
                Q_sents.append(text)
                gt_sents.append(sent.get("gt_sent", ""))
                ann_ids.append(ref["ann_id"])
                ref_ids.append(ref["ref_id"])
                sent_ids.append(idx)
                if "is_false_premise" in sent:
                    exists.append(not sent["is_false_premise"])
                elif "exist" in sent:
                    exists.append(sent["exist"])
                else:
                    exists.append(True)
        return Q_sents, gt_sents, ann_ids, exists, ref_ids, sent_ids
    
    def load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # preprocess image for clip
        if self.pad_val_clip_images:
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
        sam_input_shape = tuple(image.shape[:2])
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)
        
        return image, image_clip, sam_input_shape, clip_resize
    
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
    
    def create_conversations(self, ds, Q_sents, A_sents, exists, load_answer=True):
        # Load conversations and Q/A
        conversations = []
        questions = []
        answers = []
        for idx, text in enumerate(Q_sents):

            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            corrected_sentence = A_sents[idx].strip()
            text = text.strip()
            assert len(text.split("||")) == 1

            if exists[idx] is True:
                answer_template = random.choice(self.answer_list)
                answers.append(answer_template.format(class_name=text.lower()))
            else:
                
                # false premise correction
                if ds in ["fprefcocog", "fprefcoco", "fprefcoco+"]:
                    answer_template = random.choice(self.correct_answer_list)
                    answers.append(
                        answer_template.format(
                            class_name=text.lower(), gt_name=corrected_sentence.lower()
                        )
                    )
                else:
                    answer_template = random.choice(self.neg_answer_list)
                    answers.append(answer_template.format(class_name=text.lower()))
        
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], questions[idx])
            if load_answer is True:
                conv.append_message(conv.roles[1], answers[idx])
            else:
                conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt())
        return conversations
    
    def load_segmentation_masks(self, image_info, annotations, sam_input_shape, ann_ids, exists, include_nonexist=False):
        # Load segmentation masks
        masks = []
        for i, ann_id in enumerate(ann_ids):
            if include_nonexist is False and exists[i] is False:
                continue
            if isinstance(ann_id, list):
                combined_mask = create_zero_mask(image_info["height"], image_info["width"]) 
                if -1 not in ann_id: # valid annotations
                    for ann_id_i in ann_id:
                        combined_mask |= process_annotation(annotations[ann_id_i], image_info)
                m = combined_mask
            else:
                m = process_annotation(annotations[ann_id], image_info)
            # If include nonexist is True will also include a blank mask (for test usage)
            if exists[i] is False:
                m = np.zeros_like(m)
            masks.append(m)
        if len(masks) == 0:
            masks = np.zeros((0, *sam_input_shape))  # original input shape
        else:
            masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks)
        return masks

    def __getitem__(self, idx):
        # get one sample
        ds, image_info, refs, annotations = self.select_dataset_and_image(idx)
        # Load images and clip features
        image, image_clip, sam_input_shape, clip_resize = self.load_and_preprocess_image(image_info["file_name"])
        # load referring expression
        Q_sents, A_sents, ann_ids, exists, ref_ids, sent_ids = self.process_referring_expressions(refs)
        # create conversation Q/A (convert it to LLaVA type)
        if self.use_val_mode:
            conversations = self.create_conversations(ds, Q_sents, A_sents, exists, load_answer=True)
        if self.use_test_mode: # for test mode
            conversations = self.create_conversations(ds, Q_sents, A_sents, exists, load_answer=False)
        # load segmentation masks
        masks = self.load_segmentation_masks(image_info, annotations, sam_input_shape, ann_ids, exists, include_nonexist=True)
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        # print(masks.shape[1] == sam_mask_shape[2] and masks.shape[2] == sam_mask_shape[3], flush=True)
        return (
            image_info["file_name"],    # filename
            image,                      # raw image (for SAM)
            image_clip,                 # image clip feature (for LMMs)
            conversations,              # QA
            masks,                      # segmentation GT
            sam_mask_shape,             # input / output shape for SAM
            clip_resize,                 # clip resize
            exists,                     # object existence
            ref_ids,                    # ref id (useless now)
            sent_ids,                    # sent id (useless now)
        )


class TestReasoningDataset(ReasonSegDataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        train_test_split="val",
        pad_val_clip_images=False,
        datasetname="ReasonSeg",
        use_val_mode=True,
        use_test_mode=False,
        preprocessor_config='',
    ):
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir
        self.use_val_mode = use_val_mode
        self.use_test_mode = use_test_mode
        self.pad_val_clip_images = pad_val_clip_images
        
        # Load dataset
        self.ds = ds = datasetname
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size) 
        self.clip_image_processor = vision_tower if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])


        SHORT_QUESTION_TEMPLATE = [
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + "What is {class_name} in this image? Please output segmentation mask."
        ]
        LONG_QUESTION_TEMPLATE = [
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + "{sent} Please output segmentation mask.",
        ]
        LONG_ANSWER_TEMPLATE = ["Sure, the segmentation result is [SEG]."]

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.long_question_list = LONG_QUESTION_TEMPLATE
        self.answer_list = LONG_ANSWER_TEMPLATE
        # load dataset
        reason_seg_data, splits = datasetname, train_test_split
        if self.ds == "fpReasonSeg": reason_seg_data = reason_seg_data.replace("fp", "")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

    def __len__(self):
        return len(self.reason_seg_data[0])
    
    def load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # preprocess image for clip
        if self.pad_val_clip_images:
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
        sam_input_shape = tuple(image.shape[:2])
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)
        
        return image, image_clip, sam_input_shape, clip_resize
    
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
        
        images, jsons = self.reason_seg_data
        image_path = images[idx]
        json_path = jsons[idx]
        
        # Load images and clip features
        image, image_clip, sam_input_shape, clip_resize = self.load_and_preprocess_image(image_path)
        # Get sents and segmentation maps
        img = cv2.imread(image_path)[:, :, ::-1]
        mask, sents, fp_qa, is_sentence = get_mask_from_json_v2(json_path, img)
        # Sampling
        # assert len(sents) == len(fp_qa) == 1

        # Create Q/A Data
        conversations = []
        # True premise question
        conv = conversation_lib.default_conversation.copy()
        if is_sentence:
            question_template = random.choice(self.long_question_list)
            Q_sent = question_template.format(sent=sents[0])
        else:
            question_template = random.choice(self.short_question_list)
            Q_sent = question_template.format(class_name=sents[0].lower())
        conv.append_message(conv.roles[0], Q_sent)
        if self.use_val_mode:
            conv.append_message(conv.roles[1], random.choice(self.answer_list))
        if self.use_test_mode: 
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        
        debug = False # set true to align Lisa
        if debug:
            sampled_sents = [sents[0]]
            conversations = []
            conv = conversation_lib.default_conversation.copy()
            i = 0
            while i < len(sampled_sents):
                conv.messages = []
                text = sampled_sents[i].strip()
                if is_sentence:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n {} Please output segmentation mask.".format(text),
                    )
                    conv.append_message(conv.roles[1], "[SEG].")
                else:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n What is {} in this image? Please output segmentation mask.".format(
                            text
                        ),
                    )
                    conv.append_message(conv.roles[1], "[SEG].")
                conversations.append(conv.get_prompt())
                i += 1
        if self.ds == "fpReasonSeg":
            conv = conversation_lib.default_conversation.copy()
            # False premise question
            if fp_qa[0][1] is True:
                question_template = random.choice(self.long_question_list)
                neg_Q_sent = question_template.format(sent=fp_qa[0][0])
            else:
                question_template = random.choice(self.short_question_list)
                neg_Q_sent = question_template.format(class_name=fp_qa[0][0])
            conv.append_message(conv.roles[0], neg_Q_sent)
            
            if self.use_val_mode:
                conv.append_message(conv.roles[1], fp_qa[0][2])
            if self.use_test_mode:
                conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt())
        
        # Exists and segmentation masks
        exists = [True, False] if self.ds == "fpReasonSeg" else [True]
        masks = [ mask, np.zeros_like(mask).astype(np.float32) ] if self.ds == "fpReasonSeg" else [mask]
        masks = torch.from_numpy(np.stack(masks, axis=0))
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        ref_ids = [int(idx), int(idx)]
        sent_ids = [0, 1]
        return (
            image_path,         # filename
            image,              # raw image (for SAM)
            image_clip,         # image clip feature (for LMMs)
            conversations,      # QA
            masks,              # segmentation GT
            sam_mask_shape,     # input / output shape for SAM
            clip_resize,         # clip resize
            exists,             # object existence
            ref_ids,            # ref id (useless now)
            sent_ids,            # sent id (useless now)
        )
