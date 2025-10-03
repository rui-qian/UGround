import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide, ResizeShortestSide
# from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, SINGLE_ANSWER_LIST, MULTI_ANSWER_LIST, EXPAND_QUESTION_LIST
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, SINGLE_ANSWER_LIST, MULTI_ANSWER_LIST, EXPAND_QUESTION_LIST
from .qa_template import SHORT_QUESTION_TEMPLATE, NEG_ANSWER_TEMPLATE, CORRECT_ANSWER_TEMPLATE


def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("dataloaders/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("dataloaders/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(torch.utils.data.Dataset):
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
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
        negative_sampling_weight=-1
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
        

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.single_answer_list = SINGLE_ANSWER_LIST
        self.multi_answer_list = MULTI_ANSWER_LIST   
        self.seg_token_num = seg_token_num
        self.num_classes_per_question = num_classes_per_question

        self.negative_sampling_weight = negative_sampling_weight
        if self.negative_sampling_weight > 0:
            self.neg_answer_list = NEG_ANSWER_TEMPLATE
            self.correct_answer_list = CORRECT_ANSWER_TEMPLATE
            pos_weight = 1.0 - negative_sampling_weight
            neg_denial_weight = negative_sampling_weight * 0.5
            neg_correction_weight = negative_sampling_weight * 0.5
            self.choices = ["True_Premise", "False_Premise_Denial", "False_Premise_Correction"]  
            self.weights = [pos_weight, neg_denial_weight, neg_correction_weight]

        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])
        if use_expand_question_list:
            self.short_question_list.extend(SHORT_QUESTION_TEMPLATE)
            self.short_question_list.extend(EXPAND_QUESTION_LIST)

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

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

    def generate_negative_samples(self, ds, positive_anns=None, positive_class_ids=None, class_map=None):
        """
        独立的负例生成函数
        Args:
            ds: 数据集名称
            positive_anns: 正例annotations (for COCO-style datasets)
            positive_class_ids: 正例class IDs (for pixel-level datasets)
            class_map: 类别映射 (for COCO-style datasets)
        Returns:
            tuple: (negative_classes, negative_class_names)
        """
        if self.negative_sampling_weight < 0:
            return [], []
            
        if ds in ["paco_lvis", "pascal_part"]:
            all_cats = {cat["id"] for cat in class_map.values() if hasattr(class_map, 'values')}
            if hasattr(class_map, 'keys'):
                all_cats = set(class_map.keys())
            else:
                img_ids, coco_api = self.data2list[ds]
                all_cats = {cat["id"] for cat in coco_api.cats.values()}
            
            positive_cats = {ann["category_id"] for ann in positive_anns} if positive_anns else set()
            negative_cats = list(all_cats - positive_cats)
            
            neg_sample_size = min(len(negative_cats), self.num_classes_per_sample)
            if neg_sample_size > 0:
                neg_sampled_cats = random.sample(negative_cats, neg_sample_size)
                neg_class_names = []
                for cat_id in neg_sampled_cats:
                    if ds == "paco_lvis":
                        _, coco_api = self.data2list[ds]
                        sampled_cls = coco_api.cats[cat_id]['name'] if cat_id in coco_api.cats else str(cat_id)
                    elif ds == "pascal_part":
                        _, coco_api = self.data2list[ds]  
                        sampled_cls = coco_api.cats[cat_id]['name'] if cat_id in coco_api.cats else str(cat_id)
                    else:
                        sampled_cls = class_map[cat_id] if cat_id in class_map else str(cat_id)
                    
                    if isinstance(sampled_cls, tuple):
                        obj, part = sampled_cls
                        name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
                    else:
                        name = sampled_cls
                    neg_class_names.append(name)
                return neg_sampled_cats, neg_class_names
            else:
                return [], []
                
        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            all_class_ids = list(range(len(self.data2classes[ds])))
            negative_class_ids = list(set(all_class_ids) - set(positive_class_ids))
            
            neg_sample_size = min(len(negative_class_ids), self.num_classes_per_sample)
            if neg_sample_size > 0:
                neg_sampled_class_ids = random.sample(negative_class_ids, neg_sample_size)
                neg_class_names = [self.data2classes[ds][class_id] for class_id in neg_sampled_class_ids]
                return neg_sampled_class_ids, neg_class_names
            else:
                return [], []
        
        return [], []

    def create_negative_qa_pair(self, neg_class_name, pos_class_name=None, mode="denial"):
        """
        创建负例问答对
        Args:
            neg_class_name: 负例类别名称  
            pos_class_name: 正例类别名称 (用于correction模式)
            mode: "denial" 或 "correction"
        Returns:
            tuple: (question, answer)
        """
        question_template = random.choice(self.short_question_list)
        question = question_template.format(class_name=neg_class_name.lower())
        
        if mode == "denial":
            answer_template = random.choice(self.neg_answer_list)
            answer = answer_template.format(class_name=neg_class_name.lower())
        elif mode == "correction" and pos_class_name:
            answer_template = random.choice(self.correct_answer_list)
            answer = answer_template.format(
                class_name=neg_class_name.lower(), 
                gt_name=pos_class_name.lower()
            )
        else:
            answer_template = random.choice(self.neg_answer_list)
            answer = answer_template.format(class_name=neg_class_name.lower())
            
        return question, answer

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        use_negative = False
        mode_this_turn = "True_Premise"
        if self.negative_sampling_weight > 0:
            mode_this_turn = random.choices(self.choices, self.weights, k=1)[0]
            use_negative = (mode_this_turn != "True_Premise")

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]

            # preprocess image for clip
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
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            max_num_classes_per_sample = self.num_classes_per_question * self.num_classes_per_sample
            if len(anns) >= max_num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=max_num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)
            sampled_anns, sampled_classes = allocate_class(sampled_anns, sampled_classes, max_question_num=self.num_classes_per_sample, max_class_per_question=self.num_classes_per_question)

            neg_class_names = []
            if use_negative:
                _, neg_class_names = self.generate_negative_samples(ds, anns, None, coco_api.cats)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
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
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            max_num_classes_per_sample = self.num_classes_per_question * self.num_classes_per_sample
            if len(classes) >= max_num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=max_num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes
            _, sampled_classes = allocate_class(None, sampled_classes, max_question_num=self.num_classes_per_sample, max_class_per_question=self.num_classes_per_question)

            neg_class_names = []
            if use_negative:
                _, neg_class_names = self.generate_negative_samples(ds, None, unique_label, None)

        questions = []
        answers = []
        class_ids = []
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)
        
        if use_negative and len(neg_class_names) > 0:
            for i, sampled_classes_per_question in enumerate(sampled_classes):
                if i >= len(neg_class_names):
                    break
                    
                neg_class_name = neg_class_names[i]
                pos_class_names = sampled_classes_per_question
                
                if mode_this_turn == "False_Premise_Denial":
                    question, answer = self.create_negative_qa_pair(neg_class_name, mode="denial")
                elif mode_this_turn == "False_Premise_Correction":
                    pos_class_name = pos_class_names[0] if pos_class_names else "object"
                    question, answer = self.create_negative_qa_pair(neg_class_name, pos_class_name, mode="correction")
                
                questions.append(question)
                answers.append(answer)
        else:
            for sampled_classes_per_question in sampled_classes:
                target = ''
                _seg = []
                for i, sampled_cls in enumerate(sampled_classes_per_question):
                    text = sampled_cls
                    assert len(text.split("||")) == 1
                    if i == len(sampled_classes_per_question) - 1:
                        _seg.append('[SEG]') if self.seg_token_num == 1 else _seg.append(seg_token)
                        target = target + (' and '  + text) if i != 0 else target + text
                    elif i == 0:
                        target += text
                        _seg.append('[SEG]') if self.seg_token_num == 1 else _seg.append(seg_token)
                    else:
                        _seg.append('[SEG]') if self.seg_token_num == 1 else _seg.append(seg_token)
                        target += (', '  + text)

                    if ds in ["paco_lvis", "pascal_part"]:
                        continue
                    class_id = self.data2classes[ds].tolist().index(sampled_cls)
                    class_ids.append(class_id) 
                if len(_seg) > 1:
                    part1 = ', '.join(_seg[:-1])
                    part2 = ' and ' + _seg[-1]
                    _seg = part1 + part2 
                else:
                    _seg = _seg[0]
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=target.lower()))
                separate_answer = random.randint(0, 1)

                if len(sampled_classes_per_question) == 1:
                    choice_list = self.answer_list
                    answer_temp = random.choice(choice_list) if self.seg_token_num == 1 else random.choice(choice_list).replace('[SEG]', seg_token)
                    answer_temp = answer_temp.format(class_name=target.lower()) if "{class_name}" in answer_temp else answer_temp
                    answers.append(answer_temp)
                elif separate_answer:
                    target_answer = []
                    answer_temp = random.choice(self.single_answer_list) if self.seg_token_num == 1 else random.choice(self.single_answer_list).replace('[SEG]', seg_token)
                    for i, sampled_cls in enumerate(sampled_classes_per_question):
                        _answer_temp = answer_temp.format(class_name=sampled_cls.lower()) if "{class_name}" in answer_temp else answer_temp
                        target_answer.append(_answer_temp[:-1])
                    if len(target_answer) > 1:
                        part1 = ', '.join(target_answer[:-1])
                        part2 = ' and ' + target_answer[-1]
                        target_answer = part1 + part2 + '.'
                    else:
                        target_answer = target_answer[0] + '.'
                    answers.append(target_answer)
                else:
                    answer_temp = random.choice(self.multi_answer_list)
                    _answer_temp = answer_temp.format(class_name=target.lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                    answers.append(_answer_temp)

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        if ds in ["paco_lvis", "pascal_part"]:
            if use_negative:
                masks = torch.zeros((0, original_height, original_width)).long()
            else:
                masks = []
                for sampled_anns_per_question in sampled_anns:
                    for ann in sampled_anns_per_question:
                        try:
                            masks.append(coco_api.annToMask(ann))
                        except Exception as e:
                            print(e)
                            return self.__getitem__(0)

                masks = np.stack(masks, axis=0)
                masks = torch.from_numpy(masks.astype(np.uint8))
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        else:
            label = torch.from_numpy(label).long()
            if use_negative:
                masks = torch.zeros((0, label.shape[0], label.shape[1])).long()
            else:
                masks = []
                for class_id in class_ids:
                    masks.append(label == class_id)
                masks = torch.stack(masks, dim=0)
        
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
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize, # input for sam
            clip_resize,
            questions,
            sampled_classes,
            False,  # use_assign_list
            False   # inference
        )


def allocate_class(sampled_anns, sampled_ann_classes, max_question_num=3, max_class_per_question=3):
    if len(sampled_ann_classes) < max_question_num:
        max_question_num = len(sampled_ann_classes)
    sample_num = len(sampled_ann_classes)
    question_id = np.arange(max_question_num)
    class_counts = np.arange(max_question_num) * 0
    new_sampled_ann_ids = [[] for _ in range(max_question_num)] 
    new_sampled_ann_classes = [[] for _ in range(max_question_num)] 
    sample_ids = np.arange(sample_num)
    np.random.shuffle(sample_ids)
    for i in range(sample_num):
        if 0 in class_counts:
            choose_id = np.random.choice(np.where(class_counts == 0)[0], size=1)[0]
        else:
            choose_id = np.random.choice(np.where(class_counts < max_class_per_question)[0], size=1)[0]
        
        class_counts[choose_id] += 1
        sample_id = sample_ids[i]
        if sampled_anns is not None:
            new_sampled_ann_ids[choose_id].append(sampled_anns[sample_id])
        new_sampled_ann_classes[choose_id].append(sampled_ann_classes[sample_id])

    return new_sampled_ann_ids, new_sampled_ann_classes

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
    dataset = SemSegDataset(
        base_image_dir='../dataset_sesame', 
        tokenizer=tokenizer, 
        vision_tower='../dataset_sesame/clip-vit-large-patch14-336',
        sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        seg_token_num=seg_token_num,
        num_classes_per_question=3,
        num_classes_per_sample=3,
        negative_sampling_weight=0.5
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