import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
import re
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide

from .qa_template import (
    SHORT_QUESTION_TEMPLATE, 
    SHORT_ANSWER_TEMPLATE, 
    NEG_ANSWER_TEMPLATE, 
    CORRECT_ANSWER_TEMPLATE,
    LONG_QUESTION_TEMPLATE
)
from .utils import MR_SINGLE_ANSWER_LIST

INSTRUCTIONAL_WORDS = {
    "annotate", "annotated", "annotates", "annotating", "can you", "circle", "circled",
    "circles", "circling", "could you", "detect", "detected", "detecting", "detects",
    "draw", "drawing", "draws", "drew", "find", "finding", "finds", "found", "gave",
    "generate", "generated", "generates", "generating", "give", "gives", "giving",
    "highlight", "highlighted", "highlighting", "highlights", "identified", "identifies",
    "identify", "identifying", "indicate", "indicated", "indicates", "indicating",
    "label", "labeled", "labeling", "labels", "locate", "located", "locates", "locating",
    "mark", "marked", "marking", "marks", "mask", "outline", "outlined", "outlines",
    "outlining", "output", "outputs", "outputted", "outputting", "please", "point out",
    "provide", "provided", "provides", "providing", "return", "returned", "returning",
    "returns", "segment", "segmentation", "segmented", "segmenting", "segments", "show",
    "showed", "showing", "shows", "trace", "traced", "traces", "tracing", "would you"
}

def create_zero_mask(height, width):
    return np.zeros((height, width), dtype=np.uint8)

def decode_segmentation(ann_segmentation, height, width):
    if type(ann_segmentation[0]) == list:
        if len(ann_segmentation) > 0 and type(ann_segmentation[0][0]) == list:
            polygons = [np.array(poly, dtype=np.float64).flatten().tolist() for poly in ann_segmentation]
        else:
            poly_array = np.array(ann_segmentation, dtype=np.float64)
            flattened = poly_array.flatten().tolist()
            polygons = [flattened]
        rle = mask.frPyObjects(polygons, height, width)
    else:
        rle = ann_segmentation
        for seg in rle:
            if not isinstance(seg["counts"], bytes):
                seg["counts"] = seg["counts"].encode()
    masks = mask.decode(rle)
    return np.sum(masks, axis=2).astype(np.uint8)


class ReasonSegPlusDataset(torch.utils.data.Dataset):
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
        reason_seg_plus_data="instance_seg||cot||conversations||caption",
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False
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
        self.long_question_list = LONG_QUESTION_TEMPLATE
        self.single_answer_list = MR_SINGLE_ANSWER_LIST

        # Load reason_seg_plus data
        self.dataset_types = reason_seg_plus_data.split("||")
        self.reason_seg_data = self.load_reason_seg_plus_data()

    def __len__(self):
        # return sum([len(reason_seg_ds) for reason_seg_ds in self.reason_seg_data.values()]) # 100000
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

    def load_reason_seg_plus_data(self):
        """加载reason_seg_plus数据"""
        data_dir = os.path.join(self.base_image_dir, "reason_seg_plus")
        reason_seg_data = {}
        
        # 数据文件映射
        file_mapping = {
            "instance_seg": "LISA_Plus_Instance_Seg.json",
            "cot": "LISA_Plus_COT.json", 
            "conversations": "LISA_Plus_Conversations.json",
            "caption": "LISA_Plus_Caption.json"
        }
        
        for dataset_type in self.dataset_types:
            if dataset_type in file_mapping:
                file_path = os.path.join(data_dir, file_mapping[dataset_type])
                if os.path.exists(file_path):
                    print(f"Loading {dataset_type} data from {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    reason_seg_data[dataset_type] = data
                    print(f"Loaded {len(data)} samples for {dataset_type}")
                else:
                    print(f"File not found: {file_path}")
        
        return reason_seg_data

    def select_dataset_and_sample(self):
        """随机选择数据集类型和样本"""
        dataset_type = random.choice(self.dataset_types)
        data = self.reason_seg_data[dataset_type]
        idx = random.randint(0, len(data) - 1)
        sample = data[idx]
        
        return dataset_type, sample
    
    def contains_instructional_words(self, text):
        """检测文本中是否包含指令性词汇"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # 使用switch-case风格的字典查找来减少条件检查
        for word in INSTRUCTIONAL_WORDS:
            if word in text_lower:
                return True
                
        return False

    def extract_and_replace_segments(self, text, mask_dict):
        """提取文本中的segment ID并替换为SEG token，返回替换后的文本和对应的mask ID列表"""
        # 创建seg token字符串
        if self.seg_token_num == 1:
            seg_token = '[SEG]'
        else:
            seg_token = ' '.join(["[SEG{}]".format(i) for i in range(self.seg_token_num)])
        
        mask_ids = []
        
        # 预先计算有效mask的ID集合，避免重复检查
        valid_mask_ids = set()
        for mask_id, mask in mask_dict.items():
            try:
                if isinstance(mask, np.ndarray) and np.any(mask):  # 使用np.any代替np.sum > 0，更高效
                    valid_mask_ids.add(mask_id)
            except Exception:
                pass
        
        # 检查mask_id是否有效（非空）
        def is_valid_mask(mask_id):
            return mask_id in valid_mask_ids
        
        # 模式1: 处理各种列表格式，包括：
        # [<380| remote>, <348| remote>]
        # [pizza <1072129>, pizza <1570619>, ...]
        # [《xxx》, 《yyy》] 等
        def replace_list_pattern(match):
            content = match.group(1)
            
            # 按优先级顺序提取ID，避免重复匹配
            all_ids = []
            
            # 1. 优先匹配 word <id> 格式 (如 pizza <id>, person <id>, object <id> 等)
            if re.search(r'\w+\s*<\d+>', content):
                word_ids = re.findall(r'\w+\s*<(\d+)>', content)
                all_ids.extend(word_ids)
            # 2. 其次匹配 <id|label> 格式
            elif '<' in content and '|' in content:
                conversation_ids = re.findall(r'<(\d+)\|[^>]*>', content)
                all_ids.extend(conversation_ids)
            # 3. 再匹配 《id|label》 格式  
            elif '《' in content and '|' in content:
                unicode_ids = re.findall(r'《(\d+)\|[^》]*》', content)
                all_ids.extend(unicode_ids)
            # 4. 最后匹配直接的 id|label 格式
            elif '|' in content:
                direct_ids = re.findall(r'(\d+)\|\s*[^,\]]+', content)
                all_ids.extend(direct_ids)
            
            # 只保留有效的ID，保持原始顺序和重复
            valid_ids = [id for id in all_ids if is_valid_mask(id)]
            mask_ids.extend(valid_ids)
            
            # 生成对应数量的SEG token
            seg_count = len(valid_ids)
            if seg_count == 0:
                return ''  # 无有效ID则替换为空
            elif seg_count == 1:
                return seg_token
            elif seg_count == 2:
                return f'{seg_token} and {seg_token}'
            else:
                seg_tokens = [seg_token] * seg_count
                return ', '.join(seg_tokens[:-1]) + ' and ' + seg_tokens[-1]
        
        # 处理所有列表格式 - 使用更宽泛的正则
        text = re.sub(r'\[([^\]]+)\]', replace_list_pattern, text)
        
        # 模式2: 单独的《id|label》格式
        def replace_unicode_pattern(match):
            full_match = match.group(0)
            # 提取ID (在《和|之间)
            id_match = re.search(r'《(\d+)\|', full_match)
            if id_match:
                mask_id = id_match.group(1)
                if mask_id in valid_mask_ids:  # 直接使用预计算的valid_mask_ids
                    mask_ids.append(mask_id)
                    return seg_token
                else:
                    return ''  # 无效ID替换为空
            return ''
        
        text = re.sub(r'《[^》]*》', replace_unicode_pattern, text)
        
        # 模式3: 单独的<id|label>格式 (对话数据，不在[]内的)
        def replace_conversation_pattern(match):
            full_match = match.group(0)
            # 提取ID (在<和|之间)  
            id_match = re.search(r'<(\d+)\|', full_match)
            if id_match:
                mask_id = id_match.group(1)
                if mask_id in valid_mask_ids:  # 直接使用预计算的valid_mask_ids
                    mask_ids.append(mask_id)
                    return seg_token
                else:
                    return ''  # 无效ID替换为空
            return ''
        
        # 只匹配包含数字和竖线的尖括号格式，避免匹配<person>等
        text = re.sub(r'<\d+\|[^>]*>', replace_conversation_pattern, text)
        
        return text, mask_ids

    def extract_conversation_from_output(self, output):
        """从output字段提取对话内容"""
        conversations = []
        # 按 <person>: 和 <robot>: 分割对话
        parts = re.split(r'<(person|robot)>:', output)
        
        current_conversations = []
        for i in range(1, len(parts), 2):
            role = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            
            if role == "person":
                if current_conversations:  # 如果已有对话，先完成上一个对话
                    conversations.append(current_conversations)
                current_conversations = [content]  # 开始新对话
            elif role == "robot" and current_conversations:
                current_conversations.append(content)
                
        # 添加最后一个对话
        if len(current_conversations) == 2:
            conversations.append(current_conversations)
            
        return conversations
    
    def process_instance_seg_sample(self, sample, mask_dict):
        """处理实例分割数据样本"""
        question = sample['English Question']
        mask_ids = sample.get('ID', [])
        points_info = sample.get('points', {})
        
        # 只保留有效的mask_ids
        valid_mask_ids = [mask_id for mask_id in mask_ids if mask_id in mask_dict and np.any(mask_dict[mask_id])]
        
        # 直接从ID和points构建label names
        labels = []
        for mask_id in valid_mask_ids:
            if mask_id in points_info and 'label name' in points_info[mask_id]:
                labels.append(points_info[mask_id]['label name'])
        
        # 生成对话 - 参考multi_reason_seg_dataset.py的207-226行
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token) if self.seg_token_num > 1 else '[SEG]'
        
        if len(labels) == 1:
            # 单个label，使用MR_SINGLE_ANSWER_LIST
            answer_template = random.choice(self.single_answer_list)
            answer = answer_template.format(class_name=labels[0])
            if self.seg_token_num > 1:
                answer = answer.replace('[SEG]', seg_token)
        else:
            # 多个labels，参考207-226行的separate_answer=True逻辑
            answer_template = random.choice(self.single_answer_list)
            target_answers = []
            for label in labels:
                _answer_temp = answer_template.format(class_name=label)
                target_answers.append(_answer_temp[:-1])  # 去掉末尾的点
            
            if len(target_answers) > 1:
                part1 = ', '.join(target_answers[:-1])
                part2 = ' and ' + target_answers[-1]
                answer = part1 + part2 + '.'
            else:
                answer = target_answers[0] + '.'
                
            if self.seg_token_num > 1:
                answer = answer.replace('[SEG]', seg_token)
        
        # 创建对话
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]
            
        return conversations, valid_mask_ids

    def process_cot_sample(self, sample, mask_dict):
        """处理COT数据样本"""
        question = sample['English Question'] 
        answer = sample['English Answer']
        
        if self.contains_instructional_words(question):
            formatted_question = DEFAULT_IMAGE_TOKEN + "\n" + question
        else:
            # 使用LONG_QUESTION_TEMPLATE
            question_template = random.choice(self.long_question_list)
            formatted_question = question_template.format(sent=question)
        
        # 提取segment ID并替换为SEG token
        formatted_answer, mask_ids = self.extract_and_replace_segments(answer, mask_dict)
        
        # 创建对话
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], formatted_question)
        conv.append_message(conv.roles[1], formatted_answer)
        conversations = [conv.get_prompt()]
        
        return conversations, mask_ids

    def process_conversations_sample(self, sample, mask_dict):
        """处理多轮对话数据样本"""
        output = sample['output']
        
        # 先提取所有对话轮次
        conversation_pairs = self.extract_conversation_from_output(output)
        
        # 根据num_classes_per_sample限制对话轮数
        if len(conversation_pairs) > self.num_classes_per_sample:
            conversation_pairs = conversation_pairs[:self.num_classes_per_sample]
        
        conversations = []
        all_mask_ids = []
        
        # 只对选中的对话轮次进行segment处理
        for q, a in conversation_pairs:
            # 对问题和答案分别进行segment处理
            processed_q, q_mask_ids = self.extract_and_replace_segments(q, mask_dict)
            processed_a, a_mask_ids = self.extract_and_replace_segments(a, mask_dict)
        
            # 收集所有的mask_ids
            all_mask_ids.extend(q_mask_ids)
            all_mask_ids.extend(a_mask_ids)
            
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + processed_q)
            conv.append_message(conv.roles[1], processed_a)
            conversations.append(conv.get_prompt())
            
        return conversations, all_mask_ids

    def process_caption_sample(self, sample, mask_dict):
        """处理图片描述数据样本"""
        question = sample['English Question']
        answer = sample['English Answer']
        
        # 提取segment ID并替换为SEG token
        formatted_answer, mask_ids = self.extract_and_replace_segments(answer, mask_dict)
        # 如果 [SEG] 前面不是空白字符，补一个空格
        formatted_answer = re.sub(r'(?<!\s)(\[SEG[0-9]*\])', r' \1', formatted_answer)
        
        # 创建对话
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], formatted_answer)
        conversations = [conv.get_prompt()]
        
        return conversations, mask_ids
    
    def create_mask_dict(self, mask_info, image_height, image_width):
        """创建mask_id到mask的映射字典"""
        mask_dict = {}
        for mask_id, mask_data in mask_info.items():
            try:
                if isinstance(mask_data, dict):
                    if 'polygon' in mask_data:
                        polygon_data = mask_data['polygon']
                        # 确保polygon_data是有效的数据结构
                        if isinstance(polygon_data, list) and len(polygon_data) > 0 and all(isinstance(p, list) for p in polygon_data):
                            try:
                                mask = decode_segmentation(polygon_data, image_height, image_width)
                            except Exception as e:
                                print(f"Error decoding polygon for mask_id {mask_id}: {e}")
                                mask = create_zero_mask(image_height, image_width)
                        else:
                            mask = create_zero_mask(image_height, image_width)
                    elif 'points' in mask_data:
                        points_data = mask_data['points']
                        # 确保points_data是有效的数据结构
                        if isinstance(points_data, list) and len(points_data) > 0 and all(isinstance(p, list) for p in points_data):
                            try:
                                mask = decode_segmentation(points_data, image_height, image_width)
                            except Exception as e:
                                print(f"Error decoding points for mask_id {mask_id}: {e}")
                                mask = create_zero_mask(image_height, image_width)
                        else:
                            mask = create_zero_mask(image_height, image_width)
                    else:
                        mask = create_zero_mask(image_height, image_width)
                elif isinstance(mask_data, list) and len(mask_data) > 0:
                    try:
                        # 确保mask_data是有效的数据结构
                        if all(isinstance(p, list) for p in mask_data):
                            mask = decode_segmentation(mask_data, image_height, image_width)
                        else:
                            mask = create_zero_mask(image_height, image_width)
                    except Exception as e:
                        print(f"Error decoding list for mask_id {mask_id}: {e}")
                        mask = create_zero_mask(image_height, image_width)
                else:
                    mask = create_zero_mask(image_height, image_width)
            except Exception as e:
                print(f"Error creating mask for {mask_id}: {e}")
                mask = create_zero_mask(image_height, image_width)
            
            # 最后确认mask是一个有效的numpy数组
            if not isinstance(mask, np.ndarray):
                print(f"Warning: mask for ID {mask_id} is not a numpy array, creating zero mask")
                mask = create_zero_mask(image_height, image_width)
                
            mask_dict[mask_id] = mask
        return mask_dict
        
    def __getitem__(self, idx):
        # 选择数据集和样本
        dataset_type, sample = self.select_dataset_and_sample()
        
        # 根据数据集类型获取图像路径和mask_info
        if dataset_type == "instance_seg":
            image_path = sample.get('img_path', '')
            mask_info = sample.get('points', {})
        elif dataset_type == "cot":
            image_path = sample.get('image_path', '')
            mask_info = sample.get('info', {})
        elif dataset_type == "conversations":
            image_path = sample.get('img_pth', '')
            mask_info = sample.get('info', {})
        elif dataset_type == "caption":
            image_path = sample.get('image_path', '')
            mask_info = sample.get('info', {})
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        # 加载并预处理图像 - 所有数据集都使用COCO2017
        image_path = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2017", image_path)
        
        image = cv2.imread(image_path)
        if image is None:
            assert False, f"Warning: Could not load image {image_path}"
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # 首先创建id-->mask字典
        mask_dict = self.create_mask_dict(mask_info, original_height, original_width)
        
        # 然后根据数据集类型处理样本
        if dataset_type == "instance_seg":
            conversations, mask_ids = self.process_instance_seg_sample(sample, mask_dict)
        elif dataset_type == "cot":
            conversations, mask_ids = self.process_cot_sample(sample, mask_dict)
        elif dataset_type == "conversations":
            conversations, mask_ids = self.process_conversations_sample(sample, mask_dict)
        elif dataset_type == "caption":
            conversations, mask_ids = self.process_caption_sample(sample, mask_dict)
        
        # 收集有效的mask
        masks = []
        for mask_id in mask_ids:
            mask = mask_dict.get(mask_id, None)
            if mask is not None and np.any(mask):
                masks.append(mask)
        
        # 预处理图像用于CLIP
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            clip_resize = image_clip.shape[-2:]

        # 预处理图像用于SAM
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        
        # 转换masks为tensor
        if len(masks) == 0:
            masks = torch.zeros((0, original_height, original_width))
        else:
            masks = torch.from_numpy(np.stack(masks, axis=0).astype(np.uint8))
        
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)
        
        # 创建label tensor
        label = torch.ones(original_height, original_width) * self.ignore_label
        
        # 处理masks用于CLIP（如果需要）
        if self.masks_process_with_clip:
            mask_shape = image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = self.transform_mask(masks, mask_shape)
        
        return (
            image_path,            # filename
            image,                      # raw image (for SAM)
            image_clip,                 # image clip feature (for LMMs)
            conversations,              # QA
            masks,               # segmentation GT
            label,                      # label tensor
            resize,                     # sam input shape
            clip_resize,                # clip input shape  
            [q for conv in conversations for q in [conv.split("ASSISTANT:")[0].split("USER:")[-1].strip()]],  # questions
            [[dataset_type]],           # sampled_classes (数据集类型)
            False,                      # use_assign_list
            False                       # inference
        )

    def transform_mask(self, masks, size):
        """Transform masks to target size"""
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
        left = (orig_width - crop_width) // 2
        right = left + crop_width
        assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
        masks = masks[..., top:bottom, left:right]

        return masks


if __name__ == "__main__":
    import sys
    import os
    # 添加上级目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
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
    dataset = ReasonSegPlusDataset(
        base_image_dir='../dataset_sesame', 
        tokenizer=tokenizer, 
        vision_tower='../dataset_sesame/clip-vit-large-patch14-336',
        reason_seg_plus_data="instance_seg||cot||conversations||caption",
        seg_token_num=seg_token_num,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        # samples_per_epoch=100  # 减少样本数用于测试
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
    cnt = 0
    for batch_idx, batch_data in enumerate(dataloader):
        
        for i, (conversation, masks) in enumerate(zip(batch_data['conversations'], batch_data['masks'])):
            # print('==='*60)
            conv_text = ' '.join(conversation)
            # print(conv_text)
            seg_count = conv_text.count('[SEG') / seg_token_num
            rej_count = conv_text.count('[REJ') / seg_token_num
            valid_masks = sum(1 for j in range(masks.shape[0]) if masks[j].sum() > 0)
            empty_masks = (masks.shape[0] - valid_masks)
            
            check1 = valid_masks == seg_count 
            check2 = masks.shape[0] == seg_count + rej_count 
            check3 = empty_masks == rej_count 

            if not (check1 and check2 and check3):
                print(conv_text)
                cnt += 1
                print(cnt)
                print(f"Sample {i}: SEG={seg_count}, REJ={rej_count}, ValidMasks={valid_masks}, EmptyMasks={empty_masks}, TotalMasks={masks.shape[0]}")
                print(f"  Check: valid==SEG({check1}), total==SEG+REJ({check2}), empty==REJ({check3})")
                print(f"  Dataset type: {batch_data['sampled_sents'][i]}")
                print(f"  Image path: {batch_data['image_paths'][i]}")
                print("  ❌ CONSISTENCY CHECK FAILED!")
            
        #     print(f"Sample {i}: SEG={seg_count}, REJ={rej_count}, ValidMasks={valid_masks}, EmptyMasks={empty_masks}, TotalMasks={masks.shape[0]}")
        #     print(f"  Check: valid==SEG({check1}), total==SEG+REJ({check2}), empty==REJ({check3})")
        #     print(f"  Dataset type: {batch_data['sampled_sents'][i]}")
        #     print(f"  Image path: {batch_data['image_paths'][i]}")
            
        #     if not (check1 and check2 and check3):
        #         print("  ❌ CONSISTENCY CHECK FAILED!")
        #     else:
        #         print("  ✅ CONSISTENCY CHECK PASSED!")
        
        # if batch_idx >= 20:  # Only test first few batches
        #     break 