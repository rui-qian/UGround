IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

SINGLE_ANSWER_LIST = [
    "{class_name} is [SEG].",
    "The segmentation result of {class_name} is [SEG].",
    "[SEG]."
]

MULTI_ANSWER_LIST = [
    "{class_name} are {seg}, separately.",
    "{class_name} are {seg}.",
    "Sure, {class_name} are {seg}, separately.",
    "Sure, {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}, separately.",
    "Sure, the segmentation result of {class_name} are {seg}.",
    "Sure, the segmentation result of {class_name} are {seg}, separately.",
    "Sure, they are {seg}.",
    "They are {seg}.",
    "{seg}."
]

MR_SINGLE_ANSWER_LIST = [
    "{class_name} is [SEG].",
]

MR_MULTI_ANSWER_LIST = [
    "{class_name} are {seg}, separately.",
    "{class_name} are {seg}.",
    "Sure, {class_name} are {seg}, separately.",
    "Sure, {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}, separately.",
    "Sure, the segmentation result of {class_name} are {seg}.",
    "Sure, the segmentation result of {class_name} are {seg}, separately.",
]


EXPAND_LONG_QUESTION_LIST = [

    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Provide the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please show the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} I'd appreciate segmentation masks.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please highlight the segmentation mask.",

]

EXPAND_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you identify the {class_name} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Are you able to delineate the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you pinpoint the {class_name} in this photo?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Is it possible for you to highlight the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you discern the {class_name} in the given picture?",

    DEFAULT_IMAGE_TOKEN + "\n" + "Can you provide me with asegment of the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please perform image segmentation to isolate the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Help me segment the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Would you be willing to segment the {class_name}?",
    

    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you identify {class_name} in this picture? Please provide a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Could you point out {class_name} in this image and show it with a segmentation mask?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, where is {class_name}? I'd appreciate a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please highlight {class_name} in this image using a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In the picture provided, can you show where {class_name} is with a segmentation mask?",
    
]

# LISA Questions and GSVA questions
SHORT_QUESTION_LIST_MODE4 = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please respond with segmentation masks.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please output segmentation masks."
]


ANSWER_LIST_MODE1 = [
    "Here it is.",
    "Sure.",
    "Sure, this is the target.",
    "Sure, here is the segmentation result.",
    "Here you are."
]

ANSWER_LIST_MODE4_START = [
    "The segmentation results are",
    "Sure, they are",
    "Sure,",
    "Sure,",
    "Sure,"
]

ANSWER_LIST_MODE4_TEMPLATE = [
    "{class_name} [SEG]",
    "{class_name}:[SEG]",
    "the mask of {class_name} is [SEG]",
    "the segmentation of {class_name} is [SEG]",
    "the referred {class_name} is [SEG]"
]

ANSWER_LIST_MODE4_END = [
    ".", ".", ".", ".", "."
]

import cv2
import torch
import numpy as np

import requests
from io import BytesIO
import torch.nn.functional as F
from model.segment_anything.utils.transforms import ResizeLongestSide
def load_image(path_or_url):
    if path_or_url.startswith('http'):  # Checks if the path is a URL
        response = requests.get(path_or_url)  # Fetch the image via HTTP
        image_bytes = BytesIO(response.content)  # Convert to a Bytes stream
        image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Decode the image
    else:
        image = cv2.imread(path_or_url, cv2.IMREAD_COLOR)  # Load image from file path
    
    return image

class ImageProcessor:
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    image_size = 1024
    ignore_label = 255

    def __init__(
        self,
        vision_tower,
        image_size: int = 336,
    ):
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def load_and_preprocess_image(self, image_path):
        image = load_image(image_path)
        sam_output_shape = tuple(image.shape[:2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        sam_input_shape = tuple(image.shape[:2])
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        sam_mask_shape = [sam_input_shape, sam_output_shape]
        return image, image_clip, sam_mask_shape

def prepare_input(input_dict, precision, is_cuda=True):
    """Prepare input data based on precision."""
    if precision == "fp16":
        input_dict["images"] = input_dict["images"].half()
        input_dict["images_clip"] = input_dict["images_clip"].half()
    elif precision == "bf16":
        input_dict["images"] = input_dict["images"].bfloat16()
        input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
    else:
        input_dict["images"] = input_dict["images"].float()
        input_dict["images_clip"] = input_dict["images_clip"].float()
    if is_cuda:
        input_dict = dict_to_cuda(input_dict)
    return input_dict

import torch
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.mm_utils import tokenizer_image_token

def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat(
        [
            torch.full(
                (max_length - len(sequence),), padding_value, dtype=sequence.dtype
            ),
            sequence,
        ]
    )

def replace_image_tokens(conversation_list):
    """
    Replace <image> tokens in the conversation list with start and end image tokens.
    """
    for i in range(len(conversation_list)):
        replace_token = DEFAULT_IMAGE_TOKEN
        replace_token = (
            DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        )
        conversation_list[i] = conversation_list[i].replace(
            DEFAULT_IMAGE_TOKEN, replace_token
        )
    return conversation_list

def tokenize_and_pad(conversation_list, tokenizer, padding="right"):
    """
    Tokenize and pad the conversation list.
    Args:
        conversation_list: A list of conversation prompts to be tokenized.
        tokenizer: The tokenizer to use for tokenizing the prompts.
        padding: The direction of padding, either "right" or "left".
    Returns:
        Tuple of input_ids and attention_masks.
    """
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt").squeeze(0) for prompt in conversation_list]
    if padding == "left":
        max_len = max(len(seq) for seq in input_ids)
        input_ids = [pad_sequence_to_max_length(seq, max_len, tokenizer.pad_token_id) for seq in input_ids]
        input_ids = torch.stack(input_ids, dim=0)
    else:
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    return input_ids, attention_masks


def handle_conversation_specifics(input_ids, conversation_list, tokenizer, conv_type):
    """
    Generate targets for the model and handle conversation specifics.
    """
    # Create a copy of the default conversation structure
    conv = conversation_lib.default_conversation.copy()
    # Initialize targets with a clone of input_ids
    targets = input_ids.clone()
    # Define the separator based on conversation type
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "
    
    # Iterate through each conversation in the list and update targets
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # Mark the instruction part as IGNORE_INDEX in the target
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len
    return targets


from enum import Enum
import numpy as np
import torch
import torch.distributed as dist
import os
import atexit
import random
import glob, re
import shutil
import datetime
from pathlib import Path
import subprocess

class MetaLogRotator:
	def __init__(self, log_dir, base_name='meta.log', max_backups=99999, include_patterns=None, exclude_patterns=None):
		self.log_dir = log_dir
		self.base_name = base_name
		self.max_backups = max_backups
		self.base_path = os.path.join(self.log_dir, self.base_name)
		self.meta_dir = os.path.join(self.log_dir, 'meta')
		os.makedirs(self.meta_dir, exist_ok=True)
		
		self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
		self.project_name = os.path.basename(self.project_root)
		self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		
		# Specify files/directories (relative to project root) to include in backups
		self.include_patterns = include_patterns if include_patterns is not None else [
			"model/**",
			"dataloaders/**",
			"scripts/**",
			"configs/**",
			"/*.*",  # root-level files with any extension
			"requirements*.txt",
			"README.md"
		]
		# Explicitly excluded paths (relative to project root)
		self.exclude_patterns = exclude_patterns if exclude_patterns is not None else [
			"*.pyc",
		]

	def _memo_log(self):
		description = ""
		if os.path.exists(self.base_path):
			with open(self.base_path, 'r') as f:
				extracting = False
				for line in f:
					if '+>>' in line and not extracting:
						extracting = True
						description = line[line.find('+>>') + 3:]
						if '<<+' in description:
							description = description[:description.find('<<+')]
							break
					elif '<<+' in line and extracting:
						description += line[:line.find('<<+')]
						break
					elif extracting:
						description += line
					
					if line.strip() == "================================" and extracting:
						break
		
		description = description.strip()
		if description: 
			memo_path = os.path.join(self.log_dir, 'memo.log')
			with open(memo_path, 'a+') as memo_file:
				memo_file.write(f"{'+'*100}\n{os.readlink(self.base_path)}\n+>>{description}<<+\n{'-'*100}\n\n")

	def rotate(self):
		self._rotate_current_log()
		self._cleanup_old_logs()
		self._create_backup(self.meta_path)
		return self.meta_path

	def _rotate_current_log(self):
		i = 1
		while True:
			self.meta_path = os.path.join(self.meta_dir, f'meta_{i}.log')
			if not os.path.exists(self.meta_path):
				break
			i += 1
		self._memo_log()
		if os.path.lexists(self.base_path):
			os.remove(self.base_path)
		with open(self.meta_path, "w") as f: pass
		link_dir = os.path.dirname(self.base_path)
		src_rel = os.path.relpath(self.meta_path, link_dir)
		os.symlink(src_rel, self.base_path)
			
	def _cleanup_old_logs(self):
		pattern = os.path.join(self.meta_dir, "meta_*.log")
		all_backups = sorted(
			glob.glob(pattern),
			key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
		)
		if len(all_backups) > self.max_backups:
			for old_file in all_backups[:-self.max_backups]:
				os.remove(old_file)
				
	def _get_latest_backup(self):
		"""Find the most recent backup directory"""
		backups = [d for d in os.listdir(self.meta_dir) 
				  if os.path.isdir(os.path.join(self.meta_dir, d))]
		if not backups:
			return None
		
		backups.sort(key=lambda x: os.path.getctime(os.path.join(self.meta_dir, x)), reverse=True)
		return os.path.join(self.meta_dir, backups[0])
	
	def _should_exclude(self, path):
		"""Decide exclusion based on include patterns. Always exclude backup meta directory."""
		# Avoid backing up the backup directory itself
		abs_path = os.path.abspath(path)
		if abs_path.startswith(os.path.abspath(self.meta_dir) + os.sep):
			return True
		
		rel_path = os.path.relpath(path, self.project_root).replace('\\', '/')
		path_obj = Path(rel_path)
		
		# Apply explicit exclude patterns first
		if hasattr(self, 'exclude_patterns') and self.exclude_patterns:
			for pattern in self.exclude_patterns:
				# Root-only file patterns like "/*.py" or "/*.*"
				if pattern == '/*.*' and '/' not in rel_path and '.' in os.path.basename(rel_path):
					return True
				if pattern.startswith('/*.') and '/' not in rel_path and rel_path.endswith(pattern[2:]):
					return True
				if path_obj.match(pattern) or any(parent.match(pattern) for parent in path_obj.parents):
					return True
				if (pattern.endswith('/**') and str(rel_path).startswith(pattern[:-3])) or \
				   (pattern.endswith('/*') and str(rel_path).startswith(pattern[:-2])):
					return True
		
		# If include_patterns are provided, only include those; exclude all others
		if hasattr(self, 'include_patterns') and self.include_patterns:
			for pattern in self.include_patterns:
				# Root-only file patterns like "/*.py" or "/*.*"
				if pattern == '/*.*' and '/' not in rel_path and '.' in os.path.basename(rel_path):
					return False
				if pattern.startswith('/*.') and '/' not in rel_path and rel_path.endswith(pattern[2:]):
					return False
				# Direct match or any parent dir match
				if path_obj.match(pattern) or any(parent.match(pattern) for parent in path_obj.parents):
					return False
				# Support directory prefix globs like "dir/**" or "dir/*"
				if (pattern.endswith('/**') and str(rel_path).startswith(pattern[:-3])) or \
				   (pattern.endswith('/*') and str(rel_path).startswith(pattern[:-2])):
					return False
			return True
		
		# Fallback: no include patterns => include everything
		return False
	
	def _create_full_backup(self, log_filename):
		"""Create a full backup of the project"""
		backup_path = os.path.join(self.meta_dir, log_filename)
		os.makedirs(backup_path, exist_ok=True)
		
		rsync_args = []
		if hasattr(self, 'include_patterns') and self.include_patterns:
			for pattern in self.include_patterns:
				if pattern.endswith("/**"):
					top_dir = pattern.split("/")[0] + "/"
					rsync_args.extend(["--include", top_dir, "--include", pattern])
				elif pattern.startswith("/*."):
					# Root-level file patterns (e.g., "/*.py")
					rsync_args.extend(["--include", pattern])
				else:
					rsync_args.extend(["--include", pattern])
			# Apply explicit exclude patterns if provided
			if hasattr(self, 'exclude_patterns') and self.exclude_patterns:
				for pattern in self.exclude_patterns:
					rsync_args.extend(["--exclude", pattern])
			
			rsync_args.extend(["--exclude", "*"])
		
		cmd = ["rsync", "-a"] + rsync_args + [
			f"{self.project_root}/", 
			f"{backup_path}/"
		]
		subprocess.run(cmd, check=True)
		
		with open(os.path.join(backup_path, "manifest.txt"), "w") as f:
			f.write(f"Full backup created: {log_filename}\n")
			f.write(f"Timestamp: {self.timestamp}\n")
			f.write(f"Project: {self.project_name}\n")

		return backup_path
		
	def _get_file_hash(self, filepath):
		"""Get a content hash of a file to compare only contents, not metadata"""
		try:
			with open(filepath, 'rb') as f:
				return hash(f.read())
		except (IOError, OSError):
			raise ValueError(f"Failed to read file: {filepath}")
			
	def _get_all_backup_files(self):
		all_backup_files = {}
		all_backup_dirs = sorted([d for d in os.listdir(self.meta_dir) 
							   if os.path.isdir(os.path.join(self.meta_dir, d))],
							  key=lambda x: os.path.getctime(os.path.join(self.meta_dir, x)))
				
		for backup_dir in all_backup_dirs:
			backup_path = os.path.join(self.meta_dir, backup_dir)
			for root, _, files in os.walk(backup_path):
				rel_root = os.path.relpath(root, backup_path)
				if rel_root == '.':
					rel_root = ''
				
				for file in files:
					if file == "manifest.txt":
						continue
					
					base_name = file
					if "." + self.timestamp[:4] in file:
						parts = file.split(".")
						if len(parts) >= 3 and len(parts[-2]) == 15:
							base_name = ".".join(parts[:-2]) + "." + parts[-1]
					
					rel_path = os.path.join(rel_root, base_name).replace('\\', '/')
					full_path = os.path.join(root, file)
					all_backup_files[rel_path] = full_path
		
		return all_backup_files
		
	def _backup_changed_files(self, latest_backup, log_filename):
		"""Backup only files that have changed since the last backup"""
		backup_path = os.path.join(self.meta_dir, log_filename)
		os.makedirs(backup_path, exist_ok=True)
		
		changed_files = []
		new_files = []
		
		all_backup_files = self._get_all_backup_files()
		
		latest_file_hashes = {}
		for rel_path, full_path in all_backup_files.items():
			try:
				latest_file_hashes[rel_path] = self._get_file_hash(full_path)
			except ValueError:
				raise ValueError(f"Failed to read file: {full_path}")
		
		# Restrict traversal to whitelisted top-level dirs when include_patterns are specified
		allowed_roots = set()
		if hasattr(self, 'include_patterns') and self.include_patterns:
			for pattern in self.include_patterns:
				if '/' in pattern and (pattern.endswith('/**') or pattern.endswith('/*')):
					allowed_roots.add(pattern.split('/')[0])
		for root, dirs, files in os.walk(self.project_root):
			# At project root, limit to allowed top-level dirs (if any), and always prune excluded dirs
			if os.path.abspath(root) == os.path.abspath(self.project_root):
				dirs[:] = [d for d in dirs if (not allowed_roots or d in allowed_roots) and not self._should_exclude(os.path.join(root, d))]
			else:
				# For deeper levels, only prune based on include/exclude logic
				dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
			
			for file in files:
				src_file = os.path.join(root, file)
				
				if self._should_exclude(src_file):
					continue
				
				rel_path = os.path.relpath(src_file, self.project_root).replace('\\', '/')
				
				try:
					current_hash = self._get_file_hash(src_file)
					
					if rel_path not in all_backup_files:
						dst_dir = os.path.join(backup_path, os.path.dirname(rel_path))
						os.makedirs(dst_dir, exist_ok=True)
						dst_file = os.path.join(dst_dir, file)
						shutil.copy2(src_file, dst_file)
						new_files.append(rel_path)
					elif rel_path in latest_file_hashes and current_hash != latest_file_hashes[rel_path]:
						dst_dir = os.path.join(backup_path, os.path.dirname(rel_path))
						os.makedirs(dst_dir, exist_ok=True)
						file_base, file_ext = os.path.splitext(file)
						dst_file = os.path.join(dst_dir, f"{file_base}.{self.timestamp}{file_ext}")
						shutil.copy2(src_file, dst_file)
						changed_files.append(rel_path)
				except ValueError:
					raise ValueError(f"Failed to read file: {src_file}")
		
		if changed_files or new_files:
			with open(os.path.join(backup_path, "manifest.txt"), "w") as f:
				f.write(f"Backup created: {log_filename}\n")
				f.write(f"Previous backup: {latest_backup}\n\n")
				
				if new_files:
					f.write("=== NEW FILES ===\n")
					for file_path in sorted(new_files):
						f.write(f"{file_path}\n")
					f.write("\n")
				
				if changed_files:
					f.write("=== MODIFIED FILES ===\n")
					for file_path in sorted(changed_files):
						f.write(f"{file_path}\n")
		else:
			os.rmdir(backup_path)
			
		return backup_path if (changed_files or new_files) else None
	
	def _create_backup(self, log_filename):
		"""Create a backup of the project"""
		log_filename = os.path.basename(log_filename).split('.')[0]
		latest_backup = self._get_latest_backup()
		
		if latest_backup is None:
			return self._create_full_backup(log_filename)
		else:
			return self._backup_changed_files(latest_backup, log_filename)


def safe_get(kwargs, config, key, default):
    return kwargs.get(key, getattr(config, key, default))

def remove_epoch_ckpt(log_dir, pattern):
    files = glob.glob(os.path.join(log_dir, pattern))
    for f in files:
        os.remove(f)

def set_random_seed(seed=42):
    """
    Set all random seeds to ensure reproducible experiments.
    
    Args:
        seed (int): Random seed value, default is 42
        verbose (bool): Whether to print status messages, default is True
    
    Returns:
        dict: Information about seed settings for logging
    """
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDA settings if available
    cuda_enabled = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cuda_enabled = True
    
    # Return seed information for logging
    seed_info = {
        'Random Seed': seed,
        'CUDA Deterministic': cuda_enabled,
        'CUDNN Benchmark': not cuda_enabled if cuda_enabled else 'N/A',
        'Python Hash Seed': seed,
    }
    
    return seed_info

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class EnhancedProgressMeter(object):
    """Enhanced progress meter with beautiful formatting and additional info"""
    
    def __init__(self, meters, prefix="", program_name="PixelLM", model_name="", args=None):
        self.batch_fmtstr = self._get_batch_fmtstr(args.steps_per_epoch)
        self.meters = meters
        self.prefix = prefix
        self.program_name = program_name
        self.log_dir = args.log_dir
        self.model_name = model_name
        self.total_batches = args.steps_per_epoch
        self.args = args
        
        # Terminal width for formatting
        try:
            import shutil
            self.term_width = shutil.get_terminal_size().columns
        except:
            self.term_width = 100
        
        self.term_width = min(self.term_width, 120)  # Max width limit

    def _get_scores_comparison(self):
        """Get latest vs best scores comparison"""
        if not self.log_dir or not os.path.exists(self.log_dir):
            return ""
        
        def parse_meta_files(pattern):
            """Helper function to parse meta files and extract scores data"""
            meta_files = glob.glob(os.path.join(self.log_dir, pattern))
            if not meta_files:
                return []
            
            scores_data = []
            for meta_file in meta_files:
                filename = os.path.basename(meta_file)
                epoch_match = re.search(r'epoch(\d+)', filename)
                ciou_match = re.search(r'ciou(\d+\.\d+)', filename)
                giou_match = re.search(r'giou(\d+\.\d+)', filename)
                            
                if epoch_match and ciou_match and giou_match:
                    epoch = int(epoch_match.group(1))
                    ciou = float(ciou_match.group(1))
                    giou = float(giou_match.group(1))
                    scores_data.append((epoch, giou, ciou))
            
            return scores_data
        
        # Get latest scores
        latest_scores = parse_meta_files("epoch*_meta_log_ciou*.pth")
        if not latest_scores:
            return ""
        
        # Sort by epoch to get latest
        latest_scores.sort(key=lambda x: x[0])
        latest = latest_scores[-1]
        
        # Get best scores
        best_scores = parse_meta_files("best_epoch*_meta_log_ciou*.pth")
        if not best_scores:
            return ""
        
        # Find best score based on ciou (index 2)
        best = max(best_scores, key=lambda x: x[2])
        
        # Always show both best and latest (they may be the same)
        best_str = f"BEST E{best[0]} giou {best[1]:.4f} ciou {best[2]:.4f}"
        latest_str = f"LATEST E{latest[0]} giou {latest[1]:.4f} ciou {latest[2]:.4f}"
        return f"{self.args.val_dataset:<15}: {best_str} vs. {latest_str}"

    def display(self, batch):
        """Display training progress with enhanced formatting"""
        
        # Progress bar
        progress = batch / self.total_batches
        bar_length = 30
        filled_length = max(1, int(bar_length * progress)) if progress > 0 else 0
        bar = '█' * filled_length + '▒' * (bar_length - filled_length)
        
        # Header info - remove emoji icons
        header_parts = []
        if self.program_name:
            header_parts.append(f"{self.program_name}")
        if self.model_name:
            header_parts.append(f"{self.model_name}")
        
        header = " | ".join(header_parts) if header_parts else ""
        
        # Progress info
        progress_info = f"{self.prefix} {self.batch_fmtstr.format(batch)} {bar} {progress:.1%}"
        
        # Metrics - remove emoji icons and align with colons
        metric_parts = []
        for meter in self.meters:
            if hasattr(meter, 'name') and hasattr(meter, 'avg'):
                # Format value based on metric type
                if meter.name.lower() in ['time', 'data']:
                    formatted_val = f"{meter.avg:.3f}s"
                else:
                    formatted_val = f"{meter.avg:.4f}"
                
                metric_parts.append(f"{meter.name:<15}: {formatted_val}")
        
        # Output directory info (show every time) - remove emoji icon
        output_info = ""
        if self.log_dir:
            short_path = self.log_dir.replace(os.path.expanduser("~"), "~")
            if len(short_path) > 50:
                short_path = "..." + short_path[-47:]
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            project_name = os.path.basename(workspace_root)
            output_info = f"{'Output':<15}: {short_path}🔗{project_name}"
        
        # Scores comparison
        scores_comparison = self._get_scores_comparison()
        
        # Combine all parts
        if header:
            print(f"\n{header}")
            print("─" * len(header))
        
        print(f"{progress_info}")
        
        # Display metrics in single column with consistent alignment
        if metric_parts:
            for metric in metric_parts:
                print(f"  {metric}")
        
        # Always show output directory
        if output_info:
            print(f"  {output_info}")
        
        # Display scores comparison after output info
        if scores_comparison:
            print(f"  {scores_comparison}")
        
        print()  # Add spacing

    def display_summary(self):
        """Display final summary"""
        print("\n" + "═" * 60)
        print("Training Summary")
        print("═" * 60)
        
        entries = []
        for meter in self.meters:
            if hasattr(meter, 'summary'):
                summary = meter.summary()
                if summary:
                    entries.append(summary)
        
        if entries:
            for entry in entries:
                print(f"  {entry}")
        
        print("═" * 60)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


class TrainingLogger:
    """Unified logger for beautiful training output formatting"""
    
    def __init__(self, args, logger=None):
        """Initialize the training logger
        
        Args:
            logger: Optional logger instance for file output
        """
        self.logger = logger
        self.args = args
        # Only main process (local_rank 0 or None) should print
        self.local_rank = getattr(args, 'local_rank', 0)

    def _format_number(self, num):
        """Format numbers with appropriate units"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    def _format_value(self, key, value):
        """Format values based on their type and context"""
        if isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, float):
            if key in ['lr', 'lora_dropout']:
                return f"{value:.6f}"
            else:
                return f"{value:.3f}"
        elif isinstance(value, str) and len(value) > 50:
            # Don't truncate dataset names or other important config values
            if key.lower() in ['datasets', 'dataset', 'sample_rates', 'target_modules', 'sem_seg_data', 'refer_seg_data', 'vqa_data', 'reason_seg_data', 'multi_reason_seg_data', 'version', 'vision_pretrained', 'dataset_dir', 'log_base_dir', 'vis_save_path', 'resume', 'weight', 'preprocessor_config', '--dataset', '--sample_rates', '--sem_seg_data', '--refer_seg_data', '--vqa_data', '--reason_seg_data', '--multi_reason_seg_data', '--version', '--vision_pretrained', '--dataset_dir', '--log_base_dir', '--vis_save_path', '--resume', '--weight', '--preprocessor_config']:
                return str(value)
            else:
                return f"{value[:47]}..."
        else:
            return str(value)
    
    def _print_section(self, title, params, icon):
        """Print a formatted section with title and parameters"""
        self._output(f"{title}")
        self._output("-" * 60)
        for key, value in params.items():
            formatted_value = self._format_value(key.lower().replace(' ', '_'), value)
            self._output(f"  {key:<27}:    {formatted_value}")
        self._output("")
    
    def _output(self, message):
        """Output message to both console and logger"""
        # Only output from main process in distributed training
        if self.local_rank != 0:
            return
            
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _get_sub_datasets_info(self, dataset_obj, dataset_name):
        """Extract sub-dataset information if available"""
        sub_datasets = {}
        
        # Define extraction strategies for each dataset type
        extractors = {
            "sem_seg": self._extract_sem_seg_info,
            "refer_seg": self._extract_refer_seg_info,
            "neg_refer_seg": self._extract_neg_refer_seg_info,
            "correct_refer_seg": self._extract_correct_refer_seg_info,
            "vqa": self._extract_vqa_info,
            "reason_seg": self._extract_reason_seg_info,
            "reason_seg_plus": self._extract_reason_seg_plus_info,
            "multi_reason_seg": self._extract_multi_reason_seg_info
        }
        
        try:
            extractor = extractors.get(dataset_name)
            if extractor:
                sub_datasets = extractor(dataset_obj)
        except Exception:
            # Silently ignore errors in sub-dataset extraction
            pass
            
        return sub_datasets

    def _extract_sem_seg_info(self, dataset_obj):
        """Extract SemSegDataset sub-datasets"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'sem_seg_datas') and hasattr(dataset_obj, 'data2list'):
            for sub_ds in dataset_obj.sem_seg_datas:
                if sub_ds in dataset_obj.data2list:
                    images, labels = dataset_obj.data2list[sub_ds]
                    sub_size = len(images) if images else 0
                    if sub_size > 0:
                        sub_datasets[sub_ds] = f"{self._format_number(sub_size):>8} ({sub_size:,})"
        return sub_datasets

    def _extract_refer_seg_info(self, dataset_obj):
        """Extract ReferSegDataset sub-datasets"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'refer_seg_ds_list') and hasattr(dataset_obj, 'refer_seg_data'):
            for sub_ds in dataset_obj.refer_seg_ds_list:
                if sub_ds in dataset_obj.refer_seg_data:
                    refer_data = dataset_obj.refer_seg_data[sub_ds]
                    if 'images' in refer_data:
                        sub_size = len(refer_data['images'])
                        if sub_size > 0:
                            sub_datasets[sub_ds] = f"{self._format_number(sub_size):>8} ({sub_size:,})"
        return sub_datasets

    def _extract_neg_refer_seg_info(self, dataset_obj):
        """Extract fpReferSegDataset (neg_refer_seg) sub-datasets"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'refer_seg_ds_list') and hasattr(dataset_obj, 'refer_seg_data'):
            for sub_ds in dataset_obj.refer_seg_ds_list:
                if sub_ds in dataset_obj.refer_seg_data:
                    refer_data = dataset_obj.refer_seg_data[sub_ds]
                    if 'images' in refer_data:
                        sub_size = len(refer_data['images'])
                        if sub_size > 0:
                            sub_datasets[sub_ds] = f"{self._format_number(sub_size):>8} ({sub_size:,})"
        return sub_datasets

    def _extract_correct_refer_seg_info(self, dataset_obj):
        """Extract fpReferSegDataset (correct_refer_seg) sub-datasets"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'refer_seg_ds_list') and hasattr(dataset_obj, 'refer_seg_data'):
            for sub_ds in dataset_obj.refer_seg_ds_list:
                if sub_ds in dataset_obj.refer_seg_data:
                    refer_data = dataset_obj.refer_seg_data[sub_ds]
                    if 'images' in refer_data:
                        sub_size = len(refer_data['images'])
                        if sub_size > 0:
                            sub_datasets[sub_ds] = f"{self._format_number(sub_size):>8} ({sub_size:,})"
        return sub_datasets

    def _extract_vqa_info(self, dataset_obj):
        """Extract VQADataset info"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'vqa_data'):
            actual_size = len(dataset_obj.vqa_data)
            sub_datasets["llava_dataset"] = f"{self._format_number(actual_size):>8} ({actual_size:,})"
        return sub_datasets

    def _extract_reason_seg_info(self, dataset_obj):
        """Extract ReasonSegDataset info"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'reason_seg_data'):
            images, _ = dataset_obj.reason_seg_data
            actual_size = len(images)
            sub_datasets["ReasonSeg"] = f"{self._format_number(actual_size):>8} ({actual_size:,})"
        return sub_datasets

    def _extract_reason_seg_plus_info(self, dataset_obj):
        """Extract ReasonSegPlusDataset info"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'reason_seg_data') and hasattr(dataset_obj, 'dataset_types'):
            for dataset_type in dataset_obj.dataset_types:
                if dataset_type in dataset_obj.reason_seg_data:
                    data = dataset_obj.reason_seg_data[dataset_type]
                    actual_size = len(data)
                    if actual_size > 0:
                        sub_datasets[dataset_type] = f"{self._format_number(actual_size):>8} ({actual_size:,})"
        return sub_datasets

    def _extract_multi_reason_seg_info(self, dataset_obj):
        """Extract MultiReasonSegDataset info"""
        sub_datasets = {}
        if hasattr(dataset_obj, 'multi_reason_seg_data'):
            actual_size = len(dataset_obj.multi_reason_seg_data)
            sub_datasets["MultiReasonSeg"] = f"{self._format_number(actual_size):>8} ({actual_size:,})"
        return sub_datasets
    
    def _get_actual_dataset_size(self, dataset_obj, dataset_name):
        """Get actual dataset size instead of samples_per_epoch"""
        try:
            # For VQA, ReasonSeg, MultiReasonSeg, ReasonSegPlus - show samples_per_epoch as main size
            # The actual data size will be shown as sub-dataset info
            if dataset_name in ["vqa", "reason_seg", "multi_reason_seg", "reason_seg_plus"]:
                return len(dataset_obj)  # This is samples_per_epoch
            
            # For SemSegDataset, ReferSegDataset, fpReferSegDataset - use samples_per_epoch (it's reasonable for mixed datasets)
            else:
                return len(dataset_obj)
                
        except Exception:
            # Fallback to __len__ if anything goes wrong
            return len(dataset_obj)
    
    def print_trainable_parameters(self, model):
        """Print formatted model parameters summary"""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_param
        
        self._output("=" * 80)
        self._output("MODEL PARAMETERS SUMMARY")
        self._output("=" * 80)
        
        # Use the same style as print_training_config
        self._print_original_section("Model Parameters", [
            ('Total Parameters', self._format_number(all_param)),
            ('Trainable Parameters', self._format_number(trainable_params)),
            ('Trainable Percentage', f"{trainable_percent:.2f}%"),
        ])
        
        self._output("=" * 80)
    
    def print_dataset_summary(self, train_dataset, val_dataset=None, val_dataset_names=None):
        """Print formatted dataset summary"""
        self._output("=" * 80)
        self._output("DATASET SUMMARY")
        self._output("=" * 80)
        
        # Prepare training dataset info
        train_size = len(train_dataset)
        train_params = [('Total Training Examples', f"{self._format_number(train_size):>7} ({train_size:,})")]
        
        # Check if train_dataset is HybridDataset and show breakdown
        if hasattr(train_dataset, 'datasets') and hasattr(train_dataset, 'all_datasets') and hasattr(train_dataset, 'sample_rate'):
            for i, (dataset_name, dataset_obj) in enumerate(zip(train_dataset.datasets, train_dataset.all_datasets)):
                try:
                    # Get actual dataset size instead of samples_per_epoch
                    actual_size = self._get_actual_dataset_size(dataset_obj, dataset_name)
                    samples_per_epoch = len(dataset_obj)  # This is samples_per_epoch for most datasets
                    
                    sample_rate = train_dataset.sample_rate[i] * 100  # Convert to percentage
                    expected_samples = int(train_size * train_dataset.sample_rate[i])
                    
                    # Format display info with K alignment and ratio info
                    if actual_size != samples_per_epoch:
                        size_info = f"{self._format_number(samples_per_epoch):>7} ({samples_per_epoch:,})  - {sample_rate:>4.1f}%  ~{expected_samples:,}/{train_size}"
                    else:
                        size_info = f"{self._format_number(actual_size):>7} ({actual_size:,})  - {sample_rate:>4.1f}%  ~{expected_samples:,}/{train_size}"
                    
                    train_params.append((f'{dataset_name} Dataset', size_info))
                    
                    # Show sub-dataset breakdown if available
                    sub_datasets = self._get_sub_datasets_info(dataset_obj, dataset_name)
                    if sub_datasets:
                        for sub_name, sub_info in sub_datasets.items():
                            # Extract numbers from sub_info and reformat
                            if '(' in sub_info and ')' in sub_info:
                                # Parse the existing format like "157.71K (157,712)"
                                parts = sub_info.split('(')
                                if len(parts) == 2:
                                    formatted_num = parts[0].strip()
                                    raw_num = parts[1].replace(')', '').replace(',', '')
                                    try:
                                        # Format raw number with commas
                                        raw_num_int = int(raw_num)
                                        formatted_raw = f"{raw_num_int:,}"
                                        train_params.append((f'  └─ {sub_name}', f"{formatted_num:>7} ({formatted_raw})"))
                                    except:
                                        train_params.append((f'  └─ {sub_name}', f"{formatted_num:>7} ({raw_num})"))
                                else:
                                    train_params.append((f'  └─ {sub_name}', f"{sub_info:>7}"))
                            else:
                                train_params.append((f'  └─ {sub_name}', f"{sub_info:>7}"))
                            
                except Exception:
                    # Fallback display if anything goes wrong
                    dataset_size = len(dataset_obj)
                    sample_rate = train_dataset.sample_rate[i] * 100
                    train_params.append((f'{dataset_name} Dataset', f"{self._format_number(dataset_size):>7} ({dataset_size:,}) - {sample_rate:>4.1f}%"))
        
        # Prepare validation dataset info
        val_params = []
        if val_dataset is not None:
            if isinstance(val_dataset, list):
                total_val_size = sum(len(vd) for vd in val_dataset)
                val_params.append(('Total Validation Examples', f"{self._format_number(total_val_size):>7} ({total_val_size:,})"))
                
                # Show breakdown for multiple validation datasets
                if val_dataset_names and len(val_dataset_names) > 1:
                    for i, (dataset_name, vd) in enumerate(zip(val_dataset_names, val_dataset)):
                        val_size = len(vd)
                        val_params.append((f'{dataset_name} Dataset', f"{self._format_number(val_size):>7} ({val_size:,})"))
            else:
                val_size = len(val_dataset)
                dataset_name = val_dataset_names[0] if val_dataset_names else "Validation"
                val_params.append((f'{dataset_name} Examples', f"{self._format_number(val_size):>7} ({val_size:,})"))
            
            # Calculate train/val ratio
            total_val_size = sum(len(vd) for vd in val_dataset) if isinstance(val_dataset, list) else len(val_dataset)
            total_examples = train_size + total_val_size
            train_ratio = (train_size / total_examples) * 100
            val_ratio = (total_val_size / total_examples) * 100
            val_params.append(('Train/Val Ratio', f"{train_ratio:>4.1f}% / {val_ratio:.1f}%"))
        else:
            val_params.append(('Validation Status', "Disabled (no validation set)"))
        
        # Print sections using consistent style
        self._print_original_section("Training Dataset", train_params)
        self._print_original_section("Validation Dataset", val_params)
        
        self._output("=" * 80)
    
    def print_training_config(self):
        """Print formatted training configuration maintaining command-line style"""
        self._output("=" * 80)
        self._output("TRAINING CONFIGURATION")
        self._output("=" * 80)
        
        # Generate a command-line equivalent representation first
        self._output("Command Line Equivalent:")
        self._output("-" * 80)

        # Build command line representation
        cmd_parts = ["python train_ds.py"]
        
        # Get all attributes from self.args
        for param_name in vars(self.args):
            value = getattr(self.args, param_name)
                            
            # Handle different parameter types
            if isinstance(value, bool):
                if value:  # Only add flag for True boolean values
                    cmd_parts.append(f"--{param_name}")
            else:
                # Handle string values with spaces or pipe characters
                if isinstance(value, str) and ('|' in str(value) or ' ' in str(value)):
                    cmd_parts.append(f'--{param_name} "{value}"')
                else:
                    cmd_parts.append(f"--{param_name} {value}")
        
        # Print command in readable chunks
        current_line = cmd_parts[0]
        for part in cmd_parts[1:]:
            if len(current_line + " " + part) > 75:
                self._output(f"  {current_line} \\")
                current_line = "    " + part
            else:
                current_line += " " + part
        self._output(f"  {current_line}")
        self._output("")
        
        # Now print organized sections with original parameter style
        self._print_original_section("Dataset Configuration", [
            ('--dataset', getattr(self.args, 'dataset', None)),
            ('--sample_rates', getattr(self.args, 'sample_rates', None)),
            ('--sem_seg_data', getattr(self.args, 'sem_seg_data', None)),
            ('--refer_seg_data', getattr(self.args, 'refer_seg_data', None)),
            ('--neg_refer_seg_data', getattr(self.args, 'neg_refer_seg_data', None)),
            ('--correct_refer_seg_data', getattr(self.args, 'correct_refer_seg_data', None)),
            ('--vqa_data', getattr(self.args, 'vqa_data', None)),
            ('--reason_seg_data', getattr(self.args, 'reason_seg_data', None)),
            ('--reason_seg_plus_data', getattr(self.args, 'reason_seg_plus_data', None)),
            ('--multi_reason_seg_data', getattr(self.args, 'multi_reason_seg_data', None)),
            ('--val_dataset', getattr(self.args, 'val_dataset', None)),
        ])
        
        self._print_original_section("Model Configuration", [
            ('--model_key', getattr(self.args, 'model_key', None)),
            ('--version', getattr(self.args, 'version', None)),
            ('--vision_tower', getattr(self.args, 'vision_tower', None)),
            ('--vision_pretrained', getattr(self.args, 'vision_pretrained', None)),
            ('--conv_type', getattr(self.args, 'conv_type', None)),
            ('--model_max_length', getattr(self.args, 'model_max_length', None)),
            ('--seg_token_num', getattr(self.args, 'seg_token_num', None)),
            ('--image_feature_scale_num', getattr(self.args, 'image_feature_scale_num', None)),
        ])
        
        self._print_original_section("Training Parameters", [
            ('--epochs', getattr(self.args, 'epochs', None)),
            ('--steps_per_epoch', getattr(self.args, 'steps_per_epoch', None)),
            ('--batch_size', getattr(self.args, 'batch_size', None)),
            ('--val_batch_size', getattr(self.args, 'val_batch_size', None)),
            ('--grad_accumulation_steps', getattr(self.args, 'grad_accumulation_steps', None)),
            ('--workers', getattr(self.args, 'workers', None)),
            ('--lr', getattr(self.args, 'lr', None)),
            ('--beta1', getattr(self.args, 'beta1', None)),
            ('--beta2', getattr(self.args, 'beta2', None)),
            ('--precision', getattr(self.args, 'precision', None)),
            ('--gradient_checkpointing', getattr(self.args, 'gradient_checkpointing', None)),
        ])
        
        self._print_original_section("Model Features", [
            ('--train_mask_decoder', getattr(self.args, 'train_mask_decoder', None)),
            ('--use_mm_start_end', getattr(self.args, 'use_mm_start_end', None)),
            ('--pad_train_clip_images', getattr(self.args, 'pad_train_clip_images', None)),
            ('--masks_process_with_clip', getattr(self.args, 'masks_process_with_clip', None)),
            ('--resize_vision_tower', getattr(self.args, 'resize_vision_tower', None)),
            ('--resize_vision_tower_size', getattr(self.args, 'resize_vision_tower_size', None)),
            ('--vision_tower_for_mask', getattr(self.args, 'vision_tower_for_mask', None)),
            ('--separate_mm_projector', getattr(self.args, 'separate_mm_projector', None)),
            ('--use_expand_question_list', getattr(self.args, 'use_expand_question_list', None)),
        ])
        
        # Show LoRA Configuration
        self._print_original_section("LoRA Configuration", [
            ('--lora_r', getattr(self.args, 'lora_r', None)),
            ('--lora_alpha', getattr(self.args, 'lora_alpha', None)),
            ('--lora_dropout', getattr(self.args, 'lora_dropout', None)),
            ('--lora_target_modules', getattr(self.args, 'lora_target_modules', None)),
        ])
        
        self._print_original_section("Loss Configuration", [
            ('--ce_loss_weight', getattr(self.args, 'ce_loss_weight', None)),
            ('--dice_loss_weight', getattr(self.args, 'dice_loss_weight', None)),
            ('--bce_loss_weight', getattr(self.args, 'bce_loss_weight', None)),
            ('--out_dim', getattr(self.args, 'out_dim', None)),
        ])
        
        self._print_original_section("Other Options", [
            ('--exclude_val', getattr(self.args, 'exclude_val', None)),
            ('--no_eval', getattr(self.args, 'no_eval', None)),
            ('--eval_only', getattr(self.args, 'eval_only', None)),
            ('--auto_resume', getattr(self.args, 'auto_resume', None)),
            ('--resume_from_best', getattr(self.args, 'resume_from_best', None)),
            ('--load_in_8bit', getattr(self.args, 'load_in_8bit', None)),
            ('--load_in_4bit', getattr(self.args, 'load_in_4bit', None)),
            ('--local_rank', getattr(self.args, 'local_rank', None)),
            ('--vis_save_path', getattr(self.args, 'vis_save_path', None)),
            ('--image_size', getattr(self.args, 'image_size', None)),
            ('--dataset_dir', getattr(self.args, 'dataset_dir', None)),
            ('--log_base_dir', getattr(self.args, 'log_base_dir', None)),
            ('--exp_name', getattr(self.args, 'exp_name', None)),
            ('--resume', getattr(self.args, 'resume', None)),
            ('--print_freq', getattr(self.args, 'print_freq', None)),
            ('--start_epoch', getattr(self.args, 'start_epoch', None)),
            ('--num_classes_per_sample', getattr(self.args, 'num_classes_per_sample', None)),
            ('--num_classes_per_question', getattr(self.args, 'num_classes_per_question', None)),
            ('--preprocessor_config', getattr(self.args, 'preprocessor_config', None)),
            ('--weight', getattr(self.args, 'weight', None)),
            ('--seed', getattr(self.args, 'seed', None)),
            ('--explanatory', getattr(self.args, 'explanatory', None)),
        ])
        
        self._output("=" * 80)
    
    def _print_original_section(self, title, params):
        """Print a configuration section with original parameter style"""
        self._output(f"{title}")
        self._output("-" * 60)
        
        for param_name, value in params:
            # Keep completely original value formatting
            if value is None:
                formatted_value = "None"
            elif isinstance(value, bool):
                formatted_value = str(value)  # True or False
            elif isinstance(value, float):
                if value < 1e-3:
                    formatted_value = f"{value:.6f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                # Don't truncate any strings, show them completely
                formatted_value = str(value)
            
            self._output(f"  {param_name:<27}:    {formatted_value}")
        
        self._output("")
    
    def print_dataset_loading_start(self):
        """Print dataset loading start message"""
        self._output("=" * 60)
        self._output("Loading datasets...")
        self._output("=" * 60)
    
    def print_dataset_loading_complete(self):
        """Print dataset loading completion message"""
        self._output("All datasets loaded successfully!")
        self._output("")
    
    def print_deepspeed_init_summary(self, ds_config):
        """Print DeepSpeed initialization summary with consistent formatting"""
        import torch
        import os
        
        self._output("=" * 80)
        self._output("DEEPSPEED INITIALIZATION SUMMARY")
        self._output("=" * 80)
        
        # Extract hardware and distributed information
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        local_rank = getattr(self.args, 'local_rank', 0)
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Parse CUDA_VISIBLE_DEVICES correctly
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible_devices is None:
            visible_devices = f"all ({device_count} GPUs)"
        elif cuda_visible_devices == "":
            visible_devices = "none (CPU only)"
        else:
            device_ids = [id.strip() for id in cuda_visible_devices.split(',') if id.strip()]
            visible_count = len(device_ids)
            visible_devices = f"{cuda_visible_devices} ({visible_count} GPUs)"
        
        # Hardware Information Section
        self._print_original_section("Hardware Information", [
            ('--world_size', world_size),
            ('--local_rank', local_rank),
            ('--visible_gpu_ids', visible_devices),
            ('--total_available_gpus', device_count),
            ('--accelerator', 'cuda' if torch.cuda.is_available() else 'cpu'),
            ('--device_name', torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'),
            ('--memory_available', f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else 'N/A'),
            ('--cuda_version', torch.version.cuda if torch.cuda.is_available() else 'N/A'),
        ])
        
        # Training Configuration Section - keep all fields exactly as they are
        self._print_original_section("Training Configuration", [
            ('--train_batch_size', ds_config.get('train_batch_size')),
            ('--train_micro_batch_size_per_gpu', ds_config.get('train_micro_batch_size_per_gpu')),
            ('--gradient_accumulation_steps', ds_config.get('gradient_accumulation_steps')),
            ('--gradient_clipping', ds_config.get('gradient_clipping')),
            ('--steps_per_print', ds_config.get('steps_per_print')),
            ('--wall_clock_breakdown', ds_config.get('wall_clock_breakdown')),
        ])
        
        # ZeRO Optimization Configuration
        zero_config = ds_config.get('zero_optimization', {})
        self._print_original_section("ZeRO Optimization", [
            ('--stage', zero_config.get('stage')),
            ('--contiguous_gradients', zero_config.get('contiguous_gradients')),
            ('--reduce_scatter', zero_config.get('reduce_scatter')),
            ('--reduce_bucket_size', zero_config.get('reduce_bucket_size')),
            ('--use_multi_rank_bucket_allreduce', zero_config.get('use_multi_rank_bucket_allreduce')),
            ('--allgather_partitions', zero_config.get('allgather_partitions')),
            ('--allgather_bucket_size', zero_config.get('allgather_bucket_size')),
            ('--overlap_comm', zero_config.get('overlap_comm')),
            ('--load_from_fp32_weights', zero_config.get('load_from_fp32_weights')),
            ('--elastic_checkpoint', zero_config.get('elastic_checkpoint')),
            ('--sub_group_size', zero_config.get('sub_group_size')),
            ('--ignore_unused_parameters', zero_config.get('ignore_unused_parameters')),
            ('--legacy_stage1', zero_config.get('legacy_stage1')),
            ('--round_robin_gradients', zero_config.get('round_robin_gradients')),
            ('--zero_hpz_partition_size', zero_config.get('zero_hpz_partition_size')),
            ('--zero_quantized_weights', zero_config.get('zero_quantized_weights')),
            ('--zero_quantized_nontrainable_weights', zero_config.get('zero_quantized_nontrainable_weights')),
            ('--zero_quantized_gradients', zero_config.get('zero_quantized_gradients')),
            ('--memory_efficient_linear', zero_config.get('memory_efficient_linear')),
            ('--pipeline_loading_checkpoint', zero_config.get('pipeline_loading_checkpoint')),
            ('--override_module_apply', zero_config.get('override_module_apply')),
            ('--log_trace_cache_warnings', zero_config.get('log_trace_cache_warnings')),
        ])
        
        # Parameter Offload Configuration (ZeRO-3 specific)
        offload_param = zero_config.get('offload_param', {})
        if offload_param:
            self._print_original_section("Parameter Offload Configuration", [
                ('--device', offload_param.get('device')),
                ('--nvme_path', offload_param.get('nvme_path')),
                ('--buffer_count', offload_param.get('buffer_count')),
                ('--buffer_size', offload_param.get('buffer_size')),
                ('--max_in_cpu', offload_param.get('max_in_cpu')),
                ('--pin_memory', offload_param.get('pin_memory')),
            ])
        
        # Optimizer Offload Configuration
        offload_optimizer = zero_config.get('offload_optimizer', {})
        if offload_optimizer:
            self._print_original_section("Optimizer Offload Configuration", [
                ('--device', offload_optimizer.get('device')),
                ('--nvme_path', offload_optimizer.get('nvme_path')),
                ('--buffer_count', offload_optimizer.get('buffer_count')),
                ('--pin_memory', offload_optimizer.get('pin_memory')),
                ('--pipeline_read', offload_optimizer.get('pipeline_read')),
                ('--pipeline_write', offload_optimizer.get('pipeline_write')),
                ('--fast_init', offload_optimizer.get('fast_init')),
                ('--ratio', offload_optimizer.get('ratio')),
            ])
        
        # ZeRO-3 Stage-specific Configuration
        if zero_config.get('stage') == 3:
            self._print_original_section("ZeRO Stage-3 Configuration", [
                ('--prefetch_bucket_size', zero_config.get('prefetch_bucket_size')),
                ('--param_persistence_threshold', zero_config.get('param_persistence_threshold')),
                ('--model_persistence_threshold', zero_config.get('model_persistence_threshold')),
                ('--max_live_parameters', zero_config.get('max_live_parameters')),
                ('--max_reuse_distance', zero_config.get('max_reuse_distance')),
                ('--gather_16bit_weights_on_model_save', zero_config.get('gather_16bit_weights_on_model_save')),
                ('--module_granularity_threshold', zero_config.get('module_granularity_threshold')),
                ('--use_all_reduce_for_fetch_params', zero_config.get('use_all_reduce_for_fetch_params')),
            ])
        
        # Precision Configuration
        fp16_config = ds_config.get('fp16', {})
        bf16_config = ds_config.get('bf16', {})
        self._print_original_section("Precision Configuration", [
            ('--fp16_enabled', fp16_config.get('enabled')),
            ('--fp16_auto_cast', fp16_config.get('auto_cast')),
            ('--fp16_loss_scale', fp16_config.get('loss_scale')),
            ('--fp16_initial_scale_power', fp16_config.get('initial_scale_power')),
            ('--fp16_loss_scale_window', fp16_config.get('loss_scale_window')),
            ('--fp16_hysteresis', fp16_config.get('hysteresis')),
            ('--fp16_consecutive_hysteresis', fp16_config.get('consecutive_hysteresis')),
            ('--fp16_min_loss_scale', fp16_config.get('min_loss_scale')),
            ('--bf16_enabled', bf16_config.get('enabled')),
        ])
        
        # Optimizer Configuration
        optimizer_config = ds_config.get('optimizer', {})
        if optimizer_config:
            optimizer_params = optimizer_config.get('params', {})
            self._print_original_section("Optimizer Configuration", [
                ('--type', optimizer_config.get('type')),
                ('--lr', optimizer_params.get('lr')),
                ('--weight_decay', optimizer_params.get('weight_decay')),
                ('--betas', optimizer_params.get('betas')),
                ('--eps', optimizer_params.get('eps')),
                ('--torch_adam', optimizer_params.get('torch_adam')),
                ('--adam_w_mode', optimizer_params.get('adam_w_mode')),
            ])
        
        # Scheduler Configuration
        scheduler_config = ds_config.get('scheduler', {})
        if scheduler_config:
            scheduler_params = scheduler_config.get('params', {})
            self._print_original_section("Scheduler Configuration", [
                ('--type', scheduler_config.get('type')),
                ('--warmup_min_lr', scheduler_params.get('warmup_min_lr')),
                ('--warmup_max_lr', scheduler_params.get('warmup_max_lr')),
                ('--warmup_num_steps', scheduler_params.get('warmup_num_steps')),
                ('--warmup_type', scheduler_params.get('warmup_type')),
                ('--total_num_steps', scheduler_params.get('total_num_steps')),
            ])
        
        # Communication Configuration
        self._print_original_section("Communication Configuration", [
            ('--communication_data_type', ds_config.get('communication_data_type')),
            ('--prescale_gradients', ds_config.get('prescale_gradients')),
            ('--gradient_predivide_factor', ds_config.get('gradient_predivide_factor')),
            ('--sparse_gradients', ds_config.get('sparse_gradients')),
        ])
        
        # Activation Checkpointing Configuration
        activation_checkpointing = ds_config.get('activation_checkpointing', {})
        if activation_checkpointing:
            self._print_original_section("Activation Checkpointing", [
                ('--partition_activations', activation_checkpointing.get('partition_activations')),
                ('--cpu_checkpointing', activation_checkpointing.get('cpu_checkpointing')),
                ('--contiguous_memory_optimization', activation_checkpointing.get('contiguous_memory_optimization')),
                ('--number_checkpoints', activation_checkpointing.get('number_checkpoints')),
                ('--synchronize_checkpoint_boundary', activation_checkpointing.get('synchronize_checkpoint_boundary')),
                ('--profile', activation_checkpointing.get('profile')),
            ])
        
        self._output("Initializing DeepSpeed engine...")
        self._output("=" * 80)

    def print_deepspeed_init_complete(self, model_engine):
        """Print DeepSpeed initialization completion summary"""
        self._output("=" * 80)
        self._output("DEEPSPEED INITIALIZATION COMPLETE")
        self._output("=" * 80)
        
        # Extract model engine information
        try:
            # Get basic model information
            total_params = sum(p.numel() for p in model_engine.module.parameters())
            trainable_params = sum(p.numel() for p in model_engine.module.parameters() if p.requires_grad)
            
            # Get optimizer information
            optimizer_name = "Unknown"
            if hasattr(model_engine, 'optimizer'):
                optimizer_name = type(model_engine.optimizer).__name__
            
            # Get learning rate scheduler information  
            scheduler_name = "Unknown"
            if hasattr(model_engine, 'lr_scheduler') and model_engine.lr_scheduler is not None:
                scheduler_name = type(model_engine.lr_scheduler).__name__
            
            # Basic engine information
            self._print_original_section("Engine Information", [
                ('--total_parameters', f"{self._format_number(total_params)} ({total_params:,})"),
                ('--trainable_parameters', f"{self._format_number(trainable_params)} ({trainable_params:,})"),
                ('--optimizer', optimizer_name),
                ('--lr_scheduler', scheduler_name),
                ('--fp16_enabled', getattr(model_engine, 'fp16_enabled', lambda: False)()),
                ('--bfloat16_enabled', getattr(model_engine, 'bfloat16_enabled', lambda: False)()),
                ('--zero_optimization_stage', getattr(model_engine, 'zero_optimization_stage', lambda: 0)()),
            ])
            
            # Memory information if available
            if hasattr(model_engine, 'device') and str(model_engine.device).startswith('cuda'):             
                cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                local_rank = getattr(self.args, 'local_rank', 0)
                
                display_device_id = None
                
                if cuda_visible_devices:
                    try:
                        visible_devices = [x.strip() for x in cuda_visible_devices.split(',') if x.strip()]
                        if local_rank < len(visible_devices):
                            display_device_id = visible_devices[local_rank]
                            actual_device_id = local_rank
                        else:
                            actual_device_id = local_rank
                    except:
                        actual_device_id = model_engine.device.index if model_engine.device.index is not None else local_rank
                else:
                    actual_device_id = local_rank
                
                device_count = torch.cuda.device_count()
                if actual_device_id >= device_count:
                    actual_device_id = 0
                
                memory_allocated = torch.cuda.memory_allocated(actual_device_id) / 1e9
                memory_reserved = torch.cuda.memory_reserved(actual_device_id) / 1e9
                
                device_display = f"cuda:{display_device_id}" if display_device_id is not None else f"cuda:{actual_device_id}"
                
                self._print_original_section("Memory Information", [
                    ('--device', device_display),
                    ('--memory_allocated', f"{memory_allocated:.2f}GB"),
                    ('--memory_reserved', f"{memory_reserved:.2f}GB"),
                ])
            
        except Exception as e:
            # Fallback to basic information if detailed extraction fails
            self._print_original_section("Engine Information", [
                ('--status', "Initialized Successfully"),
                ('--engine_type', type(model_engine).__name__),
                ('--device', str(getattr(model_engine, 'device', 'Unknown'))),
            ])
        
        self._output("DeepSpeed engine initialized successfully!")
        self._output(f"Ready to start {'training' if model_engine.module.training else 'evaluation'} with model: {getattr(self.args, 'model_key', None)}!")
        self._output("=" * 80)


class SystemOutputRedirector:
    """System-level output redirector with progress bar filtering and dual output (console + file)"""
    
    def __init__(self, raw_log_filename, debug=False):
        self.raw_log_filename = raw_log_filename
        self.original_stdout_fd = None
        self.original_stderr_fd = None
        self.tee_pid = None
        self.is_active = False
        self.debug=debug
        
        # Progress bar filtering state
        self.last_progress_line = None
        self.last_progress_prefix = None
        self.logged_final_progress = set()  # Track already logged final progress bars
    
    def extract_progress_prefix(self, line):
        """Extract the prefix part of progress bar for identifying the same progress bar"""
        # Special handling for pure progress bar format: 100%|██████████| 200/200 [...]
        if re.match(r'^\d+%\|.*?\|\s*\d+/\d+', line.strip()):
            match = re.search(r'\|\s*\d+/(\d+)', line.strip())
            if match:
                return f"progress_{match.group(1)}"
        
        # Match progress bar patterns and extract prefix
        patterns = [
            r'^(.*?):\s*\d+%\|',  # "Loading checkpoint shards: 100%|"
            r'^(.*?):\s*\d+%',    # "Loading: 100%"
            r'^(.*?)\s+\d+%\|',   # "Training 100%|"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                return match.group(1).strip()
        
        return line.strip()[:30]
    
    def is_progress_line(self, line):
        """Check if the line is a progress bar line"""
        # Don't treat EnhancedProgressMeter output as progress line to be filtered
        # EnhancedProgressMeter format: "Epoch: [0] [ 17/500] █▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 3.4%"
        if re.search(r'Epoch:\s*\[\d+\]\s*\[\s*\d+/\d+\]\s*[█▒]+\s*\d+\.\d+%', line.strip()):
            return False
        
        progress_indicators = [
            '\r', '█', '▒', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '░',
            'it/s', '%|', 'ETA:', '/s]', 'B/s', 'MB/s'
        ]
        
        if not any(indicator in line for indicator in progress_indicators):
            return False
        
        stripped = line.strip()
        return len(stripped) < 150 and ('█' in stripped or '▒' in stripped or '░' in stripped or '%|' in stripped)
    
    def is_final_progress(self, line):
        """Check if the line is a final state progress bar"""
        if not line or '100%' not in line:
            return False
        
        # Check if it's truly complete (both 100% and full progress bar)
        # Look for pattern like "100%|██████████| 200/200" (full bar, same numbers)
        match = re.search(r'100%\|██████████\|\s*(\d+)/(\d+)', line.strip())
        if match:
            current, total = match.groups()
            return current == total
        
        return False
    
    def normalize_progress_line(self, line):
        """Normalize progress line by removing time variations"""
        # Remove time parts that might vary: [01:13<00:00, 2.54it/s]
        # Pattern should match: [time<time, speed]
        normalized = re.sub(r'\[[\d:]+<[\d:]+,\s*[\d.]+it/s\]', '[TIME_REMOVED]', line.strip())
        return normalized
    
    def output_stored_progress(self):
        """Output stored progress bar (if it's in final state)"""
        result = []
        if self.last_progress_line is not None and self.is_final_progress(self.last_progress_line):
            result.append(self.last_progress_line)
        self.last_progress_line = None
        self.last_progress_prefix = None
        return result
    
    def handle_progress_line(self, line):
        """Handle progress bar line - only log final state"""
        # Only process final progress bars, ignore all others
        if self.is_final_progress(line):
            formatted_line = line.replace('\r', '\n') if '\r' in line else line
            normalized_new = self.normalize_progress_line(formatted_line)
            
            if normalized_new not in self.logged_final_progress:
                self.logged_final_progress.add(normalized_new)
                return formatted_line
        
        # Ignore all non-final progress bars
        return None

    def should_log_line(self, line):
        """Determine if this line should be logged to the file"""
        # Handle carriage return lines
        if '\r' in line:
            if self.is_progress_line(line):
                return self.handle_progress_line(line)
            else:
                return line.replace('\r', '\n')
        
        # Handle regular progress bar lines
        if self.is_progress_line(line):
            return self.handle_progress_line(line)
        
        # Handle non-progress bar lines - don't output stored progress anymore
        # Just return the line as is
        return line
    
    def _get_description(self):
        if self.debug:
            return ""
        # Get user experiment description before starting redirection
        print("💬 Add a note for this run! (Enter ↵ to skip, Enter ↵ twice to finish):")
        description_lines = []
        last_line = ""
    
        import sys
        try:
            while True:
                try:
                    # Use sys.stdin.readline() instead of input() to handle encoding issues
                    line = sys.stdin.readline().rstrip('\n')
                    if not line and (not last_line or not description_lines):  # Empty line and no previous input
                        break
                    if not line and last_line == "":  # Two consecutive empty lines
                        break
                    description_lines.append(line)
                    last_line = line
                except UnicodeDecodeError:
                    print("Warning: Encoding issue detected. Continuing...")
                    continue
                except EOFError:
                    break
        except Exception as e:
            print(f"Error reading input: {e}")
            description_lines = []  # Reset in case of error
        
        return "\n".join(description_lines)

    def start_redirection(self):
        """Start redirection"""
        if self.is_active:
            return
        
        # Save original file descriptors
        self.original_stdout_fd = os.dup(1)  # stdout
        self.original_stderr_fd = os.dup(2)  # stderr
        
        # Create a pipe for tee functionality
        read_fd, write_fd = os.pipe()
                
        # Fork a child process to implement tee functionality
        self.tee_pid = os.fork()
        if self.tee_pid == 0:
            # Child process: implement tee functionality
            os.close(write_fd)  # Child process doesn't need write end
            
            with open(self.raw_log_filename, 'w') as log_file:
                log_file.write("=== Training Session Started ===\n")
                description = self._get_description()
                if description.strip():
                    log_file.write(f'+>>{description}<<+\n')
                log_file.write("================================\n\n")
                log_file.flush()
                
                buffer = ""
                last_progress_line = None
                last_progress_prefix = None
                logged_final_progress = set()  # Track already logged final progress bars
                
                def is_progress_line(line):
                    """Check if the line is a progress bar line"""
                    # Don't treat EnhancedProgressMeter output as progress line to be filtered
                    # EnhancedProgressMeter format: "Epoch: [0] [ 17/500] █▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 3.4%"
                    if re.search(r'Epoch:\s*\[\d+\]\s*\[\s*\d+/\d+\]\s*[█▒]+\s*\d+\.\d+%', line.strip()):
                        return False
                    
                    progress_indicators = [
                        '\r', '█', '▒', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '░',
                        'it/s', '%|', 'ETA:', '/s]', 'B/s', 'MB/s'
                    ]
                    
                    if not any(indicator in line for indicator in progress_indicators):
                        return False
                    
                    stripped = line.strip()
                    return len(stripped) < 150 and ('█' in stripped or '▒' in stripped or '░' in stripped or '%|' in stripped)
                
                def is_final_progress(line):
                    """Check if the line is a final state progress bar"""
                    if not line or '100%' not in line:
                        return False
                    
                    # Check if it's truly complete (both 100% and full progress bar)
                    # Look for pattern like "100%|██████████| 200/200" (full bar, same numbers)
                    match = re.search(r'100%\|██████████\|\s*(\d+)/(\d+)', line.strip())
                    if match:
                        current, total = match.groups()
                        return current == total
                    
                    return False
                
                def normalize_progress_line(line):
                    """Normalize progress line by removing time variations"""
                    # Remove time parts that might vary: [01:13<00:00, 2.54it/s]
                    # Pattern should match: [time<time, speed]
                    normalized = re.sub(r'\[[\d:]+<[\d:]+,\s*[\d.]+it/s\]', '[TIME_REMOVED]', line.strip())
                    return normalized
                
                def output_stored_progress():
                    """Output stored progress bar (if it's in final state)"""
                    nonlocal last_progress_line, last_progress_prefix
                    result = []
                    if last_progress_line is not None and is_final_progress(last_progress_line):
                        result.append(last_progress_line)
                    last_progress_line = None
                    last_progress_prefix = None
                    return result
                
                def handle_progress_line(line):
                    """Handle progress bar line - only log final state"""
                    # Only process final progress bars, ignore all others
                    if is_final_progress(line):
                        formatted_line = line.replace('\r', '\n') if '\r' in line else line
                        normalized_new = normalize_progress_line(formatted_line)
                        
                        if normalized_new not in logged_final_progress:
                            logged_final_progress.add(normalized_new)
                            return formatted_line
                    
                    # Ignore all non-final progress bars
                    return None
                
                def should_log_line(line):
                    """Determine if this line should be logged to the file"""
                    nonlocal last_progress_line, last_progress_prefix
                    
                    # Handle carriage return lines
                    if '\r' in line:
                        if is_progress_line(line):
                            return handle_progress_line(line)
                        else:
                            return line.replace('\r', '\n')
                    
                    # Handle regular progress bar lines
                    if is_progress_line(line):
                        return handle_progress_line(line)
                    
                    # Handle non-progress bar lines - don't output stored progress anymore
                    # Just return the line as is
                    return line
                
                try:
                    while True:
                        data = os.read(read_fd, 8192)
                        if not data:
                            break
                        
                        # Write to original stdout simultaneously (maintain real-time display)
                        os.write(self.original_stdout_fd, data)
                        
                        # Handle log file writing (filter progress bars)
                        text = data.decode('utf-8', errors='replace')
                        buffer += text
                        
                        # Process line by line
                        while '\n' in buffer or '\r' in buffer:
                            if '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line += '\n'
                            else:  # '\r' in buffer
                                line, buffer = buffer.split('\r', 1)
                                line += '\r'
                            
                            # Determine if this line should be logged
                            filtered_line = should_log_line(line)
                            if filtered_line is not None:
                                log_file.write(filtered_line)
                                log_file.flush()
                        
                except (OSError, KeyboardInterrupt):
                    pass
                finally:
                    # Write remaining buffer content
                    if buffer.strip():
                        log_file.write(buffer)
                    
                    # Handle possibly remaining progress bars
                    remaining_progress = output_stored_progress()
                    if remaining_progress:
                        log_file.write(''.join(remaining_progress))
                    
                    log_file.write("\n=== Training Session Ended ===\n")
            
            os.close(read_fd)
            os.close(self.original_stdout_fd)
            os.close(self.original_stderr_fd)
            os._exit(0)
        else:
            # Parent process: redirect stdout and stderr to pipe
            os.close(read_fd)  # Parent process doesn't need read end
            
            # Redirect stdout and stderr to pipe's write end
            os.dup2(write_fd, 1)  # stdout -> pipe
            os.dup2(write_fd, 2)  # stderr -> pipe
            os.close(write_fd)
            
            self.is_active = True
            
            # Register cleanup function
            atexit.register(self.cleanup)
    
    def cleanup(self):
        """Clean up resources"""
        if not self.is_active:
            return
        
        try:
            # Restore original stdout/stderr
            os.dup2(self.original_stdout_fd, 1)
            os.dup2(self.original_stderr_fd, 2)
            os.close(self.original_stdout_fd)
            os.close(self.original_stderr_fd)
            
            # Terminate tee child process
            if self.tee_pid:
                os.kill(self.tee_pid, 15)  # SIGTERM
                os.waitpid(self.tee_pid, 0)
        except:
            pass
        
        self.is_active = False
    
    def __enter__(self):
        """Support context manager"""
        self.start_redirection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager"""
        self.cleanup()
