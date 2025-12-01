import argparse
import os
import sys
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.model_factory import ModelFactory
from configs.model_config import DEFAULT_MODEL_KEY
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from model.segment_anything.utils.transforms import ResizeLongestSide
from dataloaders.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from transformers import TextStreamer

def parse_args(args):
    parser = argparse.ArgumentParser(description="UGround: Towards Unified Visual Grounding with Unrolled Transformers")
    parser.add_argument("--version", default="xinlai/PixelLM-13B-llama2-v1")
    parser.add_argument("--model_key", default=DEFAULT_MODEL_KEY, type=str,
                        help="Model key to use for chat")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)

    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def build_model(args):

    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        use_legacy=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    if args.seg_token_num * args.image_feature_scale_num == 1:
        tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        args.rej_token_idx = tokenizer("[REJ]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num * args.image_feature_scale_num)]
        tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(t, add_special_tokens=False).input_ids[0] for t in new_tokens]
        args.rej_token_idx = [tokenizer(t, add_special_tokens=False).input_ids[0] for t in new_tokens]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
        
    kwargs = {
        "torch_dtype": torch_dtype,
        "seg_token_num": args.seg_token_num,
        "image_feature_scale_num": args.image_feature_scale_num,
        "pad_train_clip_images": args.pad_train_clip_images,
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,
        "separate_mm_projector": args.separate_mm_projector,
        "rej_token_idx": args.rej_token_idx,
    }
    
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model_class, merged_kwargs = ModelFactory.create_model(
        args.model_key, 
        vision_tower=args.vision_tower, 
        seg_token_idx=args.seg_token_idx,  
        **kwargs
    )
    
    model = model_class.from_pretrained(
        args.version, device_map="auto", low_cpu_mem_usage=True, **merged_kwargs
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = (
        CLIPImageProcessor.from_pretrained(model.config.vision_tower) 
        if args.preprocessor_config == '' 
        else CLIPImageProcessor.from_pretrained(args.preprocessor_config)
    )
    
    transform = ResizeLongestSide(args.image_size)
    transform_clip = None
    if args.pad_train_clip_images:
        transform_clip = ResizeLongestSide(clip_image_processor.size['shortest_edge'])
    
    model.eval()
    
    return model, tokenizer, clip_image_processor, transform, transform_clip


def process_text_output(raw_text, args):
    COLOR_PALETTE = [
        {"rgb": [255, 0, 0], "class": "red"},
        {"rgb": [0, 255, 0], "class": "green"},
        {"rgb": [0, 0, 255], "class": "blue"},
        {"rgb": [255, 255, 0], "class": "yellow"},
        {"rgb": [128, 0, 128], "class": "purple"}
    ]

    seg_matches = list(re.finditer(r'\[SEG(\d*)\]', raw_text))
    if not seg_matches:
        return re.sub(r'\[SEG\d*\]', '', raw_text).replace('</s>', '').strip(), {}

    masks_per_object = args.seg_token_num * args.image_feature_scale_num
    color_mapping = {}
    current_color_index = 0
    
    processed_segments = []
    last_end = 0
    
    for j, match in enumerate(seg_matches):
        seg_id = match.group(1)
        start, end = match.span()
        
        if start > last_end:
            processed_segments.append(raw_text[last_end:start])
        
        obj_index = j // masks_per_object
        
        if obj_index not in color_mapping:
            color_mapping[obj_index] = {
                "color": COLOR_PALETTE[current_color_index % len(COLOR_PALETTE)]["rgb"],
                "class": COLOR_PALETTE[current_color_index % len(COLOR_PALETTE)]["class"]
            }
            current_color_index += 1
        
        processed_segments.append(f'[SEG{seg_id}]')
        last_end = end
    
    if last_end < len(raw_text):
        processed_segments.append(raw_text[last_end:])
    
    text_output = ''.join(processed_segments)
    text_output = text_output.replace('</s>', '').strip()
    return text_output, color_mapping


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    model, tokenizer, clip_image_processor, transform, transform_clip = build_model(args)

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        
        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        
        if args.pad_train_clip_images:
            image_clip = transform_clip.apply_image(image_np)
            clip_resize = image_clip.shape[:2]
            image_clip = preprocess(
                torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), 
                img_size=clip_image_processor.size['shortest_edge']
            )
            image_clip = image_clip.unsqueeze(0).cuda()
        else:
            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )
            clip_resize = image_clip.shape[-2:]
            
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        clip_resize = [clip_resize]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        
        output_ids, pred_masks, _, _ = model.evaluate(
            images_clip=image_clip,
            images=image,
            input_ids=input_ids,
            resize_list=resize_list,
            clip_resize_list=clip_resize,
            original_size_list=original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
            sam_mask_shape_list=[[*resize_list, *original_size_list]]
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output, color_mapping = process_text_output(raw_text, args)
        print("Text output: ", text_output)
        
        overlay = image_np.copy()
        valid_masks = 0
        combined_mask = np.zeros(original_size_list[0], dtype=bool)
        
        for i, _pred_mask in enumerate(pred_masks):
            if _pred_mask.shape[0] == 0:
                continue
                
            for j, pred_mask in enumerate(_pred_mask if _pred_mask.dim() == 3 else _pred_mask.unsqueeze(0)):
                pred_mask = pred_mask.float().detach().cpu().numpy()
                pred_mask = pred_mask > 0
                combined_mask = np.logical_or(combined_mask, pred_mask)
                
                save_path = "{}/{}_mask_{}.jpg".format(
                    args.vis_save_path, os.path.basename(image_path).split(".")[0], j
                )
                cv2.imwrite(save_path, pred_mask * 100)
                print("{} has been saved.".format(save_path))
                
                if len(color_mapping) == 0:
                    color_rgb = [255, 0, 0] 
                else:
                    color_rgb = color_mapping[valid_masks % len(color_mapping)]["color"]
                
                save_path = "{}/{}_masked_img_{}.jpg".format(
                    args.vis_save_path, os.path.basename(image_path).split(".")[0], j
                )
                save_img = image_np.copy()
                save_img[pred_mask] = (
                    image_np * 0.6 + 
                    pred_mask[:, :, None].astype(np.uint8) * np.array(color_rgb) * 0.4
                )[pred_mask]
                
                contour_mask = (pred_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(save_img, contours, -1, (255, 255, 255), thickness=2)
                
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, save_img)
                print("{} has been saved.".format(save_path))
                valid_masks += 1


if __name__ == "__main__":
    main(sys.argv[1:])
