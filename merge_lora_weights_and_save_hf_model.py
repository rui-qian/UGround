import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.model_factory import ModelFactory
from dataloaders.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, TrainingLogger


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./pixellm_model", type=str, required=True)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--model_key", default="pixellm", type=str, help="model key for ModelFactory")
    
    # ppm
    parser.add_argument("--num_layers", default=33, type=int)
    parser.add_argument("--strategy", default="policy_walker", type=str)
    parser.add_argument("--mode", default=3, type=int)
    parser.add_argument("--baseline_type", default="ema", type=str)
    parser.add_argument("--baseline_beta", default=1.0, type=float)
    parser.add_argument("--eval_legacy", default=True, action="store_true")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.seg_token_num*args.image_feature_scale_num == 1:
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        num_added_tokens = tokenizer.add_tokens("[REJ]")
        args.rej_token_idx = tokenizer("[REJ]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num*args.image_feature_scale_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]

        new_tokens = ["[REJ{}]".format(i) for i in range(args.seg_token_num*args.image_feature_scale_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.rej_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]


    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # Use ModelFactory to create model
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "rej_token_idx": args.rej_token_idx,
        "vision_tower": args.vision_tower,
        "seg_token_num": args.seg_token_num,
        "image_feature_scale_num": args.image_feature_scale_num,
        "pad_train_clip_images": args.pad_train_clip_images,
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,
        "separate_mm_projector": args.separate_mm_projector,
        "num_layers": args.num_layers, # ppm    
        "strategy": args.strategy, # ppm
        "mode": args.mode, # ppm
        "baseline_type": args.baseline_type, # ppm
        "baseline_beta": args.baseline_beta, # ppm
        "eval_legacy": args.eval_legacy, # ppm
    }
    
    model_class, merged_params = ModelFactory.create_model(
        args.model_key,
        **kwargs
    )

    # Create model instance with merged parameters
    model = model_class.from_pretrained(args.version, **merged_params)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model.get_model().initialize_decoder_modules(model.get_model().config)

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "mask_decoder",
                                "image_feature_neck",
                                "prompt_encoder",
                                "out_mm_projector",
                                "ppm"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        # Use our beautiful formatted print function
        training_logger = TrainingLogger(args)
        training_logger.print_trainable_parameters(model)

    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
