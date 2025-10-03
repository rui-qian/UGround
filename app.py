import os
import re
import cv2
import argparse
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from model.model_factory import ModelFactory
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from dataloaders.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

def parse_args():
    parser = argparse.ArgumentParser(description="UGround: Towards Unified Visual Grounding with Large Multimodal Models")

    parser.add_argument("--model_key", default="pixellm", type=str, help="Model key from model registry")
    parser.add_argument("--version", default="xinlai/PixelLM-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--preprocessor_config", default="", type=str)
    parser.add_argument("--resize_vision_tower", action="store_true")
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true")
    parser.add_argument("--pad_train_clip_images", action="store_true")
    parser.add_argument("--separate_mm_projector", action="store_true")
    parser.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args()

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
):
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, img_size - w, 0, img_size - h))
    return x

def build_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
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

    torch_dtype = (
        torch.bfloat16 if args.precision == "bf16"
        else torch.half if args.precision == "fp16"
        else torch.float32
    )

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
        kwargs.update({
            "load_in_4bit": True,
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "load_in_8bit": True,
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    
    model_class, merged_params = ModelFactory.create_model(
        args.model_key,
        **kwargs
    )
    
    model = model_class.from_pretrained(
        args.version,
        device_map="auto",
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        low_cpu_mem_usage=True,
        **merged_params,
    )

    model.get_model().initialize_vision_modules(model.get_model().config)
    model = model.eval().cuda().to(dtype=torch_dtype)

    clip_image_processor = (
        CLIPImageProcessor.from_pretrained(model.config.vision_tower)
        if args.preprocessor_config == ''
        else CLIPImageProcessor.from_pretrained(args.preprocessor_config)
    )
    transform = ResizeLongestSide(args.image_size)
    if args.pad_train_clip_images:
        transform_clip = ResizeLongestSide(clip_image_processor.size['shortest_edge'])

    return model, tokenizer, clip_image_processor, transform, transform_clip if args.pad_train_clip_images else None

def create_inference_function(args, model, tokenizer, clip_image_processor, transform, transform_clip=None):
    
    COLOR_PALETTE = [
        {"rgb": [255, 0, 0], "class": "highlight-red"},    
        {"rgb": [0, 255, 0], "class": "highlight-green"},  
        {"rgb": [0, 0, 255], "class": "highlight-blue"},   
        {"rgb": [255, 255, 0], "class": "highlight-yellow"}, 
        {"rgb": [128, 0, 128], "class": "highlight-purple"} 
    ]

    def process_text_output(raw_text):
    
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
            
            color_class = color_mapping[obj_index]["class"]
            processed_segments.append(f'<span class="{color_class}">[SEG{seg_id}]</span>')
            last_end = end
        
        if last_end < len(raw_text):
            processed_segments.append(raw_text[last_end:])
        
        text_output = ''.join(processed_segments)
        
        text_output = text_output.replace('</s>', '').strip()
        return text_output, color_mapping

    def inference(prompt, input_image):
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return cv2.imread("./assets/no_seg_output.png")[:, :, ::-1], "Invalid input"

        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            prompt_text = prompt_text.replace(
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
            )
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], "")
        prompt_text = conv.get_prompt()
        
        image_np = np.array(input_image)
        original_size = image_np.shape[:2]
        original_size_list = [original_size]

        # Process image for CLIP
        if args.pad_train_clip_images:
            image_clip = transform_clip.apply_image(image_np)
            clip_resize = image_clip.shape[:2]
            image_clip = preprocess(
                torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(),
                img_size=clip_image_processor.size['shortest_edge']
            ).unsqueeze(0).cuda().to(model.dtype)
        else:
            image_clip = clip_image_processor(image_np, return_tensors="pt")["pixel_values"][0]
            clip_resize = image_clip.shape[-2:]
            image_clip = image_clip.unsqueeze(0).cuda().to(model.dtype)

        # Process main image
        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        clip_resize = [clip_resize]
        image = preprocess(
            torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            img_size=args.image_size
        ).unsqueeze(0).cuda().to(model.dtype)

        input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
        
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
        text_output, color_mapping = process_text_output(raw_text.split("ASSISTANT: ")[-1])

        overlay = image_np.copy()
        valid_masks = 0
        combined_mask = np.zeros(original_size, dtype=bool)
        
        for i, _pred_mask in enumerate(pred_masks):
            if _pred_mask.shape[0] == 0:
                continue
            for j, pred_mask in enumerate(_pred_mask if _pred_mask.dim() == 3 else _pred_mask.unsqueeze(0)):
                
                pred_mask = pred_mask.float().detach().cpu().numpy()
                pred_mask = pred_mask > 0
                combined_mask = np.logical_or(combined_mask, pred_mask)
                
                if len(color_mapping) == 0:
                    color_rgb = [0, 0, 0]
                else:
                    color_rgb = color_mapping[valid_masks % len(color_mapping)]["color"]
                overlay[pred_mask] = (image_np * 0.6 + pred_mask[:, :, None] * color_rgb * 0.4)[pred_mask]
                
                contour_mask = (pred_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), thickness=2)
                valid_masks += 1

        if not np.any(combined_mask):
            return cv2.imread("./assets/no_seg_output.png")[:, :, ::-1], text_output

        return overlay, text_output

    return inference

import os
import cv2

import time
import os
import cv2
import numpy as np
import re

def save_data(input_image, seg_image, text_instruction, lang_output):
    if input_image is None:
        return "Please upload an image!", None, ""
    if seg_image is None:
        return "No segmentation result!", None, ""

    if not isinstance(input_image, np.ndarray):
        try:
            input_image = np.array(input_image)
        except Exception as e:
            return f"Image conversion failed: {e}", None, ""

    if input_image.ndim == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    save_dir = "./saved"
    os.makedirs(save_dir, exist_ok=True)

    original_path = os.path.join(save_dir, f"{timestamp}_original.png")
    cv2.imwrite(original_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

    seg_path = os.path.join(save_dir, f"{timestamp}_segmentation.png")
    if seg_image is not None:
        cv2.imwrite(seg_path, cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))

    instruction_path = os.path.join(save_dir, f"{timestamp}_instruction.txt")
    with open(instruction_path, "w", encoding="utf-8") as f:
        f.write(text_instruction or "")

    lang_output_path = os.path.join(save_dir, f"{timestamp}_language_output.txt")
    lang_text = re.sub('<[^<]+?>', '', lang_output or "")
    with open(lang_output_path, "w", encoding="utf-8") as f:
        f.write(lang_text)

    timestamp = time.strftime("%Y年%m月%d日%H时%M分%S秒")

    save_msg = (
        f"保存成功：{timestamp}<br>"
        f"<ul>"
        f"<li>原图: <code>{original_path}</code></li>"
        f"<li>分割图: <code>{seg_path}</code></li>"
        f"<li>指令: <code>{instruction_path}</code></li>"
        f"<li>语言输出: <code>{lang_output_path}</code></li>"
        f"</ul>"
    )

    history_html = load_history_from_disk()
    return save_msg, history_html, ""

import os
import glob
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime

def image_to_base64_html(img_path):
    try:
        with Image.open(img_path) as img:
            img.thumbnail((256, 256))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"<img src='data:image/png;base64,{encoded}' style='max-width:256px; margin:4px;'/>"
    except:
        return "(图像读取失败)"

def load_history_from_disk():
    save_dir = "./saved"
    if not os.path.exists(save_dir):
        return "暂无历史记录"

    original_files = sorted(glob.glob(os.path.join(save_dir, "*_original.png")), reverse=True)
    html_entries = []

    for orig_path in original_files:
        basename = os.path.basename(orig_path)
        timestamp = basename.replace("_original.png", "")

        seg_path = os.path.join(save_dir, f"{timestamp}_segmentation.png")
        instruction_path = os.path.join(save_dir, f"{timestamp}_instruction.txt")
        lang_output_path = os.path.join(save_dir, f"{timestamp}_language_output.txt")

        instruction = ""
        if os.path.exists(instruction_path):
            with open(instruction_path, "r", encoding="utf-8") as f:
                instruction = f.read()

        lang_output = ""
        if os.path.exists(lang_output_path):
            with open(lang_output_path, "r", encoding="utf-8") as f:
                lang_output = f.read()
        
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        timestamp = dt.strftime("%Y年%m月%d日 %H时%M分%S秒")

        entry_html = f"""
        <div style="border:1px solid #ccc; padding:10px; margin:10px 0;">
            <strong>保存时间：</strong> {timestamp}<br>
            <div style="display:flex; gap:10px;">
                {image_to_base64_html(orig_path)}
                {image_to_base64_html(seg_path) if os.path.exists(seg_path) else ""}
            </div>
            <div><strong>指令内容：</strong><pre style="white-space: pre-wrap;">{instruction}</pre></div>
            <div><strong>语言输出：</strong><pre style="white-space: pre-wrap;">{lang_output}</pre></div>
        </div>
        """
        html_entries.append(entry_html)

    return "".join(html_entries)

import shutil
import os

def show_confirm_btn():
    # Hide "Delete" button, show "Confirm Delete" button, clear status message
    return gr.update(visible=False), gr.update(visible=True), ""

def delete_history():
    save_dir = "./saved"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    return gr.update(visible=True), gr.update(visible=False), "历史已清空", load_history_from_disk()

def main():
    args = parse_args()
    os.makedirs(args.vis_save_path, exist_ok=True)
    model, tokenizer, clip_image_processor, transform, transform_clip = build_model(args)
    inference_fn = create_inference_function(args, model, tokenizer, clip_image_processor, transform, transform_clip)

    css = """
    .highlight-red { color: white; background-color: #FF0000; padding: 0 4px; border-radius: 4px; font-weight: bold; }
    .highlight-green { color: white; background-color: #00AA00; padding: 0 4px; border-radius: 4px; font-weight: bold; }
    .highlight-blue { color: white; background-color: #0000FF; padding: 0 4px; border-radius: 4px; font-weight: bold; }
    .highlight-yellow { color: #333; background-color: #FFFF00; padding: 0 4px; border-radius: 4px; font-weight: bold; }
    .highlight-purple { color: white; background-color: #800080; padding: 0 4px; border-radius: 4px; font-weight: bold; }
#logo-title-row {
    align-items: center;
    justify-content: center;
    gap: 12px;
}
#logo-title-row img {
    max-width: 40px;
    max-height: 40px;
}
#btn-row-left, #btn-row-right {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 10px;
}
#confirm-delete-btn {
    color: #d32f2f !important;
    font-weight: bold;
}
    """

    with gr.Blocks(css=css, title="UGround: Towards Unified Visual Grounding with Large Multimodal Models") as demo:
        with gr.Row(elem_id="logo-title-row"):
            # gr.HTML('''
            # <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
            #     <img src="file/assets/logo.png" style="height:50px; margin-right: 10px;">
            #     <h2 style="margin: 0;">PixelLM Web Demo</h2>
            # </div>
            # ''')
            gr.HTML(f'''
            <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
                <h2 style="margin: 0;">UGround: Towards Unified Visual Grounding with Large Multimodal Models - {args.model_key.upper()}</h2>
            </div>
            ''')
 
        gr.HTML('''
        <div style="display: flex; align-items: center; justify-content: flex-start;">
        <img src="file/assets/logo.png" style="height:40px; margin-right:15px;">
        <span>Upload an image and enter a question or instruction.</span>
        </div>
        ''')
        with gr.Row():
            with gr.Column(scale=1):
                inp_text = gr.Textbox(lines=2, label="Text Instruction")
                inp_img = gr.Image(type="pil", label="Upload Image")
                
                with gr.Row(elem_id="btn-row-left"):
                    clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit")

            with gr.Column(scale=1):
                out_seg = gr.Image(type="numpy", label="Segmentation Output", interactive=False)
                out_text = gr.HTML(
                    label="Language Output"
                )

                with gr.Row(elem_id="btn-row-right"):
                    save_btn = gr.Button("Save")
                save_status = gr.HTML(label="Save Status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["Which equipment is closer to the man's mouth and which clothing item is he wearing while playing? Responses with segmentation mask.", "./assets/000000398905.jpg"],
                ["If you had to differentiate between the two giraffes based on their proximity to the ostrich, which one is farther and which one is closer? Responses with segmentation mask.", "./assets/000000285788.jpg"],
                ["If you were to take a bath and then step out to dry off, which two items would you interact with directly? Responses with segmentation mask.", "./assets/000000262440.jpg"],
                ["Imagine you want to have a pleasant indoor garden scene for breakfast. Which items in the image can contribute to that ambiance? Responses with segmentation mask.", "./assets/000000136355.jpg"],
                [" If a child wanted to play with Lego while switching channels on the TV, which items from the image would he need in his hands? Responses with segmentation mask.", "./assets/000000229111.jpg"],
                ["If you had to wash your hands while observing the time, which two objects would you utilize? Responses with segmentation mask.", "./assets/000000074209.jpg"],
                ["How can I send an email without being interrupted? Responses with segmentation mask.", "./assets/cat.jpeg"],
                ["How you see a cat in the picture? Responses with segmentation mask.", "./assets/cat.jpeg"],
                ["how you see the laptop in the picture? Responses with segmentation mask.", "./assets/cat.jpeg"],
                ["who was the president of the US in this image? Please output segmentation mask.", "./assets/trump.jpg"],
                ["Can you segment the founder of Alibaba in this image?", "./assets/jackma.jpg"],
                ["Can you segment the food that tastes spicy and hot?", "./assets/example2.jpg"],
                ["Where can the driver see the car speed in this image? Please output segmentation mask.", "./assets/example1.jpg"],
            ],
            inputs=[inp_text, inp_img],
        )

        gr.Markdown("### History")
        with gr.Row():
            refresh_button = gr.Button("Refresh History")
            delete_btn = gr.Button("Delete All History")
            confirm_delete_btn = gr.Button("Confirm Delete", visible=False, elem_id="confirm-delete-btn")
        delete_status = gr.HTML()
        history_html = gr.HTML()

        refresh_button.click(fn=load_history_from_disk, outputs=history_html)

        delete_btn.click(
            fn=show_confirm_btn,
            outputs=[delete_btn, confirm_delete_btn, delete_status]
        )
        confirm_delete_btn.click(
            fn=delete_history,
            outputs=[delete_btn, confirm_delete_btn, delete_status, history_html]
        )
        demo.load(fn=load_history_from_disk, outputs=history_html)

        submit_btn.click(fn=inference_fn, inputs=[inp_text, inp_img], outputs=[out_seg, out_text])
        clear_btn.click(
            fn=lambda: (None, "", None, "", ""),
            outputs=[inp_img, inp_text, out_seg, out_text, save_status]
        )
        
        save_btn.click(
            fn=save_data,
            inputs=[inp_img, out_seg, inp_text, out_text],
            outputs=[save_status, history_html, delete_status]
        )
        
        gr.HTML(
            '''
            <div style="width:100%;text-align:center;background:#f5f5f5;color:#222;padding:8px 0;font-size:14px;">
                © 2025 Rui Qian. All rights reserved.
            </div>
            '''
        )
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_str = s.getsockname()[0]
    s.close()
    demo.queue()
    demo.launch(
        server_name=ip_str,
        share=True,
        server_port=7861,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        app_kwargs={"docs_url": None},
    )
if __name__ == "__main__":
    main()
    try:
        import time
        while True:
            time.sleep(3600) 
    except KeyboardInterrupt:
        print("Server stopped")
    