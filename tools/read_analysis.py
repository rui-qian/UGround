import os
import cv2
import json
import glob
import time
import argparse
import numpy as np
import gradio as gr
from datetime import datetime
from PIL import Image
from pathlib import Path
import threading
import queue
from collections import OrderedDict
import torch
import torch.nn.functional as F
import re
import json

class AnalysisSaver:
    def __init__(self, debug=False, use_sam=True):
        self.debug = debug
        self.use_sam = use_sam
        if not debug:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.sam = None
            self.predictor = None
            
            if use_sam:
                try:
                    from model.segment_anything import sam_model_registry, SamPredictor
                    sam_checkpoint = "../dataset_sesame/sam_vit_h_4b8939.pth"
                    model_type = "vit_h"
                    self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    self.sam.to(self.device)
                    self.predictor = SamPredictor(self.sam)
                    print("segment_anything module loaded.")
                except ImportError:
                    print("segment_anything module is not installed.")
                    self.use_sam = False
            else:
                print("SAM model loading disabled.")
            
            import os
            self.save_dir = "./vis_output"
            os.makedirs(self.save_dir, exist_ok=True)
            
            self.sample_counter = 0
            
            self.image_paths_map = {}
            
            print("Using synchronous saving mode.")
            
        except Exception as e:
            print(f"Failed to initialize visualizer: {e}")
            import traceback
            traceback.print_exc()
    
    def _do_save_sample(self, sample_id, ori_image, similarity_map, sam_mask, pred_mask, gt_mask, conversation, points, labels, dataset_path):
        try:
            ori_path = os.path.join(self.save_dir, f"{sample_id}_original.png")
            cv2.imwrite(ori_path, ori_image)
            
            sim_vis = (similarity_map * 255).astype('uint8')
            sim_vis = cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET)
            sim_vis = ori_image * 0.3 + sim_vis * 0.7
            sim_path = os.path.join(self.save_dir, f"{sample_id}_similarity.png")
            cv2.imwrite(sim_path, sim_vis)
            
            sam_path = None
            if sam_mask is not None:
                if not isinstance(sam_mask, np.ndarray) or sam_mask.dtype != np.uint8:
                    sam_mask = (sam_mask > 0).astype(np.uint8)
                sam_vis = self.process_image_for_display(ori_image, sam_mask, [0, 0, 255], points=points, labels=labels)
                sam_path = os.path.join(self.save_dir, f"{sample_id}_sam.png")
                cv2.imwrite(sam_path, cv2.cvtColor(sam_vis, cv2.COLOR_RGB2BGR))
            
            pred_path = None
            if pred_mask is not None:
                if not isinstance(pred_mask, np.ndarray) or pred_mask.dtype != np.uint8:
                    pred_mask = (pred_mask > 0).astype(np.uint8)
                pred_vis = self.process_image_for_display(ori_image, pred_mask, [0, 0, 255], points=points, labels=labels)
                pred_path = os.path.join(self.save_dir, f"{sample_id}_pred.png")
                cv2.imwrite(pred_path, cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))
            
            gt_path = None
            if gt_mask is not None:
                if not isinstance(gt_mask, np.ndarray) or gt_mask.dtype != np.uint8:
                    gt_mask = (gt_mask > 0).astype(np.uint8)
                gt_vis = self.process_image_for_display(ori_image, gt_mask, [0, 255, 0], points=points, labels=labels)
                gt_path = os.path.join(self.save_dir, f"{sample_id}_gt.png")
                cv2.imwrite(gt_path, cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))
            
            conv_path = os.path.join(self.save_dir, f"{sample_id}_conversation.txt")
            with open(conv_path, "w", encoding="utf-8") as f:
                f.write(conversation)
            
            metadata = {
                "id": sample_id,
                "conversation": conversation,
                "original_image": ori_path,
                "similarity_map": sim_path,
                "sam_prediction": sam_path,
                "model_prediction": pred_path,
                "ground_truth": gt_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_path": dataset_path if dataset_path else ""
            }
           
            meta_path = os.path.join(self.save_dir, f"{sample_id}_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"保存样本 {sample_id}:")
            print(f"  - 原始图像: {ori_path}")
            print(f"  - 相似度图: {sim_path}")
            if sam_path:
                print(f"  - SAM预测: {sam_path}")
            if pred_path:
                print(f"  - 模型预测: {pred_path}")
            if gt_path:
                print(f"  - Ground Truth: {gt_path}")
            print(f"  - 对话: {conv_path}")
            print(f"  - 元数据: {meta_path}")
            if dataset_path:
                print(f"  - Dataset路径: {dataset_path}")
        except Exception as e:
            print(f"保存样本 {sample_id} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    # def get_similarity_map(self, sm, shape, target_size):
    #     # min-max norm
    #     sm = sm[None, ...]
    #     sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
    #     # reshape
    #     side = int(sm.shape[1] ** 0.5) # square output
    #     sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    #     # interpolate
    #     sm = sm.to(torch.float32)

    #     h, w = shape
    #     scale = target_size / min(h, w)
    #     new_h, new_w = int(h * scale), int(w * scale)
    #     sm = torch.nn.functional.interpolate(sm, (target_size, target_size), mode='bilinear')
    #     pad_h = (new_h - target_size) // 2
    #     pad_w = (new_w - target_size) // 2
    #     padded_sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
    #     sm = torch.nn.functional.interpolate(padded_sm, shape, mode='bilinear')
    #     sm = sm.permute(0, 2, 3, 1)
    #     return sm
    
    def get_similarity_map(self, sm, shape, target_length = 336):
    
        # min-max norm
        sm = sm[None, ...]
        sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
        # reshape
        side = int(sm.shape[1] ** 0.5) # square output
        sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
        # interpolate
        sm = sm.to(torch.float32)
        sm = torch.nn.functional.interpolate(sm, (target_length, target_length), mode='bilinear')
        
        oldh, oldw = shape
        scale = target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        sm = sm[:, :, 0:newh, 0:neww]
        sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
        sm = sm.permute(0, 2, 3, 1)
        return sm
    
    def visualize_batch(self, 
                        offset=None, 
                        points_list=None, 
                        labels_list=None,
                        similarity=None,
                        pred_masks=None,
                        gt_masks=None,
                        **kwargs
    ):
        if not self.debug or points_list is None or len(points_list) == 0:
            return
            
        try:
            image_paths = kwargs.get('image_paths', [])
            if not image_paths:
                print("No image paths provided")
                return
                
            batch_size = len(image_paths)
            conversation_list = kwargs.get('conversation_list', [])
            clip_resize_list = kwargs.get('clip_resize_list', [])
            
            conversation_list_ = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                conversation_list_.append(conversation_list[start_i:end_i])
            conversation_list = conversation_list_
            
            for bs in range(batch_size):
                ori_image = cv2.imread(image_paths[bs])
                if ori_image is None:
                    print(f"Failed to load image: {image_paths[bs]}")
                    continue
                    
                ori_shape = ori_image.shape[:2]
                target_size = clip_resize_list[bs][0] if bs < len(clip_resize_list) else 224
                conversations = conversation_list[bs] if bs < len(conversation_list) else ["No conversation"]
                
                for N_i in range(points_list[bs].shape[0]):
                    conversation = conversations[N_i] if N_i < len(conversations) else "No conversation"
                    points = points_list[bs][N_i].detach().cpu().numpy().astype(np.float32)
                    labels = labels_list[bs][N_i].clone().cpu().numpy().astype(np.int32)
                
                    if len(labels.shape) > 1:
                        labels = labels.reshape(-1)
                    
                    similarity_map = self.get_similarity_map(similarity[N_i], ori_shape, target_size)
                    
                    sam_mask = None
                    if self.use_sam and self.predictor is not None:
                        try:
                            self.predictor.set_image(ori_image)
                            masks, scores, logits = self.predictor.predict(
                                point_labels=labels, 
                                point_coords=points, 
                                multimask_output=True
                            )
                            sam_mask = masks[np.argmax(scores)]
                            sam_mask = sam_mask.astype('uint8')
                        except Exception as e:
                            print(f"SAM预测失败: {e}")
                    
                    pred_mask = pred_masks[bs][0].detach().cpu().numpy() if pred_masks and bs < len(pred_masks) else None
                    gt_mask = gt_masks[bs][0].detach().cpu().numpy() if gt_masks and bs < len(gt_masks) else None
                    
                    self.save_sample_to_disk(conversation, ori_image, None, 
                                             similarity_map[0, ..., 0].detach().cpu().numpy(),
                                             sam_mask, pred_mask, gt_mask, points, labels,
                                             current_image_path=image_paths[bs])
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def process_image_for_display(self, ori_image, mask, color, add_contour=True, points=None, labels=None):
        vis = ori_image.copy()
        if mask is not None:
            vis[mask > 0] = vis[mask > 0] // 2 + np.array(color, dtype=np.uint8) // 2
            
            if add_contour:
                contour_mask = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (255, 255, 255), thickness=2)
        
        if points is not None and labels is not None:
                    for i, [x, y] in enumerate(points):
                        x, y = int(x), int(y)
                        cv2.circle(vis, (x, y), 3, (255, 255, 255), 3)
                        cv2.circle(vis, (x, y), 2, (0, 0, 255) if labels[i] == 1 else (255, 0, 0), 3)
        
        return cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    
    def save_sample_to_disk(self, conversation, ori_image, result_image, similarity_map, 
                           sam_mask, pred_mask=None, gt_mask=None, points=None, labels=None, **kwargs):
                
        dataset_path = ""
        try:
            if 'current_image_path' in kwargs:
                dataset_path = kwargs['current_image_path']
            elif 'image_paths' in kwargs and kwargs['image_paths']:
                dataset_path = kwargs['image_paths'][-1]
        except Exception as e:
            print(f"无法获取Dataset路径: {e}")
        
        filename = os.path.basename(dataset_path)
        sample_id, ext = os.path.splitext(filename)
        sample_id = f"{sample_id}_{self.sample_counter}"
        self.sample_counter += 1

        ori_image_copy = ori_image.copy() if ori_image is not None else None
        similarity_map_copy = similarity_map.copy() if similarity_map is not None else None
        
        sam_mask_copy = None
        if sam_mask is not None:
            sam_mask_copy = sam_mask.copy()
            if sam_mask_copy.dtype != np.uint8:
                sam_mask_copy = (sam_mask_copy > 0).astype(np.uint8)
        
        pred_mask_copy = None
        if pred_mask is not None:
            pred_mask_copy = pred_mask.copy()
            if pred_mask_copy.dtype != np.uint8:
                pred_mask_copy = (pred_mask_copy > 0).astype(np.uint8)
        
        gt_mask_copy = None
        if gt_mask is not None:
            gt_mask_copy = gt_mask.copy()
            if gt_mask_copy.dtype != np.uint8:
                gt_mask_copy = (gt_mask_copy > 0).astype(np.uint8)
        
        points_copy = points.copy() if points is not None else None
        labels_copy = labels.copy() if labels is not None else None
        
        print(f"Saving sample {sample_id}...")
        self._do_save_sample(
            sample_id,
            ori_image_copy,
            similarity_map_copy,
            sam_mask_copy,
            pred_mask_copy,
            gt_mask_copy,
            conversation,
            points_copy,
            labels_copy,
            dataset_path
        )
        
        print(f"Sample {sample_id} saved successfully")
        
        return sample_id
    
    def save_for_gradio(self, 
                        offset=None, 
                        points_list=None, 
                        labels_list=None,
                        similarity=None,
                        pred_masks=None,
                        gt_masks=None,
                        **kwargs):
        self.visualize_batch(offset, points_list, labels_list, similarity, pred_masks, gt_masks, **kwargs)
        
        print(f"Samples saved to {self.save_dir} directory")
        print("please run ./scripts/read_analysis.sh")
        
        return "Samples saved successfully"

def parse_args():
    parser = argparse.ArgumentParser(description="Reasoning to Attend: Try to Understand How <SEG> Token Works (CVPR 2025)")
    parser.add_argument("--samples_dir", default="./vis_output", type=str, help="Directory containing sample files")
    parser.add_argument("--port", default=7860, type=int, help="Port for Gradio server")
    parser.add_argument("--share", action="store_true", help="Create a public link for sharing")
    return parser.parse_args()

class AnalysisVisualizer:
    def __init__(self, samples_dir):
        self.samples_dir = samples_dir
        self.samples = OrderedDict()
        self.image_cache = {}
        self.load_samples()
        
    def load_samples(self):
        metadata_files = sorted(glob.glob(os.path.join(self.samples_dir, "*_metadata.json")), reverse=True)
        
        self.samples = OrderedDict()
        for meta_file in metadata_files:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                if not os.path.exists(metadata.get("original_image", "")):
                    continue
                    
                sample_id = metadata.get("id")                
                self.samples[sample_id] = metadata
                
            except Exception as e:
                print(f"Failed to load sample {meta_file}: {e}")
        
        print(f"Loaded {len(self.samples)} samples")
        return len(self.samples)
    
    def get_sample_by_index(self, index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = -1
            
        keys = list(self.samples.keys())
        if not keys or index < 0 or index >= len(keys):
            return index, "Sample not found", None, None, None, None, None, "Sample not found"
        
        sample_id = keys[index]
        sample = self.samples.get(sample_id)
        print(f"Loaded sample: {sample_id}, index: {index}")
        
        original_image = self._get_image_path(sample, "original_image")
        similarity_map = self._get_image_path(sample, "similarity_map")
        sam_prediction = self._get_image_path(sample, "sam_prediction")
        model_prediction = self._get_image_path(sample, "model_prediction")
        ground_truth = self._get_image_path(sample, "ground_truth")
        
        original_image_orig = self._get_image_path(sample, "original_image", use_original=True)
        similarity_map_orig = self._get_image_path(sample, "similarity_map", use_original=True)
        sam_prediction_orig = self._get_image_path(sample, "sam_prediction", use_original=True)
        model_prediction_orig = self._get_image_path(sample, "model_prediction", use_original=True)
        ground_truth_orig = self._get_image_path(sample, "ground_truth", use_original=True)
                
        conversation = ""
        conv_path = os.path.join(self.samples_dir, f"{sample['id']}_conversation.txt")
        if os.path.exists(conv_path):
            with open(conv_path, "r", encoding="utf-8") as f:
                conversation = f.read()
        
        dataset_path = sample.get("dataset_path", "")
        if not dataset_path:
            filename = os.path.basename(sample.get("original_image", ""))
            dataset_path = f"(The path in Dataset is unknown, filename: {filename})"
        
        path_info = f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <h4>样本 {sample['id']} 路径信息:</h4>
            <ul style="margin-left: 20px;">
                <li><strong>Dataset中的原始路径:</strong> {dataset_path}</li>
                <li><strong>原始图像:</strong> {sample.get("original_image", "")}</li>
                <li><strong>相似度图:</strong> {sample.get("similarity_map", "")}</li>
                <li><strong>SAM预测:</strong> {sample.get("sam_prediction", "")}</li>
                {"<li><strong>模型预测:</strong> " + sample.get("model_prediction", "") + "</li>" if sample.get("model_prediction") else ""}
                {"<li><strong>Ground Truth:</strong> " + sample.get("ground_truth", "") + "</li>" if sample.get("ground_truth") else ""}
            </ul>
        </div>
        
        <script>
            // 设置原始图像路径，用于全屏显示
            setTimeout(() => {{
                const images = document.querySelectorAll('#images-container img');
                const origPaths = [
                    "{original_image_orig or ''}",
                    "{similarity_map_orig or ''}",
                    "{sam_prediction_orig or ''}",
                    "{model_prediction_orig or ''}",
                    "{ground_truth_orig or ''}"
                ];
                
                images.forEach((img, idx) => {{
                    if (idx < origPaths.length && origPaths[idx]) {{
                        const origPath = origPaths[idx].startsWith('http') ? origPaths[idx] : '/file=' + origPaths[idx];
                        img.setAttribute('data-original-src', origPath);
                    }}
                }});
            }}, 1000);
        </script>
        """
        
        return index, conversation, original_image, similarity_map, sam_prediction, model_prediction, ground_truth, path_info
    
    def _get_image_path(self, sample, key, use_original=False):

        image_path = sample.get(key)
        if image_path and os.path.exists(image_path):
            if use_original:
                return image_path
                
            if image_path in self.image_cache:
                return self.image_cache[image_path]
            
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
            
            if file_size > 0.5:
                try:
                    cache_dir = os.path.join(self.samples_dir, "cache")
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    filename = os.path.basename(image_path)
                    compressed_path = os.path.join(cache_dir, f"compressed_{filename}")
                    
                    if os.path.exists(compressed_path):
                        self.image_cache[image_path] = compressed_path
                        return compressed_path

                    img = Image.open(image_path)
                    max_size = (600, 600)
                    img.thumbnail(max_size, Image.LANCZOS)
                    img.save(compressed_path, optimize=True, quality=80)
                    
                    print(f"Compressed image: {image_path} -> {compressed_path}")
                    self.image_cache[image_path] = compressed_path
                    return compressed_path
                except Exception as e:
                    print(f"Failed to compress image: {e}")
            
            self.image_cache[image_path] = image_path
            return image_path
        return None
    
    def get_latest_sample(self):
        if not self.samples:
            return -1, "No samples", None, None, None, None, None, "No samples"
        latest_index = 0
        return self.get_sample_by_index(latest_index)
    
    def get_sample_count(self):
        return f"当前共有 {len(self.samples)} 个样本"
    
    def load_prev_sample(self, index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        keys = list(self.samples.keys())
        new_index = min(len(keys) - 1, index + 1)  
        if new_index == index and index < len(keys) - 1:
            new_index = index + 1
        print(f"Load previous sample: from {index} to {new_index}")
        return self.get_sample_by_index(new_index)
    
    def load_next_sample(self, index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        new_index = max(0, index - 1)
        if new_index == index and index > 0:
            new_index = index - 1
        print(f"Load next sample: from {index} to {new_index}")
        return self.get_sample_by_index(new_index)
    
    def refresh_samples(self):
        count = self.load_samples()
        return self.get_sample_count()
    
    def generate_history_gallery(self):
        if not self.samples:
            return "<div>暂无历史记录</div>"
            
        html_content = ""
        
        for i, sample_id in enumerate(self.samples.keys()):
            sample = self.samples[sample_id]
            timestamp = sample.get("timestamp", "未知时间")
            
            conversation = ""
            conv_path = os.path.join(self.samples_dir, f"{sample['id']}_conversation.txt")
            if os.path.exists(conv_path):
                with open(conv_path, "r", encoding="utf-8") as f:
                    conversation = f.read()
        
            original_image = self._get_image_path(sample, "original_image")
            similarity_map = self._get_image_path(sample, "similarity_map")
            sam_prediction = self._get_image_path(sample, "sam_prediction")
            model_prediction = self._get_image_path(sample, "model_prediction")
            ground_truth = self._get_image_path(sample, "ground_truth")
            
            original_image_orig = self._get_image_path(sample, "original_image", use_original=True)
            similarity_map_orig = self._get_image_path(sample, "similarity_map", use_original=True)
            sam_prediction_orig = self._get_image_path(sample, "sam_prediction", use_original=True)
            model_prediction_orig = self._get_image_path(sample, "model_prediction", use_original=True)
            ground_truth_orig = self._get_image_path(sample, "ground_truth", use_original=True)
            
            card_html = f"""
            <div class="sample-card" onclick="loadSample({i})">
                <h4>样本 {i+1}: {sample['id']} - {timestamp}</h4>
                <div style="margin-bottom: 10px;">
                    <strong>对话内容:</strong>
                    <div style="background-color: #f5f5f5; padding: 8px; border-radius: 4px; max-height: 200px; overflow-y: auto;">
                        <pre style="white-space: pre-wrap; margin: 0;">{conversation}</pre>
                    </div>
                </div>
                <div class="images-row">
            """
            
            for img_path, orig_path, label in [
                (original_image, original_image_orig, "原始图像"),
                (similarity_map, similarity_map_orig, "相似度图"),
                (sam_prediction, sam_prediction_orig, "SAM预测"),
                (model_prediction, model_prediction_orig, "模型预测"),
                (ground_truth, ground_truth_orig, "Ground Truth")
            ]:
                if img_path:
                    img_url = img_path
                    if not img_url.startswith("http"):
                        img_url = f"/file={img_path}"
                    
                    orig_url = orig_path
                    if orig_path and not orig_url.startswith("http"):
                        orig_url = f"/file={orig_path}"
                    else:
                        orig_url = img_url
                    
                    border_style = "border: 1px solid #ddd;"
                    if label == "模型预测" or label == "SAM预测" or label == "Ground Truth":
                        border_style = "border: 2px solid white; box-shadow: 0 0 0 1px #ddd;"
                    
                    card_html += f"""
                    <div style="flex: 0 0 180px; margin-right: 10px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">{label}</div>
                        <div style="width: 180px; height: 180px; position: relative;">
                            <img src="{img_url}" style="width: 100%; height: 100%; object-fit: contain; {border_style}">
                            <a href="{orig_url}" target="_blank" style="position: absolute; top: 5px; right: 5px; background: rgba(255,255,255,0.7); border-radius: 3px; padding: 2px; cursor: pointer;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                                    <path d="M5.828 10.172a.5.5 0 0 0-.707 0l-4.096 4.096V11.5a.5.5 0 0 0-1 0v3.975a.5.5 0 0 0 .5.5H4.5a.5.5 0 0 0 0-1H1.732l4.096-4.096a.5.5 0 0 0 0-.707zm4.344 0a.5.5 0 0 1 .707 0l4.096 4.096V11.5a.5.5 0 1 1 1 0v3.975a.5.5 0 0 1-.5.5H11.5a.5.5 0 0 1 0-1h2.768l-4.096-4.096a.5.5 0 0 1 0-.707zm0-4.344a.5.5 0 0 0 .707 0l4.096-4.096V4.5a.5.5 0 1 0 1 0V.525a.5.5 0 0 0-.5-.5H11.5a.5.5 0 0 0 0 1h2.768l-4.096 4.096a.5.5 0 0 0 0 .707zm-4.344 0a.5.5 0 0 1-.707 0L1.025 1.732V4.5a.5.5 0 0 1-1 0V.525a.5.5 0 0 1 .5-.5H4.5a.5.5 0 0 1 0 1H1.732l4.096 4.096a.5.5 0 0 1 0 .707z"/>
                                </svg>
                            </a>
                        </div>
                    </div>
                    """
            
            card_html += f"""
                </div>
            </div>
            """
            
            html_content += card_html
        
        return html_content

    def search_samples(self, search_query):

        if not search_query or not search_query.strip():
            return "请输入搜索关键字"
            
        search_query = search_query.strip().lower()
        matching_samples = []
        
        for i, sid in enumerate(self.samples.keys()):
            sample = self.samples[sid]
            sample_id = sid.lower()

            conversation = ""
            conv_path = os.path.join(self.samples_dir, f"{sample['id']}_conversation.txt")
            if os.path.exists(conv_path):
                with open(conv_path, "r", encoding="utf-8") as f:
                    conversation = f.read().lower()

            if search_query in sample_id or search_query in conversation:
                matching_samples.append((i, sample))
        
        if not matching_samples:
            return f"未找到匹配'{search_query}'的样本"
        
        html_content = f"<h3>找到 {len(matching_samples)} 个匹配结果:</h3>"
        
        for i, (index, sample) in enumerate(matching_samples):
            timestamp = sample.get("timestamp", "未知时间")
            
            conversation = ""
            conv_path = os.path.join(self.samples_dir, f"{sample['id']}_conversation.txt")
            if os.path.exists(conv_path):
                with open(conv_path, "r", encoding="utf-8") as f:
                    conversation = f.read()
                    if search_query in conversation.lower():
                        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                        conversation = pattern.sub(f'<span style="background-color: yellow;">{search_query}</span>', conversation)
                    
                    if len(conversation) > 100:
                        conversation = conversation[:100] + "..."
            
            original_image = self._get_image_path(sample, "original_image")
            img_url = original_image
            if not img_url.startswith("http"):
                img_url = f"/file={original_image}"
            
            html_content += f"""
            <div class="search-result" style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                <div style="display: flex;">
                    <div style="flex: 0 0 100px; margin-right: 10px;">
                        <img src="{img_url}" style="width: 100px; height: 100px; object-fit: contain; border: 1px solid #ddd;">
                    </div>
                    <div style="flex: 1;">
                        <h4 style="margin-top: 0;">样本 {index+1}: {sample['id']} - {timestamp}</h4>
                        <div style="font-size: 0.9em; color: #666;">
                            <pre style="white-space: pre-wrap; margin: 0; max-height: 60px; overflow: hidden;">{conversation}</pre>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 10px; text-align: right;">
                    <button class="load-sample-btn" 
                            style="background-color: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;"
                            data-index="{index}">
                        加载样本 {index+1}
                    </button>
                </div>
            </div>
            """

        html_content += """
        <script>
            // 在DOM加载完成后执行
            setTimeout(function() {
                console.log("Setting up search result buttons");
                
                // 为所有加载按钮添加点击事件
                document.querySelectorAll('.load-sample-btn').forEach(btn => {
                    // 移除旧的事件监听器
                    btn.removeEventListener('click', handleLoadButtonClick);
                    // 添加新的事件监听器
                    btn.addEventListener('click', handleLoadButtonClick);
                    
                    // 添加鼠标悬停效果
                    btn.addEventListener('mouseover', function() {
                        this.style.backgroundColor = '#45a049';
                    });
                    btn.addEventListener('mouseout', function() {
                        this.style.backgroundColor = '#4CAF50';
                    });
                });
                
                // 按钮点击处理函数
                function handleLoadButtonClick(e) {
                    e.preventDefault();
                    const index = this.getAttribute('data-index');
                    console.log("Load button clicked for index:", index);
                    
                    if (index !== null) {
                        // 设置当前索引输入框的值
                        const indexInput = document.querySelector('#current_index input');
                        if (indexInput) {
                            indexInput.value = index;
                            // 触发输入事件以更新Gradio状态
                            indexInput.dispatchEvent(new Event('input', { bubbles: true }));
                            
                            // 查找并点击加载按钮 - 使用更可靠的选择器
                            const loadButton = document.querySelector('#load_sample_btn button');
                            if (loadButton) {
                                console.log("Found load button, clicking it");
                                loadButton.click();
                            } else {
                                console.error("Could not find load button with selector #load_sample_btn button");
                                
                                // 尝试查找所有按钮，找到隐藏的加载按钮
                                const allButtons = document.querySelectorAll('button');
                                console.log("Total buttons found:", allButtons.length);
                                
                                let loadButtonFound = false;
                                allButtons.forEach((btn, i) => {
                                    if (btn.textContent.includes('加载样本索引')) {
                                        console.log(`Found load button at index ${i}`);
                                        btn.click();
                                        loadButtonFound = true;
                                    }
                                });
                                
                                if (!loadButtonFound) {
                                    console.error("Could not find any button with text '加载样本索引'");
                                }
                            }
                        } else {
                            console.error("Could not find current_index input");
                        }
                    }
                }
            }, 500); // 延迟500ms确保DOM已完全加载
        </script>
        """
        
        return html_content

def create_gradio_interface(visualizer):
    css = """
    .container {
        max-width: 100%;
        margin: 0 auto;
    }
    
    /* 对话内容中的图像布局 */
    .images-row {
        display: flex;
        flex-wrap: nowrap;
        gap: 5px;
        margin-top: 10px;
        overflow-x: auto;
        width: 100%;
    }
    
    /* 调整Gradio组件的样式 */
    .gradio-container {
        max-width: 100% !important;
    }
    
    .gradio-container .prose img {
        margin: 0;
    }
    
    /* 减小图像组件的间距 */
    .gradio-container .gap-4 {
        gap: 0.5rem;
    }
    
    /* 减小标签和图像之间的间距 */
    .gradio-container label {
        margin-bottom: 0.25rem;
    }
    
    /* 强制图像容器在一行显示 */
    .gradio-container .flex.flex-wrap {
        flex-wrap: nowrap !important;
        overflow-x: auto;
    }
    
    /* 调整图像容器宽度 */
    .gradio-container .w-full {
        width: auto !important;
        min-width: 150px;
        max-width: 180px;
    }
    
    /* 样本卡片样式 */
    .sample-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 20px;
        padding: 15px;
        background-color: #f9f9f9;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .sample-card:hover {
        background-color: #f0f0f0;
    }
    
    /* 全屏按钮样式 */
    .fullscreen-btn {
        position: absolute;
        top: 5px;
        right: 5px;
        background: rgba(255,255,255,0.7);
        border-radius: 3px;
        padding: 2px;
        cursor: pointer;
        z-index: 100;
    }
    
    .fullscreen-btn:hover {
        background: rgba(255,255,255,0.9);
    }
    
    /* 图像容器相对定位，以便放置全屏按钮 */
    .image-container {
        position: relative;
    }
    
    /* 搜索结果样式 */
    .search-result {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
        transition: background-color 0.2s;
    }
    
    .search-result:hover {
        background-color: #f0f0f0;
    }
    
    .load-sample-btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .load-sample-btn:hover {
        background-color: #45a049;
    }
    """
    
    with gr.Blocks(css=css, title="READ Analysis Dashboard") as demo:
        gr.HTML("<h1 style='text-align: center;'>READ Analysis Dashboard</h1>")
        
        with gr.Row():
            with gr.Column(scale=1):
                sample_count = gr.HTML(visualizer.get_sample_count)
                refresh_button = gr.Button("刷新样本列表")
                
                with gr.Row():
                    search_input = gr.Textbox(
                        label="搜索样本ID或关键字", 
                        placeholder="输入样本ID或对话关键字",
                        show_label=True
                    )
                    search_button = gr.Button("🔍 搜索", variant="primary")
                
                with gr.Column() as search_results_container:
                    search_results_header = gr.HTML(label="搜索结果")
                    
                    search_results_dropdown = gr.Dropdown(
                        label="", 
                        choices=[], 
                        visible=False,
                        elem_id="search-results-dropdown"
                    )
                
                with gr.Row():
                    prev_button = gr.Button("上一个样本")
                    next_button = gr.Button("下一个样本")
                
                current_index = gr.Number(value=0, visible=False, elem_id="current_index")
            
            with gr.Column(scale=3):
                conversation_text = gr.Textbox(label="对话内容", lines=5)
                
                with gr.Row(elem_id="images-container"):
                    original_image = gr.Image(
                        label="原始图像", 
                        type="filepath", 
                        show_label=True, 
                        height=200, 
                        width=150,
                        show_download_button=False,
                        interactive=False,
                    )
                    similarity_map = gr.Image(
                        label="相似度图", 
                        type="filepath", 
                        show_label=True, 
                        height=200, 
                        width=150,
                        show_download_button=False,
                        interactive=False,
                    )
                    sam_prediction = gr.Image(
                        label="SAM预测", 
                        type="filepath", 
                        show_label=True, 
                        height=200, 
                        width=150,
                        show_download_button=False,
                        interactive=False,
                    )
                    model_prediction = gr.Image(
                        label="模型预测", 
                        type="filepath", 
                        show_label=True, 
                        height=200, 
                        width=150,
                        show_download_button=False,
                        interactive=False,
                    )
                    ground_truth = gr.Image(
                        label="Ground Truth", 
                        type="filepath", 
                        show_label=True, 
                        height=200, 
                        width=150,
                        show_download_button=False,
                        interactive=False,
                    )
        
        path_info = gr.HTML(label="样本路径信息")
        
        gr.Markdown("## 历史记录")
        
        history_display = gr.HTML()
        
        refresh_button.click(
            fn=visualizer.refresh_samples,
            outputs=sample_count
        )
        
        def search_and_format_results(query):
            if not query or not query.strip():
                return gr.HTML.update(value="请输入搜索关键字"), gr.Dropdown.update(choices=[], visible=False), None
                
            query = query.strip().lower()
            matching_samples = []
            
            for i, sid in enumerate(visualizer.samples.keys()):
                sample = visualizer.samples.get(sid, {})
                
                conversation = ""
                conv_path = os.path.join(visualizer.samples_dir, f"{sample.get('id', '')}_conversation.txt")
                if os.path.exists(conv_path):
                    with open(conv_path, "r", encoding="utf-8") as f:
                        conversation = f.read().lower()
                
                if query in sid.lower() or query in conversation:
                    matching_samples.append((i, sample))
            
            if not matching_samples:
                return gr.HTML.update(value=f"未找到匹配'{query}'的样本"), gr.Dropdown.update(choices=[], visible=False), None
            
            if len(matching_samples) == 1:
                index = matching_samples[0][0]
                return (
                    gr.HTML.update(value=f"找到1个匹配结果，已自动加载"),
                    gr.Dropdown.update(choices=[], visible=False),
                    index
                )
            
            dropdown_choices = []
            for i, (index, sample) in enumerate(matching_samples):
                sample_id = sample.get("id", "")
                timestamp = sample.get("timestamp", "未知时间")
                dropdown_choices.append((f"样本 {index+1}: {sample_id} - {timestamp}", index))
            
            return (
                gr.HTML.update(value=f"找到 {len(matching_samples)} 个匹配结果，默认显示第1个"), 
                gr.Dropdown.update(choices=dropdown_choices, visible=True, value=dropdown_choices[0][1], label=""),
                dropdown_choices[0][1]
            )
        
        search_result = search_button.click(
            fn=search_and_format_results,
            inputs=search_input,
            outputs=[search_results_header, search_results_dropdown, current_index]
        )
        
        search_result.then(
            fn=lambda idx: visualizer.get_sample_by_index(idx) if idx is not None else (None, None, None, None, None, None, None, None),
            inputs=current_index,
            outputs=[current_index, conversation_text, original_image, similarity_map, 
                    sam_prediction, model_prediction, ground_truth, path_info]
        )
        
        search_input_result = search_input.submit(
            fn=search_and_format_results,
            inputs=search_input,
            outputs=[search_results_header, search_results_dropdown, current_index]
        )
        
        search_input_result.then(
            fn=lambda idx: visualizer.get_sample_by_index(idx) if idx is not None else (None, None, None, None, None, None, None, None),
            inputs=current_index,
            outputs=[current_index, conversation_text, original_image, similarity_map, 
                    sam_prediction, model_prediction, ground_truth, path_info]
        )
        
        search_results_dropdown.change(
            fn=visualizer.get_sample_by_index,
            inputs=search_results_dropdown,
            outputs=[current_index, conversation_text, original_image, similarity_map, 
                    sam_prediction, model_prediction, ground_truth, path_info]
        )
        
        prev_button.click(
            fn=visualizer.load_prev_sample,
            inputs=current_index,
            outputs=[current_index, conversation_text, original_image, similarity_map, 
                    sam_prediction, model_prediction, ground_truth, path_info]
        )
        
        next_button.click(
            fn=visualizer.load_next_sample,
            inputs=current_index,
            outputs=[current_index, conversation_text, original_image, similarity_map, 
                    sam_prediction, model_prediction, ground_truth, path_info]
        )
        
        demo.load(
            fn=visualizer.get_latest_sample,
            outputs=[current_index, conversation_text, original_image, similarity_map, 
                    sam_prediction, model_prediction, ground_truth, path_info]
        )
        
        def update_history():
            history_html = visualizer.generate_history_gallery()
            
            history_display = gr.HTML(f"""
            <div class="history-container">
                <h3>历史样本</h3>
                {history_html}
            </div>
            <script>
                // 添加样本加载函数
                window.loadSample = function(index) {{
                    // 触发加载样本事件
                    const loadSampleEvent = new CustomEvent('load-sample', {{ 
                        detail: {{ index: parseInt(index) }} 
                    }});
                    document.dispatchEvent(loadSampleEvent);
                }};
            </script>
            """)
            
            return history_display
        
        refresh_button.click(
            fn=update_history,
            outputs=history_display
        )
        
        demo.load(
            fn=visualizer.get_latest_sample,
            outputs=[current_index, conversation_text, original_image, similarity_map, 
                    sam_prediction, model_prediction, ground_truth, path_info]
        )
        
        demo.load(
            fn=update_history,
            outputs=history_display
        )
        
        demo.load(
            fn=lambda: "",
            outputs=search_results_header
        )
        
        demo.load(js="""
        // 设置图像容器样式，确保图像在一行显示
        function setupImageContainers() {
            const container = document.getElementById('images-container');
            if (container) {
                // 设置容器样式
                container.style.display = 'flex';
                container.style.flexWrap = 'nowrap';
                container.style.overflowX = 'auto';
                container.style.width = '100%';
                
                // 设置子元素样式
                const children = container.children;
                for (let i = 0; i < children.length; i++) {
                    children[i].style.flexShrink = '0';
                    children[i].style.width = '180px';
                    children[i].style.minWidth = '180px';
                }
            }
            
            // 为所有图像添加全屏功能
            setupFullscreenImages();
        }
        
        // 为所有图像添加全屏功能
        function setupFullscreenImages() {
            // 处理Gradio界面中的图像
            const images = document.querySelectorAll('.gradio-container img');
            images.forEach(img => {
                if (!img.getAttribute('data-fullscreen-setup')) {
                    img.setAttribute('data-fullscreen-setup', 'true');
                    img.style.cursor = 'pointer';
                    
                    img.addEventListener('click', function(e) {
                        // 检查是否在搜索结果中
                        const isInSearchResult = !!this.closest('.search-result');
                        if (isInSearchResult) {
                            // 搜索结果中的图像不添加全屏功能
                            return;
                        }
                        
                        // 阻止事件冒泡
                        e.stopPropagation();
                        
                        // 获取原始图像URL（如果有）
                        const originalSrc = img.getAttribute('data-original-src') || img.src;
                        
                        // 创建全屏模态框
                        const modal = document.createElement('div');
                        modal.style.position = 'fixed';
                        modal.style.top = '0';
                        modal.style.left = '0';
                        modal.style.width = '100%';
                        modal.style.height = '100%';
                        modal.style.backgroundColor = 'rgba(0,0,0,0.9)';
                        modal.style.display = 'flex';
                        modal.style.justifyContent = 'center';
                        modal.style.alignItems = 'center';
                        modal.style.zIndex = '10000';
                        modal.style.cursor = 'zoom-out';
                        
                        // 创建图像元素
                        const fullImg = document.createElement('img');
                        fullImg.src = originalSrc;
                        fullImg.style.maxWidth = '95%';
                        fullImg.style.maxHeight = '95%';
                        fullImg.style.objectFit = 'contain';
                        
                        // 添加关闭按钮
                        const closeBtn = document.createElement('div');
                        closeBtn.innerHTML = '&times;';
                        closeBtn.style.position = 'absolute';
                        closeBtn.style.top = '20px';
                        closeBtn.style.right = '30px';
                        closeBtn.style.color = 'white';
                        closeBtn.style.fontSize = '35px';
                        closeBtn.style.cursor = 'pointer';
                        
                        // 添加元素到DOM
                        modal.appendChild(fullImg);
                        modal.appendChild(closeBtn);
                        document.body.appendChild(modal);
                        
                        // 添加关闭事件
                        modal.addEventListener('click', function() {
                            document.body.removeChild(modal);
                        });
                        
                        closeBtn.addEventListener('click', function(e) {
                            e.stopPropagation();
                            document.body.removeChild(modal);
                        });
                    });
                }
            });
            
            // 处理历史记录中的图像
            const historyImages = document.querySelectorAll('.history-container img');
            historyImages.forEach(img => {
                if (!img.getAttribute('data-fullscreen-setup')) {
                    img.setAttribute('data-fullscreen-setup', 'true');
                    img.style.cursor = 'pointer';
                    
                    // 查找相邻的全屏链接
                    const parentDiv = img.closest('div');
                    if (parentDiv) {
                        const fullscreenLink = parentDiv.querySelector('a[target="_blank"]');
                        if (fullscreenLink) {
                            const originalSrc = fullscreenLink.href;
                            img.setAttribute('data-original-src', originalSrc);
                            
                            // 阻止链接的默认点击行为
                            fullscreenLink.addEventListener('click', function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                
                                // 模拟点击图像
                                img.click();
                            });
                        }
                }
            }
        });
        }
        
        // 页面加载时设置
        document.addEventListener('DOMContentLoaded', function() {
            // 初始设置
            setTimeout(() => {
                setupImageContainers();
            }, 1000);
        });
            
        // 监听DOM变化，应用于新加载的内容
            const observer = new MutationObserver(function(mutations) {
            setupFullscreenImages();
            });
            
        // 开始观察整个文档的变化
        observer.observe(document.body, { childList: true, subtree: true });
        """)
    
    return demo

def main():
    args = parse_args()
    
    os.makedirs(args.samples_dir, exist_ok=True)
    
    visualizer = AnalysisVisualizer(args.samples_dir)
    
    demo = create_gradio_interface(visualizer)
    
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    
    print(f"Launching Gradio server, please visit http://{ip_address}:{args.port}")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main() 