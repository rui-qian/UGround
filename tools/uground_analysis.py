# Credit: Chuanhang Deng, © 2025 Fudan University. All rights reserved.

import os
import cv2
import json
import argparse
import numpy as np
import gradio as gr
from datetime import datetime
from PIL import Image
import threading
import queue
import math

def parse_args():
    parser = argparse.ArgumentParser(description="UGround: Towards Unified Visual Grounding with Large Multimodal Models")
    parser.add_argument("--dataset_dir", default="../dataset_sesame/reason_seg/ReasonSeg/test", type=str, help="Directory containing original sample images")
    parser.add_argument("--layers_output", default="./uground-7B@reason_seg_test", type=str, help="Directory containing layer outputs (layer0, layer1, etc.)")
    parser.add_argument("--samples_dir", default="./vis_output", type=str, help="Directory for cache and sample files")
    parser.add_argument("--port", default=7860, type=int, help="Port for Gradio server")
    parser.add_argument("--share", action="store_true", help="Create a public link for sharing")
    return parser.parse_args()

class AnalysisVisualizer:
    def __init__(self, dataset_dir, layers_output, samples_dir):
        self.dataset_dir = dataset_dir
        self.layers_output = layers_output
        self.samples_dir = samples_dir
        self.samples = []
        self.image_cache = {}
        self.viewed_samples = []
        
        self.cache_dir = os.path.join(samples_dir, "cache")
        self.all_layers_output_dir = os.path.join(samples_dir, "all_layers_output")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.all_layers_output_dir, exist_ok=True)
        
        self._layer_dirs_cache = None
        self._layer_dirs_mtime = 0
        
        self.bg_queue = queue.Queue(maxsize=50)
        self.bg_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.bg_thread.start()
        
        self.load_samples()
    
    def _get_or_generate_ground_truth_path(self, sample_id):
        gt_dir = os.path.join(self.samples_dir, "ground_truth")
        os.makedirs(gt_dir, exist_ok=True)
        target_path = os.path.join(gt_dir, f"{sample_id}.jpg")
        
        if os.path.exists(target_path):
            return target_path

        json_path = os.path.join(self.dataset_dir, f"{sample_id}.json")
        if os.path.exists(json_path):
            return target_path
        
        return None

    def _background_worker(self):
        while True:
            try:
                sample_id = self.bg_queue.get(timeout=1)
                if sample_id is None:
                    break
                    
                sample = next((s for s in self.samples if s['id'] == sample_id), None)
                if sample:
                    updated = False

                    if sample.get("ground_truth") and not os.path.exists(sample.get("ground_truth", "")):
                        print(f"Background generating GT for {sample_id}")
                        gt_path = self._generate_ground_truth_image(sample_id)
                        if gt_path:
                            for i, s in enumerate(self.samples):
                                if s['id'] == sample_id:
                                    self.samples[i]["ground_truth"] = gt_path
                                    sample = self.samples[i]
                                    updated = True
                                    break

                    if not sample.get("similarity_map") or not os.path.exists(sample.get("similarity_map", "")):
                        print(f"Background generating mosaic for {sample_id}")
                        display_path, high_res_path = self._create_layers_mosaic(sample_id)
                        if display_path and high_res_path:
                            for i, s in enumerate(self.samples):
                                if s['id'] == sample_id:
                                    self.samples[i]["similarity_map"] = display_path
                                    self.samples[i]["similarity_map_high"] = high_res_path
                                    updated = True
                                    break
                    
                    if updated:
                        print(f"Background generation completed for {sample_id}")
                            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Background generation error: {e}")
    
    def _get_mask_from_json(self, json_path, img):
        try:
            with open(json_path, "r", encoding="utf-8") as r:
                anno = json.loads(r.read())
        except:
            with open(json_path, "r", encoding="cp1252") as r:
                anno = json.loads(r.read())
        inform = anno.get("shapes", [])
        height, width = img.shape[:2]
        area_list, valid_poly_list = [], []
        for item in inform:
            label_id = item.get("label", "")
            points = item.get("points", [])
            if not points or "flag" == label_id.lower():
                continue
            tmp_mask = np.zeros((height, width), dtype=np.uint8)
            pts = np.array([points], dtype=np.int32)
            cv2.polylines(tmp_mask, pts, True, 1, 1)
            cv2.fillPoly(tmp_mask, pts, 1)
            area_list.append(tmp_mask.sum())
            valid_poly_list.append(item)
        sort_index = list(np.argsort(area_list)[::-1].astype(np.int32)) if area_list else []
        sort_inform = [valid_poly_list[idx] for idx in sort_index]
        mask = np.zeros((height, width), dtype=np.uint8)
        for item in sort_inform:
            label_id = item.get("label", "")
            points = item.get("points", [])
            if not points:
                continue
            label_value = 255 if "ignore" in label_id.lower() else 1
            pts = np.array([points], dtype=np.int32)
            cv2.polylines(mask, pts, True, label_value, 1)
            cv2.fillPoly(mask, pts, label_value)
        return mask
    
    def _generate_ground_truth_image(self, sample_id):
        gt_dir = os.path.join(self.samples_dir, "ground_truth")
        os.makedirs(gt_dir, exist_ok=True)
        target_path = os.path.join(gt_dir, f"{sample_id}.jpg")
        if os.path.exists(target_path):
            return target_path
        json_path = os.path.join(self.dataset_dir, f"{sample_id}.json")
        if not os.path.exists(json_path):
            return None

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        img_path = None
        for ext in image_extensions:
            cand = os.path.join(self.dataset_dir, f"{sample_id}{ext}")
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        img_rgb = img[:, :, ::-1]
        mask = self._get_mask_from_json(json_path, img_rgb)
        valid_mask = (mask == 1).astype(np.float32)[:, :, None]
        ignore_mask = (mask == 255).astype(np.float32)[:, :, None]
        vis_img = img_rgb * (1 - valid_mask) * (1 - ignore_mask) + (
            (np.array([0, 255, 0]) * 0.6 + img_rgb * 0.4) * valid_mask
            + (np.array([255, 0, 0]) * 0.6 + img_rgb * 0.4) * ignore_mask
        )
        cv2.imwrite(target_path, vis_img[:, :, ::-1])
        print(f"[GT生成] {sample_id}: {target_path}")
        return target_path

    def _get_layer_directories(self):
        if not os.path.exists(self.layers_output):
            return []
        
        current_mtime = os.path.getmtime(self.layers_output)
        
        if self._layer_dirs_cache is not None and current_mtime <= self._layer_dirs_mtime:
            return self._layer_dirs_cache
        
        layer_dirs = []
        for item in os.listdir(self.layers_output):
            item_path = os.path.join(self.layers_output, item)
            if os.path.isdir(item_path) and item.startswith('layer'):
                try:
                    layer_num = int(item.replace('layer', ''))
                    layer_dirs.append((layer_num, item_path))
                except ValueError:
                    continue
        
        layer_dirs.sort(key=lambda x: x[0])
        self._layer_dirs_cache = layer_dirs
        self._layer_dirs_mtime = current_mtime
        
        return layer_dirs
    
    def _create_layers_mosaic(self, sample_id):
        layer_dirs = self._get_layer_directories()
        
        high_res_path = os.path.join(self.all_layers_output_dir, f"{sample_id}_all_layers.jpg")
        display_path = os.path.join(self.cache_dir, f"{sample_id}_all_layers.jpg")

        ground_truth_path = self._generate_ground_truth_image(sample_id)

        need_regenerate = False
        if os.path.exists(high_res_path) and os.path.exists(display_path):
            mosaic_mtime = os.path.getmtime(high_res_path)

            if ground_truth_path and os.path.exists(ground_truth_path):
                gt_mtime = os.path.getmtime(ground_truth_path)
                if gt_mtime > mosaic_mtime:
                    need_regenerate = True
                    print(f"GT is newer than mosaic for {sample_id}, regenerating...")

            if not need_regenerate and layer_dirs:
                for layer_num, layer_path in layer_dirs:
                    possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                    for ext in possible_extensions:
                        candidate_path = os.path.join(layer_path, f"{sample_id}{ext}")
                        if os.path.exists(candidate_path):
                            layer_mtime = os.path.getmtime(candidate_path)
                            if layer_mtime > mosaic_mtime:
                                need_regenerate = True
                                print(f"Layer {layer_num} is newer than mosaic for {sample_id}, regenerating...")
                                break
                    if need_regenerate:
                        break
            
            if not need_regenerate:
                return display_path, high_res_path
        else:
            need_regenerate = True
        
        if not layer_dirs and not ground_truth_path:
            print(f"No layers or GT found for sample {sample_id}")
            return None, None
        
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        layer_images = []

        for layer_num, layer_path in layer_dirs:
            layer_image_path = None
            
            for ext in possible_extensions:
                candidate_path = os.path.join(layer_path, f"{sample_id}{ext}")
                if os.path.exists(candidate_path):
                    layer_image_path = candidate_path
                    break
            
            if layer_image_path and os.path.exists(layer_image_path):
                try:
                    img = Image.open(layer_image_path)
                    layer_images.append((layer_num, img))
                except Exception as e:
                    print(f"Failed to load layer {layer_num} image for sample {sample_id}: {e}")

        if ground_truth_path and os.path.exists(ground_truth_path):
            try:
                gt_img = Image.open(ground_truth_path)
                layer_images.append((9999, gt_img))
                print(f"Added GT to mosaic for sample {sample_id}")
            except Exception as e:
                print(f"Failed to load GT image for sample {sample_id}: {e}")

        if not layer_images:
            print(f"No valid images found for mosaic generation of sample {sample_id}")
            return None, None
        
        num_images = len(layer_images)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        
        sample_img = layer_images[0][1]
        img_width, img_height = sample_img.size
        
        mosaic_width = cols * img_width
        mosaic_height = rows * img_height
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), (255, 255, 255))
        
        for idx, (layer_num, img) in enumerate(layer_images):
            row = idx // cols
            col = idx % cols
            x = col * img_width
            y = row * img_height
            
            if img.size != (img_width, img_height):
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
            
            mosaic.paste(img, (x, y))

        for _, img in layer_images:
            img.close()
        
        mosaic.save(high_res_path, quality=95, optimize=True)
        
        display_mosaic = mosaic.copy()
        display_mosaic.thumbnail((600, 600), Image.Resampling.LANCZOS)
        display_mosaic.save(display_path, quality=80, optimize=True)
        
        print(f"Generated mosaic for sample {sample_id}: {len(layer_images)} images")
        return display_path, high_res_path

    def load_samples(self):
        if not os.path.exists(self.dataset_dir):
            print(f"Dataset directory not found: {self.dataset_dir}")
            return 0
        
        if not os.path.exists(self.layers_output):
            print(f"[Info] Layers output directory not found: {self.layers_output}")
        
        print(f"Building sample index from: {self.dataset_dir}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        sample_ids = set()

        for file in os.listdir(self.dataset_dir):
            if os.path.isfile(os.path.join(self.dataset_dir, file)):
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    sample_id = os.path.splitext(file)[0]
                    sample_ids.add(sample_id)
        
        print(f"Found {len(sample_ids)} samples, building metadata...")
        
        self.samples = []
        for sample_id in sample_ids:
            original_image_path = self._find_image_file(sample_id, self.dataset_dir)
            if original_image_path:
                display_path = os.path.join(self.cache_dir, f"{sample_id}_all_layers.jpg")
                high_res_path = os.path.join(self.all_layers_output_dir, f"{sample_id}_all_layers.jpg")
                
                has_mosaic = os.path.exists(display_path) and os.path.exists(high_res_path)

                gt_path = self._get_or_generate_ground_truth_path(sample_id)
                
                sample = {
                    "id": sample_id,
                    "timestamp": datetime.fromtimestamp(os.path.getmtime(original_image_path)).strftime("%Y-%m-%d %H:%M:%S"),
                    "original_image": original_image_path,
                    "dataset_path": original_image_path,
                    "similarity_map": display_path if has_mosaic else None,
                    "similarity_map_high": high_res_path if has_mosaic else None,
                    "sam_prediction": None,
                    "model_prediction": None,
                    "ground_truth": gt_path,
                }

                sam_pred, model_pred, existing_gt = self._find_other_images(sample_id)
                sample["sam_prediction"] = sam_pred
                sample["model_prediction"] = model_pred
                if not sample["ground_truth"]:
                    sample["ground_truth"] = existing_gt
                
                self.samples.append(sample)

                if not has_mosaic:
                    try:
                        self.bg_queue.put_nowait(sample_id)
                    except queue.Full:
                        pass
        
        print(f"Indexed {len(self.samples)} samples. Background generation started.")
        return len(self.samples)
    
    def _find_image_file(self, sample_id, directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        for ext in image_extensions:
            path = os.path.join(directory, f"{sample_id}{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def _find_other_images(self, sample_id):
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        sam_prediction = None
        model_prediction = None
        ground_truth = None

        sam_dir = os.path.join(self.samples_dir, "sam_prediction")
        if os.path.exists(sam_dir):
            for ext in image_extensions:
                sam_path = os.path.join(sam_dir, f"{sample_id}{ext}")
                if os.path.exists(sam_path):
                    sam_prediction = sam_path
                    break

        model_dir = os.path.join(self.samples_dir, "model_prediction")
        if os.path.exists(model_dir):
            for ext in image_extensions:
                model_path = os.path.join(model_dir, f"{sample_id}{ext}")
                if os.path.exists(model_path):
                    model_prediction = model_path
                    break

        gt_dir = os.path.join(self.samples_dir, "ground_truth")
        if os.path.exists(gt_dir):
            for ext in image_extensions:
                gt_path = os.path.join(gt_dir, f"{sample_id}{ext}")
                if os.path.exists(gt_path):
                    ground_truth = gt_path
                    break

        return sam_prediction, model_prediction, ground_truth
    
    def _load_conversation(self, sample_id):
        conv_path = os.path.join(self.samples_dir, f"{sample_id}_conversation.txt")
        if os.path.exists(conv_path):
            with open(conv_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    
    def _add_to_viewed_history(self, sample):
        self.viewed_samples = [s for s in self.viewed_samples if s["id"] != sample["id"]]
        sample_with_view_time = sample.copy()
        sample_with_view_time["view_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.viewed_samples.insert(0, sample_with_view_time)
        if len(self.viewed_samples) > 20:
            self.viewed_samples = self.viewed_samples[:20]

    def _get_gradio_file_url(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return ""
        abs_path = os.path.abspath(file_path)
        return f"/file={abs_path}"

    def create_image_html(self, display_path, high_res_path, label):
        container_style = "width: 150px; height: 220px; display: flex; flex-direction: column; align-items: center; margin-right: 5px;"
        label_style = "font-size: 0.9em; font-weight: bold; margin-bottom: 5px; text-align: center; height: 20px;"
        image_wrapper_style = "position: relative; height: 200px; width: 100%;"
        
        if not display_path or not os.path.exists(display_path):
            return f"""
            <div style='{container_style}'>
                <div style='{label_style}'>{label}</div>
                <div style='height: 200px; width: 100%; border: 1px dashed #ccc; 
                             display: flex; align-items: center; justify-content: center; 
                             color: #888; font-size: 0.9em; text-align: center; box-sizing: border-box;'>
                    无图像
                </div>
            </div>
            """

        display_url = self._get_gradio_file_url(display_path)
        high_res_url = self._get_gradio_file_url(high_res_path) if high_res_path else display_url

        return f"""
        <div style='{container_style}'>
            <div style='{label_style}'>{label}</div>
            <div style='{image_wrapper_style}'>
                <img src='{display_url}' 
                     data-original-src='{high_res_url}' 
                     style='width: 100%; height: 100%; object-fit: contain; border: 1px solid #ddd; cursor: pointer;'
                     title='点击查看大图'>
                
                <a href="{high_res_url}" 
                   target="_blank" 
                   title="在新标签页中打开高清原图"
                   style="position: absolute; top: 5px; right: 5px; background: rgba(255,255,255,0.7); border-radius: 3px; padding: 2px; line-height: 0; cursor: pointer;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M5.828 10.172a.5.5 0 0 0-.707 0l-4.096 4.096V11.5a.5.5 0 0 0-1 0v3.975a.5.5 0 0 0 .5.5H4.5a.5.5 0 0 0 0-1H1.732l4.096-4.096a.5.5 0 0 0 0-.707zm4.344 0a.5.5 0 0 1 .707 0l4.096 4.096V11.5a.5.5 0 1 1 1 0v3.975a.5.5 0 0 1-.5.5H11.5a.5.5 0 0 1 0-1h2.768l-4.096-4.096a.5.5 0 0 1 0-.707zm0-4.344a.5.5 0 0 0 .707 0l4.096-4.096V4.5a.5.5 0 1 0 1 0V.525a.5.5 0 0 0-.5-.5H11.5a.5.5 0 0 0 0 1h2.768l-4.096 4.096a.5.5 0 0 0 0 .707zm-4.344 0a.5.5 0 0 1-.707 0L1.025 1.732V4.5a.5.5 0 0 1-1 0V.525a.5.5 0 0 1 .5-.5H4.5a.5.5 0 0 1 0 1H1.732l4.096 4.096a.5.5 0 0 1 0 .707z"/>
                    </svg>
                </a>
            </div>
        </div>
        """

    def get_sample_by_index(self, index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = -1
            
        if not self.samples or index < 0 or index >= len(self.samples):
            no_image_html = self.create_image_html(None, None, "")
            return -1, "未找到样本", no_image_html, no_image_html, no_image_html, no_image_html, no_image_html, "未找到样本"
        
        sample = self.samples[index]

        if sample.get("ground_truth") and not os.path.exists(sample.get("ground_truth", "")):
            print(f"Generating GT for sample {sample['id']} (on-demand)")
            gt_path = self._generate_ground_truth_image(sample['id'])
            if gt_path:
                sample["ground_truth"] = gt_path

        if not sample.get("similarity_map") or not os.path.exists(sample.get("similarity_map", "")):
            print(f"Generating mosaic for sample {sample['id']} (on-demand)")
            display_path, high_res_path = self._create_layers_mosaic(sample['id'])
            sample["similarity_map"] = display_path
            sample["similarity_map_high"] = high_res_path
        
        print(f"Loaded sample: {sample['id']}, index: {index}")
        
        self._add_to_viewed_history(sample)

        for offset in [-1, 1]:
            adj_idx = index + offset
            if 0 <= adj_idx < len(self.samples):
                adj_sample = self.samples[adj_idx]
                needs_generation = (
                    not adj_sample.get("similarity_map") or 
                    not os.path.exists(adj_sample.get("similarity_map", "")) or
                    (adj_sample.get("ground_truth") and not os.path.exists(adj_sample.get("ground_truth", "")))
                )
                if needs_generation:
                    try:
                        self.bg_queue.put_nowait(adj_sample['id'])
                    except queue.Full:
                        pass

        original_image_path = sample.get("original_image")
        similarity_map_path = sample.get("similarity_map")
        similarity_map_high_path = sample.get("similarity_map_high")
        sam_prediction_path = sample.get("sam_prediction")
        model_prediction_path = sample.get("model_prediction")
        ground_truth_path = sample.get("ground_truth")

        original_image_display = self._get_image_path_cached(original_image_path, kind="original")
        similarity_map_display = self._get_image_path_cached(similarity_map_path, kind="similarity_map")
        sam_prediction_display = self._get_image_path_cached(sam_prediction_path, kind="sam_prediction")
        model_prediction_display = self._get_image_path_cached(model_prediction_path, kind="model_prediction")
        ground_truth_display = self._get_image_path_cached(ground_truth_path, kind="ground_truth")

        original_image_html = self.create_image_html(original_image_display, original_image_path, "原始图像")
        similarity_map_html = self.create_image_html(similarity_map_display, similarity_map_high_path, "相似度图")
        sam_prediction_html = self.create_image_html(sam_prediction_display, sam_prediction_path, "SAM预测")
        model_prediction_html = self.create_image_html(model_prediction_display, model_prediction_path, "模型预测")
        ground_truth_html = self.create_image_html(ground_truth_display, ground_truth_path, "Ground Truth")
        
        conversation = self._load_conversation(sample['id']) or ""
        
        path_info = f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <h4>样本 {sample['id']} 路径信息:</h4>
            <ul style="margin-left: 20px;">
                <li><strong>原始图像:</strong> {original_image_path or "未找到"}</li>
                <li><strong>相似度图:</strong> {similarity_map_high_path or "未找到"}</li>
                <li><strong>SAM预测:</strong> {sam_prediction_path or "未找到"}</li>
                <li><strong>模型预测:</strong> {model_prediction_path or "未找到"}</li>
                <li><strong>Ground Truth:</strong> {ground_truth_path or "未找到"}</li>
            </ul>
        </div>
        """
        
        return (index, conversation, 
                original_image_html, similarity_map_html, 
                sam_prediction_html, model_prediction_html, ground_truth_html, 
                path_info)

    def _get_image_path_cached(self, file_path, kind=None):
        if not file_path or not os.path.exists(file_path):
            return None
            
        if file_path in self.image_cache:
            return self.image_cache[file_path]
        
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        
        if file_size > 0.5:
            try:
                base = os.path.splitext(os.path.basename(file_path))[0]
                suffix = f"_{kind}" if kind else ""
                compressed_path = os.path.join(self.cache_dir, f"compressed_{base}{suffix}.jpg")

                if os.path.exists(compressed_path):
                    self.image_cache[file_path] = compressed_path
                    return compressed_path

                img = Image.open(file_path)
                max_size = (600, 600)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(compressed_path, "JPEG", optimize=True, quality=80)

                self.image_cache[file_path] = compressed_path
                return compressed_path
            except Exception as e:
                print(f"Failed to compress image {file_path}: {e}")
        
        self.image_cache[file_path] = file_path
        return file_path
    
    def get_latest_sample(self):
        if not self.samples:
            no_image_html = self.create_image_html(None, None, "")
            return -1, "没有样本", no_image_html, no_image_html, no_image_html, no_image_html, no_image_html, "没有样本"
        return self.get_sample_by_index(0)
    
    def get_sample_count(self):
        return f"当前共有 {len(self.samples)} 个样本"
    
    def load_prev_sample(self, index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        new_index = max(0, index - 1)
        print(f"Load previous sample: from {index} to {new_index}")
        return self.get_sample_by_index(new_index)
    
    def load_next_sample(self, index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        new_index = min(len(self.samples) - 1, index + 1)
        print(f"Load next sample: from {index} to {new_index}")
        return self.get_sample_by_index(new_index)
    
    def refresh_samples(self):
        count = self.load_samples()
        if self.samples:
            import random
            new_index = random.randint(0, len(self.samples) - 1)
            return self.get_sample_count(), new_index
        return self.get_sample_count(), 0
    
    def generate_history_gallery(self):
        if not self.samples:
            return "<div>暂无样本</div>"
            
        html_content = ""
        for i, sample in enumerate(self.samples):
            conversation = self._load_conversation(sample['id']) or ""

            viewed_sample = next((s for s in self.viewed_samples if s["id"] == sample["id"]), None)
            view_time = f"查看于 {viewed_sample.get('view_time', '未知时间')}" if viewed_sample else "未查看"
            
            card_html = f"""
            <div class="sample-card" onclick="loadSampleFromHistory({i})">
                <h4>样本 {i+1}: {sample['id']} - {view_time}</h4>
            """
            
            if conversation.strip():
                card_html += f"""
                <div style="margin-bottom: 10px;"><strong>对话内容:</strong>
                    <div style="background-color: #f5f5f5; padding: 8px; border-radius: 4px; max-height: 200px; overflow-y: auto;">
                        <pre style="white-space: pre-wrap; margin: 0;">{conversation}</pre>
                    </div>
                </div>"""
            
            card_html += '<div class="images-row">'

            if sample.get("ground_truth") and not os.path.exists(sample.get("ground_truth", "")):
                try:
                    self.bg_queue.put_nowait(sample['id'])
                except queue.Full:
                    pass

            image_info = {
                "原始图像": (sample["original_image"], sample["original_image"]),
                "相似度图": (sample["similarity_map"], sample.get("similarity_map_high")),
                "SAM预测": (sample.get("sam_prediction"), sample.get("sam_prediction")),
                "模型预测": (sample.get("model_prediction"), sample.get("model_prediction")),
                "Ground Truth": (sample.get("ground_truth"), sample.get("ground_truth")),
            }

            label_kind_map = {
                "原始图像": "original",
                "相似度图": "similarity_map", 
                "SAM预测": "sam_prediction",
                "模型预测": "model_prediction",
                "Ground Truth": "ground_truth",
            }
            
            for label, (display_path, high_res_path) in image_info.items():
                if display_path and os.path.exists(display_path):
                    cached_display_path = self._get_image_path_cached(display_path, kind=label_kind_map.get(label))
                    img_url = self._get_gradio_file_url(cached_display_path)
                    orig_url = self._get_gradio_file_url(high_res_path or display_path)
                    card_html += f"""
                    <div style="flex: 0 0 180px; margin-right: 10px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">{label}</div>
                        <div style="width: 180px; height: 180px; position: relative;">
                            <img src="{img_url}" style="width: 100%; height: 100%; object-fit: contain; border: 1px solid #ddd;"
                                data-original-src="{orig_url}">
                            <a href="{orig_url}" target="_blank" style="position: absolute; top: 5px; right: 5px; background: rgba(255,255,255,0.7); border-radius: 3px; padding: 2px;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5.828 10.172a.5.5 0 0 0-.707 0l-4.096 4.096V11.5a.5.5 0 0 0-1 0v3.975a.5.5 0 0 0 .5.5H4.5a.5.5 0 0 0 0-1H1.732l4.096-4.096a.5.5 0 0 0 0-.707zm4.344 0a.5.5 0 0 1 .707 0l4.096 4.096V11.5a.5.5 0 1 1 1 0v3.975a.5.5 0 0 1-.5.5H11.5a.5.5 0 0 1 0-1h2.768l-4.096-4.096a.5.5 0 0 1 0-.707zm0-4.344a.5.5 0 0 0 .707 0l4.096-4.096V4.5a.5.5 0 1 0 1 0V.525a.5.5 0 0 0-.5-.5H11.5a.5.5 0 0 0 0 1h2.768l-4.096 4.096a.5.5 0 0 0 0 .707zm-4.344 0a.5.5 0 0 1-.707 0L1.025 1.732V4.5a.5.5 0 0 1-1 0V.525a.5.5 0 0 1 .5-.5H4.5a.5.5 0 0 1 0 1H1.732l4.096 4.096a.5.5 0 0 1 0 .707z"/></svg>
                            </a>
                        </div>
                    </div>"""
                else:
                    if label == "相似度图" and (not sample.get("similarity_map") or not os.path.exists(sample.get("similarity_map", ""))):
                        status_text = "生成中..." 
                    elif label == "Ground Truth":
                        if sample.get("ground_truth") and not os.path.exists(sample.get("ground_truth", "")):
                            status_text = "生成中..."
                        elif not sample.get("ground_truth"):
                            json_path = os.path.join(self.dataset_dir, f"{sample['id']}.json")
                            status_text = "可生成" if os.path.exists(json_path) else "无标注"
                        else:
                            status_text = "无图像"
                    else:
                        status_text = "无图像"
                        
                    card_html += f"""
                    <div style="flex: 0 0 180px; margin-right: 10px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">{label}</div>
                        <div style="width: 180px; height: 180px; border: 1px dashed #ccc; 
                                    display: flex; align-items: center; justify-content: center; 
                                    color: #888; font-size: 0.9em; text-align: center; box-sizing: border-box;">
                            {status_text}
                        </div>
                    </div>"""
            card_html += "</div></div>"
            html_content += card_html
        
        return html_content
    
def create_gradio_interface(visualizer):
    css = """
    .container { max-width: 100%; margin: 0 auto; }
    .images-row { display: flex; flex-wrap: nowrap; gap: 5px; margin-top: 10px; overflow-x: auto; width: 100%; }
    .gradio-container { max-width: 100% !important; }
    .gradio-container .prose img { margin: 0; }
    .gradio-container .gap-4 { gap: 0.5rem; }
    .gradio-container label { margin-bottom: 0.25rem; }
    .gradio-container .flex.flex-wrap { flex-wrap: nowrap !important; overflow-x: auto; }
    .gradio-container .w-full { width: auto !important; min-width: 150px; max-width: 180px; }
    .sample-card { border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; padding: 15px; background-color: #f9f9f9; cursor: pointer; transition: background-color 0.2s; }
    .sample-card:hover { background-color: #f0f0f0; }
    .fullscreen-btn { position: absolute; top: 5px; right: 5px; background: rgba(255,255,255,0.7); border-radius: 3px; padding: 2px; cursor: pointer; z-index: 100; }
    .fullscreen-btn:hover { background: rgba(255,255,255,0.9); }
    .image-container { position: relative; }
    .search-result { border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #f9f9f9; transition: background-color 0.2s; }
    .search-result:hover { background-color: #f0f0f0; }
    .load-sample-btn { background-color: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }
    .load-sample-btn:hover { background-color: #45a049; }
    """
    
    with gr.Blocks(css=css, title="UGround Analysis Dashboard") as demo:
        gr.HTML("<h1 style='text-align: center;'>UGround Analysis Dashboard</h1>")
        
        outputs_list = []

        with gr.Row():
            with gr.Column(scale=1):
                sample_count = gr.HTML(value=visualizer.get_sample_count())
                refresh_button = gr.Button("随机查看样本", variant="primary")
                
                search_input = gr.Textbox(label="搜索样本ID或关键字", placeholder="输入样本ID或对话关键字", show_label=True)
                search_button = gr.Button("🔍 搜索", variant="primary")
                
                search_results_header = gr.HTML(label="搜索结果")
                search_results_dropdown = gr.Dropdown(label="", choices=[], visible=False, elem_id="search-results-dropdown")
                
                with gr.Row():
                    prev_button = gr.Button("上一个样本", variant="secondary")
                    next_button = gr.Button("下一个样本", variant="secondary")

                current_index = gr.Number(value=0, visible=False, elem_id="current_index")
                load_sample_btn = gr.Button("加载样本索引", visible=False, elem_id="load_sample_btn")

            with gr.Column(scale=4):
                conversation_text = gr.Textbox(label="对话内容", lines=5, visible=True)
                
                with gr.Row(elem_id="images-container"):
                    original_image = gr.HTML()
                    similarity_map = gr.HTML()
                    sam_prediction = gr.HTML()
                    model_prediction = gr.HTML()
                    ground_truth = gr.HTML()
        
        path_info = gr.HTML(label="样本路径信息")
        
        gr.Markdown("--- \n ## 所有样本")
        history_display = gr.HTML()
        
        outputs_list = [
            current_index, conversation_text, 
            original_image, similarity_map, sam_prediction, 
            model_prediction, ground_truth, path_info
        ]

        def update_history():
            history_html = visualizer.generate_history_gallery()
            return f"""
            <div class="history-container">{history_html}</div>
            <script>
            function loadSampleFromHistory(index) {{
                const indexInput = document.querySelector('#current_index input');
                if (indexInput) {{
                    indexInput.value = index;
                    indexInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    
                    // Use a slight delay to ensure the value is updated in Gradio's state
                    setTimeout(() => {{
                        const loadButton = document.querySelector('#load_sample_btn button');
                        if (loadButton) {{
                            loadButton.click();
                        }} else {{
                            console.error("Could not find the hidden load button.");
                        }}
                    }}, 100);
                }}
            }}
            </script>
            """

        refresh_button.click(
            fn=visualizer.refresh_samples,
            outputs=[sample_count, current_index]
        ).then(
            fn=visualizer.get_sample_by_index,
            inputs=current_index,
            outputs=outputs_list
        ).then(
            fn=update_history,
            outputs=history_display
        )
        
        def perform_search(query):
            if not query or not query.strip():
                return "请输入搜索关键字", gr.Dropdown(choices=[], visible=False), -1
            
            query = query.strip().lower()
            matching_samples = []
            for i, sample in enumerate(visualizer.samples):
                sample_id = sample.get("id", "").lower()
                conversation = visualizer._load_conversation(sample['id'])
                conversation_lower = conversation.lower() if conversation else ""
                if query in sample_id or query in conversation_lower:
                    matching_samples.append((i, sample))
            
            if not matching_samples:
                return f"未找到匹配'{query}'的样本", gr.Dropdown(choices=[], visible=False), -1
            
            if len(matching_samples) == 1:
                index = matching_samples[0][0]
                return "找到1个匹配结果，已自动加载", gr.Dropdown(choices=[], visible=False), index
            
            dropdown_choices = [(f"样本 {idx+1}: {s.get('id', '')}", idx) for idx, s in matching_samples]
            return f"找到 {len(matching_samples)} 个匹配结果，请选择：", gr.Dropdown(choices=dropdown_choices, visible=True, value=dropdown_choices[0][1]), dropdown_choices[0][1]

        search_event = search_button.click(
            fn=perform_search,
            inputs=search_input,
            outputs=[search_results_header, search_results_dropdown, current_index]
        )
        
        search_input_event = search_input.submit(
            fn=perform_search,
            inputs=search_input,
            outputs=[search_results_header, search_results_dropdown, current_index]
        )
        
        search_event.then(
            fn=lambda idx: visualizer.get_sample_by_index(idx) if idx is not None and idx >= 0 else visualizer.get_sample_by_index(-1),
            inputs=current_index,
            outputs=outputs_list
        )

        search_input_event.then(
            fn=lambda idx: visualizer.get_sample_by_index(idx) if idx is not None and idx >= 0 else visualizer.get_sample_by_index(-1),
            inputs=current_index,
            outputs=outputs_list
        )

        dropdown_change_event = search_results_dropdown.change(
            fn=visualizer.get_sample_by_index,
            inputs=search_results_dropdown,
            outputs=outputs_list
        )
        
        load_sample_btn.click(
            fn=visualizer.get_sample_by_index,
            inputs=current_index,
            outputs=outputs_list
        )

        prev_button.click(
            fn=visualizer.load_prev_sample,
            inputs=current_index,
            outputs=outputs_list
        )
        
        next_button.click(
            fn=visualizer.load_next_sample,
            inputs=current_index,
            outputs=outputs_list
        )
        
        demo.load(
            fn=visualizer.get_latest_sample,
            outputs=outputs_list
        )

        for btn in [prev_button, next_button, load_sample_btn]:
            btn.click(fn=update_history, outputs=history_display)
        
        for event in [search_event, search_input_event, dropdown_change_event]:
             event.then(fn=update_history, outputs=history_display)

        demo.load(fn=update_history, outputs=history_display)
        
        gr.HTML("<div style='text-align:center; color:#888; margin: 24px 0;'>© 2025 Qian Rui/Fudan University</div>")

        demo.load(js="""
        function setupFullscreenImages() {
            // This script now works reliably because data-original-src is embedded in the HTML from the server
            const images = document.querySelectorAll('.gradio-container img:not(.search-result img)');
            images.forEach(img => {
                if (img.getAttribute('data-fullscreen-setup')) { return; }
                img.setAttribute('data-fullscreen-setup', 'true');
                img.style.cursor = 'pointer';
                
                img.addEventListener('click', function(e) {
                    e.stopPropagation();
                    const fullscreenSrc = this.getAttribute('data-original-src') || this.src;
                    if (!fullscreenSrc || fullscreenSrc.endsWith('None')) return;

                    const modal = document.createElement('div');
                    modal.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); display:flex; justify-content:center; align-items:center; z-index:10000; cursor:zoom-out;';
                    
                    const fullImg = document.createElement('img');
                    fullImg.src = fullscreenSrc;
                    fullImg.style.cssText = 'max-width:95%; max-height:95%; object-fit:contain;';
                    
                    const closeBtn = document.createElement('div');
                    closeBtn.innerHTML = '&times;';
                    closeBtn.style.cssText = 'position:absolute; top:20px; right:30px; color:white; font-size:35px; cursor:pointer;';
                    
                    modal.appendChild(fullImg);
                    modal.appendChild(closeBtn);
                    document.body.appendChild(modal);
                    
                    const closeModal = () => document.body.removeChild(modal);
                    modal.addEventListener('click', closeModal);
                    closeBtn.addEventListener('click', e_close => {
                        e_close.stopPropagation();
                        closeModal();
                    });
                });
            });
        }
        
        const observer = new MutationObserver((mutations) => {
            setTimeout(setupFullscreenImages, 100);
        });
        observer.observe(document.body, { childList: true, subtree: true });
        window.addEventListener('load', () => setTimeout(setupFullscreenImages, 500));
        """)
    
    return demo

def main():
    args = parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return
    
    if not os.path.exists(args.layers_output):
        print(f"Warning: Layers output directory not found: {args.layers_output}.")
    
    os.makedirs(args.samples_dir, exist_ok=True)
    
    visualizer = AnalysisVisualizer(
        dataset_dir=args.dataset_dir,
        layers_output=args.layers_output,
        samples_dir=args.samples_dir
    )
    
    demo = create_gradio_interface(visualizer)
    
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        print(f"Launching Gradio server, please visit http://{ip_address}:{args.port}")
    except Exception:
        print(f"Could not determine local IP. Launching on 0.0.0.0. Please visit http://127.0.0.1:{args.port}")

    allowed_paths = [
        args.dataset_dir,
        args.layers_output,
        args.samples_dir,
        os.path.join(args.samples_dir, "cache"),
        os.path.join(args.samples_dir, "all_layers_output"),
        os.path.join(args.samples_dir, "ground_truth"),
        os.path.join(args.samples_dir, "sam_prediction"),
        os.path.join(args.samples_dir, "model_prediction"),
    ]
    allowed_paths = [p for p in allowed_paths if os.path.exists(p)]

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, allowed_paths=allowed_paths)

if __name__ == "__main__":
    main()
