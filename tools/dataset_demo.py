#!/usr/bin/env python3
import os
import cv2
import numpy as np
import gradio as gr
import tempfile
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataloaders.dataset import HybridDataset
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
 
class WebDemoHandler:
    def __init__(self, args=None):

        self.dataset_dir = args.dataset_dir
        self.vision_tower = args.vision_tower
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.samples_per_epoch = args.samples_per_epoch
        self.grad_accumulation_steps = args.grad_accumulation_steps
        self.steps_per_epoch = args.steps_per_epoch
        self.precision = args.precision
        self.image_size = args.image_size
        self.num_classes_per_sample = args.num_classes_per_sample
        self.exclude_val = args.exclude_val
        self.dataset = args.dataset
        self.sample_rates = args.sample_rates
        self.sem_seg_data = args.sem_seg_data
        self.refer_seg_data = args.refer_seg_data
        self.vqa_data = args.vqa_data
        self.reason_seg_data = args.reason_seg_data
        self.multi_reason_seg_data = args.multi_reason_seg_data
        self.explanatory = args.explanatory
        self.seg_token_num = args.seg_token_num
        self.image_feature_scale_num = args.image_feature_scale_num
        self.num_classes_per_question = args.num_classes_per_question
        self.pad_train_clip_images = args.pad_train_clip_images
        self.masks_process_with_clip = args.masks_process_with_clip
        self.preprocessor_config = args.preprocessor_config
        self.use_expand_question_list = args.use_expand_question_list
        
        self.HybridDataset = None
        self.filename_list = []
        self.current_filename = None
        self.current_filename_index = 0
        self.temp_files = []
        self.processing_lock = False
        self.tokenizer = None       
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (255, 165, 0), (255, 20, 147), (0, 191, 255)
        ]
        
        # self.dataset_mappings = {
        #     'sem_seg': ['ade20k', 'cocostuff', 'pascal_part', 'paco_lvis', 'mapillary'],
        #     'refer_seg': ['refclef', 'refcoco', 'refcoco+', 'refcocog','grefcoco', 'refzom'],
        #     'vqa': ['llava_instruct_150k'],
        #     'reason_seg': ['ReasonSeg|train'],
        #     'multi_reason_seg': ['MultiReasonSeg|train']
        # }
        
    def cleanup_temp_files(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Failed to clean temp file: {temp_file}, error: {e}")
        self.temp_files.clear()

    def load_dataset(self, sem_seg_list, refer_seg_list, neg_refer_seg_list, correct_refer_seg_list, vqa_list, reason_seg_list, reason_seg_plus_list, multi_reason_seg_list, sample_rates_str="1,1,1,1,1", samples_per_epoch_str="", num_classes_per_sample_str=""):
        try:
            self.update_params_from_selections(sem_seg_list, refer_seg_list, neg_refer_seg_list, correct_refer_seg_list, vqa_list, reason_seg_list, reason_seg_plus_list, multi_reason_seg_list, sample_rates_str, samples_per_epoch_str, num_classes_per_sample_str)
            if not self.dataset or not any([self.sem_seg_data, self.refer_seg_data, self.neg_refer_seg_data, 
                                          self.correct_refer_seg_data, self.vqa_data, 
                                          self.reason_seg_data, self.reason_seg_plus_data, self.multi_reason_seg_data]):
                return "No datasets selected. Please select at least one dataset.", ""
            
            print(f"Loading datasets:")
            print(f"  Dataset types: {self.dataset}")
            print(f"  sem_seg_data: {self.sem_seg_data}")
            print(f"  refer_seg_data: {self.refer_seg_data}")
            print(f"  neg_refer_seg_data: {self.neg_refer_seg_data}")
            print(f"  correct_refer_seg_data: {self.correct_refer_seg_data}")
            print(f"  vqa_data: {self.vqa_data}")
            print(f"  reason_seg_data: {self.reason_seg_data}")
            print(f"  reason_seg_plus_data: {self.reason_seg_plus_data}")
            print(f"  multi_reason_seg_data: {self.multi_reason_seg_data}")
                 
            print("Initializing HybridDataset...")
            self.HybridDataset = HybridDataset(
                base_image_dir=self.dataset_dir,
                tokenizer=self.tokenizer,
                vision_tower=self.vision_tower,
                samples_per_epoch=self.samples_per_epoch,
                precision=self.precision,
                image_size=self.image_size,
                num_classes_per_sample=self.num_classes_per_sample,
                exclude_val=self.exclude_val,
                dataset=self.dataset,
                sample_rate=[float(x) for x in self.sample_rates.split(",")],
                sem_seg_data=self.sem_seg_data,
                refer_seg_data=self.refer_seg_data,
                neg_refer_seg_data=self.neg_refer_seg_data,
                correct_refer_seg_data=self.correct_refer_seg_data,
                vqa_data=self.vqa_data,
                reason_seg_data=self.reason_seg_data,
                reason_seg_plus_data=self.reason_seg_plus_data,
                multi_reason_seg_data=self.multi_reason_seg_data,
                explanatory=self.explanatory,
                seg_token_num=self.seg_token_num * self.image_feature_scale_num,
                num_classes_per_question=self.num_classes_per_question,
                pad_train_clip_images=self.pad_train_clip_images,
                masks_process_with_clip=self.masks_process_with_clip,
                preprocessor_config=self.preprocessor_config,
                use_expand_question_list=self.use_expand_question_list,
            )
            
            print(f"HybridDataset initialized successfully with {len(self.HybridDataset)} samples")
            
            self.build_filename_mappings()
            
            if not self.filename_list:
                return "Dataset loaded but no valid samples found", ""
            
            load_msg = f"Dataset loaded successfully! {len(self.HybridDataset):,} total samples, {len(self.filename_list)} accessible samples"
            default_filename = self.filename_list[0] if self.filename_list else ""
            return load_msg, default_filename
                
        except Exception as e:
            error_msg = f"Dataset loading failed: {str(e)}"
            print(f"Dataset loading error: {e}")
            traceback.print_exc()
            self.HybridDataset = None
            return error_msg, ""
    
    def update_params_from_selections(
        self, 
        sem_seg_list, 
        refer_seg_list, 
        neg_refer_seg_list,
        correct_refer_seg_list,
        vqa_list, 
        reason_seg_list, 
        reason_seg_plus_list,
        multi_reason_seg_list, 
        sample_rates_str="1,1,1,1,1",
        samples_per_epoch_str="",
        num_classes_per_sample_str=""
    ):
        
        selected_types = []
        if sem_seg_list:
            selected_types.append("sem_seg")
        if refer_seg_list:
            selected_types.append("refer_seg")
        if neg_refer_seg_list:
            selected_types.append("neg_refer_seg")
        if correct_refer_seg_list:
            selected_types.append("correct_refer_seg")
        if vqa_list:
            selected_types.append("vqa")
        if reason_seg_list:
            selected_types.append("reason_seg")
        if reason_seg_plus_list:
            selected_types.append("reason_seg_plus")
        if multi_reason_seg_list:
            selected_types.append("multi_reason_seg")
        
        self.dataset = '||'.join(selected_types)
        sample_rates_str = sample_rates_str.strip()
        if sample_rates_str.isdigit():
            self.sample_rates = ",".join(sample_rates_str)
        else:
            self.sample_rates = sample_rates_str

        if samples_per_epoch_str and samples_per_epoch_str.strip():
            try:
                self.samples_per_epoch = int(samples_per_epoch_str.strip())
            except ValueError:
                print(f"Invalid samples_per_epoch value: {samples_per_epoch_str}")

        if num_classes_per_sample_str and num_classes_per_sample_str.strip():
            try:
                self.num_classes_per_sample = int(num_classes_per_sample_str.strip())
            except ValueError:
                print(f"Invalid num_classes_per_sample value: {num_classes_per_sample_str}")

        self.sem_seg_data = '||'.join(sem_seg_list or [])
        self.refer_seg_data = '||'.join(refer_seg_list or [])
        self.neg_refer_seg_data = '||'.join(neg_refer_seg_list or [])
        self.correct_refer_seg_data = '||'.join(correct_refer_seg_list or [])
        self.vqa_data = '||'.join(vqa_list or [])
        self.reason_seg_data = '||'.join(reason_seg_list or [])
        self.reason_seg_plus_data = '||'.join(reason_seg_plus_list or [])
        self.multi_reason_seg_data = '||'.join(multi_reason_seg_list or [])
    
    def build_filename_mappings(self):
        print("Building filename mappings...")
        if self.HybridDataset is None or len(self.HybridDataset) == 0:
            print("HybridDataset is None, cannot build mappings")
            return
        
        start_time = time.time()
        dataset_len = len(self.HybridDataset)
        print(f"Processing {dataset_len:,} samples...")
        
        # Choose optimization method based on dataset size
        if dataset_len < 10:
            # For small datasets, use simple sequential processing
            self._build_mappings_sequential()
        elif dataset_len < 100:
            # For medium datasets, use ThreadPoolExecutor
            self._build_mappings_threaded()
        else:
            # For large datasets, use DataLoader with multiprocessing
            self._build_mappings_dataloader()
        
        elapsed_time = time.time() - start_time
        print(f"✅ Filename mapping built successfully! Processed {len(self.filename_to_sample_map):,} samples in {elapsed_time:.2f}s")
        
        # Build filename list for navigation
        self.filename_list = list(self.filename_to_sample_map.keys())
        if self.filename_list:
            self.current_filename = self.filename_list[0]
            self.current_filename_index = 0
    
    def get_sample_by_filename(self, filename):
        if filename not in self.filename_to_sample_map:
            return None, f"Sample not found: {filename}"
        
        complete_sample = self.filename_to_sample_map[filename]
        return complete_sample['image_data'], None
    
    def get_random_sample(self):
        if not self.filename_list:
            return None, "No samples available"
        
        import random
        random_filename = random.choice(self.filename_list)
        self.current_filename = random_filename
        self.current_filename_index = self.filename_list.index(random_filename)
        
        complete_sample = self.filename_to_sample_map[random_filename]
        return complete_sample['image_data'], random_filename
    
    def get_next_sample(self):
        if not self.filename_list:
            return None, "No samples available"
        
        next_index = (self.current_filename_index + 1) % len(self.filename_list)
        next_filename = self.filename_list[next_index]
        
        self.current_filename = next_filename
        self.current_filename_index = next_index
        
        complete_sample = self.filename_to_sample_map[next_filename]
        return complete_sample['image_data'], next_filename
    
    def get_prev_sample(self):
        if not self.filename_list:
            return None, "No samples available"
        
        prev_index = (self.current_filename_index - 1) % len(self.filename_list)
        prev_filename = self.filename_list[prev_index]
        
        self.current_filename = prev_filename
        self.current_filename_index = prev_index
        
        complete_sample = self.filename_to_sample_map[prev_filename]
        return complete_sample['image_data'], prev_filename
    
    def get_file_name(self, image_path):
        file_name = os.path.basename(image_path)
        return os.path.splitext(file_name)[0]
    
    def decode_masks_from_sample_data(self, sample_data):
        try:
            masks_tensor = sample_data[4]          
            if hasattr(masks_tensor, 'numpy'):
                masks_array = masks_tensor.numpy()
            elif hasattr(masks_tensor, 'cpu'):
                masks_array = masks_tensor.cpu().numpy()
            else:
                masks_array = np.array(masks_tensor)
                        
            masks = []
            for i in range(masks_array.shape[0]):
                mask = masks_array[i]
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                if np.sum(mask > 0) == 0:
                    continue
                masks.append(mask)
            return masks
        
        except Exception as e:
            print(f"⚠️ Error decoding masks from sample_data: {e}")
            return [], []

    def visualize_sample_by_filename(self, filename):
        """Visualize sample by filename"""
        try:
            if self.processing_lock:
                return None, "⏳ Processing, please wait...", ""

            self.processing_lock = True

            if self.HybridDataset is None:
                self.processing_lock = False
                return None, "Please load dataset first", ""

            self.cleanup_temp_files()
            plt.clf()
            plt.close('all')

            sample_data, error_msg = self.get_sample_by_filename(filename)
            if error_msg:
                self.processing_lock = False
                return None, error_msg, ""

            if filename in self.filename_list:
                self.current_filename = filename
                self.current_filename_index = self.filename_list.index(filename)

            image_path = sample_data[0]
            image = cv2.imread(image_path)
            if image is None:
                self.processing_lock = False
                return None, f"❌ Cannot read image: {image_path}", ""

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.decode_masks_from_sample_data(sample_data)

            temp_files = []

            # 1. Pure binary mask image
            fig_binary, ax_binary = plt.subplots(1, 1, figsize=(12, 8))
            binary_mask_image = self.create_binary_mask_image(masks, image.shape[:2])
            ax_binary.imshow(binary_mask_image, cmap='gray')
            ax_binary.axis('off')

            temp_file_binary = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file_binary.name, dpi=100, bbox_inches='tight')
            plt.close(fig_binary)
            temp_files.append(temp_file_binary.name)
            self.temp_files.append(temp_file_binary.name)

            # 2. Mask overlay on original image
            fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(12, 8))
            overlay_image = self.create_mask_overlay_image(image, masks)
            ax_overlay.imshow(overlay_image.astype(np.uint8))
            ax_overlay.axis('off')

            temp_file_overlay = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file_overlay.name, dpi=100, bbox_inches='tight')
            plt.close(fig_overlay)
            temp_files.append(temp_file_overlay.name)
            self.temp_files.append(temp_file_overlay.name)

            # 3. Original image
            fig_original, ax_original = plt.subplots(1, 1, figsize=(12, 8))
            ax_original.imshow(image)
            ax_original.axis('off')

            temp_file_original = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file_original.name, dpi=100, bbox_inches='tight')
            plt.close(fig_original)
            temp_files.append(temp_file_original.name)
            self.temp_files.append(temp_file_original.name)

            # Generate field content list
            field_contents = self.generate_sample_details(sample_data)

            filename_display = f"{filename} ({self.current_filename_index}/{len(self.filename_list)})"
            self.processing_lock = False

            return temp_files, field_contents, filename_display

        except Exception as e:
            print(f"⚠️ Visualization error: {str(e)}")
            print(traceback.format_exc())
            plt.close('all')
            self.cleanup_temp_files()
            self.processing_lock = False
            return None, f"❌ Visualization error: {str(e)}", ""
    
    def generate_sample_details(self, sample_data):
        """Generate sample detailed information, including all field information from the dataset"""
        try:
            field_names = [
                'image_path', 'images', 'image_clip', 'conversations', 'masks',
                'label', 'resize', 'clip_resize', 'questions', 'sampled_sents',
                'use_assign_list', 'inference'
            ]
            field_contents = []
            for i, field_name in enumerate(field_names):
                if i < len(sample_data):
                    field_value = sample_data[i]
                    content = self.generate_field_content(field_name, field_value)
                    field_contents.append(content)
                else:
                    field_contents.append("N/A")
            return field_contents

        except Exception as e:
            print(f"Error generating sample details: {e}")
            return ["N/A"] * 12
    
    def visualize_jump_by_file_name(self, target_file_name):
        try:
            if self.HybridDataset is None:
                return None, "Please load dataset first", ""
            
            if not hasattr(self, 'filename_to_sample_map') or not self.filename_to_sample_map:
                return None, "Filename mapping not built", ""
            
            # If no input provided, use default filename (first one in the list)
            if not target_file_name:
                if hasattr(self, 'filename_list') and self.filename_list:
                    target_file_name = self.filename_list[0]
                else:
                    return None, "❌ Please enter filename", ""
            
            # Remove possible file extension
            target_file_name = self.get_file_name(target_file_name)
            
            # Check if filename exists
            if target_file_name not in self.filename_to_sample_map:
                return None, f"❌ Sample with filename {target_file_name} not found", ""
            
            # Use filename-based visualization method
            result = self.visualize_sample_by_filename(target_file_name)
            if result[0] is None:
                return None, result[1], ""
            
            temp_files, details, filename_display = result
            return temp_files, details, filename_display
            
        except Exception as e:
            return None, f"❌ Jump failed: {str(e)}", ""
    
    def visualize_browse_next(self):
        try:
            if self.HybridDataset is None:
                return None, "Please load dataset first", ""
            
            if not hasattr(self, 'filename_list') or not self.filename_list:
                return None, "❌ Filename mapping not built", ""
            
            # Get next sample
            sample_data, next_filename = self.get_next_sample()
            # Use filename-based visualization method
            return self.visualize_sample_by_filename(next_filename)
            
        except Exception as e:
            print(f"❌ Browse next error: {str(e)}")
            return None, f"❌ Browse next failed: {str(e)}", ""
    
    def visualize_browse_prev(self):
        try:
            if self.HybridDataset is None:
                return None, "Please load dataset first", ""
            
            if not hasattr(self, 'filename_list') or not self.filename_list:
                return None, "❌ Filename mapping not built", ""
            
            # Get previous sample
            sample_data, prev_filename = self.get_prev_sample()
            
            # Use filename-based visualization method
            return self.visualize_sample_by_filename(prev_filename)
            
        except Exception as e:
            print(f"❌ Browse previous error: {str(e)}")
            return None, f"❌ Browse previous failed: {str(e)}", ""
    
    def visualize_browse_random(self):
        try:
            if self.HybridDataset is None:
                return None, "Please load dataset first", ""
            
            if not hasattr(self, 'filename_list') or not self.filename_list:
                return None, "❌ Filename mapping not built", ""
            
            # Get random sample
            sample_data, random_filename = self.get_random_sample()
            
            # Use filename-based visualization method
            return self.visualize_sample_by_filename(random_filename)
            
        except Exception as e:
            print(f"❌ Random browse error: {str(e)}")
            return None, f"❌ Random browse failed: {str(e)}", ""
    
    def search_with_placeholder_update(self, target_file_name):
        try:
            if self.HybridDataset is None:
                return None, "Please load dataset first", "", gr.update()
            
            if not hasattr(self, 'filename_to_sample_map') or not self.filename_to_sample_map:
                return None, "❌ Filename mapping not built", "", gr.update()
            
            # If search box is empty, use first sample
            if not target_file_name or target_file_name.strip() == "":
                if hasattr(self, 'filename_list') and self.filename_list:
                    target_file_name = self.filename_list[0]
                    new_placeholder = target_file_name
                else:
                    return None, "❌ No available samples", "", gr.update()
            else:
                new_placeholder = target_file_name
            
            # Call original visualization method
            result = self.visualize_jump_by_file_name(target_file_name)
            if result is None or len(result) != 3:
                return None, "❌ Search error", "", gr.update(placeholder=new_placeholder, value="")
            
            temp_files, details, filename_display = result
            return temp_files, details, filename_display, gr.update(placeholder=new_placeholder, value="")
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return None, f"❌ Search failed: {str(e)}", "", gr.update()

    def generate_field_content(self, field_name, field_value):
        """Generate content display for specified field - show all content without truncation"""
        try:
            if field_name == 'image_path':
                return f"```\n{field_value}\n```"
            
            elif field_name in ['images', 'image_clip', 'masks', 'label']:
                # For tensor types, display shape and dtype
                if hasattr(field_value, 'shape'):
                    if field_name == 'masks':
                        # Special handling for masks - add empty mask check
                        content = f"""```
Type: {type(field_value)}
Shape: {field_value.shape if hasattr(field_value, 'shape') else 'N/A'}
Dtype: {field_value.dtype if hasattr(field_value, 'dtype') else 'N/A'}
Min: {field_value.min().item() if hasattr(field_value, 'min') and field_value.numel() > 0 else 'N/A'}
Max: {field_value.max().item() if hasattr(field_value, 'max') and field_value.numel() > 0 else 'N/A'}

=== Mask Analysis ==="""
                        
                        # Analyze each mask
                        try:
                            if hasattr(field_value, 'numpy'):
                                masks_array = field_value.numpy()
                            elif hasattr(field_value, 'cpu'):
                                masks_array = field_value.cpu().numpy()
                            else:
                                masks_array = np.array(field_value)
                            
                            for i in range(masks_array.shape[0]):
                                mask = masks_array[i]
                                if len(mask.shape) > 2:
                                    mask = mask.squeeze()
                                
                                # Check if mask is empty
                                non_zero_count = np.sum(mask > 0)
                                total_pixels = mask.size
                                is_empty = non_zero_count == 0
                                coverage_ratio = non_zero_count / total_pixels if total_pixels > 0 else 0
                                
                                content += f"""
Mask {i}: {'❌ EMPTY' if is_empty else '✅ Valid'}
  - Shape: {mask.shape}
  - Non-zero pixels: {non_zero_count:,} / {total_pixels:,}
  - Coverage ratio: {coverage_ratio:.4f} ({coverage_ratio*100:.2f}%)
  - Value range: [{mask.min():.3f}, {mask.max():.3f}]"""
                        
                        except Exception as e:
                            content += f"\nError analyzing masks: {str(e)}"
                        
                        content += "\n```"
                    else:
                        # Original handling for other tensor types
                        content = f"""```
Type: {type(field_value)}
Shape: {field_value.shape if hasattr(field_value, 'shape') else 'N/A'}
Dtype: {field_value.dtype if hasattr(field_value, 'dtype') else 'N/A'}
Min: {field_value.min().item() if hasattr(field_value, 'min') and field_value.numel() > 0 else 'N/A'}
Max: {field_value.max().item() if hasattr(field_value, 'max') and field_value.numel() > 0 else 'N/A'}
```"""
                else:
                    # Full display without truncation
                    field_str = str(field_value)
                    content = f"```\n{field_str}\n```"
                return content
            
            elif field_name == 'conversations':
                if field_value and len(field_value) > 0:
                    content = f"**{len(field_value)} conversations:**\n\n"
                    for j, conv in enumerate(field_value):
                        # Full display of conversation content without truncation
                        conv_str = str(conv)
                        content += f"**Conversation {j+1}:**\n```\n{conv_str}\n```\n\n"
                    return content
                else:
                    return "```\nNo conversations\n```"
            
            elif field_name in ['questions', 'sampled_sents', 'use_assign_list']:
                # List type - full display
                if isinstance(field_value, list) and len(field_value) > 0:
                    import json
                    try:
                        # Full JSON display without truncation
                        json_str = json.dumps(field_value, indent=2, ensure_ascii=False)
                        return f"```json\n{json_str}\n```"
                    except:
                        # If JSON serialization fails, display full string
                        field_str = str(field_value)
                        return f"```\n{field_str}\n```"
                else:
                    return f"```\n{str(field_value)}\n```"
            
            else:
                # Simple display for other types - full display without truncation
                field_str = str(field_value)
                return f"```\n{field_str}\n```"
                
        except Exception as e:
            return f"```\nError displaying field: {str(e)}\n```"

    def safe_return_accordion_data(self, gallery, sample_data, filename):
        """Return safe data for Accordion component structure"""
        try:
            if gallery is None:
                empty_fields = [""] * 12  # 12 fields
                return [None] + empty_fields
            
            # sample_data should now be a list of field contents
            if isinstance(sample_data, list) and len(sample_data) == 12:
                # Directly use returned field contents
                return [gallery] + sample_data
            else:
                # If data format is incorrect, return empty fields
                empty_fields = [""] * 12
                return [gallery] + empty_fields
                
        except Exception as e:
            print(f"safe_return_accordion_data error: {e}")
            empty_fields = [""] * 12
            return [None] + empty_fields

    def create_binary_mask_image(self, masks, image_shape):
        """Create pure binary mask image, merge multiple masks"""
        try:
            # Create black binary image
            binary_image = np.zeros(image_shape, dtype=np.uint8)
            
            for i, mask in enumerate(masks):
                # Check if mask size matches image
                if mask.shape[:2] != image_shape:
                    print(f"⚠️ Mask {i} shape {mask.shape} doesn't match image shape {image_shape}")
                    # Resize mask to match image
                    mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Ensure mask is 2D
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                
                # Set mask region to white (255)
                mask_bool = mask > 0
                binary_image[mask_bool] = 255
            
            return binary_image
            
        except Exception as e:
            print(f"⚠️ Error creating binary mask image: {e}")
            return np.zeros(image_shape, dtype=np.uint8)
    
    def create_mask_overlay_image(self, image, masks):
        try:
            overlay_image = image.copy().astype(np.float32)
            
            for i, mask in enumerate(masks):
                # Check if mask size matches image
                if mask.shape[:2] != image.shape[:2]:
                    print(f"⚠️ Mask {i} shape {mask.shape} doesn't match image shape {image.shape}")
                    # Resize mask to match image
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Ensure mask is 2D
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                
                color = np.array(self.colors[i % len(self.colors)])
                colored_mask = np.zeros_like(image, dtype=np.float32)
                
                # Safe boolean indexing
                try:
                    mask_bool = mask > 0
                    if mask_bool.shape[:2] == colored_mask.shape[:2]:
                        colored_mask[mask_bool] = color
                    else:
                        print(f"⚠️ Boolean mask shape {mask_bool.shape} doesn't match colored_mask shape {colored_mask.shape}")
                        continue
                except Exception as e:
                    print(f"⚠️ Error in mask processing for mask {i}: {e}")
                    continue
                
                # Add white edges
                kernel = np.ones((3, 3), np.uint8)
                try:
                    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
                    mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                    edge_mask = mask_dilated - mask_eroded
                    
                    # Apply segmentation mask
                    mask_area = mask > 0
                    if mask_area.shape[:2] == overlay_image.shape[:2]:
                        overlay_image[mask_area] = 0.6 * overlay_image[mask_area] + 0.4 * colored_mask[mask_area]
                        
                        # Add white edges
                        edge_area = edge_mask > 0
                        if edge_area.shape[:2] == overlay_image.shape[:2]:
                            overlay_image[edge_area] = 0.3 * overlay_image[edge_area] + 0.7 * np.array([255, 255, 255])
                except Exception as e:
                    print(f"⚠️ Error in edge processing for mask {i}: {e}")
                    continue
            
            return overlay_image
            
        except Exception as e:
            print(f"⚠️ Error creating mask overlay image: {e}")
            return image.copy()
    
    def get_labels_from_sampled_sents(self, sampled_sents):
        try:
            labels = []
            if isinstance(sampled_sents, list):
                for sent in sampled_sents:
                    if isinstance(sent, str):
                        # Truncate labels longer than 10 characters
                        label = sent[:10] if len(sent) > 10 else sent
                        labels.append(label)
                    elif isinstance(sent, list):
                        # If nested list, take first element
                        for sub_sent in sent:
                            if isinstance(sub_sent, str):
                                label = sub_sent[:10] if len(sub_sent) > 10 else sub_sent
                                labels.append(label)
                                break
            elif isinstance(sampled_sents, str):
                label = sampled_sents[:10] if len(sampled_sents) > 10 else sampled_sents
                labels.append(label)
            
            return labels
            
        except Exception as e:
            print(f"Error extracting labels from sampled_sents: {e}")
            return []

    def _build_mappings_sequential(self):
        """Sequential processing for small datasets"""
        print("Using sequential processing...")
        self.filename_to_sample_map = OrderedDict()
        
        for img_idx in range(len(self.HybridDataset)):
            try:
                sample_data = self.HybridDataset[img_idx]
                if sample_data is None or len(sample_data) == 0:
                    continue
                    
                image_path = sample_data[0] if len(sample_data) > 0 else None
                if image_path is None:
                    continue
                    
                img_filename = self.get_file_name(image_path)
                complete_sample = {
                    'filename': img_filename,
                    'image_data': sample_data,
                    'image_index': img_idx
                }
                self.filename_to_sample_map[img_filename] = complete_sample
                
            except Exception as e:
                print(f"Error processing sample {img_idx}: {e}")
                continue
    
    def _build_mappings_threaded(self):
        """Multi-threaded processing for medium datasets"""
        print("Using multi-threaded processing...")
        self.filename_to_sample_map = OrderedDict()
        
        def process_sample(img_idx):
            try:
                sample_data = self.HybridDataset[img_idx]
                if sample_data is None or len(sample_data) == 0:
                    return None
                    
                image_path = sample_data[0] if len(sample_data) > 0 else None
                if image_path is None:
                    return None
                    
                img_filename = self.get_file_name(image_path)
                return {
                    'filename': img_filename,
                    'image_data': sample_data,
                    'image_index': img_idx
                }
            except Exception as e:
                print(f"Error processing sample {img_idx}: {e}")
                return None
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_sample, i): i for i in range(len(self.HybridDataset))}
            
            # Process results
            processed_count = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                if result is not None:
                    self.filename_to_sample_map[result['filename']] = result
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count:,}/{len(self.HybridDataset):,} samples...")
    
    def _build_mappings_dataloader(self):
        """DataLoader with multiprocessing for large datasets"""
        print("Using DataLoader with multiprocessing...")
        self.filename_to_sample_map = OrderedDict()
        
        # Create a wrapper dataset that returns the sample data and index
        class MappingDataset:
            def __init__(self, hybrid_dataset):
                self.dataset = hybrid_dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                try:
                    sample_data = self.dataset[idx]
                    if sample_data is None or len(sample_data) == 0:
                        return None
                    return (idx, sample_data)
                except Exception as e:
                    print(f"Error in dataset __getitem__ for index {idx}: {e}")
                    return None
        
        def custom_collate_fn(batch):
            """Custom collate function based on dataset.py collate_fn"""
            # Filter out None values
            batch = [item for item in batch if item is not None]
            if not batch:
                return [], []
            
            # Extract data following the pattern from dataset.py
            indices = []
            image_path_list = []
            images_list = []
            images_clip_list = []
            conversation_list = []
            masks_list = []
            label_list = []
            resize_list = []
            clip_resize_list = []
            questions_list = []
            sampled_classes_list = []
            use_assign_lists = []
            inferences = []
            
            for idx, sample_data in batch:
                if sample_data is None or len(sample_data) < 12:
                    continue
                    
                indices.append(idx)
                # Following HybridDataset structure: 
                # [image_path, images, image_clip, conversations, masks, label, resize, clip_resize, questions, sampled_sents, use_assign_list, inference]
                image_path_list.append(sample_data[0])  # image_path
                images_list.append(sample_data[1])      # images
                images_clip_list.append(sample_data[2]) # image_clip
                conversation_list.append(sample_data[3]) # conversations
                masks_list.append(sample_data[4])       # masks
                label_list.append(sample_data[5])       # label
                resize_list.append(sample_data[6])      # resize
                clip_resize_list.append(sample_data[7]) # clip_resize
                questions_list.append(sample_data[8])   # questions
                sampled_classes_list.append(sample_data[9])  # sampled_sents
                use_assign_lists.append(sample_data[10]) # use_assign_list
                inferences.append(sample_data[11])      # inference
            
            if not indices:
                return [], []
            
            # Return organized batch data similar to dataset.py collate_fn
            return indices, {
                "image_paths": image_path_list,
                "images": images_list,           # Keep as list, don't stack yet
                "images_clip": images_clip_list, # Keep as list, don't stack yet  
                "conversations": conversation_list,
                "masks": masks_list,
                "labels": label_list,
                "resize_list": resize_list,
                "clip_resize_list": clip_resize_list,
                "questions": questions_list,
                "sampled_classes": sampled_classes_list,
                "use_assign_lists": use_assign_lists,
                "inferences": inferences
            }
        
        # Create DataLoader similar to refer_seg_dataset.py
        mapping_dataset = MappingDataset(self.HybridDataset)
        dataloader = DataLoader(
            mapping_dataset,
            batch_size=64,      
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn
        )
        
        processed_count = 0
        total_batches = len(dataloader)
        
        try:
            print("Processing with DataLoader (batch_size=64, num_workers=4)...")
            for batch_idx, (indices, batch_data) in enumerate(dataloader):
                # Process batch data efficiently
                if not indices or not batch_data:
                    continue
                
                image_paths = batch_data["image_paths"]
                
                # Process each sample in the batch
                for i, (idx, image_path) in enumerate(zip(indices, image_paths)):
                    try:
                        if image_path is None:
                            continue
                            
                        img_filename = self.get_file_name(image_path)
                        
                        # Reconstruct sample_data from batch_data
                        sample_data = [
                            batch_data["image_paths"][i],     # image_path
                            batch_data["images"][i],          # images
                            batch_data["images_clip"][i],     # image_clip
                            batch_data["conversations"][i],   # conversations
                            batch_data["masks"][i],           # masks
                            batch_data["labels"][i],          # label
                            batch_data["resize_list"][i],     # resize
                            batch_data["clip_resize_list"][i], # clip_resize
                            batch_data["questions"][i],       # questions
                            batch_data["sampled_classes"][i], # sampled_sents
                            batch_data["use_assign_lists"][i], # use_assign_list
                            batch_data["inferences"][i]       # inference
                        ]
                        
                        complete_sample = {
                            'filename': img_filename,
                            'image_data': sample_data,
                            'image_index': idx
                        }
                        self.filename_to_sample_map[img_filename] = complete_sample
                        
                    except Exception as e:
                        print(f"Error processing sample {idx}: {e}")
                        continue
                    
                    processed_count += 1
                
                if (batch_idx + 1) % 1 == 0:  # Show progress every 100 batches
                    print(f"Processed batch {batch_idx + 1:,}/{total_batches:,} (samples: {processed_count:,})")
        
        except Exception as e:
            print(f"Error in DataLoader processing: {e}")
            print("Falling back to threaded processing...")
            # Clean up DataLoader before fallback
            try:
                del dataloader
                del mapping_dataset
            except:
                pass
            self._build_mappings_threaded()


def create_web_demo(args=None):
    handler = WebDemoHandler(args)
    
    def safe_next():
        try:
            result = handler.visualize_browse_next()
            if result and len(result) >= 3:
                gallery, sample_data, filename = result[:3]
                accordion_result = handler.safe_return_accordion_data(gallery, sample_data, filename)
                current_index = handler.current_filename_index
                total_samples = len(handler.filename_list)
                textbox_update = gr.update(label=f"Go to File ({current_index}/{total_samples})", value=handler.current_filename)
                return accordion_result + [textbox_update]
            else:
                empty_fields = [""] * 12
                return [None] + empty_fields + [gr.update()]
        except Exception as e:
            print(f"Next event exception: {e}")
            empty_fields = [""] * 12
            return [None] + empty_fields + [gr.update()]
    
    def safe_prev():
        try:
            result = handler.visualize_browse_prev()
            if result and len(result) >= 3:
                gallery, sample_data, filename = result[:3]
                accordion_result = handler.safe_return_accordion_data(gallery, sample_data, filename)
                current_index = handler.current_filename_index
                total_samples = len(handler.filename_list)
                textbox_update = gr.update(label=f"Go to File ({current_index}/{total_samples})", value=handler.current_filename)
                return accordion_result + [textbox_update]
            else:
                empty_fields = [""] * 12
                return [None] + empty_fields + [gr.update()]
        except Exception as e:
            print(f"Prev event exception: {e}")
            empty_fields = [""] * 12
            return [None] + empty_fields + [gr.update()]
    
    def safe_random():
        try:
            result = handler.visualize_browse_random()
            if result and len(result) >= 3:
                gallery, sample_data, filename = result[:3]
                accordion_result = handler.safe_return_accordion_data(gallery, sample_data, filename)
                current_index = handler.current_filename_index
                total_samples = len(handler.filename_list)
                textbox_update = gr.update(label=f"Go to File ({current_index}/{total_samples})", value=handler.current_filename)
                return accordion_result + [textbox_update]
            else:
                empty_fields = [""] * 12
                return [None] + empty_fields + [gr.update()]
        except Exception as e:
            print(f"Random event exception: {e}")
            empty_fields = [""] * 12
            return [None] + empty_fields + [gr.update()]
    
    def safe_search(target_file_name):
        try:
            result = handler.search_with_placeholder_update(target_file_name)
            if result and len(result) >= 4:
                gallery, sample_data, filename, placeholder_update = result[:4]
                accordion_result = handler.safe_return_accordion_data(gallery, sample_data, filename)
                current_index = handler.current_filename_index
                total_samples = len(handler.filename_list)
                textbox_update = gr.update(
                    label=f"Go to File ({current_index}/{total_samples})", 
                    value=handler.current_filename,
                    placeholder=placeholder_update.get('placeholder', '') if placeholder_update else ''
                )
                return accordion_result + [textbox_update]
            else:
                empty_fields = [""] * 12
                return [None] + empty_fields + [gr.update()]
        except Exception as e:
            print(f"Search event exception: {e}")
            empty_fields = [""] * 12
            return [None] + empty_fields + [gr.update()]
    

    with gr.Blocks(
        title="UGround Dashboard",
        css="""
        /* Text and code block styles */
        .markdown, .prose {
            max-height: none !important;
            overflow: visible !important;
        }
        
        .markdown pre, .markdown code, .prose pre, .prose code,
        .gradio-container pre, .gradio-container code {
            max-height: none !important;
            overflow: visible !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            word-break: break-all !important;
        }
        
        /* Hide scrollbars */
        .gradio-container * {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
        .gradio-container *::-webkit-scrollbar {
            display: none !important;
        }
        
        /* Image display */
        #sample_gallery {
            min-height: 300px !important;
        }
        #sample_gallery .gallery img {
            max-width: 100% !important;
            max-height: 100% !important;
            object-fit: contain !important;
        }
        
        /* Modal windows */
        .modal {
            max-width: 95vw !important;
            max-height: 95vh !important;
        }
        .modal img {
            max-width: 100% !important;
            max-height: 100% !important;
            object-fit: contain !important;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🔍 UGround: Towards Unified Visual Grounding with Unrolled Transformers</h1>
        </div>
        """)

        with gr.Tabs():
            
            with gr.TabItem("📊 Dataset"):
                with gr.Row():
                    with gr.Column(scale=1):
                        load_btn = gr.Button(
                            "Dataset Loading", 
                            variant="primary", 
                            size="lg"
                        )
                        load_status = gr.Textbox(
                            label="Loading Status", 
                            interactive=False
                        )
                        selection_status = gr.Textbox(
                            label="Current Selection", 
                            interactive=False
                        )

                    with gr.Column(scale=8):
                        with gr.Row(): 
                            with gr.Row():
                                sem_seg_dropdown = gr.Dropdown(
                                    choices=["ade20k", "cocostuff", "pascal_part", "paco_lvis", "mapillary"],
                                    label="sem_seg Datasets",
                                    multiselect=True,
                                    value=[],
                                    info="Semantic segmentation datasets"
                                )
                                refer_seg_dropdown = gr.Dropdown(
                                    choices=["refclef", "refcoco", "refcoco+", "refcocog", "grefcoco", "refzom"],
                                    label="refer_seg Datasets",
                                    multiselect=True,
                                    value=[],
                                    info="Referring segmentation datasets"
                                )
                                neg_refer_seg_dropdown = gr.Dropdown(
                                    choices=["R-refcoco", "R-refcoco+", "R-refcocog"],
                                    label="neg_refer_seg Datasets",
                                    multiselect=True,
                                    value=[],
                                    info="Referring segmentation datasets"
                                )
                                correct_refer_seg_dropdown = gr.Dropdown(
                                    choices=["fprefcoco", "fprefcoco+", "fprefcocog"],
                                    label="correct_refer_seg Datasets",
                                    multiselect=True,
                                    value=[],
                                    info="Referring segmentation datasets"
                                )
                                vqa_dropdown = gr.Dropdown(
                                    choices=["llava_instruct_150k"],
                                    label="vqa Datasets", 
                                    multiselect=True,
                                    value=[],
                                    info="Visual question answering datasets"
                                )
                                reason_seg_dropdown = gr.Dropdown(
                                    choices=["ReasonSeg|train"],
                                    label="reason_seg Datasets",
                                    multiselect=True,
                                    value=[],
                                    info="Reasoning segmentation datasets"
                                )
                                reason_seg_plus_dropdown = gr.Dropdown(
                                    choices=["instance_seg", "cot", "conversations", "caption"],
                                    label="reason_seg_plus Datasets",
                                    multiselect=True,
                                    value=[],
                                    info="Reasoning segmentation datasets"
                                )
                                multi_reason_seg_dropdown = gr.Dropdown(
                                    choices=["MultiReasonSeg|train"],
                                    label="multi_reason_seg Datasets",
                                    multiselect=True,
                                    value=[],
                                    info="Multi-reasoning segmentation datasets"
                                )               
                        
                        with gr.Row():
                            sample_rates_input = gr.Textbox(
                                label="Sample Rates",
                                info="sample rates for [sem_seg, refer_seg, vqa, reason_seg, multi_reason_seg]",
                                placeholder="e.g., 1,1,1,1,1",
                            )
                            samples_per_epoch_input = gr.Textbox(
                                label="Samples Per Epoch",
                                info="number of samples per epoch",
                                placeholder="e.g., 1000",
                                value="50"
                            )
                            num_classes_per_sample_input = gr.Textbox(
                                label="Num Classes Per Sample",
                                info="number of classes per sample",
                                placeholder="e.g., 3",
                                value="3"
                            )
            
            with gr.TabItem("🖼️ Navigator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            jump_file_name_input = gr.Textbox(
                                label="Go to File by File Name", 
                                placeholder="000000571562",
                                scale=2
                            )
                            jump_to_file_btn = gr.Button("🔍 Search", variant="primary", scale=1)
                        with gr.Row():
                            random_jump_btn = gr.Button("🎲 Jump", variant="secondary")
                
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ Prev", variant="secondary")
                            next_btn = gr.Button("➡️ Next", variant="secondary")
                        
                    with gr.Column(scale=2):

                        sample_gallery = gr.Gallery(
                            label="",
                            show_label=True,
                            elem_id="sample_gallery",
                            columns=3,
                            rows=1,
                            object_fit="contain",
                            height="auto",
                            allow_preview=True,
                            preview=True
                        )
                
                with gr.Row():
                    with gr.Column():

                        with gr.Accordion("Dataset Fields", open=False):

                            with gr.Accordion("image_path", open=False):
                                field_image_path = gr.Markdown()
                            
                            with gr.Accordion("images", open=False):
                                field_images = gr.Markdown()
                            
                            with gr.Accordion("image_clip", open=False):
                                field_image_clip = gr.Markdown()
                            
                            with gr.Accordion("conversations", open=False):
                                field_conversations = gr.Markdown()
                            
                            with gr.Accordion("masks", open=False):
                                field_masks = gr.Markdown()
                            
                            with gr.Accordion("label", open=False):
                                field_label = gr.Markdown()
                            
                            with gr.Accordion("resize", open=False):
                                field_resize = gr.Markdown()
                            
                            with gr.Accordion("clip_resize", open=False):
                                field_clip_resize = gr.Markdown()
                            
                            with gr.Accordion("questions", open=False):
                                field_questions = gr.Markdown()
                            
                            with gr.Accordion("sampled_sents", open=False):
                                field_sampled_sents = gr.Markdown()
                            
                            with gr.Accordion("use_assign_list", open=False):
                                field_use_assign_list = gr.Markdown()
                            
                            with gr.Accordion("inference", open=False):
                                field_inference = gr.Markdown()
                
                field_outputs = [
                    field_image_path, field_images, field_image_clip, field_conversations, 
                    field_masks, field_label, field_resize, field_clip_resize, 
                    field_questions, field_sampled_sents, field_use_assign_list, field_inference
                ]
                
                next_btn.click(
                    safe_next,
                    outputs=[sample_gallery] + field_outputs + [jump_file_name_input]
                )
                
                prev_btn.click(
                    safe_prev,
                    outputs=[sample_gallery] + field_outputs + [jump_file_name_input]
                )
                
                random_jump_btn.click(
                    safe_random,
                    outputs=[sample_gallery] + field_outputs + [jump_file_name_input]
                )
                
                jump_to_file_btn.click(
                    safe_search,
                    inputs=jump_file_name_input,
                    outputs=[sample_gallery] + field_outputs + [jump_file_name_input]
                )
        

        def update_selection_status(sem_seg_dropdown, refer_seg_dropdown, neg_refer_seg_dropdown, correct_refer_seg_dropdown, vqa_dropdown, reason_seg_dropdown, reason_seg_plus_dropdown, multi_reason_seg_dropdown, sample_rates_input, samples_per_epoch_input, num_classes_per_sample_input):
            selected_datasets = []
            
            if sem_seg_dropdown:
                selected_datasets.extend([f"sem_seg: {ds}" for ds in sem_seg_dropdown])
            if refer_seg_dropdown:
                selected_datasets.extend([f"refer_seg: {ds}" for ds in refer_seg_dropdown])
            if neg_refer_seg_dropdown:
                selected_datasets.extend([f"neg_refer_seg: {ds}" for ds in neg_refer_seg_dropdown])
            if correct_refer_seg_dropdown:
                selected_datasets.extend([f"correct_refer_seg: {ds}" for ds in correct_refer_seg_dropdown])
            if vqa_dropdown:
                selected_datasets.extend([f"vqa: {ds}" for ds in vqa_dropdown])
            if reason_seg_dropdown:
                selected_datasets.extend([f"reason_seg: {ds}" for ds in reason_seg_dropdown])
            if reason_seg_plus_dropdown:
                selected_datasets.extend([f"reason_seg_plus: {ds}" for ds in reason_seg_plus_dropdown])
            if multi_reason_seg_dropdown:
                selected_datasets.extend([f"multi_reason_seg: {ds}" for ds in multi_reason_seg_dropdown])
            
            if not selected_datasets:
                return "No datasets selected"
            
            status = f"Datasets: {', '.join(selected_datasets)}"
            if sample_rates_input and sample_rates_input.strip():
                status += f" | Sample Rates: {sample_rates_input.strip()}"
            if samples_per_epoch_input and samples_per_epoch_input.strip():
                status += f" | Samples Per Epoch: {samples_per_epoch_input.strip()}"
            if num_classes_per_sample_input and num_classes_per_sample_input.strip():
                status += f" | Num Classes Per Sample: {num_classes_per_sample_input.strip()}"
            
            return status
        
        def load_and_update_search(sem_seg_dropdown, refer_seg_dropdown, neg_refer_seg_dropdown, correct_refer_seg_dropdown, vqa_dropdown, reason_seg_dropdown, reason_seg_plus_dropdown, multi_reason_seg_dropdown, sample_rates_input, samples_per_epoch_input, num_classes_per_sample_input):
            try:
                sem_seg_list = sem_seg_dropdown or []
                refer_seg_list = refer_seg_dropdown or []
                neg_refer_seg_list = neg_refer_seg_dropdown or []
                correct_refer_seg_list = correct_refer_seg_dropdown or []
                vqa_list = vqa_dropdown or []
                reason_seg_list = reason_seg_dropdown or []
                reason_seg_plus_list = reason_seg_plus_dropdown or []
                multi_reason_seg_list = multi_reason_seg_dropdown or []
                sample_rates_str = sample_rates_input or "1,1,1,1,1,1"
                samples_per_epoch_str = samples_per_epoch_input or ""
                num_classes_per_sample_str = num_classes_per_sample_input or ""
                
                if not any([sem_seg_list, refer_seg_list, neg_refer_seg_list, correct_refer_seg_list, vqa_list, reason_seg_list, reason_seg_plus_list, multi_reason_seg_list]):
                    return "Please select at least one dataset", gr.update()
                
                load_result = handler.load_dataset(sem_seg_list, refer_seg_list, neg_refer_seg_list, correct_refer_seg_list, vqa_list, reason_seg_list, reason_seg_plus_list, multi_reason_seg_list, sample_rates_str, samples_per_epoch_str, num_classes_per_sample_str)
                if load_result is None or len(load_result) != 2:
                    return "Loading failed", gr.update()
                
                load_msg, default_filename = load_result
                
                if default_filename:
                    return load_msg, gr.update(placeholder=default_filename, value="")
                else:
                    return load_msg, gr.update(placeholder="000000571562", value="")
            except Exception as e:
                print(f"Dataset loading exception: {e}")
                return f"Loading failed: {str(e)}", gr.update()
        
        dropdown_inputs = [sem_seg_dropdown, refer_seg_dropdown, neg_refer_seg_dropdown, correct_refer_seg_dropdown, vqa_dropdown, reason_seg_dropdown, reason_seg_plus_dropdown, multi_reason_seg_dropdown, sample_rates_input, samples_per_epoch_input, num_classes_per_sample_input]
        
        for dropdown in dropdown_inputs:
            dropdown.change(
                update_selection_status,
                inputs=dropdown_inputs,
                outputs=[selection_status]
            )
        
        load_btn.click(
            load_and_update_search,
            inputs=dropdown_inputs,
            outputs=[load_status, jump_file_name_input]
        )

        with gr.Accordion("📖 Quick Start", open=False):
            gr.Markdown("""
            ## 🎯 Quick Start Guide
            
            1. **Select Dataset Types**: Choose dataset types (A: sem_seg, B: refer_seg, C: neg_refer_seg, D: correct_refer_seg, E: vqa, F: reason_seg, G: reason_seg_plus, H: multi_reason_seg)
            2. **Select Specific Datasets**: Choose specific datasets corresponding to your selected types
            3. **Load Dataset**: Click "Load Selected Datasets" to load the chosen datasets
            4. **Navigate**: Use the navigation buttons to browse through samples
            5. **Search**: Enter a filename to jump to a specific sample
            6. **View Details**: Expand the accordion sections to see detailed dataset field information
            
            ### Dataset Mapping:
            - **A (sem_seg)**: ade20k, cocostuff, pascal_part, paco_lvis, mapillary
            - **B (refer_seg)**: refclef, refcoco, refcoco+, 1refcocog
            - **C (neg_refer_seg)**: Negative referring segmentation datasets
            - **D (correct_refer_seg)**: Corrected referring segmentation datasets
            - **E (vqa)**: llava_instruct_150k
            - **F (reason_seg)**: ReasonSeg|train
            - **G (reason_seg_plus)**: ReasonSegPlus datasets
            - **H (multi_reason_seg)**: MultiReasonSeg|train
            """)
    
    return demo


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../dataset_sesame')
    parser.add_argument('--vision_tower', default='../dataset_sesame/clip-vit-large-patch14-336')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--samples_per_epoch', type=int, default=10)
    parser.add_argument('--precision', default='fp32')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--num_classes_per_sample', type=int, default=3)
    parser.add_argument('--exclude_val', type=bool, default=True)
    parser.add_argument('--dataset', default='')
    parser.add_argument('--sample_rates', default='1,1,1,1')
    parser.add_argument('--sem_seg_data', default='')
    parser.add_argument('--refer_seg_data', default='')
    parser.add_argument('--vqa_data', default='')
    parser.add_argument('--reason_seg_data', default='')
    parser.add_argument('--multi_reason_seg_data', default='')
    parser.add_argument('--explanatory', type=float, default=-1)
    parser.add_argument('--seg_token_num', type=int, default=3)
    parser.add_argument('--image_feature_scale_num', type=int, default=2)
    parser.add_argument('--num_classes_per_question', type=int, default=3)
    parser.add_argument('--pad_train_clip_images', type=bool, default=False)
    parser.add_argument('--masks_process_with_clip', type=bool, default=False)
    parser.add_argument('--preprocessor_config', default='./configs/preprocessor_448.json')
    parser.add_argument('--use_expand_question_list', type=bool, default=False)
    args, _ = parser.parse_known_args()
    demo = create_web_demo(args)
    
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_str = s.getsockname()[0]
    s.close()
    demo.launch(
        server_name=ip_str,
        server_port=7864,
        share=False,
        debug=False,
        show_error=True
    ) 