# @inproceedings{qian2024reasoning,
#   title={Reasoning to Attend: Try to Understand How< SEG> Token Works},
#   author={Qian, Rui and Yin, Xin and Dou, Dejing},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   year={2025}
# }

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
from typing import List, Tuple, Optional
import numpy as np
from peft import LoraConfig, get_peft_model
from model.llava.model import *
from model.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from .llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)
from .segment_anything import build_sam_vit_h
from dataloaders.utils import safe_get

# from tools.read_analysis import AnalysisSaver
# analysis_saver = AnalysisSaver(debug=True, use_sam=False)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.25,  # α参数，控制正负样本的权重
    gamma: float = 2.0,   # γ参数，控制难易样本的衰减
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The raw predictions (logits) for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: Number of masks, used for normalizing the loss.
        alpha: Weighting factor for balancing positive and negative samples.
        gamma: Focusing parameter to decrease the relative loss for well-classified examples.

    Returns:
        Loss tensor
    """
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)  
    focal_factor = (1 - p_t) ** gamma 
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_factor * ce_loss
    loss = loss.flatten(1, 2) 
    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss

class READMetaModel(nn.Module):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        self.logger = kwargs.get("logger", None)
        self.local_rank = kwargs.get("local_rank", 1)
        self.register_buffer('decoder_modules_initialized', torch.tensor(False))        
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_decoder_modules(self.config)
    
    def initialize_decoder_modules(self, config):
        """Generic method name for model-specific module initialization
        
        Args:
            config: Model configuration
        """
        # Check if decoder modules have already been initialized
        if self.decoder_modules_initialized:
            if self.local_rank == 0 and self.logger is not None:
                self.logger.info("Decoder modules already initialized, skipping initialization")
            return
        
        if self.local_rank == 0 and self.logger is not None:
            self.logger.info("Initializing READ decoder modules...")
        
        result = self._initialize_read_modules(config)
        
        # Mark as initialized
        self.decoder_modules_initialized.fill_(True)
        
        if self.local_rank == 0 and self.logger is not None:
            self.logger.info("Decoder modules initialization completed successfully")
        
        return result

    def _initialize_read_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class READModel(READMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(READModel, self).__init__(config, **kwargs)
        # self.config = config
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

class READForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):  
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super(READForCausalLM, self).__init__(config)
        self.model = READModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
 
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)

        return self.model_forward(**kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs

    def generate_pred_masks(
            self, 
            pred_embeddings, 
            image_embeddings, 
            sam_mask_shape_list, 
            point_coords,
            point_labels,
        ):

        multimask_output = False
        pred_masks = []

        def get_preprocess_shape(
            oldh: int, oldw: int, long_side_length: int
        ):
            """
            Compute the output size given input size and target long side length.
            """
            scale = long_side_length * 1.0 / max(oldh, oldw)
            newh, neww = oldh * scale, oldw * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            return (newh, neww)
    
        def apply_coords(
            coords, 
            original_size,
        ) -> np.ndarray:
            """
            Expects a numpy array of length 2 in the final dimension. Requires the
            original image size in (H, W) format.
            """
            if coords.numel() ==0: return coords
            old_h, old_w = original_size
            new_h, new_w = get_preprocess_shape(
                original_size[0], original_size[1], 1024
            )
            coords = coords.clone().float()
            coords[..., 0] = coords[..., 0] * (new_w / old_w)
            coords[..., 1] = coords[..., 1] * (new_h / old_h)
            return coords
        
        for i in range(len(pred_embeddings)):
            # For inference (testing) mode only
            if pred_embeddings[i] is None:
                pred_mask = torch.zeros(sam_mask_shape_list[i][1]).to(image_embeddings.device).int()
                pred_masks.append(pred_mask)
                continue
            
            point_coords_ = apply_coords(point_coords[i], sam_mask_shape_list[i][1])
            point_coords_ = torch.as_tensor(
                point_coords_, dtype=torch.float, device=self.device
            )
            point_labels_ = torch.as_tensor(
                point_labels[i], dtype=torch.int, device=self.device
            )
            points = (point_coords_, point_labels_)
            
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=points, boxes=None, masks=None, text_embeds=pred_embeddings[i].unsqueeze(1)
            )             
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks, input_size=sam_mask_shape_list[i][0], original_size=sam_mask_shape_list[i][1]
            )
            pred_masks.append(pred_mask[:, 0])
    
        return pred_masks

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        sam_mask_shape_list: List[tuple],
        inference: bool = False,
        clip_resize_list = None,
        **kwargs,
    ):
        
        batch_size = len(sam_mask_shape_list)
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx

        # HACK: padding numer-of-token-per-image in total 
        vision_tower = self.get_vision_tower()
        num_tokens_per_image = vision_tower.num_patches
        padding_left = torch.zeros(
            seg_token_mask.shape[0],
            num_tokens_per_image - 1,
            dtype=seg_token_mask.dtype,
            device=seg_token_mask.device,
        )
        padding_right = torch.zeros(
            seg_token_mask.shape[0],
            1,
            dtype=seg_token_mask.dtype,
            device=seg_token_mask.device,
        )
        seg_token_mask = torch.cat(
            [padding_left, seg_token_mask, padding_right],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            extend_clip_resize_list = [clip_resize_list[0]] * length
            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                    clip_resize_list=extend_clip_resize_list
                )
                torch.cuda.empty_cache()
                output_hidden_states.append(output_i.hidden_states)
            output = None
        else:
            images_clip_list = []
            extend_clip_resize_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                extend_clip_resize_list.extend([clip_resize_list[i]] * (end_i - start_i))
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                clip_resize_list=extend_clip_resize_list
            )
            output_hidden_states = output.hidden_states

        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset],
            dim=0,
        )
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        points_list, labels_list, similarity = self.similarity_as_points(
            input_ids, 
            offset,
            output_hidden_states, 
            seg_token_mask,
            sam_mask_shape_list
        )
        
        # Run SAM
        image_embeddings = self.get_visual_embs(images)
        pred_masks = self.generate_pred_masks(
            pred_embeddings, 
            image_embeddings, 
            sam_mask_shape_list, 
            points_list,
            labels_list,
        )
        model_output = output
        gt_masks = masks_list

        # analysis_saver.save_for_gradio(
        #     offset=offset,
        #     points_list=points_list,
        #     labels_list=labels_list,
        #     similarity=similarity,
        #     pred_masks=pred_masks,
        #     gt_masks=gt_masks,
        #     **kwargs
        # )
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            non_empty_mask = gt_mask.sum(dim=(1, 2)) >= 1.0
            non_empty_indices = torch.where(non_empty_mask)[0]
            gt_mask = gt_mask[non_empty_indices]
            
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss 

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }
    
    def discrete_to_continuous(
        self, 
        sm: torch.Tensor, 
        selected_points: torch.Tensor,
        padding_value: float = -1) -> torch.Tensor:
        """
        将离散点根据 similarity map (sm) 转换为连续坐标。
        Args:
            sm: [H, W] similarity map
            selected_points: [N, 2]，N 个离散点坐标 (x, y)
            padding_value: padding 区域的值，用于创建 mask
        Returns:
            continuous_coordinates: [N, 2] 连续坐标
        """
        h, w = sm.shape
        device = sm.device

        # 创建 mask 来排除 padding 区域
        valid_mask = sm != padding_value  # [H, W]
        valid_mask_flat = valid_mask.view(-1)  # [H*W]
        
        # 只对有效区域计算 softmax
        sm_flat = sm.view(-1)  # [H*W]
        masked_sm = sm_flat.masked_fill(~valid_mask_flat, float('-inf'))
        softmax_probs = F.softmax(masked_sm, dim=0)
        
        grid_x, grid_y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy')
        grid_x = grid_x.contiguous().view(-1).float()
        grid_y = grid_y.contiguous().view(-1).float()
        selected_x = selected_points[:, 0].unsqueeze(1).float()
        selected_y = selected_points[:, 1].unsqueeze(1).float()
        grid_x_exp = grid_x.view(1, -1)
        grid_y_exp = grid_y.view(1, -1)
        distances = (grid_x_exp - selected_x)**2 + (grid_y_exp - selected_y)**2  # [N, H*W]
        weights = torch.exp(-distances)  # [N, H*W]
        
        # 将 padding 区域的权重设置为 0
        weights = weights * valid_mask_flat.unsqueeze(0).float()  # [N, H*W]
        
        softmax_probs = softmax_probs.unsqueeze(0).expand_as(weights)
        final_weights = weights * softmax_probs
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        continuous_x = (final_weights * grid_x_exp).sum(dim=1)
        continuous_y = (final_weights * grid_y_exp).sum(dim=1)

        return torch.stack([continuous_x, continuous_y], dim=1)  # [N, 2]

    # def points_as_prompt(
    #     self,
    #     similarity: torch.Tensor,                          # [N_seg, 576]
    #     sam_mask_shape_list: List[List[List[int]]],        # [B, [[SAM_H, SAM_W], [H, W]]]
    #     seg_token_mask: torch.Tensor,                      # [B, T]，每张图每个位置是否是 seg token
    #     offset: torch.Tensor,                              # [B]，每张图的 seg token 数量
    #     down_sample: int = 1,
    #     num_points: int = 60,
    #     t_pos: float = 0.8,
    #     t_neg: float = 0.2,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     返回每个 seg token 对应原图上的采样点坐标和标签:
    #         points_out: [N_seg, num_points, 2]
    #         labels_out: [N_seg, num_points, 1]
    #     """
    #     device = similarity.device
    #     N, T = similarity.shape
    #     side = int(T ** 0.5)
    #     d = side // down_sample

    #     seg_token_counts = seg_token_mask.int().sum(-1)  # [B]
    #     seg_token_offset = seg_token_counts.cumsum(0)
    #     seg_token_offset = torch.cat(
    #         [torch.zeros(1, dtype=torch.long, device=device), seg_token_offset], dim=0
    #     )
    #     seg_token_offset = seg_token_offset[offset]
    #     points_out = []
    #     labels_out = []

    #     for i in range(len(seg_token_offset) - 1):
    #         start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
    #         this_sim = similarity[start_i:end_i]  # [N_seg_i, 576]
    #         if this_sim.numel() == 0:
    #            points_out.append(torch.empty(0, dtype=torch.long, device=device))
    #            labels_out.append(torch.empty(0, dtype=torch.long, device=device))
    #            continue
    #         N_i = this_sim.shape[0]
            
    #         # 1. downsample + normalize
    #         sm_2d = this_sim.view(N_i, 1, side, side).float()
    #         sm_down = F.interpolate(sm_2d, size=(d, d), mode='bilinear').squeeze(1)
    #         flat = sm_down.view(N_i, -1)
    #         sm_min = flat.min(dim=1, keepdim=True)[0]
    #         sm_max = flat.max(dim=1, keepdim=True)[0]
    #         normed = (flat - sm_min) / (sm_max - sm_min + 1e-6)

    #         # 2. 获取原图大小
    #         _, [ori_h, ori_w] = sam_mask_shape_list[i]
    #         scale = d / min(ori_h, ori_w)
    #         new_h, new_w = int(ori_h * scale + 0.5), int(ori_w * scale + 0.5)
    #         pad_h = (new_h - d) // 2
    #         pad_w = (new_w - d) // 2
    #         batch_points = []
    #         batch_labels = []
    #         for j in range(N_i):
    #             norm_j = normed[j].view(d, d)
    #             padded = F.pad(norm_j, (pad_w, pad_w, pad_h, pad_h), value=-1)  # [h, w]
    #             flat_padded = padded.view(-1)

    #             pos_mask = flat_padded >= t_pos
    #             neg_mask = (flat_padded >= 0) & (flat_padded <= t_neg)
    #             neutral_mask = (flat_padded > t_neg) & (flat_padded < t_pos)

    #             pos_vals = flat_padded.masked_fill(~pos_mask, float('-inf'))
    #             neg_vals = flat_padded.masked_fill(~neg_mask, float('inf'))

    #             sorted_pos_idx = torch.argsort(pos_vals, descending=True)
    #             sorted_neg_idx = torch.argsort(neg_vals, descending=False)

    #             pos_idx = sorted_pos_idx[pos_vals[sorted_pos_idx] != float('-inf')]
    #             neg_idx = sorted_neg_idx[neg_vals[sorted_neg_idx] != float('inf')]
    #             neutral_idx = torch.nonzero(neutral_mask, as_tuple=False).squeeze()
    #             if neutral_idx.dim() == 0:
    #                 neutral_idx = neutral_idx.unsqueeze(0)

    #             selected = torch.cat([pos_idx, neg_idx, neutral_idx], dim=0)[:num_points]
    #             if selected.dim() == 0:
    #                 selected = selected.unsqueeze(0)

    #             x_coords = (selected % new_w).float() + 0.5
    #             y_coords = (selected // new_w).float() + 0.5
    #             seg_points = torch.stack([x_coords, y_coords], dim=1)
    #             seg_points[:, 0].clamp_(max=new_w - 1)
    #             seg_points[:, 1].clamp_(max=new_h - 1)

    #             seg_labels = torch.full((selected.numel(),), -1, dtype=torch.long, device=device)
    #             pos_mask = (selected.unsqueeze(1) == pos_idx[:num_points].unsqueeze(0)).any(dim=1)
    #             neg_mask = (selected.unsqueeze(1) == neg_idx[:num_points].unsqueeze(0)).any(dim=1)
    #             seg_labels[pos_mask] = 1
    #             seg_labels[neg_mask] = 0
    #             seg_points = self.discrete_to_continuous(padded, seg_points)
    #             seg_points[:, 0] = seg_points[:, 0] * (ori_w / new_w)
    #             seg_points[:, 1] = seg_points[:, 1] * (ori_h / new_h)
    #             # seg_points = seg_points.round().int().tolist() # for visualization
    #             batch_points.append(seg_points)
    #             batch_labels.append(seg_labels)
    #         points_out.append(torch.stack(batch_points, dim=0))
    #         labels_out.append(torch.stack(batch_labels, dim=0))

    #     return points_out, labels_out

    def points_as_prompt(
        self,
        similarity: torch.Tensor,                          # [N_seg, 576]
        sam_mask_shape_list: List[List[List[int]]],        # [B, [[SAM_H, SAM_W], [H, W]]]
        seg_token_mask: torch.Tensor,                      # [B, T]，每张图每个位置是否是 seg token
        offset: torch.Tensor,                              # [B]，每张图的 seg token 数量
        down_sample: int = 1,
        num_points: int = 60,
        t_pos: float = 0.8,
        t_neg: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回每个 seg token 对应原图上的采样点坐标和标签:
            points_out: [N_seg, num_points, 2]
            labels_out: [N_seg, num_points, 1]
        """
        device = similarity.device
        N, T = similarity.shape
        side = int(T ** 0.5)
        d = side // down_sample

        seg_token_counts = seg_token_mask.int().sum(-1)  # [B]
        seg_token_offset = seg_token_counts.cumsum(0)
        seg_token_offset = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), seg_token_offset], dim=0
        )
        seg_token_offset = seg_token_offset[offset]
        points_out = []
        labels_out = []

        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            this_sim = similarity[start_i:end_i]  # [N_seg_i, 576]
            if this_sim.numel() == 0:
               points_out.append(torch.empty(0, dtype=torch.long, device=device))
               labels_out.append(torch.empty(0, dtype=torch.long, device=device))
               continue
            N_i = this_sim.shape[0]
            
            # 1. downsample + normalize
            sm_2d = this_sim.view(N_i, 1, side, side).float()
            sm_down = F.interpolate(sm_2d, size=(d, d), mode='bilinear').squeeze(1)
            flat = sm_down.view(N_i, -1)
            sm_min = flat.min(dim=1, keepdim=True)[0]
            sm_max = flat.max(dim=1, keepdim=True)[0]
            normed = (flat - sm_min) / (sm_max - sm_min + 1e-6)

            # 2. 获取原图大小
            _, [ori_h, ori_w] = sam_mask_shape_list[i]
            scale = d / max(ori_h, ori_w)
            new_h, new_w = int(ori_h * scale + 0.5), int(ori_w * scale + 0.5)
            batch_points = []
            batch_labels = []
            for j in range(N_i):
                padded = normed[j].view(d, d)[0:new_h, 0:new_w].contiguous()
                flat_padded = padded.view(-1)

                pos_mask = flat_padded >= t_pos
                neg_mask = (flat_padded >= 0) & (flat_padded <= t_neg)
                neutral_mask = (flat_padded > t_neg) & (flat_padded < t_pos)

                pos_vals = flat_padded.masked_fill(~pos_mask, float('-inf'))
                neg_vals = flat_padded.masked_fill(~neg_mask, float('inf'))

                sorted_pos_idx = torch.argsort(pos_vals, descending=True)
                sorted_neg_idx = torch.argsort(neg_vals, descending=False)

                pos_idx = sorted_pos_idx[pos_vals[sorted_pos_idx] != float('-inf')]
                neg_idx = sorted_neg_idx[neg_vals[sorted_neg_idx] != float('inf')]
                neutral_idx = torch.nonzero(neutral_mask, as_tuple=False).squeeze()
                if neutral_idx.dim() == 0:
                    neutral_idx = neutral_idx.unsqueeze(0)

                selected = torch.cat([pos_idx, neg_idx, neutral_idx], dim=0)[:num_points]
                if selected.dim() == 0:
                    selected = selected.unsqueeze(0)

                x_coords = (selected % new_w).float() + 0.5
                y_coords = (selected // new_w).float() + 0.5
                seg_points = torch.stack([x_coords, y_coords], dim=1)
                seg_points[:, 0].clamp_(max=new_w - 1)
                seg_points[:, 1].clamp_(max=new_h - 1)

                seg_labels = torch.full((selected.numel(),), -1, dtype=torch.long, device=device)
                pos_mask = (selected.unsqueeze(1) == pos_idx[:num_points].unsqueeze(0)).any(dim=1)
                neg_mask = (selected.unsqueeze(1) == neg_idx[:num_points].unsqueeze(0)).any(dim=1)
                seg_labels[pos_mask] = 1
                seg_labels[neg_mask] = 0
                seg_points = self.discrete_to_continuous(padded, seg_points)
                seg_points[:, 0] = seg_points[:, 0] * (ori_w / new_w)
                seg_points[:, 1] = seg_points[:, 1] * (ori_h / new_h)
                # seg_points = seg_points.round().int().tolist() # for visualization
                batch_points.append(seg_points)
                batch_labels.append(seg_labels)
            points_out.append(torch.stack(batch_points, dim=0))
            labels_out.append(torch.stack(batch_labels, dim=0))

        return points_out, labels_out

    def similarity_as_points(
            self, 
            input_ids, 
            offset,
            output_hidden_states, 
            seg_token_mask,
            sam_mask_shape_list
    ):  

        hidden_states = output_hidden_states[-1].clone()
        
        B, T, D = hidden_states.shape
        device = hidden_states.device
        seg_token_embeds = hidden_states[seg_token_mask.bool()]  # [N_seg, D]
        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)  # [B, T]
        image_token_idx = (input_ids == IMAGE_TOKEN_INDEX).float().masked_fill(~image_token_mask, float('inf')).argmin(dim=1)  # [B]
        
        num_patches = self.get_vision_tower().num_patches
        idx_offset = torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)  # [1, 256] -> [B, 256]
        gather_idx = image_token_idx.unsqueeze(1) + idx_offset  # [B, 256]
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, 256, D]
        image_token_embeds = torch.gather(hidden_states, dim=1, index=gather_idx)  # [B, 256, D]
    
        all_batch_indices = torch.arange(B, device=seg_token_mask.device).unsqueeze(1).expand_as(seg_token_mask)  # [B, T]
        batch_idx = all_batch_indices[seg_token_mask.bool()]  # [N_seg]
        seg_image_token_embeds = image_token_embeds[batch_idx]  # [N_seg, 256, D]
        similarity = torch.einsum("sd, sid -> si", seg_token_embeds, seg_image_token_embeds)  # [N_seg, 256]
        points_list, labels_list = self.points_as_prompt(similarity, sam_mask_shape_list, seg_token_mask, offset) 
        return points_list, labels_list, similarity
        
    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        sam_mask_shape_list,
        max_new_tokens=32,
        **kwargs
    ):
        with torch.inference_mode():

            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=0.2
            )

            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx

            # HACK: padding numer-of-token-per-image in total 
            vision_tower = self.get_vision_tower()
            num_tokens_per_image = vision_tower.num_patches
            padding_left = torch.zeros(
                seg_token_mask.shape[0],
                num_tokens_per_image - 1,
                dtype=seg_token_mask.dtype,
                device=seg_token_mask.device,
            )
            seg_token_mask = torch.cat(
                [padding_left, seg_token_mask],
                dim=1,
            )
            assert len(self.model.text_hidden_fcs) == 1
            output_hidden_states = output_hidden_states.to(seg_token_mask.device)
            pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states)
            pred_embeddings = pred_embeddings.to(seg_token_mask.device)
            pred_embeddings = pred_embeddings[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset],
                dim=0,
            )

            pred_embeddings_ = []
            object_presence = []
            for i in range(len(seg_token_offset) - 1):
                if seg_token_counts[i] == 0:
                    pred_embeddings_.append(None)
                    object_presence.append(False)
                else:
                    start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                    pred_embeddings_.append(pred_embeddings[start_i:end_i])
                    object_presence.append(True)
           
            pred_embeddings = pred_embeddings_

            # Run SAM
            image_embeddings = self.get_visual_embs(images)
            offset = torch.tensor([0, 1]).to(seg_token_mask.device)
            points_list, labels_list, similarity = self.similarity_as_points(
                input_ids, 
                offset,
                output_hidden_states, 
                seg_token_mask,
                sam_mask_shape_list
            )

            pred_masks = self.generate_pred_masks(
            pred_embeddings, 
            image_embeddings, 
            sam_mask_shape_list, 
            image_path=None,
            point_coords=points_list,
            point_labels=labels_list,
            masks_list=None,
            conversation_list=None
        )  
            # pred_masks = self.generate_pred_masks(pred_embeddings, image_embeddings, sam_mask_shape_list)
            # Post processing for inference
            output_pred_masks = []
            for i, pred_mask in enumerate(pred_masks):
                if pred_embeddings[i] is not None:
                    pred_mask = (pred_mask[0] > 0).int()
                    if pred_mask.sum() == 0:
                        object_presence[i] = False
                    output_pred_masks.append(pred_mask)
                else:
                    output_pred_masks.append(pred_mask)
        
        return output_ids, output_pred_masks, object_presence
    
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []

            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings


def load_pretrained_model_READ(
    model_path,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs["device_map"] = device_map
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    model = READForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.model_max_length = kwargs.get("model_max_length", None)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))
    if "training" in kwargs and kwargs["training"] is True:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    # vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=model.dtype)
    # image_processor = vision_tower.image_processor

    if hasattr(model.config, "model_max_length"):
        context_len = model.config.model_max_length
    else:
        context_len = 2048
    
    return tokenizer, model, vision_tower, context_len

def init_READ_model(args, model_args):
    #Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        # padding_side="right",
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model_args["seg_token_idx"] = args.seg_token_idx
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16  
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = READForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True,**model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if args.use_released_param: pass # use released param by default, otherwise initialize SAM.
    else:model.get_model().initialize_read_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Configure LoRA if applicable
    if args.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            exclude_list = ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"]
            for name, module in model.named_modules():
                if isinstance(module, cls) and not any(x in name for x in exclude_list) \
                    and any([x in name for x in lora_target_modules]):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_parts = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
    for n, p in model.named_parameters():
        if any(part in n for part in trainable_parts):
            p.requires_grad = True
    return tokenizer, model, vision_tower

