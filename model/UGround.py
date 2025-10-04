# @inproceedings{qian2024uground,
#   title={UGround: Towards Unified Visual Grounding with Unrolled Transformers},
#   author={Qian, Rui and Yin, Xin and Dou, Dejing},
#   booktitle={arXiv preprint arXiv:xxxx},
#   year={2025}
# }

from typing import List
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
import torch
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
from typing import List, Tuple, Optional
from dataloaders.utils import safe_get
from model.PPM import PolicyPromptedMasking
from tools.simi_loss import SimiLoss

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


class UGroundMetaModel(nn.Module):
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
            self.logger.info("Initializing UGround decoder modules...")
        
        result = self._initialize_UGround_modules(config)
        
        # Mark as initialized
        self.decoder_modules_initialized.fill_(True)
        
        if self.local_rank == 0 and self.logger is not None:
            self.logger.info("Decoder modules initialization completed successfully")
        
        return result

    def _initialize_UGround_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
            self.visual_model.prompt_encoder.train()
            for param in self.visual_model.prompt_encoder.parameters():
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


class UGroundModel(UGroundMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(UGroundModel, self).__init__(config, **kwargs)
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


class UGroundForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.config = config        
        self.config.resize_vision_tower = safe_get(kwargs, config, "resize_vision_tower", False)
        self.config.resize_vision_tower_size = safe_get(kwargs, config, "resize_vision_tower_size", 336)
        self.config.pad_train_clip_images = safe_get(kwargs, config, "pad_train_clip_images", False)
        self.config.vision_tower_for_mask = safe_get(kwargs, config, "vision_tower_for_mask", False)
        self.config.separate_mm_projector = safe_get(kwargs, config, "separate_mm_projector", False)
        self.config.masks_process_with_clip = safe_get(kwargs, config, "masks_process_with_clip", False)
        self.config.mm_projector_hidden_dim = safe_get(kwargs, config, "mm_projector_hidden_dim", 1)
        self.config.image_feature_scale_num = safe_get(kwargs, config, "image_feature_scale_num", 1)
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
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.rej_token_idx = kwargs.pop("rej_token_idx")
        self.seg_token_num = kwargs.pop("seg_token_num", 1)
        super(UGroundForCausalLM, self).__init__(config)
        self.model = UGroundModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.config.num_layers = safe_get(kwargs, config, "num_layers", 33)
        self.config.strategy = safe_get(kwargs, config, "strategy", "policy_walker")
        self.config.mode = safe_get(kwargs, config, "mode", 3)
        self.config.baseline_type = safe_get(kwargs, config, "baseline_type", "ema")
        self.config.critic_hidden_dim = safe_get(kwargs, config, "critic_hidden_dim", 128)
        self.config.baseline_beta = safe_get(kwargs, config, "baseline_beta", 1.0)
        self.config.eval_legacy = safe_get(kwargs, config, "eval_legacy", True)
        self.similarity_for_supervision = []
        self.simi_loss = SimiLoss()
        self.ppm = PolicyPromptedMasking(
                 num_layers=self.config.num_layers, 
                 strategy=self.config.strategy, 
                 mode=self.config.mode, 
                 hidden_dim=self.config.hidden_size,
                 eval_legacy=self.config.eval_legacy,
                 baseline_type=self.config.baseline_type,
                 critic_hidden_dim=self.config.critic_hidden_dim,
                 baseline_beta=self.config.baseline_beta)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            outputs = super().forward(**kwargs)
            self.all_hidden_states = outputs.all_hidden_states
            return outputs

        return self.model_forward(**kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        clip_resize_list=None,
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
                "clip_resize_list": clip_resize_list
            }
        )
        return model_inputs

    def generate_pred_masks(self, pred_embeddings, image_embeddings, sam_mask_shape_list, similarity):
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            # For inference (testing) mode only
            if pred_embeddings[i] is None:
                pred_mask = torch.zeros(sam_mask_shape_list[i][1]).to(image_embeddings.device).int()
                pred_masks.append(pred_mask)
                continue
            
            similarity_map = self.get_similarity_map(similarity[i], sam_mask_shape_list[i][1], target_length=336)
            similarity_map = similarity_map.to(pred_embeddings[i].dtype)

            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=similarity_map, text_embeds=pred_embeddings[i].unsqueeze(1)
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
        reeval: bool = False,
        clip_resize_list = None,
        **kwargs,
    ):  
        device, dtype = images.device, images.dtype
        batch_size = len(sam_mask_shape_list)
        assert batch_size == len(offset) - 1
        vision_tower = self.get_vision_tower()
        num_tokens_per_image = vision_tower.num_patches

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            extend_clip_resize_list = [clip_resize_list[0]] * length
            output_hidden_states = []
            output_ids = []
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
            if reeval:
                for k in range(length):
                    pred_output_ids = output_i.logits[k].argmax(dim=1)
                    pred_ids = input_ids[k].clone()
                    img_idx = (pred_ids == IMAGE_TOKEN_INDEX).nonzero().item()
                    pred_ids = torch.cat([pred_ids[0:img_idx], torch.zeros(num_tokens_per_image, device=device, dtype=torch.int64), pred_ids[img_idx + 1:]], dim=0)
                    seg_index_gt = (pred_ids == self.seg_token_idx).nonzero(as_tuple=True)[0]
                    seg_index_pred = seg_index_gt - 1
                    pred_seg_values = torch.where((pred_output_ids[seg_index_pred] != self.seg_token_idx), self.rej_token_idx, self.seg_token_idx)
                    # [REJ] token prediction:
                    rej_index_gt = (pred_ids == self.rej_token_idx).nonzero(as_tuple=True)[0]
                    rej_index_pred = rej_index_gt - 1
                    pred_rej_values = torch.where((pred_output_ids[rej_index_pred] != self.rej_token_idx), self.seg_token_idx, self.rej_token_idx)
                    # Update 
                    pred_ids[seg_index_gt] = pred_seg_values
                    pred_ids[rej_index_gt] = pred_rej_values
                    # The above steps woll make the [SEG/REJ] predictions have the same number of elements to masks
                    output_ids.append(pred_ids)
                
                # Replace all [REJ] to [SEG], then re-eval
                input_ids[input_ids == self.rej_token_idx] = self.seg_token_idx
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                    clip_resize_list=extend_clip_resize_list
                )
                output_hidden_states[-1] = output_i.hidden_states
                torch.cuda.empty_cache()
            output = output_i
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
                clip_resize_list=extend_clip_resize_list,
            )
            output_hidden_states = output.hidden_states
        
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        # HACK: padding numer-of-token-per-image in total 
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

        # hidden_states = []
        # assert len(self.model.text_hidden_fcs) == 1
        # hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        # last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # pred_embeddings = last_hidden_state[seg_token_mask]

        seg_token_embeds_for_similarity, seg_image_token_embeds_for_similarity, \
            seg_token_embeds_for_sam = self.ppm(output.all_hidden_states, input_ids, seg_token_mask, num_tokens_per_image, **kwargs)
        pred_embeddings = self.model.text_hidden_fcs[0](seg_token_embeds_for_sam)

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
        
        self.kwargs = kwargs
        similarity = self.compute_similarity(
            input_ids, 
            offset,
            output_hidden_states, 
            seg_token_mask,
            seg_token_embeds_for_similarity, 
            seg_image_token_embeds_for_similarity
        )
        
        similarity_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            similarity_.append(similarity[start_i:end_i])
        similarity = similarity_
        
        # Run SAM
        image_embeddings = self.get_visual_embs(images)
        pred_masks = self.generate_pred_masks(pred_embeddings, image_embeddings, sam_mask_shape_list, similarity)

        model_output = output
        gt_masks = masks_list
        
        if inference:
            self.similarity_for_supervision = []
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "output_ids": output_ids
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        # 累计
        policy_loss_acc = 0.0
        policy_loss_count = 0
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

            gaussian_bce_loss = (self.simi_loss.compute_gaussian_bce_loss(
                self.similarity_for_supervision[batch_idx], gt_mask, num_masks=gt_mask.shape[0]) 
                * gt_mask.shape[0] 
            )
            mask_bce_loss += gaussian_bce_loss
            gaussian_dice_loss = (self.simi_loss.compute_gaussian_dice_loss(
                self.similarity_for_supervision[batch_idx], gt_mask, num_masks=gt_mask.shape[0]) 
                * gt_mask.shape[0]
            )
            mask_dice_loss += gaussian_dice_loss
            reward = -(gaussian_bce_loss + gaussian_dice_loss).detach()
            policy_loss_i = self.ppm.policy_forward(reward)
            if policy_loss_i is not None:
                policy_loss_acc = policy_loss_acc + policy_loss_i
                policy_loss_count += 1
            
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # 归一与融合
        policy_loss = (policy_loss_acc / max(1, policy_loss_count)) if policy_loss_count > 0 else torch.tensor(0.0, device=ce_loss.device)
        lambda_pg = 0.1  # 可调
        loss = ce_loss + mask_loss + lambda_pg * policy_loss
        self.similarity_for_supervision = []
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }
    
    def compute_similarity(
            self, 
            input_ids=None, 
            offset=None,
            output_hidden_states=None, 
            seg_token_mask=None,
            seg_token_embeds=None, 
            seg_image_token_embeds=None
    ):          
        # hidden_states = output_hidden_states[-1].clone()

        # B, T, D = hidden_states.shape
        # device = hidden_states.device
        # seg_token_embeds = hidden_states[seg_token_mask.bool()]  # [N_seg, D]
        # image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)  # [B, T]
        # image_token_idx = (input_ids == IMAGE_TOKEN_INDEX).float().masked_fill(~image_token_mask, float('inf')).argmin(dim=1)  # [B]
        
        # num_patches = self.get_vision_tower().num_patches
        # idx_offset = torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)  # [1, 256] -> [B, 256]
        # gather_idx = image_token_idx.unsqueeze(1) + idx_offset  # [B, 256]
        # gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, 256, D]
        # image_token_embeds = torch.gather(hidden_states, dim=1, index=gather_idx)  # [B, 256, D]
    
        # all_batch_indices = torch.arange(B, device=seg_token_mask.device).unsqueeze(1).expand_as(seg_token_mask)  # [B, T]
        # batch_idx = all_batch_indices[seg_token_mask.bool()]  # [N_seg]
        # seg_image_token_embeds = image_token_embeds[batch_idx]  # [N_seg, 256, D]
        similarity = torch.einsum("sd, sid -> si", seg_token_embeds, seg_image_token_embeds)  # [N_seg, 256]
        return similarity

    def get_similarity_map(self, sm, shape, target_length = 336):
    
        # min-max norm
        # sm = sm.sigmoid()
        sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
        # reshape
        side = int(sm.shape[1] ** 0.5) # square output
        sm = sm.reshape(sm.shape[0], side, side).unsqueeze(1).to(torch.float32)
        sm = torch.nn.functional.interpolate(sm, (target_length, target_length), mode='bilinear')
        
        oldh, oldw = shape
        scale = target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        sm = sm[:, :, 0:newh, 0:neww]
        sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
        self.similarity_for_supervision.append(sm.squeeze(1))
        # import cv2
        # import os
        # import numpy as np
        # layer_name = 'uground-7B@reason_seg_train/layer40'
        # os.makedirs(layer_name, exist_ok=True)
        # sim_vis = sm[0, 0, ...].detach().cpu().numpy()
        # sample_id, ext = os.path.splitext(os.path.basename(self.kwargs['image_paths'][0]))
        # sim_path = os.path.join(layer_name, f"{sample_id}.npy")
        # np.save(sim_path, sim_vis)
        
        # sim_vis = (sim_vis * 255).astype(np.uint8)
        # ori_image = torch.from_numpy(cv2.imread(self.kwargs['image_paths'][0])[..., ::-1].copy()).permute(2,0,1).unsqueeze(0).contiguous()
        # # import pdb; pdb.set_trace()
        # # ori_image = self.apply_image_torch(ori_image, target_length=target_length)
        # ori_image = ori_image.squeeze(0).permute(1,2,0).contiguous().numpy()
        # sim_vis = cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET)
        # sim_vis = ori_image * 0.3 + sim_vis * 0.7
        # sim_path = os.path.join(layer_name, os.path.basename(self.kwargs['image_paths'][0]))
        # cv2.imwrite(sim_path, sim_vis)

        target_length = 256 # for sam
        sm = self.apply_image_torch(sm, target_length=target_length)
        return sm
        
    def apply_image_torch(self, image: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[2], image.shape[3], target_length
        )
        image = F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

        # Pad
        h, w = image.shape[-2:]
        padh = target_length - h
        padw = target_length - w
        image = F.pad(image, (0, padw, 0, padh))
        return image

    
    def get_preprocess_shape(
        self, oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        # import pdb; pdb.set_trace()
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def evaluate(
        self,
        images_clip=None,
        images=None,
        input_ids=None,
        sam_mask_shape_list=None,
        clip_resize_list=None,
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
                temperature=0.2,
                clip_resize_list=clip_resize_list
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
            # output_hidden_states = output_hidden_states.to(seg_token_mask.device)
            # pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states)
            # pred_embeddings = pred_embeddings.to(seg_token_mask.device)
            # pred_embeddings = pred_embeddings[seg_token_mask]
            
            seg_token_embeds_for_similarity, seg_image_token_embeds_for_similarity, \
            seg_token_embeds_for_sam = self.ppm(self.all_hidden_states, input_ids, seg_token_mask, num_tokens_per_image, **kwargs)
            pred_embeddings = self.model.text_hidden_fcs[0](seg_token_embeds_for_sam)
    
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
            similarity = self.compute_similarity(
                seg_token_embeds=seg_token_embeds_for_similarity, 
                seg_image_token_embeds=seg_image_token_embeds_for_similarity
            )
            
            similarity_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                similarity_.append(similarity[start_i:end_i])
            similarity = similarity_

            pred_masks = self.generate_pred_masks(pred_embeddings, image_embeddings, sam_mask_shape_list, similarity)
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

        return output_ids, output_pred_masks, object_presence, None

    def evaluate_v2(
        self,
        images_clip=None,
        images=None,
        input_ids=None,
        resize_list=None,
        clip_resize_list=None,
        original_size_list=None,
        max_new_tokens=32,
        tokenizer=None,
        sam_mask_shape_list=None,
        **kwargs
        # instance_out=False
    ):
         
        all_pred_embeddings = []
        all_similarity = []
        all_output_ids = []
        batch_seg_token_counts = []
        with torch.no_grad():
            _, _, output_image_features = self.encode_images(images_clip, clip_resize_list)
            multi_scale_num = self.config.image_feature_scale_num
            output_image_features = torch.stack(output_image_features, dim=0)
            for idx, input_id in enumerate(input_ids):
                if 0 in input_id:
                    unk_start = torch.where(input_id==0)[0].min()
                    _input_id = input_id[:unk_start]
                else:
                    _input_id = input_id

                outputs = self.generate(
                    images=images_clip,
                    input_ids=_input_id[None],
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    clip_resize_list=clip_resize_list
                )
                output_hidden_states = outputs.hidden_states[-1]
                output_ids = outputs.sequences
                all_output_ids.append(output_ids)
       
                if isinstance(self.seg_token_idx, list):
                    seg_token_num = self.seg_token_num
                    seg_token_mask = torch.zeros_like(output_ids[:, 1:]).bool()
                      
                    for seg_token_idx in self.seg_token_idx:
                        seg_token_mask = seg_token_mask | (output_ids[:, 1:] == seg_token_idx)  
                
                else:
                    seg_token_num = self.seg_token_num
                    seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
                # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
                vision_tower = self.get_vision_tower()
                num_tokens_per_image = vision_tower.num_patches
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], num_tokens_per_image-1)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )
            
                assert len(self.model.text_hidden_fcs) == 1
                seg_token_embeds_for_similarity, seg_image_token_embeds_for_similarity, \
                seg_token_embeds_for_sam = self.ppm(self.all_hidden_states, _input_id[None], seg_token_mask, num_tokens_per_image, **kwargs)
                pred_embeddings = self.model.text_hidden_fcs[0](seg_token_embeds_for_sam)
                
                similarity = self.compute_similarity(
                    seg_token_embeds=seg_token_embeds_for_similarity, 
                    seg_image_token_embeds=seg_image_token_embeds_for_similarity
                )

                seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
                seg_token_offset = seg_token_counts.cumsum(-1)
                seg_token_offset = torch.cat(
                    [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
                )
                seg_token_offset = seg_token_offset[[0, len(seg_token_offset)-1]]
                all_pred_embeddings.extend([pred_embeddings])
                all_similarity.extend([similarity])
                batch_seg_token_counts.append(seg_token_counts)
            
            batch_seg_token_counts = [torch.tensor(batch_seg_token_counts).to(seg_token_counts)]
            pred_embeddings = [torch.cat(all_pred_embeddings, dim=0)]
            similarity = [torch.cat(all_similarity, dim=0)]
            
            image_embeddings = self.get_visual_embs(images)            
            pred_masks = self.generate_pred_masks(pred_embeddings, image_embeddings, sam_mask_shape_list, similarity)
            mask_scores = [(pred_mask[:, 0].sigmoid().flatten(1) * (pred_mask[:, 0] > 0).flatten(1)).sum(1) / ((pred_mask[:, 0] > 0).flatten(1).sum(1) + 1e-6) for pred_mask in pred_masks]
            
        return all_output_ids, pred_masks, batch_seg_token_counts, mask_scores 

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
        

def load_pretrained_model_UGround(
    model_path,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    rej_token_idx = tokenizer("[REJ]", add_special_tokens=False).input_ids[0]
    model = UGroundForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, rej_token_idx=rej_token_idx, **kwargs
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

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

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, vision_tower, context_len


def init_UGround_model(args, model_args):
    # Create model
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
    model_args["seg_token_idx"] = args.seg_token_idx
    
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = UGroundForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_UGround_modules(model.get_model().config)

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
