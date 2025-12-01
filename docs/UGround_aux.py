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
from dataloaders.utils import safe_get
from .segment_anything import build_sam_vit_h

from typing import List, Tuple, Optional
# from analysis import AnalysisSaver
# analysis_saver = AnalysisSaver(debug=True, use_sam=True)

def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor

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
        
        # Register buffer to track decoder initialization status
        '''Loading PixelLM or GSVA with the original paper settings may trigger warnings 
        about uninitialized parameters, which are safe to ignore.'''
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
        
        hidden_size = config.hidden_size
        self.point_cls = nn.Sequential(
            nn.Linear(out_dim, hidden_size), 
            nn.GELU(), 
            nn.Linear(hidden_size, 1)
        )
        self.point_cls.train()
        for param in self.self.point_cls.parameters():
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
        self.config.resize_vision_tower_size = safe_get(kwargs, config, "resize_vision_tower_size", 224)
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
        self.cnt = 0
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super(UGroundForCausalLM, self).__init__(config)
        self.model = UGroundModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)

        return self.model_forward(**kwargs)
    
    # def encode_images(self, images, clip_resize_list):

    #     vit_mask_for_llm = self._generate_vit_attention_mask(images, clip_resize_list)
    #     self.image_embeddings = self.get_visual_embs(images)
    #     num_patches_per_side = self.get_vision_tower().num_patches_per_side
    #     image_embeddings = F.interpolate(
    #         self.image_embeddings.float(), 
    #         size=(num_patches_per_side, num_patches_per_side),  
    #         mode="bilinear",align_corners=False
    #     ).to(self.image_embeddings)
    #     image_embeddings = image_embeddings.flatten(-2).permute(0, 2, 1)
    #     image_embeddings = self.model.out_mm_projector(image_embeddings)

    #     return image_embeddings, vit_mask_for_llm, []

    # def _generate_vit_attention_mask(self, images, clip_resize_list):
    #     """
    #     Generates attention masks for ViT-based vision towers.
    #     """
    #     B, _, H, W = images.shape
    #     vit_mask = torch.zeros((B, H, W), device=images.device)
    #     for i, (h, w) in enumerate(clip_resize_list):
    #         vit_mask[i, :h, :w] = 1

    #     num_patches_per_side = self.get_vision_tower().num_patches_per_side
    #     mask_for_llm = F.interpolate(vit_mask[:, None].float(), size=(num_patches_per_side, num_patches_per_side), mode="nearest")[:, 0].to(torch.bfloat16)

    #     return mask_for_llm.flatten(1)

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
    
    def build_aux_target(
        self,
        similarity_map: torch.Tensor,      # [B, 1, H, W]
        gt_mask_for_sam: torch.Tensor,     # [B, 1, H, W]
        image_token_embeds: torch.Tensor,  # [B, C, H, W]
        target_size: int
    ):
        B, _, H, W = similarity_map.shape
        C = image_token_embeds.shape[1]

        # 去掉 channel 维度，变成 [B, H, W]
        gt = gt_mask_for_sam.squeeze(1).bool()  # [B, H, W]
        sim = similarity_map.squeeze(1)         # [B, H, W]

        # 直接对每张图的 similarity_map 做 softmax
        sim_flat = sim.reshape(B, -1)            # [B, H*W]
        sim_softmax = torch.softmax(sim_flat, dim=-1)  # [B, H*W]

        # 创建坐标网格 [H, W]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=gt.device),
            torch.arange(W, device=gt.device),
            indexing='ij'
        )  # [H, W]

        x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)

        b_coords = torch.arange(B, device=gt.device).view(B, 1, 1).expand(B, H, W)

        # 扁平化所有数据
        b_idx = b_coords.reshape(-1)
        y_idx = y_coords.reshape(-1)
        x_idx = x_coords.reshape(-1)

        labels_flat = gt.reshape(-1).long()                            # [B*H*W]
        sim_softmax_flat = sim_softmax.reshape(-1)                    # [B*H*W]

        features_flat = image_token_embeds[b_idx, :, y_idx, x_idx]     # [B*H*W, C]

        # 用 softmax 权重加权特征
        weighted_features_flat = features_flat * sim_softmax_flat.unsqueeze(1)  # [B*H*W, C]

        # 计算原图坐标（对齐中心 + 放缩）
        x_abs = x_idx.float() * (target_size / W)
        y_abs = y_idx.float() * (target_size / H)
        abs_coords_flat = torch.stack([x_abs, y_abs], dim=1)           # [B*H*W, 2]

        # 重塑为 [B, N, *]
        N = H * W
        labels = labels_flat.view(B, N)                     # [B, N]
        features = weighted_features_flat.view(B, N, C)     # [B, N, C]
        abs_coords = abs_coords_flat.view(B, N, 2)          # [B, N, 2]

        return (abs_coords, labels), features 
    
    def aux_loss(self, point_cls, pts_labels):
        N = point_cls.shape[0]
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()
        pts_labels = pts_labels.float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), pts_labels, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        return aux_loss_cls

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

        self.kwargs = kwargs
        similarity, seg_image_token_embeds = self.mask_as_prompt(
            input_ids, 
            offset,
            last_hidden_state, 
            seg_token_mask,
        )

        similarity_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            similarity_.append(similarity[start_i:end_i])
        similarity = similarity_

        seg_image_token_embeds_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            seg_image_token_embeds_.append(seg_image_token_embeds[start_i:end_i])
        seg_image_token_embeds = seg_image_token_embeds_
        
        # Run SAM
        image_embeddings = self.get_visual_embs(images)
        pred_masks = self.generate_pred_masks(pred_embeddings, image_embeddings, sam_mask_shape_list, similarity)

        model_output = output
        gt_masks = masks_list

        sm = seg_image_token_embeds[i]
        side = int(sm.shape[1] ** 0.5) # square output
        sm = seg_image_token_embeds[i].permute(0, 2, 1).reshape(sm.shape[0], -1, side, side).to(torch.float32)
        image_token_embeds = self.get_image_token_embeds_map(sm, sam_mask_shape_list[i][1], target_length=336)
        gt_mask_for_sam = self.get_image_token_embeds_map(masks_list[i].unsqueeze(1), sam_mask_shape_list[i][1], target_length=336)
        points, features = self.build_aux_target(similarity_map, gt_mask_for_sam, image_token_embeds, target_size=1024)
        import pdb; pdb.set_trace()

        # points_list, labels_list, similarity = self.similarity_as_points(
        #     input_ids, 
        #     offset,
        #     output_hidden_states, 
        #     seg_token_mask,
        #     sam_mask_shape_list
        # )
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
        aux_loss = self.aux_loss(point_cls, pts_labels)
        loss = ce_loss + mask_loss + aux_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "aux_loss": aux_loss,
        }
    
    def mask_as_prompt(
            self, 
            input_ids, 
            offset,
            output_hidden_states, 
            seg_token_mask,
    ):  

        hidden_states = output_hidden_states.clone()

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
        return similarity, seg_image_token_embeds

    def get_similarity_map(self, sm, shape, target_length = 336):
    
        # min-max norm
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        target_length = 256
        sm = self.apply_image_torch(sm, target_length=target_length)
        

        # import cv2
        # import os
        # import numpy as np
        # # import pdb; pdb.set_trace()
        # sim_vis = sm[0, 0, ...].detach().cpu().numpy()
        # sim_vis = (sim_vis * 255).astype(np.uint8)
        # ori_image = torch.from_numpy(cv2.imread(self.kwargs['image_paths'][0])[..., ::-1].copy()).permute(2,0,1).unsqueeze(0).contiguous()
        # # import pdb; pdb.set_trace()
        # ori_image = self.apply_image_torch(ori_image, target_length=target_length)
        # ori_image = ori_image.squeeze(0).permute(1,2,0).contiguous().numpy()
        # # sim_vis = cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET)
        # sim_vis = ori_image * 0.3 + sim_vis * 0.7
        # sim_path = os.path.join(f"similarity_{self.cnt}.png")
        # self.cnt += 1
        # cv2.imwrite(sim_path, sim_vis)
        return sm
    
    def get_image_token_embeds_map(self, sm, shape, target_length = 336):
    
        sm = torch.nn.functional.interpolate(sm, (target_length, target_length), mode='bilinear')
        
        oldh, oldw = shape
        scale = target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        sm = sm[:, :, 0:newh, 0:neww]
        sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
        # import pdb; pdb.set_trace()
        target_length = 256
        sm = self.apply_image_torch(sm, target_length=target_length)
        

        # import cv2
        # import os
        # import numpy as np
        # # import pdb; pdb.set_trace()
        # sim_vis = sm[0, 0, ...].detach().cpu().numpy()
        # sim_vis = (sim_vis * 255).astype(np.uint8)
        # ori_image = torch.from_numpy(cv2.imread(self.kwargs['image_paths'][0])[..., ::-1].copy()).permute(2,0,1).unsqueeze(0).contiguous()
        # # import pdb; pdb.set_trace()
        # ori_image = self.apply_image_torch(ori_image, target_length=target_length)
        # ori_image = ori_image.squeeze(0).permute(1,2,0).contiguous().numpy()
        # # sim_vis = cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET)
        # sim_vis = ori_image * 0.3 + sim_vis * 0.7
        # sim_path = os.path.join(f"similarity_{self.cnt}.png")
        # self.cnt += 1
        # cv2.imwrite(sim_path, sim_vis)
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
            scale = d / min(ori_h, ori_w)
            new_h, new_w = int(ori_h * scale + 0.5), int(ori_w * scale + 0.5)
            pad_h = (new_h - d) // 2
            pad_w = (new_w - d) // 2
            batch_points = []
            batch_labels = []
            for j in range(N_i):
                norm_j = normed[j].view(d, d)
                padded = F.pad(norm_j, (pad_w, pad_w, pad_h, pad_h), value=-1)  # [h, w]
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
            pred_masks = self.generate_pred_masks(pred_embeddings, image_embeddings, sam_mask_shape_list)
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
    model = UGroundForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, **kwargs
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
