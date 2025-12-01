# @InProceedings{Xia_2024_CVPR,
#     author    = {Xia, Zhuofan and Han, Dongchen and Han, Yizeng and Pan, Xuran and Song, Shiji and Huang, Gao},
#     title     = {GSVA: Generalized Segmentation via Multimodal Large Language Models},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2024},
#     pages     = {3858-3869}
# }

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from .llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


from .llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
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
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def iou_loss(
    pred_iou: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    num_masks: float
):
    pred_iou = pred_iou.to(torch.float32).sigmoid()
    pred_mask_ = pred_mask.detach().clone()
    target_mask_ = target_mask.detach().clone()
    inter = (pred_mask_ * target_mask_).sum()
    union = pred_mask_.sum() + target_mask_.sum() - inter
    gt_iou = inter / (union + 1e-8)
    
    iou_loss = ((gt_iou - pred_iou) ** 2).sum() / (num_masks + 1e-8)
    return iou_loss

class GSVAMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        # Set local_rank and logger from kwargs
        self.local_rank = kwargs.get("local_rank", 0)
        self.logger = kwargs.get("logger", None)
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
            self.logger.info("Initializing GSVA decoder modules...")
        
        result = self._initialize_gsva_modules(config)
        
        # Mark as initialized
        self.decoder_modules_initialized.fill_(True)
        
        if self.local_rank == 0 and self.logger is not None:
            self.logger.info("Decoder modules initialization completed successfully")
        
        return result

    def _initialize_gsva_modules(self, config):
        # SAM
        vision_pretrained = self.vision_pretrained or ''
        builder_sam = build_sam_vit_b if "sam_vit_b" in vision_pretrained else \
            build_sam_vit_l if "sam_vit_l" in vision_pretrained else build_sam_vit_h
        self.visual_model = builder_sam(self.vision_pretrained)
        # Projection layer for SAM
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

class GSVAModel(GSVAMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.seg_token_idx = kwargs.get("seg_token_idx", 0)


class GSVAForCausalLM(LlavaLlamaForCausalLM):
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
        else:
            config.mm_vision_tower = config.vision_tower
        
        config.separate_mm_projector = False
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.rej_token_idx = kwargs.pop("rej_token_idx")
        self.llm_tokenizer = kwargs.get("tokenizer", None)
        super().__init__(config)

        self.model = GSVAModel(config, seg_token_idx=self.seg_token_idx, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

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

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def pad_sequnce_and_stack(self, input_ids, attention_masks, labels):
        input_ids = nn.utils.rnn.pad_sequence(input_ids, True, 0)
        attention_masks = nn.utils.rnn.pad_sequence(attention_masks, True, False)
        labels = nn.utils.rnn.pad_sequence(labels, True, IGNORE_INDEX)
        return input_ids, attention_masks, labels

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        do_segs: List[bool] = None,
        inference: bool = False,
        reeval: bool = False,
        **kwargs,
    ):
        
        device, dtype = images.device, images.dtype
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        vision_tower = self.get_vision_tower()
        num_tokens_per_image = vision_tower.num_patches
        
        assert batch_size == len(offset) - 1
        if inference: # Segmentation Eval
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            output_hidden_states = []
            output_ids = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True
                )
                output_hidden_states.append(output_i.hidden_states)
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

                if reeval:
                    # Replace all [REJ] to [SEG], then re-eval
                    input_ids[input_ids == self.rej_token_idx] = self.seg_token_idx
                    output_i_reeval = super().forward(
                        images=images_clip_extend[: end_i - start_i],
                        attention_mask=attention_masks[start_i:end_i],
                        input_ids=input_ids[start_i:end_i],
                        output_hidden_states=True
                    )
                    output_hidden_states[-1] = output_i_reeval.hidden_states
                    torch.cuda.empty_cache()
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
        else: # Training 

            images_clip_list = []
            for i in range(len(offset) - 1): # offset marks each begin and end index for each images.
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (images_clip[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous())
                images_clip_list.append(images_clip_i)

            images_clip = torch.cat(images_clip_list, dim=0)
            # VLM inference, obtain LLaVA output
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx # mask for gathering [SEG] tokens
        

        seg_token_mask = torch.cat([seg_token_mask, torch.zeros(seg_token_mask.shape[0], 1, dtype=torch.bool, device=device)], dim=1)
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat([torch.zeros(seg_token_mask.shape[0], num_tokens_per_image - 1, dtype=torch.bool, device=device), seg_token_mask], dim=1)
        # *deprecated: not used anymore.
        rej_token_mask = input_ids[:, 1:] == self.rej_token_idx
        rej_token_mask = torch.cat([rej_token_mask, torch.zeros(rej_token_mask.shape[0], 1, dtype=torch.bool, device=device)], dim=1)
        rej_token_mask = torch.cat([torch.zeros(rej_token_mask.shape[0], num_tokens_per_image - 1, dtype=torch.bool, device=device),rej_token_mask], dim=1)
        mask_list_comp = []

        for lang_i in range(len(input_ids)):
            this_seg_token_m = seg_token_mask[lang_i].long() * 2
            this_rej_token_m = rej_token_mask[lang_i].long() * 1
            this_seg_rej = this_seg_token_m + this_rej_token_m
            gathered_idx = this_seg_rej.nonzero(as_tuple=True)[0]
            this_seg_rej = this_seg_rej[gathered_idx].eq(2).nonzero(as_tuple=True)[0]
            mask_list_comp.append(this_seg_rej)        
        
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=device), seg_token_offset], dim=0
        )     
        seg_token_offset = seg_token_offset[offset]
        pred_embeddings_ = []
        num_pred_embs = len(seg_token_offset) - 1
        for i in range(num_pred_embs):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        pred_masks = []
        pred_ious = []

        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape
            )
            pred_masks.append(pred_mask[:, 0])
            pred_ious.append(iou_predictions[:, 0])

        model_output = output
        gt_masks = masks_list
       
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "output_ids": output_ids
            }
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = 0
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
        
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            non_empty_mask = gt_mask.sum(dim=(1, 2)) >= 1.0
            non_empty_indices = torch.where(non_empty_mask)[0]
            gt_mask = gt_mask[non_empty_indices]
            
            if (
                gt_mask.shape[0] != pred_mask.shape[0]
            ):
                i0, i1 = input_ids[0], input_ids[1]
                i0, i1 = i0[i0 != IMAGE_TOKEN_INDEX], i1[i1 != IMAGE_TOKEN_INDEX]
                print(f"gt: {gt_mask.shape}, pred: {pred_mask.shape}\n" + \
                    f"Prompt0: {self.llm_tokenizer.decode(i0)}\n" + \
                    f"Prompt1: {self.llm_tokenizer.decode(i1)}\n" + \
                    f"GT_MASK sum :{gt_mask.sum(dim=(1, 2))}\n"
                )
                print(kwargs)
                raise RuntimeError("Found it!")

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
            "mask_loss": mask_loss
        }


def add_task_tokens(tokenizer, args):
    # 1. pad_token set to unknown token
    tokenizer.pad_token = tokenizer.unk_token
    # 2. add a [SEG] and a [REJ] token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens("[REJ]")
    args.rej_token_idx = tokenizer("[REJ]", add_special_tokens=False).input_ids[0]
    # 3. add <im_start> and <im_end>, same as llava into tokenizer
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    return tokenizer, args

def init_vision_seg_for_model(model, tokenizer, args):
    # Register special token ids
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Set up gradckpt for saving memory
    model.gradient_checkpointing_enable()
    # Init CLIP-ViT
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=args.torch_dtype, device=args.local_rank)
    # Init segmentation module
    model.get_model().initialize_decoder_modules(model.get_model().config)
    # Freeze all parameters
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    # Get Lora model, validation lora_r must be 0
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
        print(f"LoRA finetuning with rank = {lora_r}.")
    
    model.resize_token_embeddings(len(tokenizer))
    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_parts_keys = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
    if lora_r < 0:
        trainable_parts_keys.append("model.layers")
        print("No LoRA, full LLM finetuning.")
    elif lora_r == 0:
        print("LLM left frozen.")
    if not args.eval_only:
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in trainable_parts_keys
                ]
            ):
                p.requires_grad_(True)
        # Set up input with grads
        model.enable_input_require_grads()

    return model