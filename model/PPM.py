import os
from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model.llava.constants import IMAGE_TOKEN_INDEX

class PolicyPromptedMasking(nn.Module):

    def __init__(self, 
                 num_layers=33, 
                 strategy='policy_walker', 
                 mode=0, 
                 hidden_dim=None,
                 eval_legacy=True, # always use the last layer to eval
                 baseline_type='ema',
                 critic_hidden_dim=128,
                 baseline_beta = 1.0
                ):

        super().__init__()
        self.num_layers = num_layers
        self.strategy = strategy
        self.mode = mode
        self.eval_legacy = eval_legacy
        self.hidden_dim = hidden_dim
        self.baseline_type = baseline_type
        self.reward_running_mean: float = 0.0
        self.log_probs = None
        
        # Layer gating weights (initialized with provided hidden_dim)
        self._layer_gate_W_similarity = nn.Parameter(
                torch.ones(self.num_layers, self.hidden_dim)
        )
        self.critic_hidden_dim = critic_hidden_dim
        self.baseline_beta = baseline_beta
        if self.baseline_type == 'critic':
            self.critic = nn.Sequential(
                nn.Linear(self.num_layers, self.critic_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.critic_hidden_dim, 1),
            )
        self.last_layer_probs_mean = None
        self._init_mode_handlers()
        
    def _init_mode_handlers(self):
        
        self.policy_walker_handlers = {
            1: self._policy_walker_mode1,
            2: self._policy_walker_mode2,
            3: self._policy_walker_mode3,
        }
                    
    def recurrent_unrolled(self, hidden_states, input_ids, seg_token_mask, num_patches):        
        # hidden_states: [B, L, T, D]
        hidden_states = torch.stack(hidden_states,dim=1)
        B, L, T, D = hidden_states.shape
        device = hidden_states.device

        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)         # [B, T]
        image_token_idx = image_token_mask.float().masked_fill(~image_token_mask, float('inf')).argmin(dim=1)  # [B]

        idx_offset = torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)  # [B, num_patches]
        gather_idx = image_token_idx.unsqueeze(1) + idx_offset  # [B, num_patches]
        gather_idx = gather_idx.unsqueeze(1).unsqueeze(-1).expand(-1, L, -1, D)  # [B, L, num_patches, D]

        seg_token_mask_bool = seg_token_mask.bool()  # [B, T]
        all_batch_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(seg_token_mask)  # [B, T]
        batch_idx = all_batch_indices[seg_token_mask_bool]  # [N_seg]
        # token_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)[seg_token_mask_bool]  # [N_seg]

        seg_token_embeds = hidden_states.permute(0, 2, 1, 3)  # [B, T, L, D]
        seg_token_embeds = seg_token_embeds[seg_token_mask_bool]  # [N_seg, L, D]

        image_token_embeds = torch.gather(hidden_states, dim=2, index=gather_idx)  # [B, L, num_patches, D]
        seg_image_token_embeds = image_token_embeds[batch_idx]  # [N_seg, L, num_patches, D]
    
        return seg_token_embeds, seg_image_token_embeds

     
    def forward(self, hidden_states, input_ids, seg_token_mask, num_patches, **kwargs):

        # reset per-call buffers for REINFORCE
        self.log_probs = None
        self.seg_token_embeds, self.seg_image_token_embeds = self.recurrent_unrolled(hidden_states, input_ids, seg_token_mask, num_patches)
        self.hidden_states = hidden_states
        self.N_seg, self.num_layers, self.D = self.seg_token_embeds.shape
        self.device = self.seg_token_embeds.device
        # Concise, side-effect-free handler selection
        strategy_handlers = {
            'policy_walker': self.policy_walker_handlers,
        }
        effective_mode = 1 if (not self.training and self.eval_legacy) else self.mode
        handler = strategy_handlers[self.strategy].get(effective_mode)
        return handler()
                      
    def _policy_walker_mode1(self):
        selected_indices = torch.full((self.N_seg,), self.num_layers - 1, device=self.device, dtype=torch.long)
        batch_indices = torch.arange(self.N_seg, device=self.device)
        
        seg_token_embeds_for_similarity = self.seg_token_embeds[batch_indices, selected_indices]  # [N_seg, D]
        seg_image_token_embeds_for_similarity = self.seg_image_token_embeds[batch_indices, selected_indices]  # [N_seg, 576, D]
        
        # seg_token_embeds_for_sam = self.seg_token_embeds[batch_indices, selected_indices]  # [N_seg, D]
        
        return seg_token_embeds_for_similarity, seg_image_token_embeds_for_similarity, seg_token_embeds_for_similarity
    
    def _policy_walker_mode2(self):
         # Build per-token layer logits from detached features (condition x) and learnable per-layer weights
        # seg_token_embeds: [N_seg, L, D], layer_gate_W_similarity: [L, D]
        # logits: [N_seg, L]
        logits_similarity = torch.einsum('nld,ld->nl', self.seg_token_embeds.detach(), self._layer_gate_W_similarity)
        # Compute mean layer probabilities across tokens for critic input
        if logits_similarity.size(0) > 0:
            layer_probs = torch.softmax(logits_similarity, dim=-1)  # [N_seg, L]
            self.last_layer_probs_mean = layer_probs.mean(dim=0).detach()  # [L]
        else:
            self.last_layer_probs_mean = None
            return self._policy_walker_mode1()
        
        # Per-token sampling from categorical(logits)
        dist = torch.distributions.Categorical(logits=logits_similarity)
        actions = dist.sample()  # [N_seg]
        self.log_probs = dist.log_prob(actions)  # [N_seg]
        batch_indices = torch.arange(self.seg_token_embeds.size(0), device=self.seg_token_embeds.device)
        seg_token_embeds_for_similarity = self.seg_token_embeds[batch_indices, actions]  # [N_seg, D]
        seg_image_token_embeds_for_similarity = self.seg_image_token_embeds[batch_indices, actions]  # [N_seg, P, D]
        
        # SAM 使用最后一层
        batch_indices = torch.arange(self.N_seg, device=self.device)
        selected_indices = torch.full((self.N_seg,), self.num_layers - 1, device=self.device, dtype=torch.long)
        seg_token_embeds_for_sam = self.seg_token_embeds[batch_indices, selected_indices]  # [N_seg, D]
        return seg_token_embeds_for_similarity, seg_image_token_embeds_for_similarity, seg_token_embeds_for_sam

    def _policy_walker_mode3(self):
        
        # Build per-token layer logits from detached features (condition x) and learnable per-layer weights
        # seg_token_embeds: [N_seg, L, D], layer_gate_W_similarity: [L, D]
        # logits: [N_seg, L]
        logits_similarity = torch.einsum('nld,ld->nl', self.seg_token_embeds.detach(), self._layer_gate_W_similarity)
        # Compute mean layer probabilities across tokens for critic input
        if logits_similarity.size(0) > 0:
            layer_probs = torch.softmax(logits_similarity, dim=-1)  # [N_seg, L]
            self.last_layer_probs_mean = layer_probs.mean(dim=0).detach()  # [L]
        else:
            self.last_layer_probs_mean = None
            return self._policy_walker_mode1()
        
        # Per-token sampling from categorical(logits)
        dist = torch.distributions.Categorical(logits=logits_similarity)
        actions = dist.sample()  # [N_seg]
        self.log_probs = dist.log_prob(actions)  # [N_seg]
        batch_indices = torch.arange(self.seg_token_embeds.size(0), device=self.seg_token_embeds.device)
        seg_token_embeds_for_similarity = self.seg_token_embeds[batch_indices, actions]  # [N_seg, D]
        seg_image_token_embeds_for_similarity = self.seg_image_token_embeds[batch_indices, actions]  # [N_seg, P, D]
        # Use the same chosen layer for SAM branch for consistency
        seg_token_embeds_for_sam = seg_token_embeds_for_similarity
        return seg_token_embeds_for_similarity, seg_image_token_embeds_for_similarity, seg_token_embeds_for_sam

    def policy_forward(self, reward: torch.Tensor, ema_decay: float = 0.9, world_size: int = 1, rank: int = 0) -> torch.Tensor:
        """Compute REINFORCE policy loss using stored log-probs and a baseline (EMA or Critic).
        Supports distributed training by gathering log_probs and rewards across all ranks.
        
        Args:
            reward: scalar tensor reward for current rank
            ema_decay: EMA decay factor for baseline
            world_size: number of GPUs in distributed training
            rank: current GPU rank
            
        Returns:
            scalar loss tensor or 0 if no log_probs
        """
        # Handle case with no log_probs on current rank
        if self.log_probs is None or self.log_probs.numel() == 0:
            local_log_probs = torch.zeros(0, dtype=torch.float32, device=reward.device)
            local_reward = torch.tensor(0.0, dtype=torch.float32, device=reward.device)
            local_count = torch.tensor(0, dtype=torch.long, device=reward.device)
        else:
            local_log_probs = self.log_probs.detach().to(torch.float32)
            local_reward = reward.detach().mean().to(torch.float32)
            local_count = torch.tensor(self.log_probs.numel(), dtype=torch.long, device=reward.device)
        
        # For single GPU training, use original logic
        if world_size <= 1 or not dist.is_initialized():
            if local_count.item() == 0:
                return 0
            r = local_reward
            if self.baseline_type == 'critic' and self.last_layer_probs_mean is not None:
                baseline = self.critic(self.last_layer_probs_mean.unsqueeze(0)).squeeze(0).squeeze(-1)
                adv = (r - baseline.detach()).to(self.log_probs.dtype)
                policy_loss = -(adv * self.log_probs).mean()
                critic_loss = (baseline - r).pow(2)
                return policy_loss + self.baseline_beta * critic_loss
            else:
                rm = self.reward_running_mean
                rm = ema_decay * rm + (1.0 - ema_decay) * float(r.item())
                self.reward_running_mean = rm
                adv = (r - rm)
                adv = adv.to(dtype=self.log_probs.dtype, device=self.log_probs.device)
                return -(adv.detach() * self.log_probs).mean()
        
        # === Distributed Training Logic ===
        # Step 1: Gather counts from all ranks to determine max length
        count_list = [torch.zeros(1, dtype=torch.long, device=reward.device) for _ in range(world_size)]
        dist.all_gather(count_list, local_count.unsqueeze(0))
        count_list = [c.item() for c in count_list]
        max_count = max(count_list)
        total_count = sum(count_list)
        
        # If no rank has any log_probs, return 0 on all ranks
        if total_count == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=reward.device, requires_grad=True) * 0.0
        
        # Step 2: Pad local_log_probs to max_count
        if local_count.item() < max_count:
            padding = torch.zeros(max_count - local_count.item(), dtype=torch.float32, device=reward.device)
            padded_log_probs = torch.cat([local_log_probs, padding], dim=0)
        else:
            padded_log_probs = local_log_probs
        
        # Step 3: Gather log_probs from all ranks
        gathered_log_probs_list = [torch.zeros(max_count, dtype=torch.float32, device=reward.device) for _ in range(world_size)]
        dist.all_gather(gathered_log_probs_list, padded_log_probs)
        
        # Step 4: Gather rewards from all ranks
        reward_list = [torch.zeros(1, dtype=torch.float32, device=reward.device) for _ in range(world_size)]
        dist.all_gather(reward_list, local_reward.unsqueeze(0))
        
        # Step 5: Compute global statistics
        # Unpack valid log_probs (remove padding)
        all_log_probs = []
        all_rewards = []
        for i in range(world_size):
            if count_list[i] > 0:
                all_log_probs.append(gathered_log_probs_list[i][:count_list[i]])
                all_rewards.append(reward_list[i].expand(count_list[i]))
        
        if len(all_log_probs) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=reward.device, requires_grad=True) * 0.0
        
        all_log_probs = torch.cat(all_log_probs, dim=0)  # [total_count]
        all_rewards = torch.cat(all_rewards, dim=0)  # [total_count]
        
        # Step 6: Compute global mean reward for baseline
        global_mean_reward = all_rewards.mean()
        
        # Step 7: Compute policy loss (on all ranks for gradient sync)
        if self.baseline_type == 'critic' and self.last_layer_probs_mean is not None:
            # Use critic baseline - gather layer_probs_mean from all ranks
            if self.last_layer_probs_mean is not None:
                local_layer_probs = self.last_layer_probs_mean.unsqueeze(0)  # [1, L]
            else:
                local_layer_probs = torch.zeros(1, self.num_layers, dtype=torch.float32, device=reward.device)
            
            # Gather all layer probs
            gathered_layer_probs = [torch.zeros_like(local_layer_probs) for _ in range(world_size)]
            dist.all_gather(gathered_layer_probs, local_layer_probs)
            
            # Average across ranks that have valid data
            valid_layer_probs = []
            for i in range(world_size):
                if count_list[i] > 0:
                    valid_layer_probs.append(gathered_layer_probs[i])
            
            if len(valid_layer_probs) > 0:
                mean_layer_probs = torch.stack(valid_layer_probs, dim=0).mean(dim=0)  # [1, L]
                baseline = self.critic(mean_layer_probs).squeeze(0).squeeze(-1)
            else:
                baseline = torch.tensor(0.0, dtype=torch.float32, device=reward.device)
            
            adv = (all_rewards - baseline.detach()).to(all_log_probs.dtype)
            policy_loss = -(adv * all_log_probs).mean()
            critic_loss = (baseline - global_mean_reward).pow(2)
            total_loss = policy_loss + self.baseline_beta * critic_loss
        else:
            # EMA baseline
            rm = self.reward_running_mean
            rm = ema_decay * rm + (1.0 - ema_decay) * float(global_mean_reward.item())
            self.reward_running_mean = rm
            adv = (all_rewards - rm)
            total_loss = -(adv * all_log_probs).mean()
        
        # Ensure the loss requires grad for DDP synchronization
        if not total_loss.requires_grad:
            total_loss = total_loss * torch.tensor(1.0, device=reward.device, requires_grad=True)
        
        return total_loss
