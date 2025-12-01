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
import os
import json

class GroundedMaskPropagation(nn.Module):
    """
    Grounded Mask Propagation algorithm for selecting appropriate hidden layers
    from language model for seg token embeddings and similarity map computation.
    
    Supports both single-head and multi-head Gumbel-Softmax for layer selection.
    """
    def __init__(self, 
                 num_layers=33, 
                 strategy='random_walker', 
                 mode=1, 
                 temperature=1.0, 
                 num_heads=4, 
                 hidden_dim=4096, 
                 fusion_method='concat', 
                 sparse_gumbel=False, 
                 top_k=3, 
                 log_interval=100, 
                 output_dir=None, 
                 logger=None,
                ):
        """
        Initialize the Grounded Mask Propagation module.
        
        Args:
            num_layers: Total number of hidden layers in the language model
            strategy: Strategy for layer selection - 'random_walker', 'policy_router', or 'multi_head_policy'
            mode: Selection mode
                  0: Default mode - always use the last layer (-1) for both seg token and similarity map
                  1: Fixed seg token layer (-1), variable similarity map layer
                  2: Synchronized layer selection for both seg token and similarity map
                  3: Independent layer selection for seg token and similarity map
            temperature: Temperature parameter for Gumbel-Softmax
            num_heads: Number of heads for multi-head Gumbel-Softmax
            hidden_dim: Hidden dimension of the language model
            fusion_method: Method to fuse multiple heads - 'concat', 'mean', or 'attention'
            sparse_gumbel: Whether to use Sparse Gumbel-Softmax (Top-k)
            top_k: Number of top layers to select in Sparse Gumbel-Softmax
            log_interval: Interval (in steps) for logging layer distribution during training
            output_dir: Directory to save layer distribution logs
            logger: Logger instance for printing messages
        """
        super().__init__()
        self._init_base_params(num_layers, strategy, mode, temperature, 
                              num_heads, hidden_dim, fusion_method,
                              sparse_gumbel, top_k)
        self._init_logging_params(log_interval, output_dir, logger)
        self._init_strategy_specific_params()
        self._init_mode_handlers()
        
    def _init_base_params(self, num_layers, strategy, mode, temperature, 
                         num_heads, hidden_dim, fusion_method,
                         sparse_gumbel, top_k):
        """初始化基本参数"""
        self.num_layers = num_layers
        self.strategy = strategy
        self.mode = mode
        self.temperature = temperature
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        self.sparse_gumbel = sparse_gumbel
        self.top_k = min(top_k, num_layers)  # 确保top_k不超过层数
        
    def _init_logging_params(self, log_interval, output_dir, logger):
        """初始化日志相关参数"""
        self.log_interval = log_interval
        self.output_dir = os.path.join(output_dir, "layer_distribution")
        self.logger = logger
        self.step_counter = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _init_strategy_specific_params(self):
        """初始化特定策略的参数"""
        if self.strategy == 'policy_router':
            self._init_policy_router_params()
        elif self.strategy == 'multi_head_policy':
            self._init_multi_head_policy_params()
            
    def _init_policy_router_params(self):
        """初始化policy_router策略的参数"""
        # 参考多头策略网络的实现方式
        self.seg_token_policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_layers)
        )
        
        self.similarity_policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_layers)
        )
        
        # 初始化方法
        for module in [self.seg_token_policy_net, self.similarity_policy_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
    def _init_multi_head_policy_params(self):
        """初始化multi_head_policy策略的参数"""
        # 为seg_token选择创建多头策略
        self.seg_token_multi_head_policy = self._create_multi_head_policy()
        
        # 为similarity选择创建多头策略
        self.similarity_multi_head_policy = self._create_multi_head_policy()
        
        # 根据融合方法创建额外的层
        self._init_fusion_layers()
        
    def _create_multi_head_policy(self):
        """创建多头策略网络"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_layers * self.num_heads)
        )
        
    def _init_fusion_layers(self):
        """初始化融合层"""
        if self.fusion_method == 'attention':
            self.seg_token_attn = nn.Linear(self.hidden_dim, 1)
            self.similarity_attn = nn.Linear(self.hidden_dim, 1)
            
        if self.fusion_method == 'concat':
            self.seg_token_proj = nn.Linear(self.hidden_dim * self.num_heads, self.hidden_dim)
            self.similarity_proj = nn.Linear(self.hidden_dim * self.num_heads, self.hidden_dim)
    
    def _init_mode_handlers(self):
        """初始化各种模式的处理器映射"""
        # 随机游走策略的模式处理器
        self.random_walker_mode_handlers = {
            1: self._random_walker_mode1_indices,
            2: self._random_walker_mode2_indices,
            3: self._random_walker_mode3_indices
        }
        
        # 策略路由推理阶段的模式处理器
        self.policy_router_inference_handlers = {
            0: self._policy_router_mode0_inference,
            1: self._policy_router_mode1_inference,
            2: self._policy_router_mode2_inference,
            3: self._policy_router_mode3_inference
        }
        
        # 策略路由训练阶段的模式处理器
        self.policy_router_training_handlers = {
            0: self._policy_router_mode0_training,
            1: self._policy_router_mode1_training,
            2: self._policy_router_mode2_training,
            3: self._policy_router_mode3_training
        }
        
        # 多头策略推理阶段的模式处理器
        self.multi_head_inference_handlers = {
            0: self._multi_head_mode0_inference,
            1: self._multi_head_mode1_inference,
            2: self._multi_head_mode2_inference,
            3: self._multi_head_mode3_inference
        }
        
        # 多头策略训练阶段的模式处理器
        self.multi_head_training_handlers = {
            0: self._multi_head_mode0_training,
            1: self._multi_head_mode1_training,
            2: self._multi_head_mode2_training,
            3: self._multi_head_mode3_training
        }
        
        # 随机游走策略的分布信息
        self.random_walker_distribution_info = {
            0: {
                'description': 'Always use the last layer (-1)',
                'seg_token_layer': -1,
                'similarity_layer': -1,
            },
            1: {
                'description': 'Fixed seg token layer (-1), variable similarity map layer',
                'seg_token_layer': -1,
                'similarity_layer_range': [0, self.num_layers - 1],
            },
            2: {
                'description': 'Synchronized random selection of seg token and similarity layers',
                'layer_range': [0, self.num_layers - 1],
            },
            3: {
                'description': 'Independent random selection of seg token and similarity layers',
                'seg_token_layer_range': [0, self.num_layers - 1],
                'similarity_layer_range': [0, self.num_layers - 1],
            }
        }
    
    def _get_random_walker_indices(self):
        """获取random_walker策略的层索引"""
        # 检查是否处于评估模式或者是mode=0
        if not self.training or self.mode == 0:
            # 评估状态下或mode=0，固定使用最后一层(-1)
            return -1, -1
        
        # 使用预定义的处理器映射
        handler = self.random_walker_mode_handlers.get(self.mode)
        if handler:
            return handler()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def _policy_router_inference(self, output_hidden_states):
        """Policy router strategy in inference mode"""
        with torch.no_grad():
            # 使用预定义的处理器映射
            handler = self.policy_router_inference_handlers.get(self.mode)
            if handler:
                return handler(output_hidden_states)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
                
    def _policy_router_training(self, output_hidden_states):
        """Policy router strategy in training mode"""
        # 获取最后一层的隐藏状态作为策略网络的输入
        policy_input = output_hidden_states[-1].mean(dim=1)  # [B, L, D] -> [B, D]
        
        # 使用策略网络生成层选择权重
        seg_token_logits = self.seg_token_policy_net(policy_input)  # [B, num_layers]
        similarity_logits = self.similarity_policy_net(policy_input)  # [B, num_layers]
        # 对每个样本应用Gumbel-Softmax，确保权重维度为 [B, num_layers]
        seg_token_weights = self._apply_gumbel_softmax(seg_token_logits, self.temperature)
        similarity_weights = self._apply_gumbel_softmax(similarity_logits, self.temperature)
        # 使用预定义的处理器映射
        handler = self.policy_router_training_handlers.get(self.mode)
        if handler:
            return handler(output_hidden_states, seg_token_weights, similarity_weights)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def _multi_head_policy_inference(self, output_hidden_states, hidden_states_stack, policy_input):
        """Multi-head policy strategy in inference mode"""
        B, N, L, D = hidden_states_stack.shape
        
        with torch.no_grad():
            # 使用预定义的处理器映射
            handler = self.multi_head_inference_handlers.get(self.mode)
            if handler:
                return handler(output_hidden_states, hidden_states_stack, policy_input, B)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
                
    def _multi_head_policy_training(self, output_hidden_states, hidden_states_stack, policy_input):
        """Multi-head policy strategy in training mode"""
        B, N, L, D = hidden_states_stack.shape
        
        # 使用预定义的处理器映射
        handler = self.multi_head_training_handlers.get(self.mode)
        if handler:
            return handler(output_hidden_states, hidden_states_stack, policy_input, B)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def _get_random_walker_distribution(self, result):
        """获取random_walker策略的分布信息"""
        # 使用预定义的分布信息
        mode_info = self.random_walker_distribution_info.get(self.mode, {})
        result.update(mode_info)
        
        # 添加最近选择的层信息
        if hasattr(self, 'last_seg_token_layer'):
            result['last_seg_token_layer'] = self.last_seg_token_layer
        if hasattr(self, 'last_similarity_layer'):
            result['last_similarity_layer'] = self.last_similarity_layer
            
        return result
    
    def forward(self, hidden_states, **kwargs):
        """
        Select appropriate hidden layers based on the chosen strategy and mode.
        
        Args:
            output_hidden_states: List of hidden states from language model layers
                                 [num_layers, batch_size, seq_len, hidden_dim]
            kwargs_dict: Dictionary with additional arguments
        
        Returns:
            seg_token_hidden: Hidden states for seg token embedding
            similarity_hidden: Hidden states for similarity map computation
        """
        # 根据当前是否为训练模式，决定是否记录层分布
        self._kwargs = kwargs or {}
        self._maybe_record_layer_distribution()
        
        # 根据策略调用相应的forward方法
        strategy_forward = {
            'random_walker': self._random_walker_forward,
            'policy_router': self._policy_router_forward,
            'multi_head_policy': self._multi_head_policy_forward
        }
        
        if self.strategy not in strategy_forward:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        return strategy_forward[self.strategy](hidden_states)
        
    def _maybe_record_layer_distribution(self):
        """根据条件决定是否记录层分布"""
        if not self.training:
            # 评估阶段，记录
            self._record_layer_distribution(is_eval=True)
        # 训练阶段不记录
    
    def _random_walker_forward(self, output_hidden_states):
        """Implementation of random walker strategy"""
        # 获取层索引
        seg_token_layer_idx, similarity_layer_idx = self._get_random_walker_indices()
        
        # 记录选择的层，用于日志
        self._record_selected_layers(seg_token_layer_idx, similarity_layer_idx)
        
        # 获取对应层的hidden states
        seg_token_hidden = output_hidden_states[seg_token_layer_idx]
        similarity_hidden = output_hidden_states[similarity_layer_idx]
        
        return seg_token_hidden, similarity_hidden
        
    def _random_walker_mode1_indices(self):
        """Mode 1: Fixed seg token layer (-1), variable similarity map layer"""
        return -1, torch.randint(0, self.num_layers, (1,)).item()
    
    def _random_walker_mode2_indices(self):
        """Mode 2: Synchronized random selection"""
        layer_idx = torch.randint(0, self.num_layers, (1,)).item()
        return layer_idx, layer_idx
    
    def _random_walker_mode3_indices(self):
        """Mode 3: Independent random selection"""
        return (torch.randint(0, self.num_layers, (1,)).item(), 
               torch.randint(0, self.num_layers, (1,)).item())
    
    def _record_selected_layers(self, seg_token_layer_idx, similarity_layer_idx):
        """Record selected layers for logging (only during inference)"""
        # Only record layers during inference
        if not self.training:
            self.last_seg_token_layer = seg_token_layer_idx if seg_token_layer_idx >= 0 else self.num_layers + seg_token_layer_idx
            self.last_similarity_layer = similarity_layer_idx if similarity_layer_idx >= 0 else self.num_layers + similarity_layer_idx
    
    def _policy_router_forward(self, output_hidden_states):
        """Implementation of policy router strategy using Gumbel-Softmax"""
        if not self.training:
            return self._policy_router_inference(output_hidden_states)
        else:
            return self._policy_router_training(output_hidden_states)
    
    def _policy_router_mode0_inference(self, output_hidden_states):
        """Mode 0: Default mode - always use the last layer (-1)"""
        seg_token_hidden = output_hidden_states[-1]
        similarity_hidden = output_hidden_states[-1]
        
        # 记录选择的层
        self._record_selected_layers(-1, -1)
        
        return seg_token_hidden, similarity_hidden
    
    def _policy_router_mode1_inference(self, output_hidden_states):
        """Mode 1: Fixed seg token layer (-1), learned similarity map layer"""
        seg_token_hidden = output_hidden_states[-1]
        
        # 获取最后一层的隐藏状态作为策略网络的输入
        policy_input = output_hidden_states[-1].mean(dim=1)  # [B, L, D] -> [B, D]
        
        # 对于similarity，使用argmax选择最可能的层
        similarity_logits = self.similarity_policy_net(policy_input)  # [B, num_layers]
        similarity_layer_idx = torch.argmax(similarity_logits.mean(dim=0)).item()
        similarity_hidden = output_hidden_states[similarity_layer_idx]
        
        # 记录选择的层
        self._record_selected_layers(-1, similarity_layer_idx)
        
        return seg_token_hidden, similarity_hidden
    
    def _policy_router_mode2_inference(self, output_hidden_states):
        """Mode 2: Synchronized layer selection (use same weights)"""
        # 获取最后一层的隐藏状态作为策略网络的输入
        policy_input = output_hidden_states[-1].mean(dim=1)  # [B, L, D] -> [B, D]
        
        # 使用相同的policy选择层
        seg_token_logits = self.seg_token_policy_net(policy_input)  # [B, num_layers]
        layer_idx = torch.argmax(seg_token_logits.mean(dim=0)).item()
        seg_token_hidden = output_hidden_states[layer_idx]
        similarity_hidden = seg_token_hidden  # Same layer
        
        # 记录选择的层
        self._record_selected_layers(layer_idx, layer_idx)
        
        return seg_token_hidden, similarity_hidden
    
    def _policy_router_mode3_inference(self, output_hidden_states):
        """Mode 3: Independent layer selection"""
        # 获取最后一层的隐藏状态作为策略网络的输入
        import pdb; pdb.set_trace()
        policy_input = output_hidden_states[-1].mean(dim=1)  # [B, L, D] -> [B, D]
        
        # 使用策略网络选择层
        seg_token_logits = self.seg_token_policy_net(policy_input)  # [B, num_layers]
        similarity_logits = self.similarity_policy_net(policy_input)  # [B, num_layers]
        
        seg_token_layer_idx = torch.argmax(seg_token_logits.mean(dim=0)).item()
        similarity_layer_idx = torch.argmax(similarity_logits.mean(dim=0)).item()
        
        # 记录选择的层
        self._record_selected_layers(seg_token_layer_idx, similarity_layer_idx)
        
        seg_token_hidden = output_hidden_states[seg_token_layer_idx]
        similarity_hidden = output_hidden_states[similarity_layer_idx]
        
        return seg_token_hidden, similarity_hidden
    
    def _policy_router_mode0_training(self, output_hidden_states, seg_token_weights, similarity_weights):
        """Mode 0: Default mode - always use the last layer (-1)"""
        seg_token_hidden = output_hidden_states[-1]
        similarity_hidden = output_hidden_states[-1]
        
        return seg_token_hidden, similarity_hidden
    
    def _policy_router_mode1_training(self, output_hidden_states, seg_token_weights, similarity_weights):
        """Mode 1: Fixed seg token layer (-1), learned similarity map layer"""
        seg_token_hidden = output_hidden_states[-1]
        similarity_hidden = self._weighted_hidden_states(output_hidden_states, similarity_weights)
        
        return seg_token_hidden, similarity_hidden
    
    def _policy_router_mode2_training(self, output_hidden_states, seg_token_weights, similarity_weights):
        """Mode 2: Synchronized layer selection (use same weights)"""
        weights = seg_token_weights  # Use same weights for both
        seg_token_hidden = self._weighted_hidden_states(output_hidden_states, weights)
        similarity_hidden = seg_token_hidden  # Same layer
        
        return seg_token_hidden, similarity_hidden
    
    def _policy_router_mode3_training(self, output_hidden_states, seg_token_weights, similarity_weights):
        """Mode 3: Independent layer selection"""
        seg_token_hidden = self._weighted_hidden_states(output_hidden_states, seg_token_weights)
        similarity_hidden = self._weighted_hidden_states(output_hidden_states, similarity_weights)
        return seg_token_hidden, similarity_hidden
    
    def _apply_gumbel_softmax(self, logits, temperature):
        """
        应用Gumbel-Softmax技术，使网络能够学习离散的one-hot编码
        
        Args:
            logits: 未归一化的对数概率 [batch_size, num_layers] 或 [num_layers]
            temperature: 温度参数，控制分布的平滑度，越低越接近one-hot
            
        Returns:
            近似one-hot的概率分布
        """
        # 添加Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        gumbel_logits = (logits + gumbel_noise) / temperature
        
        # 应用softmax得到近似one-hot向量
        if self.sparse_gumbel:
            # 使用稀疏Gumbel-Softmax（只保留top-k个值）
            return self._sparse_gumbel_softmax(gumbel_logits)
        else:
            # 标准Gumbel-Softmax
            return F.softmax(gumbel_logits, dim=-1)

    def _sparse_gumbel_softmax(self, gumbel_logits, softmax_dim=-1):
        """
        实现稀疏Gumbel-Softmax，只保留top-k个值
        
        Args:
            gumbel_logits: 添加了Gumbel噪声的logits
            softmax_dim: 应用softmax的维度
            
        Returns:
            稀疏的近似one-hot向量
        """
        # 获取原始形状和设备
        original_shape = gumbel_logits.shape
        device = gumbel_logits.device
        
        # 展平处理多维情况
        if len(original_shape) > 2:
            # 如果是多维张量，先展平处理
            flat_logits = gumbel_logits.view(-1, original_shape[-1])
        else:
            flat_logits = gumbel_logits
            
        batch_size = flat_logits.shape[0] if len(flat_logits.shape) > 1 else 1
        num_classes = flat_logits.shape[-1]
        
        # 确保top_k不超过类别数
        k = min(self.top_k, num_classes)
        
        # 获取top-k的值和索引
        if batch_size > 1:
            # 批量处理
            top_values, top_indices = torch.topk(flat_logits, k=k, dim=-1)
            
            # 创建掩码张量并填充top-k位置
            sparse_mask = torch.zeros_like(flat_logits).to(device)
            
            # 使用scatter_填充掩码
            sparse_mask.scatter_(1, top_indices, 1.0)
            
            # 应用掩码并执行softmax
            masked_logits = flat_logits * sparse_mask - 1e10 * (1 - sparse_mask)
            sparse_probs = F.softmax(masked_logits, dim=softmax_dim)
        else:
            # 单样本处理
            top_values, top_indices = torch.topk(flat_logits, k=k)
            
            # 创建掩码张量并填充top-k位置
            sparse_mask = torch.zeros_like(flat_logits).to(device)
            sparse_mask[top_indices] = 1.0
            
            # 应用掩码并执行softmax
            masked_logits = flat_logits * sparse_mask - 1e10 * (1 - sparse_mask)
            sparse_probs = F.softmax(masked_logits, dim=softmax_dim)
        
        # 恢复原始形状
        if len(original_shape) > 2:
            sparse_probs = sparse_probs.view(original_shape)
            
        return sparse_probs
    
    def _weighted_hidden_states(self, output_hidden_states, weights):
        """
        Compute weighted sum of hidden states based on layer weights
        
        Args:
            output_hidden_states: List of hidden states [num_layers, batch_size, seq_len, hidden_dim]
            weights: Layer selection weights from Gumbel-Softmax [batch_size, num_layers] or [num_layers]
            
        Returns:
            Weighted hidden states [batch_size, seq_len, hidden_dim]
        """
        # 检查weights的维度
        if len(weights.shape) == 1:
            # 如果是1D权重 [num_layers]，使用原来的逻辑
            device = output_hidden_states[0].device
            weighted_hidden = torch.zeros_like(output_hidden_states[0]).to(device)
            
            # Weighted sum of hidden states
            for i, hidden in enumerate(output_hidden_states):
                weighted_hidden += weights[i] * hidden
                
            return weighted_hidden
        else:
            # 如果是2D权重 [batch_size, num_layers]，使用批量处理
            batch_size = weights.shape[0]
            device = output_hidden_states[0].device
            
            # 将输出隐藏状态堆叠为 [num_layers, batch_size, seq_len, hidden_dim]
            stacked_hidden = torch.stack(output_hidden_states, dim=0)
            
            # 调整权重形状为 [batch_size, num_layers, 1, 1] 用于广播
            weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
            
            # 调整隐藏状态形状为 [num_layers, batch_size, seq_len, hidden_dim]
            # 然后转置为 [batch_size, num_layers, seq_len, hidden_dim]
            hidden_transposed = stacked_hidden.transpose(0, 1)
            
            # 应用权重并沿层维度求和
            # [batch_size, num_layers, seq_len, hidden_dim] * [batch_size, num_layers, 1, 1]
            # -> [batch_size, num_layers, seq_len, hidden_dim]
            # -> sum -> [batch_size, seq_len, hidden_dim]
            weighted_hidden = (hidden_transposed * weights_expanded).sum(dim=1)
            
            return weighted_hidden
    
    def _multi_head_policy_forward(self, output_hidden_states):
        """Implementation of multi-head policy router strategy using Gumbel-Softmax"""
        # 准备数据
        hidden_states_stack, policy_input = self._prepare_multi_head_input(output_hidden_states)
        
        if not self.training:
            return self._multi_head_policy_inference(output_hidden_states, hidden_states_stack, policy_input)
        else:
            return self._multi_head_policy_training(output_hidden_states, hidden_states_stack, policy_input)
    
    def _prepare_multi_head_input(self, output_hidden_states):
        """准备多头策略的输入数据"""
        # Stack hidden states for easier processing [B, N, L, D]
        hidden_states_stack = torch.stack(output_hidden_states, dim=1)
        B, N, L, D = hidden_states_stack.shape
        
        # Get CLS token or mean pooling for policy input
        # Using mean pooling across sequence length for simplicity
        pooled_hidden = hidden_states_stack.mean(dim=2)  # [B, N, D]
        
        # Use the last layer's pooled representation to predict policy
        policy_input = pooled_hidden[:, -1, :]  # [B, D]
        
        return hidden_states_stack, policy_input
    
    def _multi_head_mode0_inference(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 0: Default mode - always use the last layer (-1)"""
        seg_token_hidden = output_hidden_states[-1]
        similarity_hidden = output_hidden_states[-1]
        return seg_token_hidden, similarity_hidden
    
    def _multi_head_mode1_inference(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 1: Fixed seg token layer (-1), multi-head similarity map layer"""
        seg_token_hidden = output_hidden_states[-1]
        
        # 计算并选择similarity层
        similarity_hidden = self._select_multi_head_layers(
            output_hidden_states, 
            policy_input, 
            B, 
            self.similarity_multi_head_policy,
            "similarity"
        )
        
        return seg_token_hidden, similarity_hidden
    
    def _multi_head_mode2_inference(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 2: Synchronized multi-head selection (use same weights for both)"""
        # 计算并选择共享层
        selected_hidden_list = self._compute_multi_head_selection(
            output_hidden_states,
            policy_input,
            B,
            self.seg_token_multi_head_policy,
            store_layers=True,
            is_shared=True
        )
        
        # 融合结果
        seg_token_hidden = self._fuse_inference_hidden_states(selected_hidden_list, "seg_token")
        similarity_hidden = seg_token_hidden  # Same representation
        
        return seg_token_hidden, similarity_hidden
    
    def _multi_head_mode3_inference(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 3: Independent multi-head selection"""
        # 计算并选择seg_token层
        seg_token_hidden_list = self._compute_multi_head_selection(
            output_hidden_states,
            policy_input,
            B,
            self.seg_token_multi_head_policy,
            store_layers=True,
            layer_name="seg_token"
        )
        
        # 计算并选择similarity层
        similarity_hidden_list = self._compute_multi_head_selection(
            output_hidden_states,
            policy_input,
            B,
            self.similarity_multi_head_policy,
            store_layers=True,
            layer_name="similarity"
        )
        
        # 融合结果
        seg_token_hidden = self._fuse_inference_hidden_states(seg_token_hidden_list, "seg_token")
        similarity_hidden = self._fuse_inference_hidden_states(similarity_hidden_list, "similarity")
        
        return seg_token_hidden, similarity_hidden
    
    def _select_multi_head_layers(self, output_hidden_states, policy_input, B, policy_network, selection_type):
        """选择多头层并融合结果"""
        selected_hidden_list = self._compute_multi_head_selection(
            output_hidden_states,
            policy_input,
            B,
            policy_network,
            store_layers=True,
            layer_name=selection_type
        )
        
        return self._fuse_inference_hidden_states(selected_hidden_list, selection_type)
    
    def _compute_multi_head_selection(self, output_hidden_states, policy_input, B, policy_network, 
                                     store_layers=False, is_shared=False, layer_name=None):
        """计算多头选择并收集结果"""
        # 计算logits
        logits = policy_network(policy_input)  # [B, N*H]
        logits = logits.view(B, self.num_heads, self.num_layers)  # [B, H, N]
        
        # 使用argmax选择每个head最可能的层
        selected_layer_idx = logits.argmax(dim=-1)  # [B, H]
        
        # 记录每个头选择的层，用于日志（仅在非训练阶段）
        if store_layers and not self.training:
            layers = selected_layer_idx.detach().cpu().tolist()
            if is_shared:
                self.last_seg_token_head_layers = layers
                self.last_similarity_head_layers = layers  # 同步模式下相同
            elif layer_name == "seg_token":
                self.last_seg_token_head_layers = layers
            elif layer_name == "similarity":
                self.last_similarity_head_layers = layers
        
        # 为每个head收集对应层的hidden states
        selected_hidden_list = []
        for h in range(self.num_heads):
            # 获取当前head选择的层索引 [B]
            head_layer_idx = selected_layer_idx[:, h]
            
            # 为每个样本在batch中收集对应层的hidden states
            head_hidden_list = []
            for b in range(B):
                layer_idx = head_layer_idx[b].item()
                head_hidden_list.append(output_hidden_states[layer_idx][b:b+1])
            
            # 将batch中所有样本拼接起来
            head_hidden = torch.cat(head_hidden_list, dim=0)  # [B, L, D]
            selected_hidden_list.append(head_hidden)
            
        return selected_hidden_list
    
    def _multi_head_mode0_training(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 0: Default mode - always use the last layer (-1)"""
        seg_token_hidden = output_hidden_states[-1]
        similarity_hidden = output_hidden_states[-1]
        return seg_token_hidden, similarity_hidden
            
    def _multi_head_mode1_training(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 1: Fixed seg token layer (-1), multi-head similarity map layer"""
        seg_token_hidden = output_hidden_states[-1]
        
        # 计算similarity的logits和权重
        similarity_logits = self.similarity_multi_head_policy(policy_input)  # [B, N*H]
        similarity_logits = similarity_logits.view(B, self.num_heads, self.num_layers)  # [B, H, N]
        similarity_weights = self._apply_gumbel_softmax(similarity_logits, self.temperature)  # [B, H, N]
        
        # 融合hidden states
        similarity_hidden = self._fuse_multi_head_hidden_states(hidden_states_stack, similarity_weights, "similarity")
        
        return seg_token_hidden, similarity_hidden
            
    def _multi_head_mode2_training(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 2: Synchronized multi-head selection (use same weights for both)"""
        # 计算共享的logits和权重
        shared_logits = self.seg_token_multi_head_policy(policy_input)  # [B, N*H]
        shared_logits = shared_logits.view(B, self.num_heads, self.num_layers)  # [B, H, N]
        shared_weights = self._apply_gumbel_softmax(shared_logits, self.temperature)  # [B, H, N]
        
        # 融合hidden states
        seg_token_hidden = self._fuse_multi_head_hidden_states(hidden_states_stack, shared_weights, "seg_token")
        similarity_hidden = seg_token_hidden  # Same representation
        
        return seg_token_hidden, similarity_hidden
            
    def _multi_head_mode3_training(self, output_hidden_states, hidden_states_stack, policy_input, B):
        """Mode 3: Independent multi-head selection"""
        # 计算seg_token的logits和权重
        seg_token_logits = self.seg_token_multi_head_policy(policy_input)  # [B, N*H]
        seg_token_logits = seg_token_logits.view(B, self.num_heads, self.num_layers)  # [B, H, N]
        seg_token_weights = self._apply_gumbel_softmax(seg_token_logits, self.temperature)  # [B, H, N]
        
        # 计算similarity的logits和权重
        similarity_logits = self.similarity_multi_head_policy(policy_input)  # [B, N*H]
        similarity_logits = similarity_logits.view(B, self.num_heads, self.num_layers)  # [B, H, N]
        similarity_weights = self._apply_gumbel_softmax(similarity_logits, self.temperature)  # [B, H, N]
        
        # 融合hidden states
        seg_token_hidden = self._fuse_multi_head_hidden_states(hidden_states_stack, seg_token_weights, "seg_token")
        similarity_hidden = self._fuse_multi_head_hidden_states(hidden_states_stack, similarity_weights, "similarity")
        
        return seg_token_hidden, similarity_hidden
    
    def _fuse_hidden_states(self, hidden_states, selection_type, is_inference=False):
        """
        通用的隐藏状态融合方法
        
        Args:
            hidden_states: 隐藏状态，可能是张量[B, H, L, D]或列表[H x [B, L, D]]
            selection_type: 选择类型，"seg_token"或"similarity"
            is_inference: 是否为推理阶段
            
        Returns:
            融合后的隐藏状态[B, L, D]
        """
        # 确保hidden_states的格式一致
        if is_inference:
            # 从列表转换为张量 [H x [B, L, D]] -> [H, B, L, D] -> [B, H, L, D]
            hidden_states = torch.stack(hidden_states, dim=0).transpose(0, 1)
        
        B = hidden_states.shape[0]
        H = self.num_heads
        L = hidden_states.shape[2]
        D = hidden_states.shape[3]
        
        # 根据融合方法融合隐藏状态
        if self.fusion_method == 'concat':
            return self._fuse_by_concat(hidden_states, B, L, H, D, selection_type)
        elif self.fusion_method == 'mean':
            return hidden_states.mean(dim=1)
        elif self.fusion_method == 'attention':
            return self._fuse_by_attention(hidden_states, selection_type, is_inference)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
    def _fuse_by_concat(self, hidden_states, B, L, H, D, selection_type):
        """通过拼接融合隐藏状态"""
        # [B, H, L, D] -> [B, L, H*D]
        fused_hidden = hidden_states.transpose(1, 2).contiguous().view(B, L, H * D)
        
        # 投影回原始维度
        if selection_type == "seg_token":
            return self.seg_token_proj(fused_hidden)
        else:
            return self.similarity_proj(fused_hidden)
            
    def _fuse_by_attention(self, hidden_states, selection_type, is_inference=False):
        """通过注意力机制融合隐藏状态"""
        # 计算注意力分数
        if selection_type == "seg_token":
            attn_scores = self.seg_token_attn(hidden_states)
        else:
            attn_scores = self.similarity_attn(hidden_states)
            
        # 应用softmax
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 记录注意力权重（_record_attention_weights内部会检查是否为训练阶段）
        if not is_inference:
            self._record_attention_weights(attn_weights, selection_type)
            
        # 加权求和
        return (hidden_states * attn_weights).sum(dim=1)
        
    def _record_attention_weights(self, attn_weights, selection_type):
        """记录注意力权重（仅在非训练阶段）"""
        if not self.training:
            weights = attn_weights.mean(dim=2).squeeze(-1).detach().cpu().tolist()
            if selection_type == "seg_token":
                self.last_seg_token_head_weights = weights
            else:
                self.last_similarity_head_weights = weights
    
    def _fuse_multi_head_hidden_states(self, hidden_states_stack, weights, selection_type):
        """
        融合多头隐藏状态（训练阶段）
        
        Args:
            hidden_states_stack: 堆叠的隐藏状态 [B, N, L, D]
            weights: Gumbel-Softmax权重 [B, H, N]
            selection_type: 选择类型，"seg_token"或"similarity"
            
        Returns:
            融合后的隐藏状态 [B, L, D]
        """
        B, N, L, D = hidden_states_stack.shape
        H = self.num_heads
        
        # 扩展维度用于批量矩阵乘法
        # weights: [B, H, N] -> [B, H, N, 1, 1]
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        
        # hidden_states_stack: [B, N, L, D] -> [B, 1, N, L, D] -> [B, H, N, L, D]
        hidden_states_expanded = hidden_states_stack.unsqueeze(1).expand(-1, H, -1, -1, -1)
        
        # 加权求和
        # [B, H, N, L, D] * [B, H, N, 1, 1] -> [B, H, N, L, D] -> sum -> [B, H, L, D]
        selected_hidden = (weights * hidden_states_expanded).sum(dim=2)
        
        # 使用通用融合方法
        return self._fuse_hidden_states(selected_hidden, selection_type, is_inference=False)
    
    def _fuse_inference_hidden_states(self, hidden_states_list, selection_type):
        """
        融合多头隐藏状态（推理阶段）
        
        Args:
            hidden_states_list: 隐藏状态列表 [H x [B, L, D]]
            selection_type: 选择类型，"seg_token"或"similarity"
            
        Returns:
            融合后的隐藏状态 [B, L, D]
        """
        # 使用通用融合方法
        return self._fuse_hidden_states(hidden_states_list, selection_type, is_inference=True)
    
    def _record_layer_distribution(self, is_eval=False, step=None):
        """
        记录层选择分布并保存到JSON文件
        
        Args:
            is_eval: 是否为评估阶段
            step: 当前训练步数（如果是评估阶段，则为None）
        """
        # 获取层分布
        layer_dist = self.get_layer_distribution()
        if layer_dist is None:
            return
            
        # 添加元数据
        layer_dist = self._add_metadata_to_layer_dist(layer_dist, is_eval, step)
        
        # 处理张量数据
        layer_dist = self._process_tensor_data(layer_dist)
        
        # 生成文件名并保存
        filepath = self._save_layer_dist_to_file(layer_dist, is_eval, step)
                    
        return layer_dist
        
    def _add_metadata_to_layer_dist(self, layer_dist, is_eval, step):
        """添加元数据到层分布"""
        layer_dist['is_eval'] = is_eval
        layer_dist['mode'] = self.mode
        layer_dist['strategy'] = self.strategy
        
        if step is not None:
            layer_dist['step'] = step
            
        return layer_dist
        
    def _process_tensor_data(self, layer_dist):
        """处理张量数据，转换为列表"""
        for key, value in layer_dist.items():
            if hasattr(value, 'tolist'):
                layer_dist[key] = value.tolist()
                
        return layer_dist
        
    def _save_layer_dist_to_file(self, layer_dist, is_eval, step):
        """保存层分布到文件"""
        # 生成文件名，使用序号
        filename = os.path.basename(self._kwargs.get("image_paths")[0])
        filename = filename.split(".")[0] + ".json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存到JSON文件
        with open(filepath, 'w') as f:
            json.dump(layer_dist, f, indent=2)
            
        return filepath
    
    def get_layer_distribution(self):
        """获取层选择分布信息"""
        # 创建基本结果字典
        result = self._create_base_result_dict()
        
        # 根据策略添加特定信息
        strategy_handlers = {
            'random_walker': self._get_random_walker_distribution,
            'policy_router': self._get_policy_router_distribution,
            'multi_head_policy': self._get_multi_head_policy_distribution
        }
        
        if self.strategy in strategy_handlers:
            result = strategy_handlers[self.strategy](result)
            
        return result
        
    def _create_base_result_dict(self):
        """创建基本结果字典"""
        result = {
            'strategy': self.strategy,
            'mode': self.mode,
            'temperature': self.temperature,
        }
        
        # 添加稀疏Gumbel信息
        if self.sparse_gumbel:
            result.update({
                'sparse_gumbel': True,
                'top_k': self.top_k
            })
            
        return result
        
    def _get_random_walker_distribution(self, result):
        """获取random_walker策略的分布信息"""
        # 使用预定义的分布信息
        mode_info = self.random_walker_distribution_info.get(self.mode, {})
        result.update(mode_info)
        
        # 添加最近选择的层信息
        if hasattr(self, 'last_seg_token_layer'):
            result['last_seg_token_layer'] = self.last_seg_token_layer
        if hasattr(self, 'last_similarity_layer'):
            result['last_similarity_layer'] = self.last_similarity_layer
            
        return result
        
    def _get_policy_router_distribution(self, result):
        """获取policy_router策略的分布信息"""
        # 由于没有实际的输入，创建一个全1的dummy输入
        # 获取网络参数的数据类型和设备
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype
        
        # 创建与网络参数相同数据类型的dummy输入
        dummy_input = torch.ones(1, self.hidden_dim, device=device, dtype=dtype)
        
        # 计算概率分布
        seg_token_logits = self.seg_token_policy_net(dummy_input)  # [1, num_layers]
        similarity_logits = self.similarity_policy_net(dummy_input)  # [1, num_layers]
        
        seg_token_probs = F.softmax(seg_token_logits, dim=1).squeeze(0)  # [num_layers]
        similarity_probs = F.softmax(similarity_logits, dim=1).squeeze(0)  # [num_layers]
        
        # 获取最可能的层
        seg_token_top_layer = torch.argmax(seg_token_probs).item()
        similarity_top_layer = torch.argmax(similarity_probs).item()
        
        # 计算熵，衡量分布的不确定性
        seg_token_entropy = -torch.sum(seg_token_probs * torch.log(seg_token_probs + 1e-10))
        similarity_entropy = -torch.sum(similarity_probs * torch.log(similarity_probs + 1e-10))
        
        # 获取top-k层及其概率
        k = min(5, self.num_layers)
        seg_token_topk = self._get_topk_distribution(seg_token_probs, k)
        similarity_topk = self._get_topk_distribution(similarity_probs, k)
        
        # 更新结果字典
        result.update({
            'seg_token_distribution': seg_token_probs.detach().cpu(),
            'similarity_distribution': similarity_probs.detach().cpu(),
            'seg_token_top_layer': seg_token_top_layer,
            'similarity_top_layer': similarity_top_layer,
            'seg_token_entropy': seg_token_entropy.item(),
            'similarity_entropy': similarity_entropy.item(),
            'seg_token_topk': seg_token_topk,
            'similarity_topk': similarity_topk
        })
        
        return result
        
    def _get_topk_distribution(self, probs, k):
        """获取top-k分布"""
        topk_values, topk_indices = torch.topk(probs, k)
        return {idx.item(): val.item() for idx, val in zip(topk_indices, topk_values)}
        
    def _get_multi_head_policy_distribution(self, result):
        """获取multi_head_policy策略的分布信息"""
        # 添加多头策略的基本信息
        result.update({
            'num_heads': self.num_heads,
            'fusion_method': self.fusion_method
        })
        
        # 添加每个头选择的层信息
        if hasattr(self, 'last_seg_token_head_layers'):
            result['last_seg_token_head_layers'] = self.last_seg_token_head_layers
        if hasattr(self, 'last_similarity_head_layers'):
            result['last_similarity_head_layers'] = self.last_similarity_head_layers
            
        # 添加注意力权重信息（如果使用attention融合）
        if self.fusion_method == 'attention':
            if hasattr(self, 'last_seg_token_head_weights'):
                result['last_seg_token_head_weights'] = self.last_seg_token_head_weights
            if hasattr(self, 'last_similarity_head_weights'):
                result['last_similarity_head_weights'] = self.last_similarity_head_weights
                
        return result


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

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super(UGroundForCausalLM, self).__init__(config)
        self.model = UGroundModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize Grounded Mask Propagation
        # gmp_strategy = kwargs.get("gmp_strategy", "random_walker")
        gmp_strategy = kwargs.get("gmp_strategy", "policy_router")
        # gmp_strategy = kwargs.get("gmp_strategy", "multi_head_policy")
        gmp_mode = kwargs.get("gmp_mode", 3)
        gmp_temperature = kwargs.get("gmp_temperature", 1.0)
        gmp_num_heads = kwargs.get("gmp_num_heads", 4)
        gmp_fusion_method = kwargs.get("gmp_fusion_method", "mean")
        gmp_sparse_gumbel = kwargs.get("gmp_sparse_gumbel", False)
        gmp_top_k = kwargs.get("gmp_top_k", 3)  # 增加默认top_k值从3到5
        gmp_log_interval = kwargs.get("gmp_log_interval", 100)
        gmp_output_dir = kwargs.get("log_dir", None)
        logger = kwargs.get("logger", None)
        
        self.ppm = GroundedMaskPropagation(
            num_layers=33,  # Assuming 33 layers in the LLM
            strategy=gmp_strategy,
            mode=gmp_mode,
            temperature=gmp_temperature,
            num_heads=gmp_num_heads,
            hidden_dim=config.hidden_size,
            fusion_method=gmp_fusion_method,
            sparse_gumbel=gmp_sparse_gumbel,
            top_k=gmp_top_k,
            log_interval=gmp_log_interval,
            output_dir=gmp_output_dir,
            logger=logger
        )

        # Initialize weights and apply final processing
        self.post_init()
        self.cnt = 0

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
            output_hidden_states = output_i.hidden_states
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
        
        # Use Grounded Mask Propagation to select appropriate layers
        seg_token_hidden, similarity_hidden = self.ppm(output_hidden_states, **kwargs)
        import pdb; pdb.set_trace()
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        # Use selected hidden state for seg token embedding
        hidden_states.append(self.model.text_hidden_fcs[0](seg_token_hidden))
        # hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
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
        # Pass the similarity_hidden from ppm to compute_similarity
        similarity = self.compute_similarity(
            input_ids, 
            offset,
            output_hidden_states,  # Still pass all hidden states
            seg_token_mask,
            similarity_hidden,  # Pass the selected hidden state for similarity computation
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
    
    def compute_similarity(
            self, 
            input_ids, 
            offset,
            output_hidden_states, 
            seg_token_mask,
            similarity_hidden=None,
    ):  
        # Use the selected hidden state from ppm if provided, otherwise use the last layer
        hidden_states = similarity_hidden.clone() if similarity_hidden is not None else output_hidden_states[-1].clone()

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
        # sim_vis = cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET)
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
