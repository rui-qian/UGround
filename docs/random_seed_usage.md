# 🎲 随机数种子控制功能使用指南

## 概述

为了确保实验的可复现性，我们在 `utils/utils.py` 中添加了随机数种子控制功能。该功能可以设置所有相关的随机数种子，包括 Python、NumPy、PyTorch 和 CUDA。

## 功能特性

✅ **全面覆盖**: 设置 Python、NumPy、PyTorch、CUDA 的随机数种子  
✅ **CUDA优化**: 启用确定性模式，禁用benchmark模式  
✅ **环境变量**: 设置 PYTHONHASHSEED 确保完全一致性  
✅ **多种接口**: 提供详细输出和静默两种模式  
✅ **向后兼容**: 保留原始 `random_seed` 函数名  

## 可用函数

### 1. `set_random_seed(seed=42)`
**详细输出版本**，会打印设置过程和结果：
```python
from dataloaders.utils import set_random_seed

set_random_seed(42)
# 输出：
# 🎲 Setting random seed to 42 for reproducibility...
# 🔧 CUDA deterministic mode enabled
# ✅ Random seed set successfully
```

### 2. `set_random_seed_silent(seed=42)`
**静默版本**，不产生任何输出：
```python
from dataloaders.utils import set_random_seed_silent

set_random_seed_silent(42)  # 静默设置
```

### 3. `random_seed(seed=42)` 
**别名函数**，与 `set_random_seed` 完全相同：
```python
from dataloaders.utils import random_seed

random_seed(42)  # 等同于 set_random_seed(42)
```

## 训练脚本集成

### 自动使用
在 `train_ds.py` 中已经集成了随机数种子控制：

```bash
# 使用默认种子 42
python train_ds.py [其他参数]

# 使用自定义种子
python train_ds.py --seed 123 [其他参数]
```

### 手动集成到其他脚本
在任何Python脚本的开头添加：

```python
from dataloaders.utils import set_random_seed

def main():
    # 在所有其他操作之前设置种子
    set_random_seed(42)
    
    # 其他代码...
    pass

if __name__ == "__main__":
    main()
```

## 技术细节

### 设置的随机数源
- **Python built-in**: `random.seed()`
- **NumPy**: `np.random.seed()`
- **PyTorch**: `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
- **CUDA**: 确定性模式，禁用benchmark
- **环境变量**: `PYTHONHASHSEED`

### CUDA确定性设置
```python
torch.backends.cudnn.deterministic = True  # 确保确定性
torch.backends.cudnn.benchmark = False     # 禁用性能优化
```

⚠️ **注意**: 启用确定性模式可能会轻微影响训练速度，但能确保完全可复现的结果。

## 测试和验证

### 运行测试脚本
```bash
cd utils
python test_random_seed.py
```

### 测试结果示例
```
🎲 Random Seed Control Function Test
==================================================
🧪 Testing reproducibility with random seed...
🎉 Reproducibility test PASSED!
```

## 最佳实践

### 1. 在训练开始前设置
```python
from dataloaders.utils import set_random_seed
import torch
import numpy as np

# 在训练开始前设置种子
set_random_seed(42)

# 开始训练
model.train()
```

### 2. 多GPU环境
函数自动处理多GPU环境，无需额外配置：
```python
# 自动设置所有GPU的种子
set_random_seed(42)
```

### 3. 实验记录
建议在实验日志中记录使用的种子：
```python
seed = 42
set_random_seed(seed)
logger.info(f"Using random seed: {seed}")
```

### 4. 种子选择
- **默认种子**: 42（通用良好选择）
- **实验对比**: 使用相同种子确保公平比较
- **随机探索**: 使用不同种子测试模型稳定性

## 故障排除

### 如果结果仍不一致
1. **检查数据加载**: 确保数据加载顺序一致
2. **检查分布式训练**: 确保所有进程使用相同种子
3. **检查第三方库**: 某些库可能有自己的随机数生成器

### 性能考虑
- 确定性模式可能导致 5-10% 的性能下降
- 如果需要最佳性能，可以在生产环境中禁用确定性模式

## 示例用法

### 基础使用
```python
from dataloaders.utils import set_random_seed
import torch
import numpy as np

# 设置种子
set_random_seed(42)

# 现在所有随机操作都是可复现的
tensor1 = torch.randn(3, 3)
array1 = np.random.randn(3, 3)

# 重新设置相同种子
set_random_seed(42)

# 将产生相同的结果
tensor2 = torch.randn(3, 3)  # 与 tensor1 相同
array2 = np.random.randn(3, 3)  # 与 array1 相同
```

### 在训练脚本中使用
```python
from dataloaders.utils import set_random_seed

def train_model(seed=42):
    # 设置随机种子
    set_random_seed(seed)
    
    # 创建模型和数据
    model = create_model()
    train_loader = create_dataloader()
    
    # 开始训练
    for epoch in range(num_epochs):
        # 训练循环...
        pass
```

---

💡 **提示**: 使用固定的随机种子是科学实验的重要组成部分，它确保了结果的可复现性和实验的可信度。 