# Port Selector Usage Guide

自动端口选择器，用于为 DeepSpeed 训练自动分配可用的 `master_port`。

## 🚀 快速开始

### 基本用法

在任何需要使用 DeepSpeed 的脚本中添加：

```bash
#!/bin/bash

# 引入端口选择器
source scripts/port_selector.sh

# 现在 $MASTER_PORT 变量包含了一个可用端口
deepspeed --include "localhost:0" --master_port="$MASTER_PORT" train_ds.py
```

### 高级用法

```bash
#!/bin/bash

# 设置首选端口（可选）
export MASTER_PORT=25000

# 引入端口选择器
source scripts/port_selector.sh

# 验证端口是否可用
verify_port_selection

# 使用选定的端口
echo "Using port: $MASTER_PORT"
```

## 📋 功能特性

### 🔍 端口检测方法

脚本使用多种方法检测端口可用性：

1. **`ss` 命令**（推荐，现代Linux系统）
2. **`netstat` 命令**（传统方法，兼容性好）
3. **Python socket**（备用方案）
4. **假设可用**（最后备选）

### 🎯 端口选择策略

1. **首选端口检查**：先检查默认端口 24994 或环境变量 `MASTER_PORT`
2. **顺序搜索**：从首选端口开始逐个向上搜索
3. **范围搜索**：在指定范围内搜索（24990-25990）
4. **随机搜索**：在高端口段随机搜索（29000-39000）
5. **备用端口**：最终回退到 24994

### 📊 输出信息

- ℹ️  使用首选端口：当首选端口可用时
- ⚠️  端口占用警告：当首选端口被占用时
- ✅ 找到可用端口：成功找到替代端口时
- 🚀 端口设置完成：最终确认信息

## 🛠️ 可用函数

### 基础函数

```bash
# 检查端口是否可用
is_port_available 24994

# 查找可用端口（范围）
find_available_port 25000 26000

# 从指定端口开始查找下一个可用端口
get_next_available_port 24994

# 验证当前选择的端口
verify_port_selection
```

### 高级函数

```bash
# 预留连续的多个端口
reserve_port_range 25000 3  # 从25000开始预留3个端口
```

## 📝 集成示例

### 在现有脚本中集成

**方法1：直接引入**
```bash
#!/bin/bash

# 现有的训练脚本
MODEL_PATH="./model"
DATA_PATH="./data"

# 添加端口选择器
source scripts/port_selector.sh

# 现有的训练命令，只需要添加 --master_port="$MASTER_PORT"
deepspeed --include "localhost:0" --master_port="$MASTER_PORT" train_ds.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH"
```

**方法2：条件集成**
```bash
#!/bin/bash

# 检查是否需要自动端口选择
if [ -z "$MASTER_PORT" ]; then
    echo "🔍 Auto-selecting port..."
    source scripts/port_selector.sh
else
    echo "ℹ️  Using provided port: $MASTER_PORT"
fi

# 训练命令
deepspeed --include "localhost:0" --master_port="$MASTER_PORT" train_ds.py
```

### 多GPU配置示例

```bash
#!/bin/bash

# 多GPU训练脚本
GPU_COUNT=4
GPU_LIST="localhost:0,1,2,3"

# 自动端口选择
source scripts/port_selector.sh

# 可能需要多个端口的情况
if [ $GPU_COUNT -gt 1 ]; then
    # 为多GPU预留额外端口
    ADDITIONAL_PORTS=$(reserve_port_range $((MASTER_PORT + 1)) 2)
    echo "Additional ports reserved: $ADDITIONAL_PORTS"
fi

# 启动训练
deepspeed --include "$GPU_LIST" --master_port="$MASTER_PORT" train_ds.py
```

## 🔧 环境变量

| 变量名 | 描述 | 默认值 | 示例 |
|--------|------|--------|------|
| `MASTER_PORT` | 首选端口号 | 24994 | `export MASTER_PORT=25000` |

## 📊 故障排除

### 常见问题

**Q: 脚本报告端口可用，但 DeepSpeed 仍然报端口占用？**

A: 可能存在端口被快速占用的情况，可以尝试：
```bash
# 验证端口状态
verify_port_selection

# 重新选择端口
unset MASTER_PORT
source scripts/port_selector.sh
```

**Q: 在高负载系统上端口选择很慢？**

A: 可以指定一个更小的搜索范围：
```bash
# 修改 port_selector.sh 中的范围
find_available_port() {
    local start_port=${1:-25000}  # 提高起始端口
    local end_port=${2:-25100}    # 缩小搜索范围
    # ...
}
```

**Q: 如何在集群环境中使用？**

A: 确保所有节点都可以访问相同的端口：
```bash
# 在主节点选择端口
source scripts/port_selector.sh
echo "Selected port: $MASTER_PORT" > /shared/port.txt

# 在其他节点读取端口
MASTER_PORT=$(cat /shared/port.txt | grep "Selected port:" | cut -d' ' -f3)
```

## 🧪 测试

运行测试脚本验证功能：

```bash
bash scripts/test_port_selector.sh
```

测试内容包括：
- ✅ 基本端口选择
- ✅ 手动端口可用性检查  
- ✅ 范围内端口查找
- ✅ 多端口预留
- ✅ 最终验证

## 📖 相关文档

- [DeepSpeed 配置指南](https://www.deepspeed.ai/getting-started/)
- [分布式训练最佳实践](./distributed_training.md)
- [故障排除指南](./troubleshooting.md)

---

**注意**：该脚本设计为幂等性，可以安全地多次引入而不会产生副作用。 