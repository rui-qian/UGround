#!/bin/bash

# 路径配置（根据实际情况改）
MY_CONDA="/data/xinyin/qianrui/anaconda3"

# 清理当前可能残留的 conda 变量
unset CONDA_SHLVL
unset CONDA_PROMPT_MODIFIER
unset CONDA_PREFIX
unset _CONDA_EXE
unset _CONDA_ROOT
unset _CONDA_PYTHON_EXE
hash -r  # 清理 shell 中的缓存命令

# 激活你的 Conda
echo "🔄 Switching to your Conda at: $MY_CONDA"
eval "$($MY_CONDA/bin/conda shell.bash hook)"

