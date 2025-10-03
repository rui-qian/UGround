# UGround: Towards Unified Visual Grounding with Unrolled Transformers

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)  [![arXiv](https://img.shields.io/badge/arXiv-2311.16090-red)](https://arxiv.org/abs/2412.17741) 

This repo provides the PyTorch source code of our paper: [UGround: Towards Unified Visual Grounding with Unrolled Transformers](https://arxiv.org/abs/2412.17741).

**Authors**: 
[Rui Qian](https://scholar.google.com.hk/citations?user=z3sAW3oAAAAJ&hl=zh-CN), 
[Xin Yin](https://scholar.google.com.hk/citations?hl=zh-CN&user=v3OOQQkAAAAJ), 
[Chuanhang Deng](xxx),
[Zhiyuan Peng](https://scholar.google.com.hk/citations?hl=zh-CN&user=kfiyUgIAAAAJ),
[Jian Xiong](https://scholar.google.com.hk/citations?hl=zh-CN&user=ePOXfkAAAAAJ),
[Wei Zhai](https://scholar.google.com.hk/citations?hl=zh-CN&user=seIo-acAAAAJ),
[Dejing Dou†](https://scholar.google.com.hk/citations?hl=zh-CN&user=qBHsQ04AAAAJ). 

## Abstract
We present UGround, a **U**nified visual \textbf{Ground}ing paradigm that dynamically selects 
intermediate layers across \textbf{U}nrolled transformers as ''mask as prompt'', diverging from the 
prevailing pipeline that leverages the fixed last hidden layer as ''\<SEG\> as prompt''. UGround addresses two primary challenges posed by the prevailing paradigm: (1) its reliance on the fixed last hidden layer, which sequentially amplifies cumulative errors arising from layer-by-layer propagation without intermediate correction, and (2) its use of \<SEG\> as a prompt, which implicitly projects textual embeddings into visual space without explicit spatial cues (e.g., coordinates). Central to UGround is Policy-Prompted Masking, which comprises two key components: Stochastic Skip Connection (SSC) and Mask as Prompt (MasP). SSC is a reinforcement learning policy that, via stochastic sampling, allows each \<SEG\> token to slide across unrolled transformer layers, enabling dynamic layer selection at which it connects to the 
vision model (e.g., SAM) in a skip-connection fashion. Given the selected hidden layer, MasP uses the similarity map derived from the \<SEG\> token and image tokens as a soft logit 
mask to prompt SAM for mask generation, offering explicit spatial cues through its 
activation regions. To validate the effectiveness of UGround, we, for the first time, have unified 
visual grounding within a single framework from an attribute perspective, spanning from 
traditional refer expression segmentation to newly proposed reasoning segmentation, 
single-target to multi-target, positive query to false premise (empty target). 
All codes and models are publicly available at https://github.com/rui-qian/UGround.

TBD