# 1. 项目概述

## 1.1 什么是 BitNet

BitNet 是微软研究院开发的 **1-bit 大语言模型推理框架**，专门为 BitNet b1.58 架构设计。这是一种革命性的模型量化技术，将模型权重压缩到仅使用 **1.58 bit**（三值：-1, 0, 1）。

### 核心论文

| 发布时间 | 论文标题 | 链接 |
|---------|---------|------|
| 2023.10 | BitNet: Scaling 1-bit Transformers for Large Language Models | [arXiv](https://arxiv.org/abs/2310.11453) |
| 2024.02 | The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits | [arXiv](https://arxiv.org/abs/2402.17764) |
| 2024.10 | 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs | [arXiv](https://arxiv.org/abs/2410.16144) |
| 2025.02 | Bitnet.cpp: Efficient Edge Inference for Ternary LLMs | [arXiv](https://arxiv.org/abs/2502.11880) |

## 1.2 技术特点

### 1.58-bit 量化原理

BitNet b1.58 使用三值量化方案：
- **权重值**: {-1, 0, 1}
- **位宽**: 实际使用 2 bit 存储（log₂(3) ≈ 1.58 bit）
- **激活值**: 保持 8-bit 精度

```
传统模型:  FP32/FP16 权重 × FP32/FP16 激活
BitNet:    1.58-bit 权重 × 8-bit 激活
```

### 性能优势

#### ARM CPU (Apple M2)
| 指标 | 改进幅度 |
|------|---------|
| 速度提升 | 1.37x - 5.07x |
| 能耗降低 | 55.4% - 70.0% |

#### x86 CPU (Intel)
| 指标 | 改进幅度 |
|------|---------|
| 速度提升 | 2.37x - 6.17x |
| 能耗降低 | 71.9% - 82.2% |

#### GPU (NVIDIA A100)
| 指标 | 改进幅度 |
|------|---------|
| Kernel 延迟降低 | 1.27x - 3.63x |
| 端到端生成延迟 | 约 3x 加速 |

## 1.3 支持的模型

### 官方模型

| 模型名称 | 参数量 | 平台支持 |
|---------|--------|---------|
| [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | 2.4B | ARM (I2_S, TL1) / x86 (I2_S, TL2) |

### 社区模型

| 模型名称 | 参数量 | 来源 |
|---------|--------|------|
| bitnet_b1_58-large | 0.7B | 1bitLLM |
| bitnet_b1_58-3B | 3.3B | 1bitLLM |
| Llama3-8B-1.58-100B-tokens | 8.0B | HF1BitLLM |
| Falcon3 Family | 1B-10B | tiiuae |
| Falcon-E Family | 1B-3B | tiiuae |

## 1.4 量化类型

bitnet.cpp 支持三种量化内核类型：

| 类型 | 描述 | 平台 |
|------|------|------|
| **I2_S** | 2-bit 整数量化，标准实现 | ARM/x86 |
| **TL1** | Lookup Table Level 1，ARM 优化 | ARM only |
| **TL2** | Lookup Table Level 2，x86 优化 | x86 only |

## 1.5 项目结构

```
BitNet/
├── 3rdparty/llama.cpp/     # llama.cpp 依赖库
├── Analysis/               # 代码分析文档
├── assets/                 # 资源文件（图片等）
├── build/                  # 编译输出目录
├── docs/                   # 官方文档
├── gpu/                    # GPU 推理内核
├── include/                # C++ 头文件
├── preset_kernels/         # 预调优的内核参数
├── src/                    # 核心源代码
├── utils/                  # 工具脚本
├── setup_env.py            # 环境配置脚本
├── run_inference.py        # 推理运行脚本
├── run_inference_server.py # 推理服务器
├── requirements.txt        # Python 依赖
└── CMakeLists.txt         # CMake 构建配置
```

## 1.6 依赖关系

### 核心依赖

- **llama.cpp**: 基于 ggerganov 的 llama.cpp 框架修改
- **T-MAC**: Lookup Table 方法论来自 T-MAC 项目

### 系统要求

- Python >= 3.9
- CMake >= 3.22
- Clang >= 18
- Conda (推荐)

## 1.7 使用流程

```mermaid
graph LR
    A[下载模型] --> B[setup_env.py]
    B --> C[生成内核代码]
    C --> D[编译项目]
    D --> E[转换模型]
    E --> F[run_inference.py]
```

### 快速开始

```bash
# 1. 克隆仓库
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

# 2. 安装依赖
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp
pip install -r requirements.txt

# 3. 构建项目
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# 4. 运行推理
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "Hello" -cnv
```
