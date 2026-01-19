# 2. 核心架构

## 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        BitNet.cpp 架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ HuggingFace │───▶│  转换工具   │───▶│   GGUF 模型文件     │ │
│  │   Models    │    │ (Python)    │    │                     │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                    │            │
│                                                    ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      推理引擎                               ││
│  │  ┌─────────────────┬─────────────────┬───────────────────┐ ││
│  │  │   llama.cpp     │   BitNet 内核   │    代码生成器     │ ││
│  │  │   (3rdparty)    │   (src/)        │    (utils/)       │ ││
│  │  └─────────────────┴─────────────────┴───────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      硬件后端                               ││
│  │  ┌─────────────────┬─────────────────┬───────────────────┐ ││
│  │  │   ARM CPU       │    x86 CPU      │      GPU          │ ││
│  │  │   (TL1/I2_S)    │    (TL2/I2_S)   │    (CUDA W2A8)   │ ││
│  │  └─────────────────┴─────────────────┴───────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 模块划分

### 2.2.1 核心模块

| 模块 | 路径 | 功能描述 |
|------|------|---------|
| **BitNet 内核** | `src/` | 1.58-bit 矩阵乘法实现 |
| **头文件** | `include/` | API 接口定义 |
| **代码生成器** | `utils/codegen_*.py` | 生成优化内核代码 |
| **模型转换** | `utils/convert-*.py` | HF→GGUF 格式转换 |

### 2.2.2 依赖模块

| 模块 | 路径 | 功能描述 |
|------|------|---------|
| **llama.cpp** | `3rdparty/llama.cpp/` | 基础推理框架 |
| **GGUF** | `3rdparty/llama.cpp/gguf-py/` | 模型格式库 |

### 2.2.3 GPU 模块

| 模块 | 路径 | 功能描述 |
|------|------|---------|
| **CUDA 内核** | `gpu/bitnet_kernels/` | W2A8 GEMV 实现 |
| **模型定义** | `gpu/model.py` | PyTorch 模型封装 |
| **权重转换** | `gpu/convert_*.py` | 权重格式转换 |

## 2.3 数据流

### 2.3.1 模型转换流程

```
HuggingFace Model (.safetensors/.bin)
         │
         ▼
┌─────────────────────────────────────┐
│  convert-hf-to-gguf-bitnet.py      │
│  - 加载 HF 权重                     │
│  - 提取模型参数                     │
│  - 量化权重 (TL1/TL2/I2_S)         │
│  - 生成 GGUF 文件                   │
└─────────────────────────────────────┘
         │
         ▼
    GGUF Model (.gguf)
         │
         ▼
┌─────────────────────────────────────┐
│  llama-quantize (for I2_S)         │
│  - F32 GGUF → I2_S GGUF            │
└─────────────────────────────────────┘
         │
         ▼
  Quantized GGUF Model
```

### 2.3.2 推理流程

```
User Prompt
    │
    ▼
┌─────────────────────────────────────┐
│  llama-cli / run_inference.py      │
│  - Tokenization                     │
│  - Model Loading                    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  BitNet Matrix Multiplication      │
│  ┌───────────────────────────────┐ │
│  │ Input Quantization            │ │
│  │ FP32 Activation → INT8       │ │
│  └───────────────────────────────┘ │
│  ┌───────────────────────────────┐ │
│  │ LUT/MAD Kernel               │ │
│  │ INT8 × INT2 → INT32          │ │
│  └───────────────────────────────┘ │
│  ┌───────────────────────────────┐ │
│  │ Output Dequantization        │ │
│  │ INT32 → FP32                 │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
Generated Tokens
```

## 2.4 关键数据结构

### 2.4.1 bitnet_tensor_extra

定义在 `include/ggml-bitnet.h`:

```cpp
struct bitnet_tensor_extra {
    int lut_scales_size;      // LUT 缩放因子大小
    int BK;                   // 块大小 K
    int n_tile_num;           // tile 数量
    uint8_t * qweights;       // 量化权重指针
    bitnet_float_type * scales; // 缩放因子
};
```

### 2.4.2 量化格式

#### I2_S 格式 (2-bit 量化)

```
每 128 个权重为一组 (QK_I2_S = 128)
存储布局:
┌────────────────────────────────────────┐
│  32 bytes: 128 个 2-bit 权重压缩存储   │
│  4 bytes: float32 scale               │
└────────────────────────────────────────┘

权重值映射:
  0b00 → -1
  0b01 →  0
  0b10 → +1
```

#### TL1/TL2 格式 (Lookup Table)

```
使用查找表加速计算:
- 预计算所有可能的 2-bit 权重组合结果
- 推理时直接查表获取部分和
```

## 2.5 API 接口

### 2.5.1 核心 API (ggml-bitnet.h)

```cpp
// 初始化 BitNet
void ggml_bitnet_init(void);

// 释放资源
void ggml_bitnet_free(void);

// 检查是否可以使用 BitNet 矩阵乘法
bool ggml_bitnet_can_mul_mat(
    const struct ggml_tensor * src0,  // 权重
    const struct ggml_tensor * src1,  // 输入
    const struct ggml_tensor * dst    // 输出
);

// 获取工作空间大小
size_t ggml_bitnet_mul_mat_get_wsize(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
);

// 初始化矩阵乘法任务
void ggml_bitnet_mul_mat_task_init(
    void * src1,        // 输入激活
    void * qlut,        // 量化 LUT
    void * lut_scales,  // LUT 缩放因子
    void * lut_biases,  // LUT 偏置
    int n, int k, int m, int bits
);

// 执行矩阵乘法
void ggml_bitnet_mul_mat_task_compute(
    void * src0,        // 量化权重
    void * scales,      // 权重缩放因子
    void * qlut,        // 量化 LUT
    void * lut_scales,  // LUT 缩放因子
    void * lut_biases,  // LUT 偏置
    void * dst,         // 输出
    int n, int k, int m, int bits
);

// 转换张量格式
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor);

// 获取类型位宽
int ggml_bitnet_get_type_bits(enum ggml_type type);
```

### 2.5.2 平台特定 API

#### ARM TL1
```cpp
void ggml_qgemm_lut(int m, int k, void* A, void* LUT, void* Scales, 
                    void* LUT_Scales, void* C);
void ggml_preprocessor(int m, int k, void* B, void* LUT_Scales, void* QLUT);
```

#### x86 TL2
```cpp
void ggml_qgemm_lut(int bs, int m, int k, int BK, void* A, void* sign, 
                    void* LUT, void* Scales, void* LUT_Scales, void* C);
void ggml_preprocessor(int bs, int m, int three_k, int two_k, void* B, 
                       void* LUT_Scales, void* Three_QLUT, void* Two_QLUT);
```

## 2.6 编译选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `BITNET_ARM_TL1` | 启用 ARM TL1 内核 | OFF |
| `BITNET_X86_TL2` | 启用 x86 TL2 内核 | OFF |

编译时自动根据目标平台选择合适的内核。
