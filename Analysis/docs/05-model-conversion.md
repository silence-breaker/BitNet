# 5. 模型转换工具

## 5.1 转换流程概述

```
┌────────────────────┐
│  HuggingFace Model │
│  (.safetensors)    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────────────────────────────┐
│  convert-hf-to-gguf-bitnet.py             │
│  - 读取模型配置和权重                       │
│  - 提取词表和特殊标记                       │
│  - 量化权重 (F32/TL1/TL2)                  │
│  - 写入 GGUF 格式                          │
└─────────┬──────────────────────────────────┘
          │
          ▼
┌────────────────────┐
│  GGUF Model (F32)  │
└─────────┬──────────┘
          │
          ▼ (仅 I2_S 量化类型)
┌────────────────────────────────────────────┐
│  llama-quantize                           │
│  - F32 -> I2_S 量化                        │
└─────────┬──────────────────────────────────┘
          │
          ▼
┌────────────────────────┐
│  Quantized GGUF Model  │
│  (ggml-model-*.gguf)   │
└────────────────────────┘
```

## 5.2 convert-hf-to-gguf-bitnet.py

### 5.2.1 主要功能

该脚本负责将 HuggingFace 格式的 BitNet 模型转换为 GGUF 格式：

1. **加载模型配置** (`config.json`)
2. **处理词表** (tokenizer)
3. **转换权重格式**
4. **写入 GGUF 文件**

### 5.2.2 核心类结构

```python
class Model(ABC):
    """模型基类"""
    
    def __init__(self, dir_model, ftype, fname_out, is_big_endian, use_temp_file):
        self.dir_model = dir_model
        self.ftype = ftype  # 量化类型
        self.fname_out = fname_out
        self.hparams = Model.load_hparams(self.dir_model)
        self.gguf_writer = gguf.GGUFWriter(fname_out, ...)
        
    @property
    @abstractmethod
    def model_arch(self) -> gguf.MODEL_ARCH:
        """返回模型架构类型"""
        pass
    
    def find_hparam(self, keys, optional=False):
        """查找超参数"""
        pass
    
    def set_vocab(self):
        """设置词表"""
        pass
    
    def get_tensors(self):
        """迭代获取所有张量"""
        pass
    
    def set_gguf_parameters(self):
        """设置 GGUF 元数据"""
        pass
```

### 5.2.3 支持的模型架构

通过装饰器注册模型类：

```python
@Model.register("LlamaForCausalLM", "BitnetForCausalLM")
class LlamaModel(Model):
    model_arch = gguf.MODEL_ARCH.LLAMA
    
@Model.register("FalconForCausalLM")
class FalconModel(Model):
    model_arch = gguf.MODEL_ARCH.FALCON
```

### 5.2.4 参数提取

```python
def set_gguf_parameters(self):
    self.gguf_writer.add_name(self.dir_model.name)
    self.gguf_writer.add_block_count(self.block_count)
    
    # 上下文长度
    if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)):
        self.gguf_writer.add_context_length(n_ctx)
    
    # 嵌入维度
    n_embd = self.find_hparam(["hidden_size", "n_embd"])
    self.gguf_writer.add_embedding_length(n_embd)
    
    # FFN 维度
    if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)):
        self.gguf_writer.add_feed_forward_length(n_ff)
    
    # 注意力头数
    n_head = self.find_hparam(["num_attention_heads", "n_head"])
    self.gguf_writer.add_head_count(n_head)
    
    # KV 头数 (GQA)
    if (n_head_kv := self.hparams.get("num_key_value_heads")):
        self.gguf_writer.add_head_count_kv(n_head_kv)
    
    # RoPE theta
    if (rope_theta := self.hparams.get("rope_theta")):
        self.gguf_writer.add_rope_freq_base(rope_theta)
    
    # RMS Norm epsilon
    if (f_rms_eps := self.hparams.get("rms_norm_eps")):
        self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
```

### 5.2.5 权重转换流程

```python
def write_tensors(self):
    for name, data_torch in self.get_tensors():
        # 跳过不需要的张量
        if name.endswith((".attention.masked_bias", ".attention.bias")):
            continue
            
        # 映射张量名称
        new_name = self.map_tensor_name(name)
        
        # 处理特殊张量
        # 1. 词嵌入
        if self.match_model_tensor_name(name, gguf.MODEL_TENSOR.TOKEN_EMBD, None):
            # 量化或保留原精度
            pass
            
        # 2. 线性层权重
        elif self.match_model_tensor_name(name, gguf.MODEL_TENSOR.ATTN_Q, bid):
            # 量化为 TL1/TL2/I2_S
            pass
            
        # 3. LayerNorm 权重
        elif "norm" in new_name:
            # 保留 F32 精度
            pass
        
        # 写入 GGUF
        self.gguf_writer.add_tensor(new_name, data)
```

## 5.3 量化类型详解

### 5.3.1 F32 (无量化)

```python
# 直接保存 FP32 权重
data = data_torch.numpy().astype(np.float32)
self.gguf_writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
```

### 5.3.2 TL1/TL2 (Lookup Table 量化)

```python
# 转换为 TL1/TL2 格式
if self.ftype == "tl1":
    # ARM 优化格式
    data_type = gguf.GGMLQuantizationType.TL1
elif self.ftype == "tl2":
    # x86 优化格式
    data_type = gguf.GGMLQuantizationType.TL2
    
# 量化过程
data = quantize_to_tl(data_torch, self.ftype)
self.gguf_writer.add_tensor(name, data, raw_dtype=data_type)
```

### 5.3.3 I2_S (2-bit 整数量化)

I2_S 量化通过 `llama-quantize` 工具完成：

```bash
# F32 GGUF -> I2_S GGUF
./build/bin/llama-quantize \
    models/model-f32.gguf \
    models/model-i2_s.gguf \
    I2_S 1
```

## 5.4 辅助转换工具

### 5.4.1 convert-helper-bitnet.py

封装了整个转换流程的辅助脚本：

```python
# 使用示例
python ./utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16
```

### 5.4.2 convert-ms-to-gguf-bitnet.py

用于转换微软内部格式的模型。

### 5.4.3 preprocess-huggingface-bitnet.py

预处理 HuggingFace 模型，准备转换所需的文件。

## 5.5 GPU 模型转换

### 5.5.1 convert_safetensors.py

将 safetensors 格式转换为 PyTorch 状态字典：

```python
# 使用示例
python ./convert_safetensors.py \
    --safetensors_file ./checkpoints/model.safetensors \
    --output checkpoints/model_state.pt \
    --model_name 2B
```

### 5.5.2 convert_checkpoint.py

对权重进行排列优化，以适应 GPU 内核的内存访问模式：

```python
# 权重按 16x32 块重排
# 每个块内的值按特定顺序存储

def permute_weight(weight):
    """
    优化权重布局:
    - 16 × 32 分块
    - 块内交错存储以优化 CUDA 内存访问
    """
    # 重排逻辑
    pass
```

## 5.6 GGUF 格式说明

### 5.6.1 文件结构

```
GGUF 文件结构:
┌────────────────────────────────┐
│  Magic Number (4 bytes)        │  'GGUF'
├────────────────────────────────┤
│  Version (4 bytes)             │  版本号
├────────────────────────────────┤
│  Tensor Count (8 bytes)        │  张量数量
├────────────────────────────────┤
│  Metadata KV Count (8 bytes)   │  元数据数量
├────────────────────────────────┤
│  Metadata Key-Value Pairs      │  模型配置
│  - architecture                │
│  - context_length              │
│  - embedding_length            │
│  - ...                         │
├────────────────────────────────┤
│  Tensor Infos                  │  张量元信息
│  - name                        │
│  - n_dims                      │
│  - dims[]                      │
│  - type                        │
│  - offset                      │
├────────────────────────────────┤
│  Tensor Data (aligned)         │  实际权重数据
└────────────────────────────────┘
```

### 5.6.2 量化类型枚举

```python
class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # ...
    I2_S = 26    # BitNet 2-bit 量化
    TL1 = 27     # ARM LUT 格式
    TL2 = 28     # x86 LUT 格式
```

## 5.7 使用示例

### 5.7.1 完整 CPU 转换流程

```bash
# 1. 下载模型
huggingface-cli download microsoft/BitNet-b1.58-2B-4T \
    --local-dir models/BitNet-b1.58-2B-4T

# 2. 转换为 F32 GGUF
python utils/convert-hf-to-gguf-bitnet.py \
    models/BitNet-b1.58-2B-4T \
    --outtype f32

# 3. 量化为 I2_S
./build/bin/llama-quantize \
    models/BitNet-b1.58-2B-4T/ggml-model-f32.gguf \
    models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    I2_S 1
```

### 5.7.2 TL1/TL2 直接转换

```bash
# ARM TL1
python utils/convert-hf-to-gguf-bitnet.py \
    models/BitNet-b1.58-2B-4T \
    --outtype tl1 \
    --quant-embd

# x86 TL2
python utils/convert-hf-to-gguf-bitnet.py \
    models/BitNet-b1.58-2B-4T \
    --outtype tl2 \
    --quant-embd
```

### 5.7.3 完整 GPU 转换流程

```bash
# 1. 下载 BF16 模型
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 \
    --local-dir ./checkpoints/bitnet-b1.58-2B-4T-bf16

# 2. 转换 safetensors
python gpu/convert_safetensors.py \
    --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors \
    --output checkpoints/model_state.pt \
    --model_name 2B

# 3. 权重排列优化
python gpu/convert_checkpoint.py \
    --input ./checkpoints/model_state.pt

# 4. 清理临时文件
rm ./checkpoints/model_state.pt
```
