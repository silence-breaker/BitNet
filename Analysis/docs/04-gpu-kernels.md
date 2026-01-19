# 4. GPU 推理内核

## 4.1 概述

BitNet GPU 内核专为 **W2A8 (2-bit Weight × 8-bit Activation)** GEMV 运算优化，针对 BitNet-b1.58-2B-4T 模型设计。

**文件位置**: `gpu/`

```
gpu/
├── bitnet_kernels/
│   ├── bitnet_kernels.cu    # CUDA 内核实现
│   ├── bitnet_kernels.h     # 内核头文件
│   ├── compile.sh           # 编译脚本
│   └── setup.py             # Python 绑定
├── model.py                 # PyTorch 模型封装
├── convert_checkpoint.py    # 权重转换
├── convert_safetensors.py   # safetensors 转换
├── generate.py              # 推理脚本
└── README.md
```

## 4.2 CUDA 内核架构

### 4.2.1 内核模板

```cuda
template <int M, int N, int K, int ws_num, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(
    int8_t* __restrict__ A,           // 输入激活 (INT8)
    int8_t* __restrict__ B,           // 权重 (INT2 packed)
    __nv_bfloat16* __restrict__ dtype_transform,  // 输出
    __nv_bfloat16* __restrict__ s,    // 输入缩放因子
    __nv_bfloat16* __restrict__ ws    // 权重缩放因子
);
```

**模板参数**:
| 参数 | 描述 |
|------|------|
| M | 批大小 |
| N | 输出维度 |
| K | 输入维度 |
| ws_num | 权重缩放分组数 |
| K_block_size | K 方向块大小 |
| N_block_size | N 方向块大小 |

### 4.2.2 支持的矩阵尺寸

```cuda
extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, 
    __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, 
    int M, int N, int K, cudaStream_t stream) {
    
    // 2B 模型层尺寸
    if (M == 1 && N == 3840 && K == 2560) {
        ladder_int8xint2_kernel<1, 3840, 2560, 3, 8, 16>
            <<<dim3(240, 1, 1), dim3(8, 16, 1), 0, stream>>>(...)
    }
    else if (M == 1 && N == 2560 && K == 2560) {
        ladder_int8xint2_kernel<1, 2560, 2560, 1, 8, 16>
            <<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(...)
    }
    else if (M == 1 && N == 13824 && K == 2560) {
        ladder_int8xint2_kernel<1, 13824, 2560, 2, 8, 16>
            <<<dim3(864, 1, 1), dim3(8, 16, 1), 0, stream>>>(...)
    }
    // ... 更多尺寸配置
}
```

## 4.3 内核实现详解

### 4.3.1 INT2 解码函数

```cuda
template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16) {
    uint *i8s = reinterpret_cast<uint *>(_i8s);
    uint const i2s = *_i2s;

    // LOP3 指令: 三操作数逻辑运算
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;  // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x03030303;       // 提取 2-bit
    static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; 

    #pragma unroll
    for (int i = 0; i < (N / 4); i++) {
        // 使用 LOP3 指令提取 2-bit 值
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(i8s[i])
            : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), 
              "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
        
        // 将 {0,1,2} 映射到 {-2,-1,0}，再用 -1 偏移得到 {-1,0,1}
        i8s[i] = __vsubss4(i8s[i], 0x02020202);
    }
}
```

**解码原理**:

```
原始存储: 16 个 2-bit 值打包成 32-bit
i2s = [w0, w4, w8, w12, w1, w5, w9, w13, w2, w6, w10, w14, w3, w7, w11, w15]

交错存储布局便于使用单条指令提取 4 个值
```

### 4.3.2 主计算循环

```cuda
// 局部变量
int in_thread_C_local[1];
signed char A_local[K_per_loop];
int B_reshape_local[1];
signed char B_decode_local[K_per_loop];
int red_buf0[1];

in_thread_C_local[0] = 0;

// K 维度循环
#pragma unroll
for (int k_0 = 0; k_0 < K/(K_per_loop * K_block_size); ++k_0) {
    // 加载输入激活 (连续 16 个 INT8)
    *(int4*)(A_local + 0) = *(int4*)(A + 
        ((k_0 * K_per_loop * K_block_size) + (threadIdx.x * K_per_loop)));
    
    // 加载权重 (32-bit = 16 个 INT2)
    B_reshape_local[0] = *(int*)(B + 
        (blockIdx.x * N_block_size * K / 4) + 
        (k_0 * K_block_size * K_per_loop * wmma_N / 4) +
        ((threadIdx.x >> 1) * wmma_K * wmma_N / 4) +
        ((threadIdx.y >> 3) * (wmma_K * wmma_N / 2) / 4) + 
        ((threadIdx.x & 1) * (wmma_K * wmma_N / 4) / 4) + 
        ((threadIdx.y & 7) * (wmma_K / 2) / 4));
    
    // 解码 INT2 -> INT8
    decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
    
    // 使用 dp4a 指令计算点积
    #pragma unroll
    for (int k_2_0 = 0; k_2_0 < 4; ++k_2_0) {
        in_thread_C_local[0] = __dp4a(
            *(int *)&A_local[k_2_0 * 4],
            *(int *)&B_decode_local[k_2_0 * 4], 
            in_thread_C_local[0]);
    }
}

// Warp 内规约
red_buf0[0] = in_thread_C_local[0];
#pragma unroll
for (int offset = K_block_size/2; offset > 0; offset /= 2) {
    red_buf0[0] += __shfl_down_sync(__activemask(), red_buf0[0], offset, K_block_size);
}

// 写回结果
int out_idx = (blockIdx.x * N_block_size) + threadIdx.y;
int ws_idx = out_idx / (N / ws_num);
if (threadIdx.x == 0)
    dtype_transform[out_idx] = (__nv_bfloat16)(
        ((float)red_buf0[0]) / (float)s[0] * (float)ws[ws_idx]);
```

### 4.3.3 dp4a 指令

`dp4a` (Dot Product 4 Accumulate) 是 NVIDIA GPU 的重要指令：

```cuda
int __dp4a(int a, int b, int c);
// 等价于:
// c += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
// 其中 a, b 被解释为 4 个 signed int8
```

**性能优势**:
- 单条指令完成 4 次乘法 + 3 次加法
- 专用硬件单元，无需使用 FMA 单元

## 4.4 权重排列优化

### 4.4.1 16×32 分块

```python
# convert_checkpoint.py
# 权重矩阵按 16×32 块划分
# 块内值按特定顺序排列以优化内存访问

block_size = (16, 32)
```

### 4.4.2 INT2 打包布局

```
16 个 2-bit 值打包成 32-bit:
索引: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

这种交错布局使得:
- 每次解码可以高效提取 4 个连续值
- 利用 LOP3 指令的位操作能力
```

## 4.5 模型封装 (model.py)

### 4.5.1 BitLinearKernel 类

```python
class BitLinearKernel(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重存储为 INT8 (每个 byte 存 4 个 INT2)
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features//4, dtype=torch.int8), 
            requires_grad=False)
        # 权重缩放因子 (BF16)
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(4, dtype=torch.bfloat16), 
            requires_grad=False)

    @torch.compile
    def quant_input(self, input):
        # 动态量化输入激活到 INT8
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        return bitnet_int8xint2_linear(input, self.weight, s, self.weight_scale)
```

### 4.5.2 CUDA 调用接口

```python
import ctypes
bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')

def bitnet_int8xint2_linear(input0, input1, s, ws):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]
    
    stream = torch.cuda.current_stream()
    
    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4  # 解压后的 K 维度
    
    ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=input0.device)
    
    bitnet_lib.bitlinear_int8xint2(
        ctypes.c_void_p(input0.data_ptr()), 
        ctypes.c_void_p(input1.data_ptr()), 
        ctypes.c_void_p(ret.data_ptr()), 
        ctypes.c_void_p(s.data_ptr()), 
        ctypes.c_void_p(ws.data_ptr()), 
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), 
        ctypes.c_void_p(stream.cuda_stream))
    
    return ret
```

## 4.6 性能数据

### 4.6.1 内核基准测试 (A100 40GB)

| 矩阵尺寸 (N×K) | W2A8 延迟 (μs) | BF16 延迟 (μs) | 加速比 |
|----------------|---------------|----------------|--------|
| 2560 × 2560 | 13.32 | 18.32 | 1.38x |
| 3840 × 2560 | 14.90 | 18.87 | 1.27x |
| 13824 × 2560 | 18.75 | 59.51 | 3.17x |
| 2560 × 6912 | 14.49 | 37.78 | 2.61x |
| 20480 × 3200 | 30.99 | 112.39 | 3.63x |

### 4.6.2 端到端生成延迟

对比 Gemma-2-2B (BF16, vLLM):

| 输入长度 | 输出长度 | BF16 延迟 (ms) | W2A8 延迟 (ms) | 加速比 |
|---------|---------|---------------|---------------|--------|
| 64 | 16 | 187.64 | 57.40 | 3.27x |
| 64 | 32 | 353.50 | 112.22 | 3.15x |
| 64 | 64 | 683.23 | 221.08 | 3.09x |
| 256 | 16 | 183.14 | 61.24 | 2.99x |

## 4.7 使用指南

### 4.7.1 编译内核

```bash
cd gpu/bitnet_kernels
bash compile.sh
```

### 4.7.2 模型转换

```bash
# 1. 下载模型
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 \
    --local-dir ./checkpoints/bitnet-b1.58-2B-4T-bf16

# 2. 转换 safetensors
python ./convert_safetensors.py \
    --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors \
    --output checkpoints/model_state.pt \
    --model_name 2B

# 3. 权重排列优化
python ./convert_checkpoint.py --input ./checkpoints/model_state.pt
```

### 4.7.3 运行推理

```bash
python3 ./generate.py ./checkpoints/ --interactive --chat_format
```
