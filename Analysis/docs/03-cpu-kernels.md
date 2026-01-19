# 3. CPU 推理内核

## 3.1 内核类型概述

BitNet.cpp 在 CPU 上提供两类内核实现：

| 内核类型 | 文件 | 原理 | 适用场景 |
|---------|------|------|---------|
| **LUT (Lookup Table)** | `ggml-bitnet-lut.cpp` | 查表法 | 高性能推理 |
| **MAD (Multiply-Add)** | `ggml-bitnet-mad.cpp` | 直接计算 | 通用兼容 |

## 3.2 MAD 内核 (ggml-bitnet-mad.cpp)

### 3.2.1 量化函数

```cpp
size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, 
                     int64_t n_per_row, const float * quant_weights)
```

**功能**: 将 FP32 权重量化为 2-bit 整数格式

**量化流程**:

```
1. 找到最大绝对值作为缩放因子
   max = max(|src[i]|) for all i
   scale = max

2. 量化到三值 {-1, 0, +1}
   if |src[i]| < 1e-6:
       q = 1  (代表 0)
   elif src[i] > 0:
       q = 2  (代表 +1)
   else:
       q = 0  (代表 -1)

3. 打包存储 (4个2-bit值存入1个byte)
   每 128 个权重为一组 (QK_I2 = 128)
```

**存储布局**:

```
输入: 128 个 float32 权重
输出: 32 bytes 压缩权重 + 4 bytes scale

┌─────────────────────────────────────────────────────┐
│ byte[0]: w[0:1], w[32:33], w[64:65], w[96:97]      │
│ byte[1]: w[1:2], w[33:34], w[65:66], w[97:98]      │
│ ...                                                 │
│ byte[31]: w[31:32], w[63:64], w[95:96], w[127:128] │
│ scale: float32                                      │
└─────────────────────────────────────────────────────┘
```

### 3.2.2 向量点积函数

```cpp
void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, 
                          const void * vx, size_t bx, 
                          const void * vy, size_t by, int nrc)
```

**功能**: 计算 2-bit 权重和 8-bit 激活的点积

#### AVX2 实现 (x86)

```cpp
// 使用 AVX2 SIMD 指令
__m256i mask = _mm256_set1_epi8(0x03);  // 2-bit 掩码
__m256i accu = _mm256_setzero_si256();

for (int i=0; i < group32_num; i++) {
    __m256i accu32 = _mm256_setzero_si256();
    for (int j=0; j < 32; j++) {
        // 加载 32 bytes 压缩权重
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + ...));
        
        // 解包 4 组 2-bit 值
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);  // 第1组
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);  // 第2组
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);  // 第3组
        // xq8_3 已经是第4组
        
        // 应用掩码
        xq8_0 = _mm256_and_si256(xq8_0, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        
        // 加载对应的 8-bit 激活值
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + ...));
        // ...
        
        // 乘累加
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        // ...
        
        accu32 = _mm256_add_epi16(accu32, ...);
    }
    accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
}

int sumi = hsum_i32_8(accu);  // 水平求和
*s = (float)sumi;
```

#### ARM NEON 实现

```cpp
// 使用 ARM NEON SIMD 指令
int32x4_t accu_0 = vdupq_n_s32(0);
int32x4_t accu_1 = vdupq_n_s32(0);
// ...
const uint8x16_t mask = vdupq_n_u8(3);

for (int i=0; i < group32_num; i++) {
    for (int j=0; j < 32; j++) {
        // 加载压缩权重
        uint8x16_t xq8_6 = vld1q_u8(x + ...);
        uint8x16_t xq8_7 = vld1q_u8(x + ... + 16);
        
        // 解包
        uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
        uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);
        // ...
        
        // 掩码
        int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
        // ...
        
        // 加载激活值
        const int8x16_t yq8_0 = vld1q_s8(y + ...);
        // ...
        
#if defined(__ARM_FEATURE_DOTPROD)
        // 使用 dot product 指令 (ARMv8.2+)
        accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
#else
        // 传统乘累加
        accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
#endif
    }
}

int sumi = vaddlvq_s32(accu_0);
*s = (float)sumi;
```

## 3.3 LUT 内核 (ggml-bitnet-lut.cpp)

### 3.3.1 LUT 原理

Lookup Table (LUT) 方法通过预计算来加速三值权重的矩阵乘法：

```
传统方法: 
  result = Σ(weight[i] × activation[i])
  weight ∈ {-1, 0, +1}

LUT 方法:
  1. 预计算所有可能的部分和
  2. 根据权重组合直接查表
```

对于 2 个连续的三值权重，共有 3² = 9 种组合：

| 组合 | weight[0] | weight[1] | 结果 |
|------|-----------|-----------|------|
| 0 | -1 | -1 | -act[0] - act[1] |
| 1 | -1 | 0 | -act[0] |
| 2 | -1 | +1 | -act[0] + act[1] |
| 3 | 0 | -1 | -act[1] |
| 4 | 0 | 0 | 0 |
| 5 | 0 | +1 | act[1] |
| 6 | +1 | -1 | act[0] - act[1] |
| 7 | +1 | 0 | act[0] |
| 8 | +1 | +1 | act[0] + act[1] |

### 3.3.2 TL1 内核 (ARM)

**文件**: `utils/codegen_tl1.py` 生成

**特点**:
- 针对 ARM NEON 优化
- 使用 `vqtbl1q_s8` 进行快速查表
- 支持批量处理

**LUT 构建函数**:

```cpp
template<int act_k>
inline void lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {
#ifdef __ARM_NEON
    int16x8_t vec_lut[16];
    float32_t scales = *lut_scales;
    
    for (int k = 0; k < act_k / 16; ++k) {
        // 加载并量化激活值
        float32x4x2_t vec_bs_x0 = vld2q_f32(b + k * 16);
        float32x4x2_t vec_bs_x1 = vld2q_f32(b + k * 16 + 8);
        
        float32x4_t vec_f_0 = vmulq_n_f32(vec_bs_x0.val[0], scales);
        // ...
        
        int32x4_t vec_b_0 = vcvtnq_s32_f32(vec_f_0);
        // ...
        
        // 构建 9 种组合的 LUT
        vec_lut[0] = vdupq_n_s16(0) - vec_bs_0 - vec_bs_1;  // (-1,-1)
        vec_lut[1] = vdupq_n_s16(0) - vec_bs_0;              // (-1, 0)
        vec_lut[2] = vdupq_n_s16(0) - vec_bs_0 + vec_bs_1;  // (-1,+1)
        // ...
        
        // 转置并存储
        Transpose_8_8(&vec_lut[0], ..., &vec_lut[7]);
    }
#endif
}
```

**Per-Tensor 量化**:

```cpp
void per_tensor_quant(int k, void* lut_scales_, void* b_) {
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    bitnet_float_type* b = (bitnet_float_type*)b_;
    
#ifdef __ARM_NEON
    float32x4_t temp_max = vdupq_n_f32(0);
    for (int i=0; i < k / 4; i++) {
        float32x4_t vec_bs = vld1q_f32(b + 4 * i);
        float32x4_t abssum = vabsq_f32(vec_bs);
        temp_max = vmaxq_f32(abssum, temp_max);
    }
    float32_t scales = 127 / vmaxvq_f32(temp_max);
    *lut_scales = scales;
#endif
}
```

### 3.3.3 TL2 内核 (x86)

**文件**: `utils/codegen_tl2.py` 生成

**特点**:
- 针对 AVX2/AVX512 优化
- 支持更大的 LUT 尺寸
- 优化的内存访问模式

## 3.4 内核选择逻辑

```cpp
bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, 
                              const struct ggml_tensor * src1, 
                              const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
        
#if defined(GGML_BITNET_ARM_TL1)
        // TL1: 仅支持单 batch
        if (src1->ne[1] <= 1) {
            return true;
        }
#elif defined(GGML_BITNET_X86_TL2)
        // TL2: 支持多 batch
        return true;
#endif
    }
    return false;
}
```

## 3.5 工作空间计算

```cpp
size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, 
                                      const struct ggml_tensor * src1, 
                                      const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];  // 输出维度
    const size_t ne10 = src1->ne[0];  // 输入维度
    const size_t ne11 = src1->ne[1];  // batch size
    
#if defined(GGML_BITNET_ARM_TL1)
    // TL1 工作空间: LUT + 缩放因子
    size_t wsize = ne10 * ne11 * 15 * sizeof(int8_t) + 
                   1 * ne11 * 2 * sizeof(bitnet_float_type);
#elif defined(GGML_BITNET_X86_TL2)
    // TL2 工作空间
    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 
                   2 * ne11 * 2 * sizeof(bitnet_float_type);
#endif
    
    // FP16 转换空间
    if (sizeof(bitnet_float_type) == 2) {
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    
    // 64 字节对齐
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}
```

## 3.6 性能优化技巧

### 3.6.1 内存对齐

```cpp
static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
#endif
}
```

### 3.6.2 分块处理

- 将大矩阵划分为小块 (tiles)
- 利用缓存局部性
- 参数: BM, BK, bm 控制块大小

### 3.6.3 SIMD 利用

| 平台 | SIMD 指令集 | 向量宽度 |
|------|------------|---------|
| ARM | NEON | 128-bit |
| ARM | dot product | 4x 加速 |
| x86 | AVX2 | 256-bit |
| x86 | AVX512 | 512-bit |
