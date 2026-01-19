# BitNet 仓库代码分析文档

本文档详细分析了 Microsoft BitNet (bitnet.cpp) 仓库的代码结构、核心功能和实现原理。

## 目录

1. [项目概述](./01-overview.md)
2. [核心架构](./02-architecture.md)
3. [CPU推理内核](./03-cpu-kernels.md)
4. [GPU推理内核](./04-gpu-kernels.md)
5. [模型转换工具](./05-model-conversion.md)
6. [代码生成系统](./06-codegen.md)
7. [构建系统](./07-build-system.md)
8. [BitNet.cpp 优化后 CPU 推理性能分析报告](./BitNet.cpp优化后CPU推理性能分析报告.md)

---

## 快速导航

| 文档 | 描述 |
|------|------|
| [项目概述](./01-overview.md) | BitNet b1.58 的背景、目标和性能指标 |
| [核心架构](./02-architecture.md) | 整体代码组织和模块划分 |
| [CPU推理内核](./03-cpu-kernels.md) | LUT和MAD内核的详细实现 |
| [GPU推理内核](./04-gpu-kernels.md) | CUDA W2A8 GEMV内核分析 |
| [模型转换工具](./05-model-conversion.md) | HuggingFace到GGUF的转换流程 |
| [代码生成系统](./06-codegen.md) | TL1/TL2内核代码生成器 |
| [构建系统](./07-build-system.md) | CMake配置和编译流程 |
| [BitNet.cpp 优化后 CPU 推理性能分析报告](./BitNet.cpp优化后CPU推理性能分析报告.md) | 详细的性能测试数据和分析结论 |
