# 7. 构建系统

## 7.1 概述

BitNet.cpp 使用 CMake 作为构建系统，支持多平台编译（Linux、macOS、Windows）。

## 7.2 CMakeLists.txt 分析

### 7.2.1 顶层 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.14)
project("bitnet.cpp" C CXX)
include(CheckIncludeFileCXX)

# 导出编译命令（用于 IDE 支持）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 默认构建类型
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS 
                 "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
```

### 7.2.2 BitNet 选项

```cmake
# BitNet 特定选项
option(BITNET_ARM_TL1    "bitnet.cpp: use tl1 on arm platform"    OFF)
option(BITNET_X86_TL2    "bitnet.cpp: use tl2 on x86 platform"    OFF)

# 传递给 ggml
set(GGML_BITNET_ARM_TL1    ${BITNET_ARM_TL1})
set(GGML_BITNET_X86_TL2    ${BITNET_X86_TL2})

# 编译定义
if (GGML_BITNET_ARM_TL1)
    add_compile_definitions(GGML_BITNET_ARM_TL1)
endif()
if (GGML_BITNET_X86_TL2)
    add_compile_definitions(GGML_BITNET_X86_TL2)
endif()
```

### 7.2.3 子目录和依赖

```cmake
# 编译器兼容性
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fpermissive)
endif()

# 线程库
find_package(Threads REQUIRED)

# 添加子目录
add_subdirectory(src)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "Build llama.cpp server" FORCE)
add_subdirectory(3rdparty/llama.cpp)
```

## 7.3 src/CMakeLists.txt

BitNet 核心库的构建配置：

```cmake
# 源文件列表
set(BITNET_SOURCES
    ggml-bitnet-lut.cpp
    ggml-bitnet-mad.cpp
)

# 创建静态库
add_library(bitnet STATIC ${BITNET_SOURCES})

# 包含目录
target_include_directories(bitnet PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp/include
)

# 链接 ggml
target_link_libraries(bitnet PUBLIC ggml)

# 平台特定设置
if(GGML_BITNET_ARM_TL1)
    target_compile_options(bitnet PRIVATE -march=armv8-a+dotprod)
endif()

if(GGML_BITNET_X86_TL2)
    target_compile_options(bitnet PRIVATE -mavx2 -mfma)
endif()
```

## 7.4 setup_env.py 构建流程

### 7.4.1 主函数

```python
def main():
    setup_gguf()      # 安装 gguf Python 包
    gen_code()        # 生成内核代码
    compile()         # CMake 编译
    prepare_model()   # 准备模型文件
```

### 7.4.2 编译函数

```python
def compile():
    # 检查 CMake
    cmake_exists = subprocess.run(["cmake", "--version"], capture_output=True)
    if cmake_exists.returncode != 0:
        logging.error("Cmake is not available.")
        sys.exit(1)
    
    _, arch = system_info()
    if arch not in COMPILER_EXTRA_ARGS.keys():
        logging.error(f"Arch {arch} is not supported yet")
        exit(0)
    
    # CMake 配置
    logging.info("Compiling the code using CMake.")
    run_command([
        "cmake", "-B", "build",
        *COMPILER_EXTRA_ARGS[arch],           # 平台特定参数
        *OS_EXTRA_ARGS.get(platform.system(), []),  # OS 特定参数
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++"
    ], log_step="generate_build_files")
    
    # CMake 编译
    run_command([
        "cmake", "--build", "build", 
        "--config", "Release"
    ], log_step="compile")
```

### 7.4.3 平台特定参数

```python
# 编译器参数
COMPILER_EXTRA_ARGS = {
    "arm64": ["-DBITNET_ARM_TL1=ON"],
    "x86_64": ["-DBITNET_X86_TL2=ON"]
}

# 操作系统参数
OS_EXTRA_ARGS = {
    "Windows": ["-T", "ClangCL"],
}

# 架构别名
ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}
```

## 7.5 llama.cpp 集成

### 7.5.1 作为子模块

llama.cpp 作为 git submodule 集成在 `3rdparty/llama.cpp/` 目录：

```bash
git clone --recursive https://github.com/microsoft/BitNet.git
```

### 7.5.2 修改点

BitNet.cpp 对 llama.cpp 进行了以下修改：

1. **添加 BitNet 量化类型** (`GGML_TYPE_TL1`, `GGML_TYPE_TL2`, `GGML_TYPE_I2_S`)
2. **矩阵乘法路由** 到 BitNet 内核
3. **模型加载** 支持 BitNet 格式

## 7.6 构建命令参考

### 7.6.1 自动构建（推荐）

```bash
# ARM 平台
python setup_env.py -md models/BitNet-b1.58-2B-4T -q tl1

# x86 平台
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

### 7.6.2 手动构建

```bash
# 1. 生成内核代码（ARM 示例）
python utils/codegen_tl1.py --model bitnet_b1_58-3B \
    --BM "160,320,320" --BK "64,128,64" --bm "32,64,32"

# 2. CMake 配置
cmake -B build \
    -DBITNET_ARM_TL1=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# 3. 编译
cmake --build build --config Release
```

### 7.6.3 Windows 构建

```powershell
# 使用 Visual Studio Developer Command Prompt
cmake -B build -T ClangCL -DBITNET_X86_TL2=ON
cmake --build build --config Release
```

## 7.7 构建输出

### 7.7.1 目录结构

```
build/
├── bin/
│   ├── llama-cli          # 主推理程序
│   ├── llama-quantize     # 量化工具
│   ├── llama-server       # 服务器
│   └── ...                # 其他工具
├── CMakeCache.txt
├── CMakeFiles/
├── src/
│   └── libbitnet.a        # BitNet 静态库
└── 3rdparty/
    └── llama.cpp/
        └── ...
```

### 7.7.2 关键可执行文件

| 文件 | 描述 |
|------|------|
| `llama-cli` | 命令行推理工具 |
| `llama-quantize` | 模型量化工具 |
| `llama-server` | HTTP API 服务器 |
| `llama-bench` | 性能测试工具 |

## 7.8 GPU 构建（独立）

GPU 内核使用独立的构建流程：

### 7.8.1 编译 CUDA 内核

```bash
cd gpu/bitnet_kernels
bash compile.sh
```

### 7.8.2 compile.sh 内容

```bash
#!/bin/bash
nvcc -shared -o libbitnet.so bitnet_kernels.cu \
    -Xcompiler -fPIC \
    -arch=sm_80 \
    -O3
```

### 7.8.3 Python 绑定

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bitnet_kernels',
    ext_modules=[
        CUDAExtension(
            name='bitnet_kernels',
            sources=['bitnet_kernels.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

## 7.9 依赖管理

### 7.9.1 Python 依赖

```
# requirements.txt
numpy
torch
transformers
safetensors
huggingface_hub
```

### 7.9.2 系统依赖

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.9 | Python 运行时 |
| CMake | 3.22 | 构建系统 |
| Clang | 18 | C/C++ 编译器 |
| CUDA | 11.4 | GPU 支持（可选）|

### 7.9.3 安装系统依赖

**Ubuntu/Debian:**
```bash
# Clang 18
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# CMake
sudo apt install cmake
```

**macOS:**
```bash
xcode-select --install
brew install cmake
```

**Windows:**
- 安装 Visual Studio 2022
- 勾选 "C++ Clang Compiler for Windows"
- 勾选 "C++-CMake Tools for Windows"

## 7.10 故障排除

### 7.10.1 常见问题

**Q: CMake 找不到 Clang**
```bash
# 确认 Clang 已安装
clang --version

# 显式指定编译器
cmake -B build \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++
```

**Q: Windows 编译失败**
```powershell
# 使用 Developer Command Prompt
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```

**Q: llama.cpp std::chrono 错误**
参考 [修复 commit](https://github.com/tinglou/llama.cpp/commit/4e3db1e3d78cc1bcd22bcb3af54bd2a4628dd323)

### 7.10.2 日志查看

```bash
# 查看编译日志
cat logs/generate_build_files.log
cat logs/compile.log
cat logs/convert_to_f32_gguf.log
```
