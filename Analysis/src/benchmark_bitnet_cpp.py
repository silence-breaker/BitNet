#!/usr/bin/env python3
"""
BitNet.cpp CPU 推理性能测试脚本

测试条件与 PyTorch 基准测试对齐:
- 输入 Prompt: "Microsoft is" (3 tokens)
- 生成 Token 数: 可配置
- 测试指标: 延迟、吞吐量、内存占用
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
import platform
import psutil
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"

def get_llama_cli_path():
    """获取 llama-cli 可执行文件路径"""
    if platform.system() == "Windows":
        cli_path = BUILD_DIR / "bin" / "Release" / "llama-cli.exe"
        if not cli_path.exists():
            cli_path = BUILD_DIR / "bin" / "llama-cli.exe"
    else:
        cli_path = BUILD_DIR / "bin" / "llama-cli"
    
    if not cli_path.exists():
        raise FileNotFoundError(f"llama-cli not found at {cli_path}")
    return str(cli_path)

def get_system_info():
    """获取系统信息"""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    # 尝试获取 CPU 型号
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
    except:
        pass
    
    return info

def run_benchmark(model_path, prompt, n_predict, threads, n_runs=5, ctx_size=2048):
    """
    运行基准测试
    
    Args:
        model_path: 模型文件路径
        prompt: 输入 prompt
        n_predict: 生成的 token 数量
        threads: 使用的线程数
        n_runs: 运行次数（取平均）
        ctx_size: 上下文大小
    
    Returns:
        dict: 测试结果
    """
    cli_path = get_llama_cli_path()
    
    results = {
        "prompt": prompt,
        "n_predict": n_predict,
        "threads": threads,
        "n_runs": n_runs,
        "runs": [],
        "summary": {}
    }
    
    print(f"\n{'='*60}")
    print(f"BitNet.cpp 性能测试")
    print(f"{'='*60}")
    print(f"Prompt: \"{prompt}\"")
    print(f"生成 Token 数: {n_predict}")
    print(f"线程数: {threads}")
    print(f"运行次数: {n_runs}")
    print(f"{'='*60}\n")
    
    for run_idx in range(n_runs):
        print(f"[Run {run_idx + 1}/{n_runs}]", end=" ", flush=True)
        
        # 记录内存使用（运行前）
        mem_before = psutil.virtual_memory().used / (1024**3)
        
        # 构建命令
        command = [
            cli_path,
            '-m', model_path,
            '-n', str(n_predict),
            '-t', str(threads),
            '-p', prompt,
            '-ngl', '0',  # CPU only
            '-c', str(ctx_size),
            '--temp', '0.0',  # 确定性输出
            '-b', '1',  # batch size = 1
            '--no-display-prompt',  # 不显示 prompt
        ]
        
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 分钟超时
            )
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            # 记录内存使用（运行后）
            mem_after = psutil.virtual_memory().used / (1024**3)
            
            # 解析 llama.cpp 输出中的性能指标
            stderr_output = result.stderr
            stdout_output = result.stdout
            
            # 提取性能指标
            run_result = {
                "run_idx": run_idx + 1,
                "total_time_s": elapsed,
                "memory_before_gb": round(mem_before, 2),
                "memory_after_gb": round(mem_after, 2),
                "memory_delta_gb": round(mem_after - mem_before, 2),
            }
            
            # 解析 llama.cpp 的统计输出
            # 示例: "llama_perf_sampler_print:    sampling time =      18.55 ms"
            # 示例: "llama_perf_context_print:        load time =    1311.12 ms"
            # 示例: "llama_perf_context_print: prompt eval time =      37.19 ms /     3 tokens"
            # 示例: "llama_perf_context_print:        eval time =    3927.89 ms /   127 runs"
            
            combined_output = stderr_output + stdout_output
            
            # 提取 prompt eval time
            prompt_match = re.search(r'prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens', combined_output)
            if prompt_match:
                prompt_time_ms = float(prompt_match.group(1))
                prompt_tokens = int(prompt_match.group(2))
                run_result["prompt_eval_time_ms"] = prompt_time_ms
                run_result["prompt_tokens"] = prompt_tokens
                run_result["prompt_tokens_per_sec"] = prompt_tokens / (prompt_time_ms / 1000) if prompt_time_ms > 0 else 0
            
            # 提取 eval time (生成阶段)
            eval_match = re.search(r'eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)', combined_output)
            if eval_match:
                eval_time_ms = float(eval_match.group(1))
                eval_tokens = int(eval_match.group(2))
                run_result["eval_time_ms"] = eval_time_ms
                run_result["eval_tokens"] = eval_tokens
                run_result["tokens_per_sec"] = eval_tokens / (eval_time_ms / 1000) if eval_time_ms > 0 else 0
                run_result["ms_per_token"] = eval_time_ms / eval_tokens if eval_tokens > 0 else 0
            
            # 提取 load time
            load_match = re.search(r'load time\s*=\s*([\d.]+)\s*ms', combined_output)
            if load_match:
                run_result["load_time_ms"] = float(load_match.group(1))
            
            # 提取 sampling time
            sampling_match = re.search(r'sampling time\s*=\s*([\d.]+)\s*ms', combined_output)
            if sampling_match:
                run_result["sampling_time_ms"] = float(sampling_match.group(1))
            
            # 保存生成的文本
            run_result["generated_text"] = stdout_output.strip()
            
            results["runs"].append(run_result)
            
            if "tokens_per_sec" in run_result:
                print(f"✓ {run_result['tokens_per_sec']:.2f} tokens/sec, "
                      f"{run_result.get('ms_per_token', 0):.2f} ms/token")
            else:
                print(f"✓ {elapsed:.2f}s")
            
        except subprocess.TimeoutExpired:
            print(f"✗ 超时")
            results["runs"].append({"run_idx": run_idx + 1, "error": "timeout"})
        except Exception as e:
            print(f"✗ 错误: {e}")
            results["runs"].append({"run_idx": run_idx + 1, "error": str(e)})
    
    # 计算汇总统计
    valid_runs = [r for r in results["runs"] if "error" not in r]
    if valid_runs:
        results["summary"] = {
            "valid_runs": len(valid_runs),
            "total_runs": n_runs,
        }
        
        # 平均各项指标
        metrics = ["tokens_per_sec", "ms_per_token", "prompt_tokens_per_sec", 
                   "prompt_eval_time_ms", "eval_time_ms", "load_time_ms", "sampling_time_ms"]
        for metric in metrics:
            values = [r[metric] for r in valid_runs if metric in r]
            if values:
                results["summary"][f"avg_{metric}"] = round(sum(values) / len(values), 2)
                results["summary"][f"min_{metric}"] = round(min(values), 2)
                results["summary"][f"max_{metric}"] = round(max(values), 2)
    
    return results

def run_layer_profiling(model_path, prompt, threads):
    """
    运行逐层分析（使用 llama-cli 的详细日志）
    
    注意: 这需要 llama.cpp 编译时启用 LLAMA_PERF 宏
    """
    # TODO: 实现逐层分析
    # 目前 llama.cpp 标准版本不直接输出逐层耗时
    # 需要修改源码或使用特殊编译选项
    pass

def generate_report(results, system_info, output_path):
    """生成性能报告"""
    
    report = f"""# BitNet.cpp CPU 推理性能测试报告

**测试日期**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 操作系统 | {system_info.get('platform', 'N/A')} {system_info.get('platform_release', '')} |
| CPU | {system_info.get('cpu_model', system_info.get('processor', 'N/A'))} |
| 物理核心数 | {system_info.get('cpu_count_physical', 'N/A')} |
| 逻辑核心数 | {system_info.get('cpu_count_logical', 'N/A')} |
| 总内存 | {system_info.get('memory_total_gb', 'N/A')} GB |
| 架构 | {system_info.get('architecture', 'N/A')} |

## 2. 测试配置

| 参数 | 值 |
|------|-----|
| 输入 Prompt | "{results.get('prompt', 'N/A')}" |
| 输入 Token 数 | {results.get('summary', {}).get('avg_prompt_tokens', 'N/A') if 'summary' in results else 'N/A'} |
| 生成 Token 数 | {results.get('n_predict', 'N/A')} |
| 线程数 | {results.get('threads', 'N/A')} |
| 测试运行次数 | {results.get('n_runs', 'N/A')} |

## 3. 性能结果汇总

### 3.1 核心指标

| 指标 | 平均值 | 最小值 | 最大值 |
|------|--------|--------|--------|
| **生成速度** | {results.get('summary', {}).get('avg_tokens_per_sec', 'N/A')} tokens/sec | {results.get('summary', {}).get('min_tokens_per_sec', 'N/A')} | {results.get('summary', {}).get('max_tokens_per_sec', 'N/A')} |
| **单 Token 延迟** | {results.get('summary', {}).get('avg_ms_per_token', 'N/A')} ms | {results.get('summary', {}).get('min_ms_per_token', 'N/A')} | {results.get('summary', {}).get('max_ms_per_token', 'N/A')} |
| **Prompt 处理速度** | {results.get('summary', {}).get('avg_prompt_tokens_per_sec', 'N/A')} tokens/sec | - | - |

### 3.2 时间分解

| 阶段 | 平均耗时 (ms) |
|------|--------------|
| 模型加载 | {results.get('summary', {}).get('avg_load_time_ms', 'N/A')} |
| Prompt 处理 | {results.get('summary', {}).get('avg_prompt_eval_time_ms', 'N/A')} |
| Token 生成 | {results.get('summary', {}).get('avg_eval_time_ms', 'N/A')} |
| 采样 | {results.get('summary', {}).get('avg_sampling_time_ms', 'N/A')} |

## 4. 各次运行详情

| Run | 生成速度 (tok/s) | 单 Token 延迟 (ms) | Prompt 处理 (ms) | 生成耗时 (ms) |
|-----|------------------|-------------------|------------------|--------------|
"""
    
    for run in results.get("runs", []):
        if "error" not in run:
            report += f"| {run.get('run_idx', 'N/A')} | {run.get('tokens_per_sec', 'N/A'):.2f} | {run.get('ms_per_token', 'N/A'):.2f} | {run.get('prompt_eval_time_ms', 'N/A'):.2f} | {run.get('eval_time_ms', 'N/A'):.2f} |\n"
        else:
            report += f"| {run.get('run_idx', 'N/A')} | Error: {run.get('error', 'unknown')} | - | - | - |\n"
    
    report += f"""
## 5. 与 PyTorch 基准对比

> 注: PyTorch 数据来自《在Pytorch框架下 BitNet b1.58 模型 CPU 推理性能分析报告》
> 测试条件: 相同输入 "{results.get('prompt', 'Microsoft is')}" (3 tokens)

| 指标 | PyTorch (基准) | BitNet.cpp | 加速比 |
|------|---------------|------------|--------|
| 单层耗时 | ~76.85 ms | ~{results.get('summary', {}).get('avg_ms_per_token', 0) / 30:.2f} ms (估算) | **~{76.85 / (results.get('summary', {}).get('avg_ms_per_token', 1) / 30):.0f}x** |
| 生成速度 | ~0.4 tok/s | ~{results.get('summary', {}).get('avg_tokens_per_sec', 'N/A')} tok/s | **~{results.get('summary', {}).get('avg_tokens_per_sec', 0) / 0.4:.0f}x** |
| 30层遍历 | ~2.3 s | ~{results.get('summary', {}).get('avg_ms_per_token', 'N/A')} ms | **~{2300 / results.get('summary', {}).get('avg_ms_per_token', 1):.0f}x** |

## 6. 生成示例

**Prompt**: "{results.get('prompt', 'N/A')}"

**Generated**:
```
{results.get('runs', [{}])[0].get('generated_text', 'N/A') if results.get('runs') else 'N/A'}
```

---
*报告由 benchmark_bitnet_cpp.py 自动生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存至: {output_path}")
    return report

def main():
    parser = argparse.ArgumentParser(description='BitNet.cpp CPU 性能测试')
    parser.add_argument('-m', '--model', type=str, 
                        default=str(PROJECT_ROOT / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"),
                        help='模型文件路径')
    parser.add_argument('-p', '--prompt', type=str, default="Microsoft is",
                        help='输入 prompt (默认: "Microsoft is" 与 PyTorch 基准一致)')
    parser.add_argument('-n', '--n-predict', type=int, default=128,
                        help='生成的 token 数量')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='使用的线程数')
    parser.add_argument('-r', '--runs', type=int, default=5,
                        help='运行次数')
    parser.add_argument('-o', '--output', type=str, 
                        default=str(PROJECT_ROOT / "Analysis" / "docs" / "BitNet.cpp性能测试报告.md"),
                        help='输出报告路径')
    parser.add_argument('--json', type=str,
                        help='输出 JSON 格式的原始数据')
    
    args = parser.parse_args()
    
    # 获取系统信息
    print("获取系统信息...")
    system_info = get_system_info()
    print(f"CPU: {system_info.get('cpu_model', system_info.get('processor', 'Unknown'))}")
    print(f"内存: {system_info.get('memory_total_gb')} GB")
    
    # 运行基准测试
    results = run_benchmark(
        model_path=args.model,
        prompt=args.prompt,
        n_predict=args.n_predict,
        threads=args.threads,
        n_runs=args.runs
    )
    
    # 输出 JSON 数据
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({
                "system_info": system_info,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"JSON 数据已保存至: {args.json}")
    
    # 生成报告
    generate_report(results, system_info, args.output)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")
    if results.get("summary"):
        print(f"平均生成速度: {results['summary'].get('avg_tokens_per_sec', 'N/A')} tokens/sec")
        print(f"平均单 Token 延迟: {results['summary'].get('avg_ms_per_token', 'N/A')} ms")

if __name__ == "__main__":
    main()
