"""
Operator Performance Delay Factor Benchmark for LIFE Paper Operators
Measures performance delay factors (actual_time / theoretical_time) for each operator type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class BenchmarkResult:
    """Results from benchmarking an operator."""
    operator_name: str
    input_shape: Tuple[int, ...]
    theoretical_flops: int
    actual_time_ms: float
    theoretical_time_ms: float
    delay_factor: float
    achieved_tflops: float


class OperatorDelayFactorBenchmark:
    """Benchmark suite for measuring operator performance delay factors on GPU."""
    
    def __init__(self, device='cuda', dtype=torch.float16, warmup_runs=5, benchmark_runs=10):
        self.device = device
        self.dtype = dtype
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        # GPU specifications (A100 example - adjust for your GPU)
        self.gpu_specs = {
            'peak_fp16_tflops': 312.0,  # A100 Tensor performance
            'memory_bandwidth_gb_s': 1600.0,  # A100 HBM bandwidth
            'sm_count': 108,
            'cuda_cores_per_sm': 64
        }
        
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("❌ CUDA not available")
    
    def measure_time(self, func, *args, **kwargs) -> float:
        """Measure execution time of a function with proper GPU synchronization."""
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)
        
        torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        return np.median(times), result
    
    def calculate_delay_factor(self, theoretical_flops: int, actual_time_ms: float) -> Tuple[float, float, float]:
        """Calculate performance delay factor and achieved TFLOPS."""
        
        theoretical_time_ms = theoretical_flops / (self.gpu_specs['peak_fp16_tflops'] * 1e12) * 1000
        delay_factor = actual_time_ms / theoretical_time_ms if theoretical_time_ms > 0 else float('inf')
        achieved_tflops = theoretical_flops / (actual_time_ms / 1000) / 1e12
        
        return delay_factor, achieved_tflops, theoretical_time_ms
    
    # ================================
    # FOUNDATIONAL OPERATORS (Table 1)
    # ================================
    
    def benchmark_linear_gemm(self, m: int, k: int, n: int) -> BenchmarkResult:
        """Benchmark Linear/GEMM operation: A @ B where A(m,k), B(k,n)"""
        
        A = torch.randn(m, k, device=self.device, dtype=self.dtype)
        B = torch.randn(k, n, device=self.device, dtype=self.dtype)
        
        def gemm_op():
            return torch.mm(A, B)
        
        actual_time_ms, _ = self.measure_time(gemm_op)
        theoretical_flops = 2 * m * k * n  # GEMM: 2*m*k*n operations
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="Linear_GEMM",
            input_shape=(m, k, n),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_bmm(self, batch: int, m: int, k: int, n: int) -> BenchmarkResult:
        """Benchmark Batch Matrix Multiply: batched A @ B"""
        
        A = torch.randn(batch, m, k, device=self.device, dtype=self.dtype)
        B = torch.randn(batch, k, n, device=self.device, dtype=self.dtype)
        
        def bmm_op():
            return torch.bmm(A, B)
        
        actual_time_ms, _ = self.measure_time(bmm_op)
        theoretical_flops = batch * 2 * m * k * n
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="BMM",
            input_shape=(batch, m, k, n),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_elementwise_add(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark elementwise addition."""
        
        A = torch.randn(shape, device=self.device, dtype=self.dtype)
        B = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        def add_op():
            return A + B
        
        actual_time_ms, _ = self.measure_time(add_op)
        theoretical_flops = torch.numel(A)  # One add operation per element
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="Elementwise_Add",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_elementwise_mul(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark elementwise multiplication."""
        
        A = torch.randn(shape, device=self.device, dtype=self.dtype)
        B = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        def mul_op():
            return A * B
        
        actual_time_ms, _ = self.measure_time(mul_op)
        theoretical_flops = torch.numel(A)
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="Elementwise_Mul",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_nonlinear_silu(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark SiLU activation function."""
        
        x = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        def silu_op():
            return F.silu(x)
        
        actual_time_ms, _ = self.measure_time(silu_op)
        theoretical_flops = torch.numel(x) * 4  # Approximate: sigmoid + mul
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="NonLinear_SiLU",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_embedding(self, vocab_size: int, embed_dim: int, seq_len: int) -> BenchmarkResult:
        """Benchmark embedding lookup."""
        
        embedding = nn.Embedding(vocab_size, embed_dim).to(self.device).to(self.dtype)
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=self.device)
        
        def embed_op():
            return embedding(input_ids)
        
        actual_time_ms, _ = self.measure_time(embed_op)
        theoretical_flops = seq_len * embed_dim  # Lookup operations
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="Embedding",
            input_shape=(vocab_size, embed_dim, seq_len),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    # ==============================
    # DERIVED OPERATORS (Table 2)
    # ==============================
    
    def benchmark_multi_head_attention(self, batch: int, seq_len: int, num_heads: int, head_dim: int) -> BenchmarkResult:
        """Benchmark Multi-Head Attention."""
        
        hidden_dim = num_heads * head_dim
        
        # Linear projections
        q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
        k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
        v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
        o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
        
        x = torch.randn(batch, seq_len, hidden_dim, device=self.device, dtype=self.dtype)
        
        def mha_op():
            # Q, K, V projections
            q = q_proj(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k_proj(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v_proj(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            
            # Output projection
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
            return o_proj(out)
        
        actual_time_ms, _ = self.measure_time(mha_op)
        
        # FLOP calculation
        linear_flops = 4 * batch * seq_len * hidden_dim * hidden_dim * 2  # Q,K,V,O projections
        attn_flops = batch * num_heads * (2 * seq_len * seq_len * head_dim)  # Q@K^T + Attn@V
        softmax_flops = batch * num_heads * seq_len * seq_len * 5  # Approximate
        theoretical_flops = linear_flops + attn_flops + softmax_flops
        
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="Multi_Head_Attention",
            input_shape=(batch, seq_len, num_heads, head_dim),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_grouped_query_attention(self, batch: int, seq_len: int, num_q_heads: int, num_kv_heads: int, head_dim: int) -> BenchmarkResult:
        """Benchmark Grouped Query Attention."""
        
        hidden_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        
        q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
        k_proj = nn.Linear(hidden_dim, kv_dim, bias=False).to(self.device).to(self.dtype)
        v_proj = nn.Linear(hidden_dim, kv_dim, bias=False).to(self.device).to(self.dtype)
        o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
        
        x = torch.randn(batch, seq_len, hidden_dim, device=self.device, dtype=self.dtype)
        
        def gqa_op():
            q = q_proj(x).view(batch, seq_len, num_q_heads, head_dim).transpose(1, 2)
            k = k_proj(x).view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v_proj(x).view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # Repeat K,V for grouped attention
            group_size = num_q_heads // num_kv_heads
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
            return o_proj(out)
        
        actual_time_ms, _ = self.measure_time(gqa_op)
        
        # FLOP calculation (similar to MHA but with different K,V dimensions)
        q_flops = batch * seq_len * hidden_dim * hidden_dim * 2
        kv_flops = 2 * batch * seq_len * hidden_dim * kv_dim * 2
        o_flops = batch * seq_len * hidden_dim * hidden_dim * 2
        attn_flops = batch * num_q_heads * (2 * seq_len * seq_len * head_dim)
        softmax_flops = batch * num_q_heads * seq_len * seq_len * 5
        theoretical_flops = q_flops + kv_flops + o_flops + attn_flops + softmax_flops
        
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="Grouped_Query_Attention",
            input_shape=(batch, seq_len, num_q_heads, num_kv_heads, head_dim),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_swiglu_mlp(self, batch: int, seq_len: int, hidden_dim: int, intermediate_dim: int) -> BenchmarkResult:
        """Benchmark SwiGLU MLP."""
        
        gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(self.device).to(self.dtype)
        up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(self.device).to(self.dtype)
        down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
        
        x = torch.randn(batch, seq_len, hidden_dim, device=self.device, dtype=self.dtype)
        
        def swiglu_mlp_op():
            gate = gate_proj(x)
            up = up_proj(x)
            return down_proj(F.silu(gate) * up)
        
        actual_time_ms, _ = self.measure_time(swiglu_mlp_op)
        
        # FLOP calculation
        gate_flops = batch * seq_len * hidden_dim * intermediate_dim * 2
        up_flops = batch * seq_len * hidden_dim * intermediate_dim * 2
        down_flops = batch * seq_len * intermediate_dim * hidden_dim * 2
        silu_flops = batch * seq_len * intermediate_dim * 4
        mul_flops = batch * seq_len * intermediate_dim
        theoretical_flops = gate_flops + up_flops + down_flops + silu_flops + mul_flops
        
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="SwiGLU_MLP",
            input_shape=(batch, seq_len, hidden_dim, intermediate_dim),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_rms_norm(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark RMS Normalization."""
        
        x = torch.randn(shape, device=self.device, dtype=self.dtype)
        hidden_dim = shape[-1]
        weight = torch.ones(hidden_dim, device=self.device, dtype=self.dtype)
        eps = 1e-6
        
        def rms_norm_op():
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + eps) * weight
        
        actual_time_ms, _ = self.measure_time(rms_norm_op)
        
        # FLOP calculation
        numel = torch.numel(x)
        theoretical_flops = numel * 5  # pow, mean, rsqrt, mul operations
        
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="RMS_Norm",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def benchmark_softmax(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark Softmax operation."""
        
        x = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        def softmax_op():
            return torch.softmax(x, dim=-1)
        
        actual_time_ms, _ = self.measure_time(softmax_op)
        
        # FLOP calculation
        numel = torch.numel(x)
        theoretical_flops = numel * 5  # exp, sum, div operations
        
        delay_factor, achieved_tflops, theoretical_time_ms = self.calculate_delay_factor(theoretical_flops, actual_time_ms)
        
        return BenchmarkResult(
            operator_name="Softmax",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_time_ms,
            delay_factor=delay_factor,
            achieved_tflops=achieved_tflops
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across all operators and sizes."""
        
        print("🚀 Starting Comprehensive Operator Efficiency Benchmark")
        print("=" * 80)
        
        results = {}
        
        # Real LLaMA model configurations
        llama_configs = self._get_realistic_llama_configs()
        
        return llama_configs

    def _get_realistic_llama_configs(self):
        """Get realistic test configurations based on actual LLaMA model architectures."""
        
        # Real LLaMA model specifications
        llama_models = {
            'LLaMA_3.2_1B': {
                'hidden_size': 2048,
                'intermediate_size': 5632,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'head_dim': 64,  # 2048 / 32
                'layers': 16,
                'vocab_size': 128256
            },
            'LLaMA_3_8B': {
                'hidden_size': 4096,
                'intermediate_size': 14336,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'head_dim': 128,  # 4096 / 32
                'layers': 32,
                'vocab_size': 128256
            },
            'LLaMA_3_70B': {
                'hidden_size': 8192,
                'intermediate_size': 28672,
                'num_attention_heads': 64,
                'num_key_value_heads': 8,
                'head_dim': 128,  # 8192 / 64
                'layers': 80,
                'vocab_size': 128256
            }
        }
        
        # Common sequence lengths used in practice
        sequence_lengths = [512, 1024, 2048, 4096, 8192]
        batch_sizes = [1, 4, 8]  # Common inference batch sizes
        
        configs = {}
        
        for model_name, specs in llama_models.items():
            model_configs = {
                'linear_projections': [],
                'bmm_operations': [],
                'elementwise_ops': [],
                'attention_blocks': [],
                'gqa_blocks': [],
                'mlp_blocks': [],
                'norm_ops': []
            }
            
            hidden = specs['hidden_size']
            intermediate = specs['intermediate_size']
            num_q_heads = specs['num_attention_heads']
            num_kv_heads = specs['num_key_value_heads']
            head_dim = specs['head_dim']
            
            # Test different sequence lengths and batch sizes
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    # Linear projections (most common in LLaMA)
                    # Q, K, V projections
                    model_configs['linear_projections'].extend([
                        (batch_size * seq_len, hidden, hidden),  # Q projection
                        (batch_size * seq_len, hidden, num_kv_heads * head_dim),  # K projection  
                        (batch_size * seq_len, hidden, num_kv_heads * head_dim),  # V projection
                        (batch_size * seq_len, hidden, hidden),  # O projection
                    ])
                    
                    # MLP projections
                    model_configs['linear_projections'].extend([
                        (batch_size * seq_len, hidden, intermediate),  # Gate projection
                        (batch_size * seq_len, hidden, intermediate),  # Up projection
                        (batch_size * seq_len, intermediate, hidden),  # Down projection
                    ])
                    
                    # BMM operations in attention
                    model_configs['bmm_operations'].extend([
                        (batch_size * num_q_heads, seq_len, head_dim, seq_len),  # Q @ K^T
                        (batch_size * num_q_heads, seq_len, seq_len, head_dim),  # Attn @ V
                    ])
                    
                    # Elementwise operations
                    model_configs['elementwise_ops'].extend([
                        (batch_size, seq_len, hidden),  # Residual connections
                        (batch_size, seq_len, intermediate),  # SwiGLU multiplication
                    ])
                    
                    # Complete attention blocks
                    model_configs['attention_blocks'].append(
                        (batch_size, seq_len, num_q_heads, head_dim)
                    )
                    
                    # GQA blocks
                    model_configs['gqa_blocks'].append(
                        (batch_size, seq_len, num_q_heads, num_kv_heads, head_dim)
                    )
                    
                    # MLP blocks
                    model_configs['mlp_blocks'].append(
                        (batch_size, seq_len, hidden, intermediate)
                    )
                    
                    # Normalization operations
                    model_configs['norm_ops'].extend([
                        (batch_size, seq_len, hidden),  # Pre-attention norm
                        (batch_size, seq_len, hidden),  # Pre-MLP norm
                    ])
            
            configs[model_name] = model_configs
        
        return configs

    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across all operators and realistic sizes."""
        
        print("🚀 Starting Comprehensive Operator Delay Factor Benchmark")
        print("=" * 80)
        
        results = {}
        
        # Get realistic LLaMA configurations
        llama_configs = self._get_realistic_llama_configs()
        
        # Benchmark each LLaMA model size
        for model_name, configs in llama_configs.items():
            print(f"\n📊 Benchmarking {model_name} Operations")
            print("-" * 50)
            
            category_results = []
            
            # Linear projections (Q, K, V, O, Gate, Up, Down)
            print("  🔧 Linear Projections...")
            for i, (m, k, n) in enumerate(configs['linear_projections'][:6]):  # Sample first 6
                result = self.benchmark_linear_gemm(m, k, n)
                category_results.append(result)
                if i < 3:  # Only print first few to avoid spam
                    print(f"    ✓ {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
            
            # BMM operations
            print("  🔧 Batch Matrix Multiply...")
            for i, (batch, m, k, n) in enumerate(configs['bmm_operations'][:4]):  # Sample first 4
                result = self.benchmark_bmm(batch, m, k, n)
                category_results.append(result)
                if i < 2:
                    print(f"    ✓ {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
            
            # Elementwise operations
            print("  🔧 Elementwise Operations...")
            for i, shape in enumerate(configs['elementwise_ops'][:4]):  # Sample first 4
                result = self.benchmark_elementwise_add(shape)
                category_results.append(result)
                if i < 2:
                    print(f"    ✓ {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
            
            # GQA blocks
            print("  🔧 Grouped Query Attention...")
            for i, (batch, seq_len, num_q_heads, num_kv_heads, head_dim) in enumerate(configs['gqa_blocks'][:3]):  # Sample first 3
                result = self.benchmark_grouped_query_attention(batch, seq_len, num_q_heads, num_kv_heads, head_dim)
                category_results.append(result)
                if i < 2:
                    print(f"    ✓ {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
            
            # MLP blocks
            print("  🔧 SwiGLU MLP...")
            for i, (batch, seq_len, hidden_dim, intermediate_dim) in enumerate(configs['mlp_blocks'][:3]):  # Sample first 3
                result = self.benchmark_swiglu_mlp(batch, seq_len, hidden_dim, intermediate_dim)
                category_results.append(result)
                if i < 2:
                    print(f"    ✓ {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
            
            # Normalization
            print("  🔧 RMS Normalization...")
            for i, shape in enumerate(configs['norm_ops'][:4]):  # Sample first 4
                result = self.benchmark_rms_norm(shape)
                category_results.append(result)
                if i < 2:
                    print(f"    ✓ {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
            
            # Softmax
            print("  🔧 Softmax...")
            for i, shape in enumerate(configs['norm_ops'][:2]):  # Sample first 2
                result = self.benchmark_softmax(shape)
                category_results.append(result)
                if i < 1:
                    print(f"    ✓ {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
            
            results[model_name] = category_results
        
        return results
    
    def analyze_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Analyze benchmark results and generate efficiency factors."""
        
        print(f"\n📈 EFFICIENCY ANALYSIS")
        print("=" * 80)
        
        # Group results by operator type
        operator_stats = {}
        
        for category, category_results in results.items():
            for result in category_results:
                op_name = result.operator_name
                if op_name not in operator_stats:
                    operator_stats[op_name] = []
                operator_stats[op_name].append(result)
        
        # Calculate statistics for each operator
        print(f"\n{'Operator':<25} {'Count':<8} {'Avg Delay':<10} {'Min Delay':<10} {'Max Delay':<10} {'Avg TFLOPS':<12}")
        print("-" * 90)
        
        delay_factors = {}
        
        for op_name, op_results in operator_stats.items():
            delays = [r.delay_factor for r in op_results]
            tflops = [r.achieved_tflops for r in op_results]
            
            avg_delay = np.mean(delays)
            min_delay = np.min(delays)
            max_delay = np.max(delays)
            avg_tflops = np.mean(tflops)
            
            delay_factors[op_name] = {
                'avg_delay_factor': avg_delay,
                'min_delay_factor': min_delay,
                'max_delay_factor': max_delay,
                'avg_tflops': avg_tflops,
                'sample_count': len(op_results)
            }
            
            print(f"{op_name:<25} {len(op_results):<8} {avg_delay:<10.2f}x {min_delay:<10.2f}x {max_delay:<10.2f}x {avg_tflops:<12.2f}")
        
        # Save results
        output_data = {
            'gpu_specs': self.gpu_specs,
            'benchmark_config': {
                'dtype': str(self.dtype),
                'warmup_runs': self.warmup_runs,
                'benchmark_runs': self.benchmark_runs
            },
            'delay_factors': delay_factors,
            'detailed_results': {
                category: [
                    {
                        'operator_name': r.operator_name,
                        'input_shape': r.input_shape,
                        'delay_factor': r.delay_factor,
                        'achieved_tflops': r.achieved_tflops,
                        'actual_time_ms': r.actual_time_ms
                    } for r in category_results
                ] for category, category_results in results.items()
            }
        }
        
        with open('delay_factor_benchmark_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to delay_factor_benchmark_results.json")
        
        return delay_factors


def main():
    """Run the comprehensive operator performance delay factor benchmark."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires a GPU.")
        return
    
    # Initialize benchmark
    benchmark = OperatorDelayFactorBenchmark(
        device='cuda',
        dtype=torch.float16,
        warmup_runs=3,
        benchmark_runs=5
    )
    
    # Run benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Analyze results
    delay_factors = benchmark.analyze_results(results)
    
    print(f"\n🎯 BENCHMARK COMPLETE!")
    print(f"   • Measured performance delay factors for all LIFE operators")
    print(f"   • Results saved for integration into performance models")
    print(f"   • Use these factors to improve latency predictions")


if __name__ == "__main__":
    main()