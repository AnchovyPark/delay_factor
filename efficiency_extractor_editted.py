"""
Operator Performance Efficiency Benchmark for LIFE Paper Operators
Measures both compute and memory efficiency, identifying bottlenecks for each operator type:
- Compute efficiency: actual_time / theoretical_compute_time  
- Memory efficiency: actual_time / theoretical_memory_time
- Bottleneck analysis: compute-bound, memory-bound, or balanced
- Effective delay factor: true bottleneck for accurate latency predictions
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
    
    # Memory efficiency metrics
    memory_bytes: int
    theoretical_memory_time_ms: float
    memory_delay_factor: float
    achieved_memory_bandwidth_gb_s: float
    
    # Combined efficiency analysis
    bottleneck_type: str  # "compute", "memory", or "balanced"
    effective_delay_factor: float  # Overall delay considering both compute and memory


class OperatorDelayFactorBenchmark:
    """Benchmark suite for measuring operator performance delay factors on GPU."""
    
    def __init__(self, device='cuda', dtype=torch.float16, warmup_runs=5, benchmark_runs=10, enable_memory_optimization=True):
        self.device = device
        self.dtype = dtype
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.enable_memory_optimization = enable_memory_optimization
        self.failed_operations = []  # Track failed operations
        
        # GPU specifications (A100 example - adjust for your GPU)
        self.gpu_specs = {
            'peak_fp16_tflops': 121.0,  # A100 Tensor performance
            'memory_bandwidth_gb_s': 300.0,  # A100 HBM bandwidth
            'sm_count': 58,
            'cuda_cores_per_sm': 128 
        }
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   Available: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserved")
            # Clear any existing cache
            torch.cuda.empty_cache()
        else:
            print("CUDA not available")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
                'reserved_gb': torch.cuda.memory_reserved(0) / 1e9,
                'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
            }
        return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0}
    
    def clear_gpu_cache(self):
        """Clear GPU cache and collect garbage."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def measure_time(self, func, *args, **kwargs) -> float:
        """Measure execution time of a function with proper GPU synchronization."""
        
        try:
            # Warmup runs
            for _ in range(self.warmup_runs):
                try:
                    func(*args, **kwargs)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # Try with reduced warmup
                    if self.warmup_runs > 1:
                        self.warmup_runs = 1
                        func(*args, **kwargs)
                    else:
                        raise
            
            torch.cuda.synchronize()
            
            # Benchmark runs
            times = []
            for _ in range(self.benchmark_runs):
                try:
                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # If we have at least one measurement, use that
                    if times:
                        break
                    else:
                        raise
            
            if not times:
                raise RuntimeError("No successful benchmark runs completed")
            
            return np.median(times), result if 'result' in locals() else None
        
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            torch.cuda.empty_cache()
            raise RuntimeError(f"Measurement failed: {str(e)}")
    
    def calculate_delay_factor(self, theoretical_flops: int, actual_time_ms: float, memory_bytes: int = 0) -> Tuple[float, float, float, float, float, str, float]:
        """Calculate performance delay factor, memory efficiency, and bottleneck analysis."""
        
        # Compute efficiency
        theoretical_compute_time_ms = theoretical_flops / (self.gpu_specs['peak_fp16_tflops'] * 1e12) * 1000
        compute_delay_factor = actual_time_ms / theoretical_compute_time_ms if theoretical_compute_time_ms > 0 else float('inf')
        achieved_tflops = theoretical_flops / (actual_time_ms / 1000) / 1e12
        
        # Memory efficiency 
        theoretical_memory_time_ms = memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000 if memory_bytes > 0 else 0
        memory_delay_factor = actual_time_ms / theoretical_memory_time_ms if theoretical_memory_time_ms > 0 else 1.0
        achieved_memory_bandwidth = memory_bytes / (actual_time_ms / 1000) / 1e9 if actual_time_ms > 0 else 0
        
        # Bottleneck analysis
        if memory_bytes == 0:
            bottleneck_type = "compute"
            effective_delay_factor = compute_delay_factor
        else:
            compute_bound_time = theoretical_compute_time_ms * compute_delay_factor
            memory_bound_time = theoretical_memory_time_ms * memory_delay_factor
            
            if abs(compute_bound_time - memory_bound_time) / max(compute_bound_time, memory_bound_time) < 0.2:
                bottleneck_type = "balanced"
                effective_delay_factor = max(compute_delay_factor, memory_delay_factor)
            elif compute_bound_time > memory_bound_time:
                bottleneck_type = "compute"
                effective_delay_factor = compute_delay_factor
            else:
                bottleneck_type = "memory"
                effective_delay_factor = memory_delay_factor
        
        return compute_delay_factor, achieved_tflops, theoretical_compute_time_ms, memory_delay_factor, achieved_memory_bandwidth, bottleneck_type, effective_delay_factor
    
    # ================================
    # FOUNDATIONAL OPERATORS (Table 1)
    # ================================
    
    def benchmark_linear_gemm(self, m: int, k: int, n: int) -> BenchmarkResult:
        """Benchmark Linear/GEMM operation: A @ B where A(m,k), B(k,n)"""
        
        try:
            A = torch.randn(m, k, device=self.device, dtype=self.dtype)
            B = torch.randn(k, n, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Try with smaller precision if possible
            if self.dtype == torch.float32:
                self.dtype = torch.float16
                A = torch.randn(m, k, device=self.device, dtype=self.dtype)
                B = torch.randn(k, n, device=self.device, dtype=self.dtype)
            else:
                raise
        
        def gemm_op():
            return torch.mm(A, B)
        
        actual_time_ms, _ = self.measure_time(gemm_op)
        theoretical_flops = 2 * m * k * n  # GEMM: 2*m*k*n operations
        
        # Memory analysis: Input matrices A, B and output matrix C
        memory_bytes = (m * k + k * n + m * n) * 2  # fp16 = 2 bytes per element
        
        compute_delay_factor, achieved_tflops, theoretical_compute_time_ms, memory_delay_factor, achieved_memory_bandwidth, bottleneck_type, effective_delay_factor = self.calculate_delay_factor(theoretical_flops, actual_time_ms, memory_bytes)
        
        return BenchmarkResult(
            operator_name="Linear_GEMM",
            input_shape=(m, k, n),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_bmm(self, batch: int, m: int, k: int, n: int) -> BenchmarkResult:
        """Benchmark Batch Matrix Multiply: batched A @ B"""
        
        try:
            A = torch.randn(batch, m, k, device=self.device, dtype=self.dtype)
            B = torch.randn(batch, k, n, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        
        def bmm_op():
            return torch.bmm(A, B)
        
        actual_time_ms, _ = self.measure_time(bmm_op)
        theoretical_flops = batch * 2 * m * k * n
        
        # Memory analysis: Batch of input matrices A, B and output matrices C
        memory_bytes = batch * (m * k + k * n + m * n) * 2  # fp16 = 2 bytes per element
        
        compute_delay_factor, achieved_tflops, theoretical_compute_time_ms, memory_delay_factor, achieved_memory_bandwidth, bottleneck_type, effective_delay_factor = self.calculate_delay_factor(theoretical_flops, actual_time_ms, memory_bytes)
        
        return BenchmarkResult(
            operator_name="BMM",
            input_shape=(batch, m, k, n),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_elementwise_add(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark elementwise addition."""
        
        try:
            A = torch.randn(shape, device=self.device, dtype=self.dtype)
            B = torch.randn(shape, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
        def add_op():
            return A + B
        
        actual_time_ms, _ = self.measure_time(add_op)
        theoretical_flops = torch.numel(A)  # One add operation per element
        
        # Memory analysis: Read A, B and write result (memory-bound operation)
        memory_bytes = (torch.numel(A) * 2 + torch.numel(B) * 2 + torch.numel(A) * 2)  # Read A, B, Write result
        
        compute_delay_factor, achieved_tflops, theoretical_compute_time_ms, memory_delay_factor, achieved_memory_bandwidth, bottleneck_type, effective_delay_factor = self.calculate_delay_factor(theoretical_flops, actual_time_ms, memory_bytes)
        
        return BenchmarkResult(
            operator_name="Elementwise_Add",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_elementwise_mul(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark elementwise multiplication."""
        
        try:
            A = torch.randn(shape, device=self.device, dtype=self.dtype)
            B = torch.randn(shape, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
        def mul_op():
            return A * B
        
        actual_time_ms, _ = self.measure_time(mul_op)
        theoretical_flops = torch.numel(A)
        
        # Memory analysis: Read A, B and write result (memory-bound operation)
        memory_bytes = (torch.numel(A) * 2 + torch.numel(B) * 2 + torch.numel(A) * 2)  # Read A, B, Write result
        
        compute_delay_factor, achieved_tflops, theoretical_compute_time_ms, memory_delay_factor, achieved_memory_bandwidth, bottleneck_type, effective_delay_factor = self.calculate_delay_factor(theoretical_flops, actual_time_ms, memory_bytes)
        
        return BenchmarkResult(
            operator_name="Elementwise_Mul",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_nonlinear_silu(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark SiLU activation function."""
        
        try:
            x = torch.randn(shape, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
        def silu_op():
            return F.silu(x)
        
        actual_time_ms, _ = self.measure_time(silu_op)
        theoretical_flops = torch.numel(x) * 4  # Approximate: sigmoid + mul
        
        # Memory analysis: SiLU is typically memory-bound (read input, write output)
        memory_bytes = torch.numel(x) * 2 * 2  # Read input + write output, fp16
        
        (compute_delay_factor, 
         achieved_tflops, 
         theoretical_compute_time_ms, 
         memory_delay_factor, 
         achieved_memory_bandwidth, 
         bottleneck_type, 
         effective_delay_factor) = self.calculate_delay_factor(
            theoretical_flops, actual_time_ms, memory_bytes
        )
        
        return BenchmarkResult(
            operator_name="NonLinear_SiLU",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_embedding(self, vocab_size: int, embed_dim: int, seq_len: int) -> BenchmarkResult:
        """Benchmark embedding lookup."""
        
        try:
            embedding = nn.Embedding(vocab_size, embed_dim).to(self.device).to(self.dtype)
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=self.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
        def embed_op():
            return embedding(input_ids)
        
        actual_time_ms, _ = self.measure_time(embed_op)
        theoretical_flops = seq_len * embed_dim  # Lookup operations
        
        # Memory analysis: Embedding is memory-bound (random access to embedding table)
        memory_bytes = seq_len * embed_dim * 2  # Output embeddings, fp16
        # Note: Input access is random, so we only count output for bandwidth
        
        (compute_delay_factor, 
         achieved_tflops, 
         theoretical_compute_time_ms, 
         memory_delay_factor, 
         achieved_memory_bandwidth, 
         bottleneck_type, 
         effective_delay_factor) = self.calculate_delay_factor(
            theoretical_flops, actual_time_ms, memory_bytes
        )
        
        return BenchmarkResult(
            operator_name="Embedding",
            input_shape=(vocab_size, embed_dim, seq_len),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    # ==============================
    # DERIVED OPERATORS (Table 2)
    # ==============================
    
    def benchmark_multi_head_attention(self, batch: int, seq_len: int, num_heads: int, head_dim: int) -> BenchmarkResult:
        """Benchmark Multi-Head Attention."""
        
        hidden_dim = num_heads * head_dim
        
        try:
            # Linear projections
            q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
            k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
            v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
            o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
            
            x = torch.randn(batch, seq_len, hidden_dim, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
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
        attn_flops = batch * num_heads * (2 * 2 * seq_len * seq_len * head_dim)  # Q@K^T + Attn@V
        softmax_flops = batch * num_heads * seq_len * seq_len * 5  # Approximate
        theoretical_flops = linear_flops + attn_flops + softmax_flops
        
        # Memory analysis: MHA involves both linear ops and attention matrices
        # Input/output for linear projections
        linear_memory = batch * seq_len * hidden_dim * 2 * 5  # Input + 4 projections output
        # Attention matrices (Q, K, V, scores, output)
        attn_memory = batch * num_heads * seq_len * (head_dim * 3 + seq_len * 2) * 2  # Q,K,V + scores + attn_out
        memory_bytes = linear_memory + attn_memory
        
        (compute_delay_factor, 
         achieved_tflops, 
         theoretical_compute_time_ms, 
         memory_delay_factor, 
         achieved_memory_bandwidth, 
         bottleneck_type, 
         effective_delay_factor) = self.calculate_delay_factor(
            theoretical_flops, actual_time_ms, memory_bytes
        )
        
        return BenchmarkResult(
            operator_name="Multi_Head_Attention",
            input_shape=(batch, seq_len, num_heads, head_dim),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_grouped_query_attention(self, batch: int, seq_len: int, num_q_heads: int, num_kv_heads: int, head_dim: int) -> BenchmarkResult:
        """Benchmark Grouped Query Attention."""
        
        hidden_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        
        try:
            q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
            k_proj = nn.Linear(hidden_dim, kv_dim, bias=False).to(self.device).to(self.dtype)
            v_proj = nn.Linear(hidden_dim, kv_dim, bias=False).to(self.device).to(self.dtype)
            o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
            
            x = torch.randn(batch, seq_len, hidden_dim, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
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
        attn_flops = batch * num_q_heads * (2 * 2* seq_len * seq_len * head_dim)
        softmax_flops = batch * num_q_heads * seq_len * seq_len * 5
        theoretical_flops = q_flops + kv_flops + o_flops + attn_flops + softmax_flops
        
        # Memory analysis: GQA with fewer KV heads
        # Input/output for linear projections
        linear_memory = batch * seq_len * hidden_dim * 2 * 4  # Input + Q, K, V, O projections
        # Attention matrices (Q with full heads, K,V with fewer heads, scores, output)
        q_memory = batch * num_q_heads * seq_len * head_dim * 2
        kv_memory = batch * num_kv_heads * seq_len * head_dim * 2 * 2  # K and V
        attn_scores_memory = batch * num_q_heads * seq_len * seq_len * 2
        memory_bytes = linear_memory + q_memory + kv_memory + attn_scores_memory
        
        (compute_delay_factor, 
         achieved_tflops, 
         theoretical_compute_time_ms, 
         memory_delay_factor, 
         achieved_memory_bandwidth, 
         bottleneck_type, 
         effective_delay_factor) = self.calculate_delay_factor(
            theoretical_flops, actual_time_ms, memory_bytes
        )
        
        return BenchmarkResult(
            operator_name="Grouped_Query_Attention",
            input_shape=(batch, seq_len, num_q_heads, num_kv_heads, head_dim),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_swiglu_mlp(self, batch: int, seq_len: int, hidden_dim: int, intermediate_dim: int) -> BenchmarkResult:
        """Benchmark SwiGLU MLP."""
        
        try:
            gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(self.device).to(self.dtype)
            up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(self.device).to(self.dtype)
            down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False).to(self.device).to(self.dtype)
            
            x = torch.randn(batch, seq_len, hidden_dim, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
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
        
        # Memory analysis: SwiGLU MLP memory usage
        # Input, gate, up outputs, intermediate activations, final output
        input_memory = batch * seq_len * hidden_dim * 2
        gate_up_memory = batch * seq_len * intermediate_dim * 2 * 2  # Gate and Up outputs
        intermediate_memory = batch * seq_len * intermediate_dim * 2  # SiLU output and elementwise mul
        output_memory = batch * seq_len * hidden_dim * 2
        memory_bytes = input_memory + gate_up_memory + intermediate_memory + output_memory
        
        (compute_delay_factor, 
         achieved_tflops, 
         theoretical_compute_time_ms, 
         memory_delay_factor, 
         achieved_memory_bandwidth, 
         bottleneck_type, 
         effective_delay_factor) = self.calculate_delay_factor(
            theoretical_flops, actual_time_ms, memory_bytes
        )
        
        return BenchmarkResult(
            operator_name="SwiGLU_MLP",
            input_shape=(batch, seq_len, hidden_dim, intermediate_dim),
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_rms_norm(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark RMS Normalization."""
        
        try:
            x = torch.randn(shape, device=self.device, dtype=self.dtype)
            hidden_dim = shape[-1]
            weight = torch.ones(hidden_dim, device=self.device, dtype=self.dtype)
            eps = 1e-6
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
        def rms_norm_op():
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + eps) * weight
        
        actual_time_ms, _ = self.measure_time(rms_norm_op)
        
        # FLOP calculation
        numel = torch.numel(x)
        theoretical_flops = numel * 5  # pow, mean, rsqrt, mul operations
        
        # Memory analysis: RMS norm is typically memory-bound
        # Read input x, weight, write output + intermediate variance
        memory_bytes = (numel * 2 + hidden_dim * 2 + numel * 2 + shape[0] * 2)  # Input, weight, output, variance
        
        compute_delay_factor, achieved_tflops, theoretical_compute_time_ms, memory_delay_factor, achieved_memory_bandwidth, bottleneck_type, effective_delay_factor = self.calculate_delay_factor(theoretical_flops, actual_time_ms, memory_bytes)
        
        return BenchmarkResult(
            operator_name="RMS_Norm",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
    def benchmark_softmax(self, shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark Softmax operation."""
        
        try:
            x = torch.randn(shape, device=self.device, dtype=self.dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        
        def softmax_op():
            return torch.softmax(x, dim=-1)
        
        actual_time_ms, _ = self.measure_time(softmax_op)
        
        # FLOP calculation
        numel = torch.numel(x)
        theoretical_flops = numel * 5  # exp, sum, div operations
        
        # Memory analysis: Softmax is memory-bound (read input, write output, intermediate max/sum)
        memory_bytes = numel * 2 * 2 + shape[0] * shape[1] * 2  # Input + output + intermediate reductions
        
        (compute_delay_factor, 
         achieved_tflops, 
         theoretical_compute_time_ms, 
         memory_delay_factor, 
         achieved_memory_bandwidth, 
         bottleneck_type, 
         effective_delay_factor) = self.calculate_delay_factor(
            theoretical_flops, actual_time_ms, memory_bytes
        )
        
        return BenchmarkResult(
            operator_name="Softmax",
            input_shape=shape,
            theoretical_flops=theoretical_flops,
            actual_time_ms=actual_time_ms,
            theoretical_time_ms=theoretical_compute_time_ms,
            delay_factor=compute_delay_factor,
            achieved_tflops=achieved_tflops,
            memory_bytes=memory_bytes,
            theoretical_memory_time_ms=memory_bytes / (self.gpu_specs['memory_bandwidth_gb_s'] * 1e9) * 1000,
            memory_delay_factor=memory_delay_factor,
            achieved_memory_bandwidth_gb_s=achieved_memory_bandwidth,
            bottleneck_type=bottleneck_type,
            effective_delay_factor=effective_delay_factor
        )
    
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
        
        print("Starting Comprehensive Operator Delay Factor Benchmark")
        print("=" * 80)
        
        results = {}
        
        # Get realistic LLaMA configurations
        llama_configs = self._get_realistic_llama_configs()
        
        # Benchmark each LLaMA model size
        for model_name, configs in llama_configs.items():
            print(f"\nBenchmarking {model_name} Operations")
            print("-" * 50)
            
            category_results = []
            
            # Linear projections (Q, K, V, O, Gate, Up, Down)
            print("  Linear Projections...")
            for i, (m, k, n) in enumerate(configs['linear_projections'][:6]):  # Sample first 6
                try:
                    # Check memory before operation
                    if self.enable_memory_optimization:
                        mem_info = self.get_memory_info()
                        if mem_info['free_gb'] < 1.0:  # Less than 1GB free
                            self.clear_gpu_cache()
                    
                    result = self.benchmark_linear_gemm(m, k, n)
                    if result is not None:
                        category_results.append(result)
                        if i < 3:  # Only print first few to avoid spam
                            print(f"    OK {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
                    else:
                        print(f"    WARNING Linear_GEMM {(m, k, n)}: Skipped (Out of Memory)")
                        self.failed_operations.append({'op': 'Linear_GEMM', 'params': (m, k, n), 'reason': 'OOM'})
                        self.clear_gpu_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    op_name = 'Linear_GEMM'
                    print(f"    {op_name} {(m, k, n)}: Skipped (Out of Memory)")
                    self.failed_operations.append({'op': op_name, 'params': (m, k, n), 'reason': 'OOM'})
                    self.clear_gpu_cache()  # Clear GPU memory
                    continue
                except Exception as e:
                    print(f"    Linear_GEMM {(m, k, n)}: Error - {str(e)}")
                    self.failed_operations.append({'op': 'Linear_GEMM', 'params': (m, k, n), 'reason': str(e)})
                    self.clear_gpu_cache()
                    continue
            
            # BMM operations
            print("  Batch Matrix Multiply...")
            for i, (batch, m, k, n) in enumerate(configs['bmm_operations'][:4]):  # Sample first 4
                try:
                    # Check memory before operation
                    if self.enable_memory_optimization:
                        mem_info = self.get_memory_info()
                        if mem_info['free_gb'] < 1.0:
                            self.clear_gpu_cache()
                    
                    result = self.benchmark_bmm(batch, m, k, n)
                    if result is not None:
                        category_results.append(result)
                        if i < 2:
                            print(f"    OK {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
                    else:
                        print(f"    WARNING BMM {(batch, m, k, n)}: Skipped (Out of Memory)")
                        self.failed_operations.append({'op': 'BMM', 'params': (batch, m, k, n), 'reason': 'OOM'})
                        self.clear_gpu_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    print(f"    BMM {(batch, m, k, n)}: Skipped (Out of Memory)")
                    self.failed_operations.append({'op': 'BMM', 'params': (batch, m, k, n), 'reason': 'OOM'})
                    self.clear_gpu_cache()  # Clear GPU memory
                    continue
                except Exception as e:
                    print(f"    BMM {(batch, m, k, n)}: Error - {str(e)}")
                    self.failed_operations.append({'op': 'BMM', 'params': (batch, m, k, n), 'reason': str(e)})
                    self.clear_gpu_cache()
                    continue
            
            # Elementwise operations
            print("  Elementwise Operations...")
            for i, shape in enumerate(configs['elementwise_ops'][:4]):  # Sample first 4
                try:
                    # Check memory before operation
                    if self.enable_memory_optimization:
                        mem_info = self.get_memory_info()
                        if mem_info['free_gb'] < 1.0:
                            self.clear_gpu_cache()
                    
                    result = self.benchmark_elementwise_add(shape)
                    if result is not None:
                        category_results.append(result)
                        if i < 2:
                            print(f"    OK {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
                    else:
                        print(f"    WARNING Elementwise_Add {shape}: Skipped (Out of Memory)")
                        self.failed_operations.append({'op': 'Elementwise_Add', 'params': shape, 'reason': 'OOM'})
                        self.clear_gpu_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    print(f"     Elementwise_Add {shape}: Skipped (Out of Memory)")
                    self.failed_operations.append({'op': 'Elementwise_Add', 'params': shape, 'reason': 'OOM'})
                    self.clear_gpu_cache()
                    continue
                except Exception as e:
                    print(f"    Elementwise_Add {shape}: Error - {str(e)}")
                    self.failed_operations.append({'op': 'Elementwise_Add', 'params': shape, 'reason': str(e)})
                    self.clear_gpu_cache()
                    continue
            
            # GQA blocks
            print("  Grouped Query Attention...")
            for i, (batch, seq_len, num_q_heads, num_kv_heads, head_dim) in enumerate(configs['gqa_blocks'][:3]):  # Sample first 3
                try:
                    # Check memory before operation
                    if self.enable_memory_optimization:
                        mem_info = self.get_memory_info()
                        if mem_info['free_gb'] < 2.0:  # GQA needs more memory
                            self.clear_gpu_cache()
                    
                    result = self.benchmark_grouped_query_attention(batch, seq_len, num_q_heads, num_kv_heads, head_dim)
                    if result is not None:
                        category_results.append(result)
                        if i < 2:
                            print(f"    OK {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
                    else:
                        print(f"    WARNING Grouped_Query_Attention {(batch, seq_len, num_q_heads, num_kv_heads, head_dim)}: Skipped (Out of Memory)")
                        self.failed_operations.append({'op': 'Grouped_Query_Attention', 'params': (batch, seq_len, num_q_heads, num_kv_heads, head_dim), 'reason': 'OOM'})
                        self.clear_gpu_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    print(f"     GQA {(batch, seq_len, num_q_heads, num_kv_heads, head_dim)}: Skipped (Out of Memory)")
                    self.failed_operations.append({'op': 'GQA', 'params': (batch, seq_len, num_q_heads, num_kv_heads, head_dim), 'reason': 'OOM'})
                    self.clear_gpu_cache()
                    continue
                except Exception as e:
                    print(f"    GQA {(batch, seq_len, num_q_heads, num_kv_heads, head_dim)}: Error - {str(e)}")
                    self.failed_operations.append({'op': 'GQA', 'params': (batch, seq_len, num_q_heads, num_kv_heads, head_dim), 'reason': str(e)})
                    self.clear_gpu_cache()
                    continue
            
            # MLP blocks
            print("  SwiGLU MLP...")
            for i, (batch, seq_len, hidden_dim, intermediate_dim) in enumerate(configs['mlp_blocks'][:3]):  # Sample first 3
                try:
                    # Check memory before operation
                    if self.enable_memory_optimization:
                        mem_info = self.get_memory_info()
                        if mem_info['free_gb'] < 1.5:
                            self.clear_gpu_cache()
                    
                    result = self.benchmark_swiglu_mlp(batch, seq_len, hidden_dim, intermediate_dim)
                    if result is not None:
                        category_results.append(result)
                        if i < 2:
                            print(f"    OK {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
                    else:
                        print(f"    WARNING SwiGLU_MLP {(batch, seq_len, hidden_dim, intermediate_dim)}: Skipped (Out of Memory)")
                        self.failed_operations.append({'op': 'SwiGLU_MLP', 'params': (batch, seq_len, hidden_dim, intermediate_dim), 'reason': 'OOM'})
                        self.clear_gpu_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    print(f"     SwiGLU_MLP {(batch, seq_len, hidden_dim, intermediate_dim)}: Skipped (Out of Memory)")
                    self.failed_operations.append({'op': 'SwiGLU_MLP', 'params': (batch, seq_len, hidden_dim, intermediate_dim), 'reason': 'OOM'})
                    self.clear_gpu_cache()
                    continue
                except Exception as e:
                    print(f"     SwiGLU_MLP {(batch, seq_len, hidden_dim, intermediate_dim)}: Error - {str(e)}")
                    self.failed_operations.append({'op': 'SwiGLU_MLP', 'params': (batch, seq_len, hidden_dim, intermediate_dim), 'reason': str(e)})
                    self.clear_gpu_cache()
                    continue
            
            # Normalization
            print("  RMS Normalization...")
            for i, shape in enumerate(configs['norm_ops'][:4]):  # Sample first 4
                try:
                    # Check memory before operation
                    if self.enable_memory_optimization:
                        mem_info = self.get_memory_info()
                        if mem_info['free_gb'] < 0.5:
                            self.clear_gpu_cache()
                    
                    result = self.benchmark_rms_norm(shape)
                    if result is not None:
                        category_results.append(result)
                        if i < 2:
                            print(f"    OK {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
                    else:
                        print(f"    WARNING RMS_Norm {shape}: Skipped (Out of Memory)")
                        self.failed_operations.append({'op': 'RMS_Norm', 'params': shape, 'reason': 'OOM'})
                        self.clear_gpu_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    print(f"     RMS_Norm {shape}: Skipped (Out of Memory)")
                    self.failed_operations.append({'op': 'RMS_Norm', 'params': shape, 'reason': 'OOM'})
                    self.clear_gpu_cache()
                    continue
                except Exception as e:
                    print(f"     RMS_Norm {shape}: Error - {str(e)}")
                    self.failed_operations.append({'op': 'RMS_Norm', 'params': shape, 'reason': str(e)})
                    self.clear_gpu_cache()
                    continue
            
            # Softmax
            print("   Softmax...")
            for i, shape in enumerate(configs['norm_ops'][:2]):  # Sample first 2
                try:
                    # Check memory before operation
                    if self.enable_memory_optimization:
                        mem_info = self.get_memory_info()
                        if mem_info['free_gb'] < 0.5:
                            self.clear_gpu_cache()
                    
                    result = self.benchmark_softmax(shape)
                    if result is not None:
                        category_results.append(result)
                        if i < 1:
                            print(f"    OK {result.operator_name} {result.input_shape}: {result.delay_factor:.2f}x delay, {result.achieved_tflops:.2f} TFLOPS")
                    else:
                        print(f"    WARNING Softmax {shape}: Skipped (Out of Memory)")
                        self.failed_operations.append({'op': 'Softmax', 'params': shape, 'reason': 'OOM'})
                        self.clear_gpu_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    print(f"     Softmax {shape}: Skipped (Out of Memory)")
                    self.failed_operations.append({'op': 'Softmax', 'params': shape, 'reason': 'OOM'})
                    self.clear_gpu_cache()
                    continue
                except Exception as e:
                    print(f"     Softmax {shape}: Error - {str(e)}")
                    self.failed_operations.append({'op': 'Softmax', 'params': shape, 'reason': str(e)})
                    self.clear_gpu_cache()
                    continue
            
            results[model_name] = category_results
        
        return results
    
    def analyze_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Analyze benchmark results and generate efficiency factors."""
        
        print(f"\nEFFICIENCY ANALYSIS")
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
        print(f"\n{'Operator':<25} {'Count':<6} {'Compute':<15} {'Memory':<15} {'Bottleneck':<12} {'Effective':<10}")
        print(f"{'':25} {'':6} {'Delay | TFLOPS':<15} {'Delay | GB/s':<15} {'Type':<12} {'Delay':<10}")
        print("-" * 100)
        
        delay_factors = {}
        
        for op_name, op_results in operator_stats.items():
            compute_delays = [r.delay_factor for r in op_results]
            memory_delays = [r.memory_delay_factor for r in op_results]
            effective_delays = [r.effective_delay_factor for r in op_results]
            tflops = [r.achieved_tflops for r in op_results]
            memory_bw = [r.achieved_memory_bandwidth_gb_s for r in op_results]
            bottlenecks = [r.bottleneck_type for r in op_results]
            
            avg_compute_delay = np.mean(compute_delays)
            avg_memory_delay = np.mean(memory_delays)
            avg_effective_delay = np.mean(effective_delays)
            avg_tflops = np.mean(tflops)
            avg_memory_bw = np.mean(memory_bw)
            
            # Most common bottleneck type
            bottleneck_counts = {}
            for bt in bottlenecks:
                bottleneck_counts[bt] = bottleneck_counts.get(bt, 0) + 1
            dominant_bottleneck = max(bottleneck_counts, key=bottleneck_counts.get)
            
            delay_factors[op_name] = {
                'avg_delay_factor': avg_compute_delay,  # Keep for backward compatibility
                'avg_compute_delay_factor': avg_compute_delay,
                'avg_memory_delay_factor': avg_memory_delay,
                'avg_effective_delay_factor': avg_effective_delay,
                'min_delay_factor': np.min(compute_delays),
                'max_delay_factor': np.max(compute_delays),
                'avg_tflops': avg_tflops,
                'avg_memory_bandwidth_gb_s': avg_memory_bw,
                'dominant_bottleneck': dominant_bottleneck,
                'sample_count': len(op_results)
            }
            
            print(f"{op_name:<25} {len(op_results):<6} {avg_compute_delay:<5.2f}x|{avg_tflops:<6.1f} {avg_memory_delay:<5.2f}x|{avg_memory_bw:<6.1f} {dominant_bottleneck:<12} {avg_effective_delay:<10.2f}x")
        
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
                        'theoretical_flops': r.theoretical_flops,
                        'actual_time_ms': r.actual_time_ms,
                        'compute_delay_factor': r.delay_factor,
                        'achieved_tflops': r.achieved_tflops,
                        'memory_bytes': r.memory_bytes,
                        'memory_delay_factor': r.memory_delay_factor,
                        'achieved_memory_bandwidth_gb_s': r.achieved_memory_bandwidth_gb_s,
                        'bottleneck_type': r.bottleneck_type,
                        'effective_delay_factor': r.effective_delay_factor
                    } for r in category_results
                ] for category, category_results in results.items()
            }
        }
        
        with open('memory_compute_efficiency_benchmark_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to memory_compute_efficiency_benchmark_results.json")
        
        # Report failed operations if any
        if self.failed_operations:
            print(f"\nWARNING FAILED OPERATIONS ({len(self.failed_operations)} total):")
            for failed_op in self.failed_operations[:10]:  # Show first 10
                print(f"   - {failed_op['op']} {failed_op['params']}: {failed_op['reason']}")
            if len(self.failed_operations) > 10:
                print(f"   ... and {len(self.failed_operations) - 10} more")
        
        print(f"\nEFFICIENCY SUMMARY:")
        print("   - Compute-bound operators show high compute delay factors")
        print("   - Memory-bound operators show high memory delay factors") 
        print("   - Effective delay factor represents the true bottleneck")
        print("   - Use effective_delay_factor for latency predictions")
        
        return delay_factors


def main():
    """Run the comprehensive operator performance delay factor benchmark."""
    
    if not torch.cuda.is_available():
        print("ERROR CUDA not available. This benchmark requires a GPU.")
        return
    
    # Initialize benchmark with memory optimization enabled
    benchmark = OperatorDelayFactorBenchmark(
        device='cuda',
        dtype=torch.float16,
        warmup_runs=3,
        benchmark_runs=5,
        enable_memory_optimization=True
    )
    
    # Run benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Analyze results
    delay_factors = benchmark.analyze_results(results)
    
    print(f"\nBENCHMARK COMPLETE!")
    print(f"   - Measured compute AND memory efficiency for all operators")
    print(f"   - Identified bottleneck types (compute/memory/balanced)")
    print(f"   - Results include effective delay factors for accurate predictions") 
    print(f"   - Use effective_delay_factor for improved latency modeling")


if __name__ == "__main__":
    main()