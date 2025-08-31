"""
Delay Factor Extractor

This module benchmarks actual GPU performance to extract delay factors
for different FLOPS and memory operations, then saves to JSON format.
"""

import torch
import time
import json
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DelayFactorExtractor:
    """Extract delay factors by benchmarking actual GPU performance."""
    
    def __init__(self, warmup_iterations: int = 10, measure_iterations: int = 50):
        """
        Initialize the extractor.
        
        Args:
            warmup_iterations: Number of warmup iterations before measurement
            measure_iterations: Number of iterations to average for measurement
        """
        self.warmup_iterations = warmup_iterations
        self.measure_iterations = measure_iterations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU benchmarking requires CUDA.")
        
        # Get GPU information
        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_properties = torch.cuda.get_device_properties(0)
        
        # Get theoretical GPU specifications
        self.theoretical_specs = self._get_theoretical_specs()
        
        print(f"Initialized DelayFactorExtractor for GPU: {self.gpu_name}")
        print(f"Theoretical FP16 TFLOPS: {self.theoretical_specs['fp16_tflops']:.2f}")
        print(f"Theoretical Memory Bandwidth: {self.theoretical_specs['memory_bandwidth_gb_s']:.2f} GB/s")
    
    def _get_theoretical_specs(self) -> Dict[str, float]:
        """Get theoretical GPU specifications based on GPU name."""
        # Known GPU specifications (you can expand this)
        gpu_specs = {
            'Tesla T4': {'fp16_tflops': 65.13, 'memory_bandwidth_gb_s': 300.0},
            'L4': {'fp16_tflops': 121.0, 'memory_bandwidth_gb_s': 300.0},
            'A10G': {'fp16_tflops': 125.0, 'memory_bandwidth_gb_s': 600.0},
            'L40': {'fp16_tflops': 181.05, 'memory_bandwidth_gb_s': 864.0},
            'L40S': {'fp16_tflops': 362.07, 'memory_bandwidth_gb_s': 864.0},
            'A100-SXM4-40GB': {'fp16_tflops': 77.97, 'memory_bandwidth_gb_s': 1555.0},
            'A100-SXM4-80GB': {'fp16_tflops': 77.97, 'memory_bandwidth_gb_s': 1935.0},
            'H100-PCIe': {'fp16_tflops': 204.9, 'memory_bandwidth_gb_s': 2000.0},
            'H100-SXM': {'fp16_tflops': 267.6, 'memory_bandwidth_gb_s': 3350.0},
        }
        
        # Try to match GPU name with known specs
        for known_gpu, specs in gpu_specs.items():
            if known_gpu.lower() in self.gpu_name.lower():
                return specs
        
        # Fallback: estimate from GPU properties
        print(f"Warning: Unknown GPU '{self.gpu_name}'. Using estimated specifications.")
        
        # Rough estimation based on memory and compute capability
        memory_gb = self.gpu_properties.total_memory / (1024**3)
        major, minor = self.gpu_properties.major, self.gpu_properties.minor
        
        # Very rough estimates - these should be replaced with actual specs
        estimated_tflops = self.gpu_properties.multi_processor_count * 2.0  # rough estimate
        estimated_bandwidth = memory_gb * 50  # rough estimate
        
        return {
            'fp16_tflops': estimated_tflops,
            'memory_bandwidth_gb_s': estimated_bandwidth
        }
    
    def benchmark_flops(self, flops_targets: List[float]) -> Dict[float, float]:
        """
        Benchmark FLOPS performance and calculate delay factors.
        
        Args:
            flops_targets: List of FLOPS values to benchmark
            
        Returns:
            Dictionary mapping FLOPS to delay factors
        """
        print("\\n" + "="*50)
        print("Benchmarking FLOPS Performance")
        print("="*50)
        
        flops_delay_factors = {}
        
        for target_flops in flops_targets:
            print(f"\\nBenchmarking {target_flops:.0e} FLOPs...")
            
            # Calculate matrix dimensions for target FLOPS
            # For A(M,K) @ B(K,N): FLOPs = 2*M*K*N
            # We'll use square matrices for simplicity: M=N=K=cube_root(FLOPs/2)
            dim = max(1, int((target_flops / 2) ** (1/3)))
            actual_flops = 2 * dim * dim * dim
            
            print(f"  Matrix dimensions: {dim}x{dim} @ {dim}x{dim}")
            print(f"  Actual FLOPs: {actual_flops:.0e}")
            
            try:
                # Create random matrices
                A = torch.randn(dim, dim, dtype=torch.float16, device=self.device)
                B = torch.randn(dim, dim, dtype=torch.float16, device=self.device)
                
                # Warmup
                for _ in range(self.warmup_iterations):
                    _ = torch.matmul(A, B)
                torch.cuda.synchronize()
                
                # Measure
                times = []
                for _ in range(self.measure_iterations):
                    start_time = time.perf_counter()
                    result = torch.matmul(A, B)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Calculate actual TFLOPS
                actual_tflops = (actual_flops / avg_time) / 1e12
                
                # Calculate delay factor
                theoretical_tflops = self.theoretical_specs['fp16_tflops']
                delay_factor = theoretical_tflops / actual_tflops
                
                print(f"  Average time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
                print(f"  Actual TFLOPS: {actual_tflops:.2f}")
                print(f"  Theoretical TFLOPS: {theoretical_tflops:.2f}")
                print(f"  Delay factor: {delay_factor:.2f}")
                
                flops_delay_factors[actual_flops] = delay_factor
                
                # Clean up
                del A, B, result
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"  ❌ GPU 메모리 부족으로 건너뜀 (행렬 크기: {dim}x{dim})")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  ❌ 측정 실패: {e}")
                torch.cuda.empty_cache()
                continue
        
        return flops_delay_factors
    
    def benchmark_memory(self, bytes_targets: List[float]) -> Dict[float, float]:
        """
        Benchmark memory performance and calculate delay factors.
        
        Args:
            bytes_targets: List of byte counts to benchmark
            
        Returns:
            Dictionary mapping bytes to delay factors
        """
        print("\\n" + "="*50)
        print("Benchmarking Memory Performance")
        print("="*50)
        
        memory_delay_factors = {}
        bytes_per_element = 2  # FP16
        
        for target_bytes in bytes_targets:
            print(f"\\nBenchmarking {target_bytes:.0e} bytes...")
            
            # Calculate tensor size
            num_elements = int(target_bytes / bytes_per_element)
            actual_bytes = num_elements * bytes_per_element
            
            print(f"  Tensor elements: {num_elements}")
            print(f"  Actual bytes: {actual_bytes:.0e}")
            
            try:
                # Create source tensor
                src_tensor = torch.randn(num_elements, dtype=torch.float16, device=self.device)
                
                # Warmup - memory copy operations
                for _ in range(self.warmup_iterations):
                    _ = src_tensor.clone()
                torch.cuda.synchronize()
                
                # Measure memory copy time
                times = []
                for _ in range(self.measure_iterations):
                    start_time = time.perf_counter()
                    dst_tensor = src_tensor.clone()  # This involves read + write
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                    del dst_tensor
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Calculate actual bandwidth (read + write = 2 * actual_bytes)
                total_bytes_transferred = 2 * actual_bytes  # read source + write destination
                actual_bandwidth_gb_s = (total_bytes_transferred / avg_time) / 1e9
                
                # Calculate delay factor
                theoretical_bandwidth = self.theoretical_specs['memory_bandwidth_gb_s']
                delay_factor = theoretical_bandwidth / actual_bandwidth_gb_s
                
                print(f"  Average time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
                print(f"  Actual bandwidth: {actual_bandwidth_gb_s:.2f} GB/s")
                print(f"  Theoretical bandwidth: {theoretical_bandwidth:.2f} GB/s")
                print(f"  Delay factor: {delay_factor:.2f}")
                
                memory_delay_factors[actual_bytes] = delay_factor
                
                # Clean up
                del src_tensor
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"  ❌ GPU 메모리 부족으로 건너뜀 (텐서 크기: {num_elements} elements)")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  ❌ 측정 실패: {e}")
                torch.cuda.empty_cache()
                continue
        
        return memory_delay_factors
    
    def extract_delay_factors(self, 
                            flops_range: Tuple[float, float] = (1e6, 1e12),
                            bytes_range: Tuple[float, float] = (1e4, 1e9),
                            num_points: int = 10) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Extract delay factors for a range of FLOPS and bytes values.
        
        Args:
            flops_range: (min_flops, max_flops) range to benchmark
            bytes_range: (min_bytes, max_bytes) range to benchmark  
            num_points: Number of points to sample in each range
            
        Returns:
            Dictionary in the format expected by DelayFactorManager
        """
        print(f"\\nExtracting delay factors for GPU: {self.gpu_name}")
        print(f"FLOPS range: {flops_range[0]:.0e} to {flops_range[1]:.0e}")
        print(f"Bytes range: {bytes_range[0]:.0e} to {bytes_range[1]:.0e}")
        print(f"Number of sample points: {num_points}")
        
        # Generate logarithmically spaced sample points
        flops_targets = np.logspace(
            np.log10(flops_range[0]), 
            np.log10(flops_range[1]), 
            num_points
        )
        
        bytes_targets = np.logspace(
            np.log10(bytes_range[0]), 
            np.log10(bytes_range[1]), 
            num_points
        )
        
        # Benchmark FLOPS
        flops_delay_factors = self.benchmark_flops(flops_targets)
        
        # Benchmark Memory
        memory_delay_factors = self.benchmark_memory(bytes_targets)
        
        # Convert to string keys for JSON serialization
        result = {
            self._normalize_gpu_name(): {
                "flops_delay_factors": {
                    str(int(flops)): delay_factor 
                    for flops, delay_factor in flops_delay_factors.items()
                },
                "memory_delay_factors": {
                    str(int(bytes_val)): delay_factor 
                    for bytes_val, delay_factor in memory_delay_factors.items()
                }
            }
        }
        
        return result
    
    def _normalize_gpu_name(self) -> str:
        """Normalize GPU name for consistent naming."""
        name = self.gpu_name.upper()
        
        # Common normalizations
        if 'TESLA T4' in name:
            return 'T4'
        elif 'L4' in name:
            return 'L4'
        elif 'A10G' in name:
            return 'A10G'
        elif 'L40S' in name:
            return 'L40S'
        elif 'L40' in name:
            return 'L40'
        elif 'A100' in name:
            if '80GB' in name or '80G' in name:
                return 'A100-80GB'
            else:
                return 'A100-40GB'
        elif 'H100' in name:
            if 'SXM' in name:
                return 'H100-SXM'
            else:
                return 'H100-PCIe'
        else:
            # Return cleaned up version of the original name
            return name.replace(' ', '-')
    
    def save_delay_factors(self, delay_factors: Dict, output_file: str = None):
        """
        Save delay factors to JSON file.
        
        Args:
            delay_factors: Delay factors dictionary
            output_file: Output file path (optional)
        """
        if output_file is None:
            gpu_name = self._normalize_gpu_name()
            output_file = f"/Users/anchovy-mac/Desktop/calculating/data/{gpu_name}_extracted_delay_factors.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if file exists
        existing_data = {}
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
            except:
                print("Warning: Could not load existing delay factors file.")
        
        # Merge with existing data
        for gpu_name, factors in delay_factors.items():
            existing_data[gpu_name] = factors
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2, sort_keys=True)
        
        print(f"\\nSaved delay factors to: {output_path}")
        print("File contents preview:")
        print(json.dumps(delay_factors, indent=2)[:500] + "...")
    
    def quick_benchmark(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run a quick benchmark with predefined ranges.
        
        Returns:
            Delay factors dictionary
        """
        return self.extract_delay_factors(
            flops_range=(1e3, 1e13),    # 1K to 10T FLOPs (더 넓은 범위)
            bytes_range=(1e3, 1e10),    # 1K to 10G bytes (더 넓은 범위)  
            num_points=12               # 더 많은 측정 포인트
        )


def main():
    """Main function to extract delay factors."""
    print("="*80)
    print("GPU Delay Factor Extractor")
    print("="*80)
    
    try:
        # Initialize extractor
        extractor = DelayFactorExtractor(warmup_iterations=5, measure_iterations=20)
        
        # Extract delay factors
        print("\\nStarting benchmark...")
        delay_factors = extractor.quick_benchmark()
        
        # Save results
        extractor.save_delay_factors(delay_factors)
        
        # Print summary
        gpu_name = list(delay_factors.keys())[0]
        flops_factors = delay_factors[gpu_name]["flops_delay_factors"]
        memory_factors = delay_factors[gpu_name]["memory_delay_factors"]
        
        print("\\n" + "="*50)
        print("Benchmarking Complete!")
        print("="*50)
        print(f"GPU: {gpu_name}")
        print(f"FLOPS delay factors: {len(flops_factors)} points")
        print(f"Memory delay factors: {len(memory_factors)} points")
        print(f"FLOPS range: {min(flops_factors.values()):.2f} - {max(flops_factors.values()):.2f}")
        print(f"Memory range: {min(memory_factors.values()):.2f} - {max(memory_factors.values()):.2f}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise


if __name__ == "__main__":
    main()