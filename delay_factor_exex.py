"""
Delay Factor Extractor

This module benchmarks actual GPU performance to extract delay factors
for different FLOPS and memory operations, then saves to JSON format.
"""

import torch
import time
import json
import csv
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import nvtx 

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
    
    def benchmark_flops(self, flops_targets: List[float]) -> List[Dict]:
        """
        Benchmark FLOPS performance and record actual/theoretical times.

        Args:
            flops_targets: List of FLOPS values to benchmark

        Returns:
            List of dictionaries with benchmark results
        """
        print("\\n" + "="*50)
        print("Benchmarking FLOPS Performance")
        print("="*50)

        results = []

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
                # Measure using CUDA events
                times = []
                for _ in range(self.measure_iterations):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    with nvtx.annotate(f"FLOPS_{target_flops:.0e}", color="blue"):
                        start_event.record()
                        result = torch.matmul(A, B)
                        end_event.record()

                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
                    times.append(elapsed_time)
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)

                # Calculate actual TFLOPS
                actual_tflops = (actual_flops / avg_time) / 1e12

                # Calculate theoretical latency
                theoretical_tflops = self.theoretical_specs['fp16_tflops']
                theoretical_time = actual_flops / (theoretical_tflops * 1e12)  # 이론적 latency
                delay_factor = avg_time / theoretical_time  # 실제시간 / 이론시간

                print(f"  Actual time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
                print(f"  Theoretical time: {theoretical_time*1000:.2f}ms")
                print(f"  Actual TFLOPS: {actual_tflops:.2f}")
                print(f"  Theoretical TFLOPS: {theoretical_tflops:.2f}")
                print(f"  Delay factor: {delay_factor:.2f} (actual/theoretical time)")

                # Store result
                results.append({
                    'operation_type': 'flops',
                    'log_value': np.log10(actual_flops),
                    'actual_value': actual_flops,
                    'matrix_dim': dim,
                    'actual_time_ms': avg_time * 1000,
                    'theoretical_time_ms': theoretical_time * 1000,
                    'std_time_ms': std_time * 1000,
                    'actual_tflops': actual_tflops,
                    'theoretical_tflops': theoretical_tflops,
                    'delay_factor': delay_factor
                })

                # Clean up
                del A, B, result
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"   GPU 메모리 부족으로 건너뜀 (행렬 크기: {dim}x{dim})")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"   측정 실패: {e}")
                torch.cuda.empty_cache()
                continue

        return results
    
    # # def benchmark_memory_read(self, bytes_targets: List[float]) -> Dict[float, float, float]:
    #     """
    #     Benchmark memory READ performance and calculate delay factors.
        
    #     Args:
    #         bytes_targets: List of byte counts to benchmark
            
    #     Returns:
    #         Dictionary mapping bytes to delay factors
    #     """
    #     print("\\n" + "="*50)
    #     print("Benchmarking Memory READ Performance")
    #     print("="*50)
        
    #     # Create dummy data to fill L2 cache and force DRAM access
    #     l2_cache_size = 40 * 1024 * 1024  # 40MB (typical L2 cache size)
    #     dummy_elements = l2_cache_size // 2  # FP16
    #     dummy_tensor = torch.randn(dummy_elements, dtype=torch.float16, device=self.device)
    #     print(f"Created {l2_cache_size/1024/1024:.1f}MB dummy tensor to invalidate L2 cache")
        
    #     read_delay_factors = {}
    #     bytes_per_element = 2  # FP16
        
    #     for target_bytes in bytes_targets:
    #         print(f"\\nBenchmarking READ {target_bytes:.0e} bytes...")
            
    #         # Calculate tensor size
    #         num_elements = int(target_bytes / bytes_per_element)
    #         actual_bytes = num_elements * bytes_per_element
            
    #         print(f"  Tensor elements: {num_elements}")
    #         print(f"  Actual bytes: {actual_bytes:.0e}")
            
    #         try:
    #             # Create source tensor
    #             src_tensor = torch.randn(num_elements, dtype=torch.float16, device=self.device)
                
    #             # Warmup - memory read operations (계산 결과를 변수에 저장하여 실제 read 강제)
    #             for _ in range(self.warmup_iterations):
    #                 # Fill L2 cache with dummy data to force DRAM access
    #                 _ = torch.sum(dummy_tensor)  # Fill L2 with dummy data
    #                 _ = torch.sum(src_tensor)  # Read all elements from DRAM
    #             torch.cuda.synchronize()
                
    #             # Measure memory read time
    #             times = []
    #             for _ in range(self.measure_iterations):
    #                 # Fill L2 cache with dummy data before each measurement
    #                 _ = torch.sum(dummy_tensor)  # Fill L2 with dummy data
    #                 torch.cuda.synchronize()
    #                 with nvtx.annotate(f"MemRead_{actual_bytes:.0e}B", color="purple"):
    #                     start_time = time.perf_counter()
    #                     result = torch.sum(src_tensor)  # Forces reading all elements from DRAM
    #                     torch.cuda.synchronize()
    #                     end_time = time.perf_counter()
    #                 times.append(end_time - start_time)
                
    #             # Calculate statistics
    #             avg_time = np.mean(times)
    #             std_time = np.std(times)
                
    #             # Calculate actual READ bandwidth
    #             # actual_bandwidth_gb_s = (actual_bytes / avg_time) / 1e9
                
    #             # Calculate theoretical latency and delay factor
    #             theoretical_bandwidth = self.theoretical_specs['memory_bandwidth_gb_s']
    #             theoretical_time = actual_bytes / (theoretical_bandwidth * 1e9)  # 이론적 latency
    #             delay_factor = avg_time / theoretical_time  # 실제시간 / 이론시간
                
    #             print(f"  Actual time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
    #             print(f"  Theoretical time: {theoretical_time*1000:.2f}ms")
    #             # print(f"  Actual READ bandwidth: {actual_bandwidth_gb_s:.2f} GB/s")
    #             print(f"  Theoretical bandwidth: {theoretical_bandwidth:.2f} GB/s")
    #             print(f"  READ delay factor: {delay_factor:.2f} (actual/theoretical time)")
                
    #             read_delay_factors[actual_bytes] = delay_factor
                
    #             # Clean up
    #             del src_tensor
    #             torch.cuda.empty_cache()
                
    #         except torch.cuda.OutOfMemoryError:
    #             print(f"   GPU 메모리 부족으로 건너뜀 (텐서 크기: {num_elements} elements)")
    #             torch.cuda.empty_cache()
    #             continue
    #         except Exception as e:
    #             print(f"   측정 실패: {e}")
    #             torch.cuda.empty_cache()
    #             continue
        
    #     # Clean up dummy tensor
    #     del dummy_tensor
    #     torch.cuda.empty_cache()
        
    #     return read_delay_factors, avg_time
    
    # def benchmark_memory_write(self, bytes_targets: List[float]) -> Dict[float, float]:
    #     """
    #     Benchmark memory WRITE performance and calculate delay factors.
        
    #     Args:
    #         bytes_targets: List of byte counts to benchmark
            
    #     Returns:
    #         Dictionary mapping bytes to delay factors
    #     """
    #     print("\\n" + "="*50)
    #     print("Benchmarking Memory WRITE Performance")
    #     print("="*50)
        
    #     # Create dummy data to fill L2 cache and force DRAM access
    #     l2_cache_size = 40 * 1024 * 1024  # 40MB (typical L2 cache size)
    #     dummy_elements = l2_cache_size // 2  # FP16
    #     dummy_tensor = torch.randn(dummy_elements, dtype=torch.float16, device=self.device)
    #     print(f"Created {l2_cache_size/1024/1024:.1f}MB dummy tensor to invalidate L2 cache")
        
    #     write_delay_factors = {}
    #     bytes_per_element = 2  # FP16
        
    #     for target_bytes in bytes_targets:
    #         print(f"\\nBenchmarking WRITE {target_bytes:.0e} bytes...")
            
    #         # Calculate tensor size
    #         num_elements = int(target_bytes / bytes_per_element)
    #         actual_bytes = num_elements * bytes_per_element
            
    #         print(f"  Tensor elements: {num_elements}")
    #         print(f"  Actual bytes: {actual_bytes:.0e}")
            
    #         try:
    #             # Create empty tensor for writing
    #             dst_tensor = torch.empty(num_elements, dtype=torch.float16, device=self.device)
                
    #             # Warmup - memory write operations
    #             for _ in range(self.warmup_iterations):
    #                 # Fill L2 cache with dummy data to force DRAM access
    #                 _ = torch.sum(dummy_tensor)  # Fill L2 with dummy data
    #                 dst_tensor.fill_(1.0)  # Write same value to all elements to DRAM
    #             torch.cuda.synchronize()
                
    #             # Measure memory write time
    #             times = []
    #             for _ in range(self.measure_iterations):
    #                 # Fill L2 cache with dummy data before each measurement
    #                 _ = torch.sum(dummy_tensor)  # Fill L2 with dummy data
    #                 torch.cuda.synchronize()
                    

    #                 with nvtx.annotate(f"MemWrite_{actual_bytes:.0e}B", color="red"):
    #                     start_time = time.perf_counter()
    #                     dst_tensor.fill_(1.0)  # Forces writing to all elements to DRAM
    #                     torch.cuda.synchronize()
    #                     end_time = time.perf_counter()

    #                 times.append(end_time - start_time)
                
    #             # Calculate statistics
    #             avg_time = np.mean(times)
    #             std_time = np.std(times)
                
    #             # Calculate actual WRITE bandwidth
    #             actual_bandwidth_gb_s = (actual_bytes / avg_time) / 1e9
                
    #             # Calculate theoretical latency and delay factor
    #             theoretical_bandwidth = self.theoretical_specs['memory_bandwidth_gb_s']
    #             theoretical_time = actual_bytes / (theoretical_bandwidth * 1e9)  # 이론적 latency
    #             delay_factor = avg_time / theoretical_time  # 실제시간 / 이론시간
                
    #             print(f"  Actual time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
    #             print(f"  Theoretical time: {theoretical_time*1000:.2f}ms")
    #             print(f"  Actual WRITE bandwidth: {actual_bandwidth_gb_s:.2f} GB/s")
    #             print(f"  Theoretical bandwidth: {theoretical_bandwidth:.2f} GB/s")
    #             print(f"  WRITE delay factor: {delay_factor:.2f} (actual/theoretical time)")
                
    #             write_delay_factors[actual_bytes] = delay_factor
                
    #             # Clean up
    #             del dst_tensor
    #             torch.cuda.empty_cache()
                
    #         except torch.cuda.OutOfMemoryError:
    #             print(f"  ❌ GPU 메모리 부족으로 건너뜀 (텐서 크기: {num_elements} elements)")
    #             torch.cuda.empty_cache()
    #             continue
    #         except Exception as e:
    #             print(f"  ❌ 측정 실패: {e}")
    #             torch.cuda.empty_cache()
    #             continue
        
    #     # Clean up dummy tensor
    #     del dummy_tensor
    #     torch.cuda.empty_cache()
        
    #     return write_delay_factors
    
    def benchmark_memory(self, bytes_targets: List[float]) -> List[Dict]:
        """
        Benchmark memory DRAM bandwidth by measuring memory copy with cache flushing.

        Args:
            bytes_targets: List of byte counts to benchmark

        Returns:
            List of dictionaries with benchmark results
        """
        print("\\n" + "="*50)
        print("Benchmarking Memory DRAM Bandwidth")
        print("="*50)

        # Create dummy data to fill L2 cache and force DRAM access
        l2_cache_size = 40 * 1024 * 1024  # 40MB (typical L2 cache size for most GPUs)
        dummy_elements = l2_cache_size // 2  # FP16
        dummy_tensor = torch.randn(dummy_elements, dtype=torch.float16, device=self.device)
        print(f"Created {l2_cache_size/1024/1024:.1f}MB dummy tensor to flush L2 cache")

        results = []
        bytes_per_element = 2  # FP16

        for target_bytes in bytes_targets:
            print(f"\\nBenchmarking {target_bytes:.0e} bytes...")

            # Calculate tensor size
            num_elements = int(target_bytes / bytes_per_element)
            actual_bytes = num_elements * bytes_per_element

            print(f"  Tensor elements: {num_elements}")
            print(f"  Actual bytes: {actual_bytes:.0e}")

            try:
                # Create source and destination tensors
                src_tensor = torch.randn(num_elements, dtype=torch.float16, device=self.device)
                dst_tensor = torch.empty(num_elements, dtype=torch.float16, device=self.device)

                # Warmup - memory copy operations
                for _ in range(self.warmup_iterations):
                    # Fill L2 cache with dummy data to force DRAM access
                    _ = torch.sum(dummy_tensor)
                    dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()

                # Measure memory copy time using CUDA events (read from DRAM + write to DRAM)
                times = []
                for _ in range(self.measure_iterations):
                    # Fill L2 cache with dummy data before each measurement
                    _ = torch.sum(dummy_tensor)
                    torch.cuda.synchronize()

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    with nvtx.annotate(f"MemCopy_{target_bytes:.0e}B", color="green"):
                        start_event.record()
                        dst_tensor.copy_(src_tensor)  # Read from DRAM + Write to DRAM
                        end_event.record()

                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
                    times.append(elapsed_time)

                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)

                # Calculate actual bandwidth (read + write = 2 * actual_bytes)
                total_bytes_transferred = 2 * actual_bytes  # read source + write destination
                actual_bandwidth_gb_s = (total_bytes_transferred / avg_time) / 1e9

                # Calculate theoretical latency
                theoretical_bandwidth = self.theoretical_specs['memory_bandwidth_gb_s']
                theoretical_time = total_bytes_transferred / (theoretical_bandwidth * 1e9)  # 이론적 latency
                delay_factor = avg_time / theoretical_time  # 실제시간 / 이론시간

                print(f"  Average time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
                print(f"  Theoretical time: {theoretical_time*1000:.2f}ms")
                print(f"  Actual bandwidth: {actual_bandwidth_gb_s:.2f} GB/s")
                print(f"  Theoretical bandwidth: {theoretical_bandwidth:.2f} GB/s")
                print(f"  Delay factor: {delay_factor:.2f} (actual/theoretical time)")

                # Store result
                results.append({
                    'operation_type': 'memory',
                    'log_value': np.log10(actual_bytes),
                    'actual_value': actual_bytes,
                    'tensor_elements': num_elements,
                    'actual_time_ms': avg_time * 1000,
                    'theoretical_time_ms': theoretical_time * 1000,
                    'std_time_ms': std_time * 1000,
                    'actual_bandwidth_gb_s': actual_bandwidth_gb_s,
                    'theoretical_bandwidth_gb_s': theoretical_bandwidth,
                    'delay_factor': delay_factor
                })

                # Clean up
                del src_tensor, dst_tensor
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"GPU 메모리 부족으로 건너뜀 (텐서 크기: {num_elements} elements)")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"측정 실패: {e}")
                torch.cuda.empty_cache()
                continue

        # Clean up dummy tensor
        del dummy_tensor
        torch.cuda.empty_cache()

        return results
    
    def extract_delay_factors(self, 
                            flops_range: Tuple[float, float] = (1e6, 1e12),
                            bytes_range: Tuple[float, float] = (1e4, 1e9)) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Extract delay factors for a range of FLOPS and bytes values.
        
        Args:
            flops_range: (min_flops, max_flops) range to benchmark
            bytes_range: (min_bytes, max_bytes) range to benchmark
            
        Returns:
            Dictionary in the format expected by DelayFactorManager
        """
        print(f"\\nExtracting delay factors for GPU: {self.gpu_name}")
        print(f"FLOPS range: {flops_range[0]:.0e} to {flops_range[1]:.0e}")
        print(f"Bytes range: {bytes_range[0]:.0e} to {bytes_range[1]:.0e}")
        
        # Generate logarithmically spaced sample points with 0.5 increments
        # Create log range with 0.5 increments: 3.0, 3.5, 4.0, 4.5, ...
        flops_log_start = np.log10(flops_range[0])
        flops_log_end = np.log10(flops_range[1])
        flops_log_points = np.arange(flops_log_start, flops_log_end + 0.1, 0.5)
        flops_targets = 10 ** flops_log_points
        
        bytes_log_start = np.log10(bytes_range[0])
        bytes_log_end = np.log10(bytes_range[1])
        bytes_log_points = np.arange(bytes_log_start, bytes_log_end + 0.1, 0.5)
        bytes_targets = 10 ** bytes_log_points
        
        print(f"FLOPS sample points: {len(flops_targets)} points")
        print(f"Bytes sample points: {len(bytes_targets)} points")
        
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
    
    def extract_and_save_csv(self,
                            flops_range: Tuple[float, float] = (1e3, 1e13),
                            bytes_range: Tuple[float, float] = (1e3, 1e10),
                            output_file: str = None) -> str:
        """
        Extract benchmark data and save to CSV.

        Args:
            flops_range: (min_flops, max_flops) range to benchmark
            bytes_range: (min_bytes, max_bytes) range to benchmark
            output_file: Output CSV file path (optional)

        Returns:
            Path to the saved CSV file
        """
        print(f"\\nExtracting benchmark data for GPU: {self.gpu_name}")
        print(f"FLOPS range: {flops_range[0]:.0e} to {flops_range[1]:.0e}")
        print(f"Bytes range: {bytes_range[0]:.0e} to {bytes_range[1]:.0e}")

        # Generate logarithmically spaced sample points with 0.1 increments for fine granularity
        # Create log range with 0.1 increments: 3.0, 3.1, 3.2, 3.3, ...
        flops_log_start = np.log10(flops_range[0])
        flops_log_end = np.log10(flops_range[1])
        flops_log_points = np.arange(flops_log_start, flops_log_end + 0.01, 0.1)
        flops_targets = 10 ** flops_log_points

        bytes_log_start = np.log10(bytes_range[0])
        bytes_log_end = np.log10(bytes_range[1])
        bytes_log_points = np.arange(bytes_log_start, bytes_log_end + 0.01, 0.1)
        bytes_targets = 10 ** bytes_log_points

        print(f"FLOPS sample points: {len(flops_targets)} points (0.1 log increments)")
        print(f"Bytes sample points: {len(bytes_targets)} points (0.1 log increments)")

        # Benchmark FLOPS and Memory
        flops_results = self.benchmark_flops(flops_targets)
        memory_results = self.benchmark_memory(bytes_targets)

        # Combine all results
        all_results = flops_results + memory_results

        # Determine output file path
        if output_file is None:
            gpu_name = self._normalize_gpu_name()
            output_file = f"/home/ubuntu/delay_factor/data/{gpu_name}_benchmark_results.csv"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        # Define fieldnames based on operation type
        fieldnames = ['gpu_name', 'operation_type', 'log_value', 'actual_value',
                     'actual_time_ms', 'theoretical_time_ms', 'std_time_ms', 'delay_factor']

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for result in all_results:
                result['gpu_name'] = self._normalize_gpu_name()
                writer.writerow(result)

        print(f"\\n" + "="*80)
        print(f"Benchmark results saved to: {output_path}")
        print(f"Total measurements: {len(all_results)}")
        print(f"  - FLOPS measurements: {len(flops_results)}")
        print(f"  - Memory measurements: {len(memory_results)}")
        print("="*80)

        return str(output_path)
    
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
    
    


def main():
    """Main function to benchmark GPU and save results to CSV."""
    print("="*80)
    print("GPU Benchmark - CSV Output Mode")
    print("="*80)

    try:
        # Initialize extractor
        extractor = DelayFactorExtractor(warmup_iterations=5, measure_iterations=20)

        # Extract and save benchmark data to CSV
        print("\\nStarting benchmark...")
        output_file = extractor.extract_and_save_csv(
            flops_range=(1e3, 1e14),    # 1K to 100T FLOPs (더 넓은 범위)
            bytes_range=(1e3, 1e11)     # 1K to 100G bytes (더 넓은 범위)
        )

        print("\\n" + "="*80)
        print("Benchmarking Complete!")
        print(f"Results saved to: {output_file}")
        print("="*80)

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise


if __name__ == "__main__":
    main()