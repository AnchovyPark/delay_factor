"""
실제 LLM 추론 시간 측정기

시나리오 CSV를 읽어서 실제 GPU에서 LLM 추론을 실행하고
PREFILL/DECODE 단계별 시간을 정밀하게 측정합니다.
"""

import csv
import time
import torch
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import json

# GPU 메모리 정리용
import gc


class ActualInferenceRunner:
    """실제 LLM 추론 실행 및 시간 측정 클래스"""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
        # GPU 사용 가능 여부 확인
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU가 필요합니다!")
        
        self.device = torch.device('cuda')
        print(f"  사용 중인 GPU: {torch.cuda.get_device_name()}")
        print(f" GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model(self, model_name: str) -> bool:
        """모델 로드 (필요시에만)"""
        
        if self.current_model_name == model_name:
            print(f" {model_name} 이미 로드됨 (재사용)")
            return True
        
        # 기존 모델 메모리 해제
        if self.current_model is not None:
            print(f"  기존 모델 ({self.current_model_name}) 메모리 해제")
            del self.current_model
            del self.current_tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f" {model_name} 모델 로딩 중...")
        
        try:
            # 모델명 매핑 (로컬 경로도 지원)
            model_paths = {
                'LLaMA_3.2_1B': 'meta-llama/Llama-3.2-1B-Instruct',
                'LLaMA_3_8B': 'meta-llama/Llama-3-8B-Instruct',
                'LLaMA_3.1_8B': 'meta-llama/Llama-3.1-8B-Instruct',
                'LLaMA_3_70B': 'meta-llama/Llama-3-70B-Instruct'
            }
            
            if model_name not in model_paths:
                print(f" 지원하지 않는 모델: {model_name}")
                return False
            
            model_path = model_paths[model_name]
            
            # Transformers 라이브러리 사용
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # 토크나이저 로드
            print(f" 토크나이저 로딩...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            # 현재 GPU 메모리 확인
            memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            memory_free_gb = memory_free / 1e9
            print(f" 사용 가능한 GPU 메모리: {memory_free_gb:.1f} GB")
            
            # 모델 로드 (GPU 메모리에 맞춰 설정)
            print(f" 모델 로딩...")
            if '70B' in model_name:
                # 70B 모델: 메모리 최적화 필요
                if memory_free_gb < 40:  # 40GB 미만이면 8bit 로딩
                    print(f"  8bit 양자화 로딩 (메모리 절약)")
                    self.current_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map='auto',
                        load_in_8bit=True,
                        trust_remote_code=True
                    )
                else:
                    self.current_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map='auto',
                        trust_remote_code=True
                    )
            elif '8B' in model_name:
                # 8B 모델: 대부분 GPU에서 fp16 가능
                if memory_free_gb < 16:
                    print(f" 8bit 양자화 로딩 (메모리 절약)")
                    self.current_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map='auto',
                        load_in_8bit=True,
                        trust_remote_code=True
                    )
                else:
                    self.current_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map='auto',
                        trust_remote_code=True
                    )
            else:
                # 1B 모델: 가벼워서 대부분 문제없음
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map='auto',
                    trust_remote_code=True
                )
            
            self.current_model_name = model_name
            print(f" {model_name} 로딩 완료")
            
            # 메모리 사용량 출력
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f" GPU 메모리 사용량: {memory_used:.1f}/{memory_total:.1f} GB ({memory_used/memory_total*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f" 모델 로딩 실패: {e}")
            return False
    
    def measure_inference_time(self, input_text: str, max_new_tokens: int, batch_size: int = 1) -> Dict:
        """실제 추론 시간 측정 (PREFILL + DECODE 분리)"""
        
        if self.current_model is None or self.current_tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다!")
        
        # 입력 토크나이징
        inputs = self.current_tokenizer(
            input_text, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=4096  # 최대 길이 제한
        ).to(self.device)
        
        input_length = inputs.input_ids.shape[1]
        
        # GPU 워밍업 (첫 실행시 초기화 시간 제거)
        with torch.no_grad():
            _ = self.current_model(**inputs)
        
        torch.cuda.synchronize()  # GPU 동기화
        
        # PREFILL 단계 측정 (입력 처리)
        prefill_start = time.perf_counter()
        
        with torch.no_grad():
            # KV cache를 포함한 첫 forward pass
            outputs = self.current_model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values
        
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        prefill_time_ms = (prefill_end - prefill_start) * 1000
        
        # DECODE 단계 측정 (토큰 생성)
        generated_tokens = []
        decode_times = []
        
        current_input_ids = inputs.input_ids
        current_past_key_values = past_key_values
        
        for i in range(max_new_tokens):
            decode_start = time.perf_counter()
            
            with torch.no_grad():
                # 다음 토큰 예측
                outputs = self.current_model(
                    input_ids=current_input_ids[:, -1:],  # 마지막 토큰만
                    past_key_values=current_past_key_values,
                    use_cache=True
                )
                
                # 다음 토큰 선택 (temperature sampling)
                next_token_logits = outputs.logits[:, -1, :] / 0.8  # temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # KV cache 업데이트
                current_past_key_values = outputs.past_key_values
                current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            
            torch.cuda.synchronize()
            decode_end = time.perf_counter()
            
            decode_time_ms = (decode_end - decode_start) * 1000
            decode_times.append(decode_time_ms)
            generated_tokens.append(next_token.item())
            
            # EOS 토큰이면 중단
            if next_token.item() == self.current_tokenizer.eos_token_id:
                break
        
        # 결과 정리
        actual_output_tokens = len(generated_tokens)
        total_decode_time_ms = sum(decode_times)
        avg_decode_per_token_ms = total_decode_time_ms / actual_output_tokens if actual_output_tokens > 0 else 0
        total_time_ms = prefill_time_ms + total_decode_time_ms
        
        # 생성된 텍스트 디코딩
        generated_text = self.current_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'input_length': input_length,
            'actual_output_length': actual_output_tokens,
            'requested_output_length': max_new_tokens,
            
            'prefill_time_ms': round(prefill_time_ms, 2),
            'total_decode_time_ms': round(total_decode_time_ms, 2),
            'avg_decode_per_token_ms': round(avg_decode_per_token_ms, 4),
            'total_time_ms': round(total_time_ms, 2),
            'throughput_tokens_per_sec': round(actual_output_tokens / (total_time_ms / 1000), 2) if total_time_ms > 0 else 0,
            
            'generated_text': generated_text[:200] + '...' if len(generated_text) > 200 else generated_text,
            'decode_times_per_token': [round(t, 4) for t in decode_times]
        }


def load_scenarios(csv_file: str) -> List[Dict]:
    """시나리오 CSV 파일 로드"""
    
    filepath = f'/Users/anchovy-mac/Desktop/calculating/experiment/{csv_file}'
    scenarios = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row['scenario_id'] = int(row['scenario_id'])
            row['input_length'] = int(row['input_length'])
            row['output_length'] = int(row['output_length'])
            row['batch_size'] = int(row['batch_size'])
            scenarios.append(row)
    
    return scenarios


def generate_input_text(length: int) -> str:
    """지정된 토큰 길이의 입력 텍스트 생성 (단순 반복)"""
    
    # 단순한 반복 텍스트 생성 (latency는 내용과 무관하므로)
    # "a" 문자를 반복하여 대략적인 토큰 길이 맞춤
    base_word = "a"
    
    # 대략 토큰당 1-2개 문자로 추정하여 생성
    # 실제로는 토크나이저가 정확히 계산하지만, 대략적으로 맞춤
    repeated_text = base_word * (length * 2)  # 여유있게 생성
    
    return repeated_text


def filter_scenarios_by_current_gpu(scenarios: List[Dict]) -> List[Dict]:
    """현재 GPU에서 실행 가능한 시나리오만 필터링"""
    
    current_gpu_name = torch.cuda.get_device_name()
    current_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🖥️  현재 GPU: {current_gpu_name} ({current_memory_gb:.1f} GB)")
    
    # GPU별 대략적인 이름 매칭
    gpu_mapping = {
        'Tesla T4': ['T4'],
        'A10G': ['A10G'], 
        'L40S': ['L40S', 'L40s'],
        'A100': ['A100'],
        'A100-SXM4-40GB': ['A100'],
        'A100-SXM4-80GB': ['A100-SXM4-80GB']
    }
    
    # 현재 GPU에 해당하는 시나리오 찾기
    matching_gpu_names = []
    for gpu_key, aliases in gpu_mapping.items():
        for alias in aliases:
            if alias in current_gpu_name:
                matching_gpu_names.append(gpu_key)
                break
    
    if not matching_gpu_names:
        print(f"⚠️  현재 GPU ({current_gpu_name})에 대한 시나리오를 찾을 수 없습니다.")
        print("🔧 모든 시나리오를 실행합니다.")
        return scenarios
    
    # 필터링된 시나리오
    filtered_scenarios = []
    for scenario in scenarios:
        if scenario['gpu'] in matching_gpu_names:
            filtered_scenarios.append(scenario)
    
    print(f"✅ {len(filtered_scenarios)}개 시나리오가 현재 GPU에 적합합니다.")
    return filtered_scenarios


def run_scenario_measurements(scenarios: List[Dict], runner: ActualInferenceRunner, max_scenarios: int = None) -> List[Dict]:
    """시나리오들에 대해 실제 추론 시간 측정"""
    
    # 현재 GPU에 맞는 시나리오만 필터링
    scenarios = filter_scenarios_by_current_gpu(scenarios)
    
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]
    
    results = []
    total_scenarios = len(scenarios)
    
    print(f"⏱️  {total_scenarios}개 시나리오에 대해 실제 추론 시간 측정 시작")
    print("-" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        try:
            print(f"\n🔄 [{i}/{total_scenarios}] 시나리오 {scenario['scenario_id']} 처리 중...")
            print(f"   GPU: {scenario['gpu']}, 모델: {scenario['model']}")
            print(f"   입력: {scenario['input_length']}토큰, 출력: {scenario['output_length']}토큰")
            
            # 모델 로드
            if not runner.load_model(scenario['model']):
                raise Exception(f"모델 {scenario['model']} 로딩 실패")
            
            # 입력 텍스트 생성
            input_text = generate_input_text(scenario['input_length'])
            
            # 실제 추론 실행 및 시간 측정
            measurement = runner.measure_inference_time(
                input_text=input_text,
                max_new_tokens=scenario['output_length'],
                batch_size=scenario['batch_size']
            )
            
            # 결과 저장
            result = {
                'scenario_id': scenario['scenario_id'],
                'gpu': scenario['gpu'],
                'model': scenario['model'],
                'use_case': scenario['use_case'],
                'batch_size': scenario['batch_size'],
                
                # 요청 파라미터
                'requested_input_length': scenario['input_length'],
                'requested_output_length': scenario['output_length'],
                
                # 실제 측정 결과
                'actual_input_length': measurement['input_length'],
                'actual_output_length': measurement['actual_output_length'],
                
                'measured_prefill_ms': measurement['prefill_time_ms'],
                'measured_decode_per_token_ms': measurement['avg_decode_per_token_ms'],
                'measured_total_decode_ms': measurement['total_decode_time_ms'],
                'measured_total_ms': measurement['total_time_ms'],
                'measured_throughput_tokens_per_sec': measurement['throughput_tokens_per_sec'],
                
                'generated_text_preview': measurement['generated_text'],
                'measurement_timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"   ✅ 완료: {measurement['total_time_ms']:.1f}ms "
                  f"(PREFILL: {measurement['prefill_time_ms']:.1f}ms, "
                  f"DECODE: {measurement['avg_decode_per_token_ms']:.2f}ms/token)")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
            
            # 실패 기록
            error_result = {
                'scenario_id': scenario['scenario_id'],
                'gpu': scenario['gpu'],
                'model': scenario['model'],
                'use_case': scenario['use_case'],
                'error': str(e),
                'measurement_timestamp': datetime.now().isoformat()
            }
            results.append(error_result)
    
    return results


def save_measurements_to_csv(measurements: List[Dict], filename: str):
    """측정 결과를 CSV로 저장"""
    
    filepath = f'/Users/anchovy-mac/Desktop/calculating/experiment/{filename}'
    
    if not measurements:
        print("❌ 저장할 측정 결과가 없습니다.")
        return
    
    fieldnames = list(measurements[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measurements)
    
    print(f"💾 측정 결과가 {filepath}에 저장되었습니다")


def main():
    """메인 실행 함수"""
    
    print("⏱️  LLM 실제 추론 시간 측정 시스템")
    print("=" * 60)
    
    try:
        # 추론 실행기 초기화
        runner = ActualInferenceRunner()
        
        # 현재 GPU 정보 확인
        current_gpu = torch.cuda.get_device_name()
        print(f"\n🖥️  현재 GPU: {current_gpu}")
        
        # 시나리오 파일 선택
        scenario_files = [
            ('length_combination_scenarios.csv', 'length_combination_measurements.csv')
        ]
        
        for scenario_file, output_file in scenario_files:
            print(f"\n📋 {scenario_file} 처리 중...")
            
            try:
                scenarios = load_scenarios(scenario_file)
                print(f"📖 {len(scenarios)}개 시나리오 로드됨")
                
                # 모든 시나리오 실행
                measurements = run_scenario_measurements(
                    scenarios, runner, max_scenarios=None  # 모든 시나리오 실행
                )
                
                save_measurements_to_csv(measurements, output_file)
                
                # 간단한 통계
                successful = [m for m in measurements if 'error' not in m]
                if successful:
                    avg_time = sum(m['measured_total_ms'] for m in successful) / len(successful)
                    print(f"📊 평균 추론 시간: {avg_time:.1f}ms")
                
            except FileNotFoundError:
                print(f"⚠️  {scenario_file}을 찾을 수 없습니다.")
            except Exception as e:
                print(f"❌ {scenario_file} 처리 중 오류: {e}")
        
        print(f"\n🎉 실제 측정 완료!")
        
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        print("💡 CUDA GPU와 transformers 라이브러리가 필요합니다.")


if __name__ == "__main__":
    main()