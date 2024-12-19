import torch
import time
import GPUtil
import numpy as np
import traceback
import threading
from transformers import TextIteratorStreamer
from typing import Optional, Dict, Any, List, Union

class GPUMonitor:
    """
    GPU monitoring class using GPUtil.
    Tracks peak GPU memory and utilization.
    """
    def __init__(self, monitoring_interval: float = 0.1):
        """
        Initialize GPU Monitor

        Args:
            monitoring_interval: Time between GPU usage checks (in seconds)
        """
        self.monitoring_interval = monitoring_interval
        self._gpu_memory_usage = []
        self._gpu_utilization = []
        self._is_monitoring = False
        self._monitoring_thread = None

    def start(self):
        """Start GPU monitoring"""
        self._is_monitoring = True
        self._gpu_memory_usage = []
        self._gpu_utilization = []

        def monitor_gpu():
            while self._is_monitoring:
                try:
                    # Get GPU information
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Assuming first GPU
                        self._gpu_memory_usage.append(gpu.memoryUsed)
                        self._gpu_utilization.append(gpu.load * 100)

                    # Wait for next interval
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    print(f"GPU monitoring error: {e}")
                    break

        self._monitoring_thread = threading.Thread(target=monitor_gpu)
        self._monitoring_thread.start()

    def stop(self):
        """Stop GPU monitoring"""
        self._is_monitoring = False
        if self._monitoring_thread:
            self._monitoring_thread.join()

    def get_peak_usage(self) -> float:
        """
        Get peak GPU memory usage in MB

        Returns:
            Peak GPU memory usage
        """
        return max(self._gpu_memory_usage) if self._gpu_memory_usage else 0

    def get_peak_utilization(self) -> float:
        """
        Get peak GPU utilization percentage

        Returns:
            Peak GPU utilization
        """
        return max(self._gpu_utilization) if self._gpu_utilization else 0

def benchmark_single_prompt(
        model,
        tokenizer,
        input_prompt_text: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 100,
        device: Optional[str] = None) -> Dict[str, Any]:
    """
    Benchmark a language model's performance for a single prompt.

    Args:
        model: Hugging Face model to benchmark
        tokenizer: Corresponding tokenizer
        input_prompt_text: Input text prompt for generation
        temperature: Sampling temperature for randomness
        top_p: Nucleus sampling threshold
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run inference on (e.g., 'cuda', 'cpu')

    Returns:
        Dict containing detailed benchmark metrics for the prompt
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # GPU monitoring setup
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    # Tokenize input
    start_input_process = time.time()
    inputs = tokenizer(input_prompt_text, return_tensors="pt").to(device)
    input_process_time = time.time() - start_input_process

    # Streaming generation setup
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False)
    generation_start_time = time.time()
    first_token_time = None
    generated_decoded_tokens = []

    # Streaming generation loop
    try:
        # Create generation kwargs
        generation_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': temperature > 0
        }

        # Start generation in a separate thread
        def generate():
            model.generate(**generation_kwargs, streamer=streamer)

        generation_thread = threading.Thread(target=generate)
        generation_thread.start()

        for token in streamer:
            # Record time to first token
            if first_token_time is None:
                first_token_time = time.time() - generation_start_time

            # Accumulate tokens
            generated_decoded_tokens.append(token)

    except Exception as e:
        print(f"Generation error: {e}")
        print(f"Error trace :\n{traceback.format_exc()}")
        return {}

    # Stop GPU monitoring
    gpu_monitor.stop()

    # Total generation metrics
    total_generation_time = time.time() - generation_start_time

    # Calculate metrics
    input_tokens = inputs.input_ids.shape[1]
    output_tokens = len(generated_decoded_tokens)

    # Get peak GPU usage
    peak_gpu_usage = gpu_monitor.get_peak_usage()

    # Prepare benchmark results
    benchmark_results = {
        # 'input_prompt': input_prompt_text,
        'total_time_seconds': total_generation_time,
        'time_to_first_token_seconds': first_token_time,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'tokens_per_second': output_tokens / total_generation_time,
        'input_process_time_seconds': input_process_time,
        'peak_gpu_memory_mb': peak_gpu_usage,
        # 'generated_text': " ".join(generated_decoded_tokens)
    }

    return benchmark_results

def benchmark_language_model(
        model,
        tokenizer,
        prompts: List[str],
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 100,
        device: Optional[str] = None) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
    """
    Benchmark a language model's performance across multiple prompts.

    Args:
        model: Hugging Face model to benchmark
        tokenizer: Corresponding tokenizer
        prompts: List of input text prompts for generation
        temperature: Sampling temperature for randomness
        top_p: Nucleus sampling threshold
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run inference on (e.g., 'cuda', 'cpu')

    Returns:
        Dict containing aggregate benchmark metrics across all prompts
    """
    # Run benchmark for each prompt
    prompt_results = []
    for prompt in prompts:
        result = benchmark_single_prompt(
            model,
            tokenizer,
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            device
        )
        if result:  # Only add non-empty results
            prompt_results.append(result)

    # Calculate aggregate metrics
    if not prompt_results:
        return {}

    # Extract metric lists for aggregation
    tps_list = [result['tokens_per_second'] for result in prompt_results]
    ttft_list = [result['time_to_first_token_seconds'] for result in prompt_results]
    gpu_usage_list = [result['peak_gpu_memory_mb'] for result in prompt_results]

    # Aggregate metrics
    aggregate_results = {
        # Individual prompt results
        # 'prompt_results': prompt_results,

        # Tokens Per Second (TPS) metrics
        'median_tps': round(np.median(tps_list), 3),
        'mean_tps': round(np.mean(tps_list), 3),
        'min_tps': round(np.min(tps_list), 3),
        'max_tps': round(np.max(tps_list), 3),

        # Time to First Token (TTFT) metrics
        'median_ttft_seconds': round(np.median(ttft_list), 3),
        'mean_ttft_seconds': round(np.mean(ttft_list), 3),
        'min_ttft_seconds': round(np.min(ttft_list), 3),
        'max_ttft_seconds': round(np.max(ttft_list), 3),

        # GPU Usage metrics
        'median_gpu_usage_mb': round(np.median(gpu_usage_list), 3),
        'mean_gpu_usage_mb': round(np.mean(gpu_usage_list), 3),
        'min_gpu_usage_mb': round(np.min(gpu_usage_list), 3),
        'max_gpu_usage_mb': round(np.max(gpu_usage_list), 3)
    }

    return aggregate_results