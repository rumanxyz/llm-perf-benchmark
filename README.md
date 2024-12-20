# Benchmarking Language Model Performance

This script allows you to benchmark the performance of a given Language Model (LLM) or Small Language Model (SLM). It provides key performance metrics such as Transactions Per Second (TPS), Time to First Token (TTFT), and Peak GPU Usage.

## Requirements

To measure GPU usage, you need to install the `GPUtil` package:
```bash
pip install gputil
```

## How to Use

1. Download the `bench_llm.py` file from the repository.
2. Import the `benchmark_language_model` function into your code.

Example usage:
```python
from bench_llm import benchmark_language_model

# Parameters:
# model -> The language model to benchmark
# tokenizer -> The tokenizer associated with the model
# input_prompt_list -> A list of input prompts of varying sizes

bench_result = benchmark_language_model(model, tokenizer, [input_prompt])
```

### Example Output

The `benchmark_language_model` will return a dictionary containing the benchmark results. Example:
```json
{'median_decode_tps': 29.985,
 'mean_decode_tps': 25.269,
 'median_tps': 37.19,
 'mean_tps': 31.207,
 'median_ttft_seconds': 0.002,
 'mean_ttft_seconds': 0.048,
 'median_gpu_usage_mb': 1445.0,
 'mean_gpu_usage_mb': 1445.0}
```

### Reference Notebook

You can refer to this Colab notebook for a detailed example of using the code for performance benchmarking:

[Colab Notebook: Performance Benchmarking](https://colab.research.google.com/drive/1OTf3v3kJepj7j_XwIQDrNTdKjxbbR1V-?usp=sharing)

---

Feel free to customize and extend the script as needed for your benchmarking purposes :)
