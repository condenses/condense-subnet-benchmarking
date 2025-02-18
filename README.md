# Condense Benchmarking System

This repository contains tools for benchmarking the Condense API (Bittensor) against standard causal language models.

![Latest Bench](./results/bench-dec-04.png)

*Ruler-4k-qa* means RULER subset 4096 and only `qa_1` and `qa_2` task. https://huggingface.co/datasets/simonjegou/ruler

| Method | Accuracy | Input Tokens | Compression Rate |
|--------|----------|--------------|------------------|
| Original | 93% | 3,361 | 1.00 |
| Condense (Top-1) | 73% | 963 ± 156 | Reduce 72% |
| Condense (Top-5) | 59% | 958 ± 148 | Reduce 72% |
| Knorm | 29% | - | Reduce 72% |
| ExpectedAttentionPress | 70% | - | Reduce 72% |

**Notes:**
- Input tokens are reported as mean ± standard deviation where applicable
- Compression Rate = compressed size / original size
- Top-x refers to Elo-based miner sampling strategy
- All experiments conducted on the RULER-4K question-answering dataset

This table provides a clear comparison of different compression methods' performance across key metrics: accuracy, token efficiency, and compression rates.

# Reproduce

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- Condense API key
- OpenAI API key (for accuracy evaluation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/condenses/condense-subnet-benchmarking
cd condense_subnet_benchmarking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export CONDENSE_API_KEY="your-condense-api-key"
export TOGETHER_API_KEY="together-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

*Contact `subnet@condenses.ai` for `CONDENSE_API_KEY`*
If you're an validator. You can spin up your own api server
- https://github.com/condenses/subnet-organic

## Running Benchmarks

The benchmarking process consists of three main steps:

### 1. Run the Main Benchmark

```bash
python main.py 
```

This script:
- Loads the specified dataset
- Processes each sample through both Condense and causal inference
- Saves individual results in `results/<dataset>/<subset>/sample_*.json`
- Saves aggregate results in `results/<dataset>/<subset>/results.json`

### 2. Evaluate Accuracy

```bash
python get_accuracy.py --tier tier_use_to_bench_mark
```

This script:
- Uses GPT-4 to evaluate the accuracy of responses
- Generates comparison plots in `results/comparison_plots.png`
- Outputs accuracy and token usage statistics

### 3. Generate Fancy Plots

```bash
python plot.py
```

Generates detailed visualizations:
- Accuracy comparison
- Token usage analysis
- Time series analysis (if applicable)
- Saves plots in both HTML and PNG formats in the `results/` directory

## Configuration

### Dataset Configuration

Modify the dataset configs in `main.py`:

```python
dataset_configs = [
    {
        'dataset': 'simonjegou/ruler',
        'subsets': ['4096'],
        "folder_id": "ruler-4096"
    }
]
```

### Benchmark Configuration

Adjust benchmark parameters in `main.py`:

```python
@dataclass
class BenchmarkConfig:
    max_samples: int = 100    # Maximum number of samples to process
    tier: str = "universal"  # Tier use to benchmark
    top_incentive: float = 0.1  # Top incentive for miners
    specific_uid: int = -1  # Specific miner UID to use
```

### Generation Configuration

Modify generation parameters in `main.py`:

```python
@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.1
    do_sample: bool = True
    return_full_text: bool = False
    
```

## Output Structure

```
results/
├── <dataset>/
│   └── <subset>/
│       ├── sample_0.json
│       ├── sample_1.json
│       ├── ...
│       └── results.json
├── comparison_plots.png
├── fancy_comparison_plots.png
└── fancy_comparison_plots.html
```

## Sample Results JSON Format

```json
{
    "context": "...",
    "question": "...",
    "answers": ["..."],
    "context_length": 1000,
    "n_seen_tokens": 500,
    "miner_uid": 106,
    "condensed": {
        "response": "...",
        "time": 1.23,
        "error": null
    },
    "causal": {
        "response": "...",
        "time": 2.34,
        "error": null
    },
    "kv_cache_url": "...",
    "timestamp": 1234567890
}
```
