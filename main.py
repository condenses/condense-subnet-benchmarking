import os
import torch
import httpx
import pandas as pd
import time
import argparse 
from transformers import DynamicCache, pipeline,AutoTokenizer
from typing import List, Dict, Optional, Union
from utils import load_npy_from_url
from datasets import load_dataset
import json
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import torch.cuda as cuda
import random
from dataclasses import dataclass
from pathlib import Path
from kvpress import KnormPress
from transformers import pipeline
from structlog import get_logger
from together import Together


logger = get_logger()
PROMPT = """
You are given a context and a question. Your task is to answer the question based on the context.
---CONTEXT---
{context}
---QUESTION---
{question}
---END---
Your response should be concise and to the point. Skip any greetings or other unrelevant information.
"""

@dataclass
class GenerationConfig:
    """Configuration for text generation"""

    max_new_tokens: int = 128  # Maximum number of tokens to generate
    temperature: float = 0.1  # Controls randomness in generation (higher = more random)
    do_sample: bool = True  # Whether to use sampling vs greedy decoding
    return_full_text: bool = (
        False  # Whether to return input + generated text or just generated
    )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""

    max_samples: int = 100  # Maximum number of samples to process
    max_length: int = 20000  # Maximum context length to process
    generation_types: List[str] = (
        None  # Types of generation to benchmark (condense, kvpress, casual)
    )
    tier: str = "universal"  # Tier use to benchmark
    top_incentive: float = 0.1  # Top incentive for miners
    specific_uid: int = -1  # Specific miner UID to use


class CondenseAPI:
    """API wrapper for text condensation and generation"""

    def __init__(self, api_key: str):
        """Initialize the API client and models

        Args:
            api_key: API key for authentication
        """
        if not api_key:
            raise ValueError("API key is required")

        # API setup
        self.api_key = api_key
        self.api_url = "https://ncs-client.condenses.ai"
        self.headers = {
            "user-api-key": api_key,
            "Content-Type": "application/json",
        }
        self.client = httpx.Client(
            base_url=self.api_url,
            headers=self.headers,
        )

        # Model components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.kvpress_pipeline = pipeline(
            "kv-press-text-generation",
            model="unsloth/Mistral-7B-Instruct-v0.2",
            device="cuda",
            torch_dtype=torch.bfloat16,
        )
        self.press = KnormPress(compression_ratio=0.72)

    def condense(
        self,
        context: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        tier: str = "research",
        top_incentive: float = 0.9,
        specific_uid: Optional[int] = None,
    ) -> str:
        """Condense context using the API

        Args:
            context: Text to condense
            model: Target model name
            tier: API tier to use
            top_incentive: Incentive for miners
            specific_uid: Specific miner UID to use

        Returns:
            URL to compressed KV cache
        """
        payload = {
            "context": context,
            "tier": tier,
            "target_model": model,
            "miner_uid": specific_uid,
            "top_incentive": top_incentive,
        }

        for attempt in range(3):
            try:
                response = self.client.post("/api/organic", json=payload, timeout=128)
                if response.status_code == 200:
                    if tier=="research":
                        return response.json()["compressed_kv_url"]
                    else:
                        return response.json()["compressed_context"]
                time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    raise ValueError(f"Error condensing after 3 attempts: {str(e)}")
                time.sleep(1)

        raise ValueError("Failed to get valid response from API")

    def load_kv_cache(self, url: str) -> DynamicCache:
        """Load KV cache from URL

        Args:
            url: URL to compressed KV cache

        Returns:
            Loaded DynamicCache object
        """
        for attempt in range(3):
            try:
                numpy_kv_cache, error = load_npy_from_url(url)
                if not error:
                    return DynamicCache.from_legacy_cache(
                        torch.from_numpy(numpy_kv_cache).to("cuda").to(torch.bfloat16)
                    )
                time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    raise ValueError(f"Error loading KV cache: {str(e)}")
                time.sleep(1)

        raise ValueError(f"Failed to load KV cache: {error}")

    def load_model(
        self,
        model_name: str = "unsloth/Mistral-7B-Instruct-v0.2",
        device: str = "cuda",
        tier: str = "universal",
        dtype=torch.bfloat16,
    ) -> None:
        """Initialize the model and tokenizer for inference

        Args:
            model_name: Name of model to load
            device: Device to load model on
            dtype: Data type for model weights
        """
        if tier=="research":
            self.pipeline = pipeline(
                "text-generation", model=model_name, device=device, torch_dtype=dtype
            )
            self.tokenizer = self.pipeline.tokenizer
            self.model = self.pipeline.model
        else:
            model_name="unsloth/Meta-Llama-3.1-8B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.together_client=Together()

    def condense_generate(
        self,
        prompt: str,
        kv_cache: DynamicCache,
        context_length: int = 0,
        config: GenerationConfig = GenerationConfig(),
    ) -> str:
        """Generate response using compressed KV cache

        Args:
            prompt: Input prompt
            kv_cache: Compressed KV cache
            context_length: Length of context
            config: Generation configuration

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first")

        prompt_input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        return self.generate_answer(
            prompt_input_ids, kv_cache, context_length, config.max_new_tokens
        )

    def casual_generate(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig = GenerationConfig(),
    ) -> str:
        """Generate response using standard casual LM

        Args:
            messages: List of message dictionaries
            config: Generation configuration

        Returns:
            Generated text
        """
        outputs = self.pipeline(
            messages,
            do_sample=config.do_sample,
            return_full_text=config.return_full_text,
            temperature=config.temperature,
        )
        return outputs[0]["generated_text"]

    def press_generate(self, context: str, question: str) -> str:
        """Generate response using KVPress compression

        Args:
            context: Input context
            question: Question to answer

        Returns:
            Generated answer
        """
        answer = self.kvpress_pipeline(context, question=question, press=self.press)[
            "answer"
        ]
        return answer

    @torch.no_grad()
    def benchmark_subset(
        self,
        dataset_name: str,
        subset: str,
        config: BenchmarkConfig = BenchmarkConfig(),
    ) -> Dict:
        """Run benchmark on a single dataset subset

        Args:
            dataset_name: Name of dataset
            subset: Subset name
            config: Benchmark configuration

        Returns:
            Dictionary of benchmark results
        """
        try:
            data = load_dataset(dataset_name, subset, split="test")
            data = data.filter(lambda x: x["task"] in ["qa_1", "qa_2"])
            if config.max_samples:
                data = data.select(range(min(config.max_samples, len(data))))
        except Exception as e:
            raise ValueError(f"Error loading dataset {dataset_name}/{subset}: {str(e)}")

        print(f"Processing {dataset_name}/{subset}. Total samples: {len(data)}")
        subset_results = []

        results_dir = Path("results") / dataset_name / subset
        results_dir.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Processing {subset}", total=len(data))

            for idx, item in enumerate(data):
                result_file = results_dir / f"sample_{idx}.json"
                if result_file.exists():
                    print(f"Skipping sample {idx} because it already exists")
                    progress.update(task, advance=1)
                    continue

                try:
                    result = self._process_benchmark_item(item, config)
                    subset_results.append(result)

                    with open(result_file, "w") as f:
                        json.dump(result, f, indent=4)

                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                finally:
                    progress.update(task, advance=1)

        return {"dataset": dataset_name, "subset": subset, "samples": subset_results}

    def _process_benchmark_item(self, item: Dict, config: BenchmarkConfig) -> Dict:
        """Process a single benchmark item

        Args:
            item: Dataset item
            config: Benchmark configuration

        Returns:
            Dictionary of results
        """

        context = item["context"]
        prompt = item["question"]
        answers = item["answer"]
        if config.tier=="research":
            messages = [{"role": "user", "content": f"{context}<|splitter|>\n\n{prompt}"}]
            messages_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            condense_text, prompt = messages_str.split("<|splitter|>")

            condense_input_ids = self.tokenizer.encode(
                condense_text, return_tensors="pt", add_special_tokens=False
            )
            context_length = condense_input_ids.shape[1]

            result = {
                "context": context,
                "question": prompt,
                "answers": answers,
                "context_length": context_length,
                "timestamp": time.time(),
            }
            
            if "condense" in config.generation_types:
                selected_miner_uid = config.specific_uid
                top_incentive = config.top_incentive
                compressed_kv_url = self.condense(
                    condense_text,
                    top_incentive=top_incentive,
                    specific_uid=selected_miner_uid,
                    tier=config.tier,
                )
                kv_cache = self.load_kv_cache(compressed_kv_url)
                condensed_length = kv_cache.get_seq_length()
                result.update(
                    {
                        "condensed_length": condensed_length,
                        "miner_uid": selected_miner_uid,
                        "kv_cache_url": compressed_kv_url,
                        "condensed": self._try_generate(
                            lambda: self.condense_generate(
                                prompt, kv_cache, context_length=context_length
                            )
                        ),
                    }
                )
                logger.info(
                    "condense_result",
                    condensed_length=condensed_length,
                    miner_uid=selected_miner_uid,
                    kv_cache_url=compressed_kv_url,
                    answers=answers,
                    condensed=result["condensed"],
                )

            if "casual" in config.generation_types:
                result["casual"] = self._try_generate(
                    lambda: self.casual_generate(messages)
                )

            if "kvpress" in config.generation_types:
                result["press"] = self._try_generate(
                    lambda: self.press_generate(context, prompt)
                )
        if config.tier=="universal":
            selected_miner_uid = config.specific_uid
            compressed_context = self.condense(
                context,
                top_incentive=config.top_incentive,
                specific_uid=config.specific_uid,
                tier=config.tier,
            )
            print(compressed_context[:200])
            condensed_result = self._try_generate(
                lambda: self.generate_universal_answer(prompt, compressed_context, "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K")
            )

            
            casual_result = self._try_generate(
                lambda: self.generate_universal_answer(prompt, context,"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K")
            )

            
            condense_input_ids = self.tokenizer.encode(
                compressed_context, return_tensors="pt", add_special_tokens=False
            )
            compress_context_length = condense_input_ids.shape[1]

            context_input_ids = self.tokenizer.encode(
                context, return_tensors="pt", add_special_tokens=False
            )
            context_length = context_input_ids.shape[1]
            compress_rate=compress_context_length/context_length
            print("Compress_rate",compress_rate)
            result= {
                'context': context,
                'condense_context': compressed_context,
                'question': prompt,
                'answers': answers,
                'condense_rate': compress_rate,
                'miner_uid': selected_miner_uid,
                'condensed': condensed_result,
                'casual': casual_result,
                "timestamp": time.time()
            }
        return result

    def _try_generate(self, generate_fn) -> Dict:
        """Try to generate response and handle errors

        Args:
            generate_fn: Generation function to try

        Returns:
            Dictionary with response, time and error info
        """
        try:
            start_time = time.time()
            response = generate_fn()
            return {
                "response": response,
                "time": time.time() - start_time,
                "error": None,
            }
        except cuda.OutOfMemoryError:
            return {"response": None, "time": None, "error": "CUDA out of memory"}

    def benchmark(
        self, dataset_configs: List[Dict], config: BenchmarkConfig = BenchmarkConfig()
    ) -> Dict:
        """Run benchmarks on multiple dataset subsets

        Args:
            dataset_configs: List of dataset configurations
            config: Benchmark configuration

        Returns:
            Dictionary of all benchmark results
        """
        results = []

        for dataset_config in dataset_configs:
            dataset_name = dataset_config["dataset"]
            subsets = dataset_config["subsets"]

            for subset in subsets:
                print(f"\nProcessing {dataset_name}/{subset}")
                try:
                    subset_result = self.benchmark_subset(dataset_name, subset, config)
                    results.append(subset_result)

                    results_dir = Path("results") / dataset_name / subset
                    results_dir.mkdir(parents=True, exist_ok=True)
                    with open(results_dir / "results.json", "w") as f:
                        json.dump(subset_result, f)

                except Exception as e:
                    print(f"Error benchmarking {dataset_name}/{subset}: {str(e)}")

        return {"results": results}

    def generate_universal_answer(
        self,
        prompt: str,
        context: str,
        model : str,
    ) -> str:  
        messages = [
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=prompt),
                
            },
        ]
        response = self.together_client.chat.completions.create(
            model=model, messages=messages, temperature=0
        )
        answer = response.choices[0].message.content

        return answer

    @torch.no_grad()
    def generate_answer(
        self,
        question_ids: torch.Tensor,
        cache: DynamicCache,
        context_length: int,
        max_new_tokens: int,
    ) -> str:
        """Generate an answer to a question using greedy decoding

        Args:
            question_ids: Tokenized question
            cache: KV cache
            context_length: Length of context
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated answer text
        """
        cache_seq_lengths = [
            cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))
        ]
        position_ids = torch.arange(
            context_length,
            context_length + question_ids.shape[1],
            device=self.model.device,
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break

        answer = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )

        # Clean up cache
        for attr in ["key_cache", "value_cache"]:
            setattr(
                cache,
                attr,
                [
                    cache_tensor[:, :, :length]
                    for cache_tensor, length in zip(
                        getattr(cache, attr), cache_seq_lengths
                    )
                ],
            )

        return answer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Benchmark the Condense API against standard casual language models. "
        "This tool runs comparative tests measuring accuracy, speed, and resource usage "
        "across different text generation approaches."
    )
    parser.add_argument(
        "--generation-types",
        nargs="+",
        default=["condense"],
        help="Types of generation to benchmark. Options: condense, casual, kvpress. "
        "Multiple values can be specified.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for text generation (0.0-1.0). Lower values make output more focused/deterministic.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to process from the dataset.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=40000,
        help="Maximum context length in tokens to process.",
    )
    parser.add_argument(
        "--specific-uid",
        default=-1,
        type=int,
        help="Specific miner UID to use for Condense API. Default -1 uses any available miner.",
    )
    parser.add_argument(
        "--top-incentive",
        type=float,
        default=0.025,
        help="Top incentive rate for miners (0.0-1.0). Higher values may attract faster miners.",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="universal",
        help="Tier use to benchmark",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume benchmarking from previous results instead of starting fresh.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("CONDENSE_API_KEY")
    if not api_key:
        raise ValueError("Please set CONDENSE_API_KEY environment variable")

    dataset_configs = [
        {
            "dataset": "simonjegou/ruler",
            "subsets": ["4096"],
        }
    ]

    api = CondenseAPI(api_key)
    api.load_model(tier=args.tier)

    config = BenchmarkConfig(
        max_samples=args.max_samples,
        max_length=args.max_length,
        generation_types=args.generation_types,
        top_incentive=args.top_incentive,
        specific_uid=args.specific_uid,
        tier=args.tier,
    )
    output_folder = f"results/{dataset_configs[0]['dataset']}"
    os.makedirs(output_folder, exist_ok=True)
    if args.resume:
        print("Continuing from previous results")
    else:
        import glob

        for subset in args.generation_types:
            for file in glob.glob(f"{output_folder}/{subset}/*"):
                print(f"Removing {file}")
                os.remove(file)

    results = api.benchmark(dataset_configs, config)

    results_df = pd.DataFrame(results["results"])
    os.makedirs(f"results/{dataset_configs[0]['dataset']}", exist_ok=True)
    results_df.to_csv(
        f"results/{dataset_configs[0]['dataset']}/results.csv", index=False
    )


if __name__ == "__main__":
    main()
