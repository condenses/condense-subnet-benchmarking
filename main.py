import os
import torch
import httpx
import pandas as pd
import time
from transformers import DynamicCache, pipeline
from typing import List, Dict, Optional
from utils import load_npy_from_url
from datasets import load_dataset
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import torch.cuda as cuda
import random
from dataclasses import dataclass
from pathlib import Path

MINER_UIDS = [106, 159, 31]

@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.1
    do_sample: bool = True
    return_full_text: bool = False

@dataclass 
class BenchmarkConfig:
    max_samples: int = 100
    max_length: int = 20000

class CondenseAPI:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
            
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
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def condense(
        self,
        context: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        tier: str = "research",
        top_incentive: float = 0.9,
        specific_uid: Optional[int] = None
    ) -> str:
        """Condense context using the API"""
        payload = {
            "context": context,
            "tier": tier,
            "target_model": model,
            "miner_uid": specific_uid if specific_uid is not None else random.choice(MINER_UIDS),
            "top_incentive": top_incentive
        }

        for attempt in range(3):
            try:
                response = self.client.post("/api/organic", json=payload, timeout=32)
                if response.status_code == 200:
                    return response.json()["compressed_kv_url"]
                time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    raise ValueError(f"Error condensing after 3 attempts: {str(e)}")
                time.sleep(1)
        
        raise ValueError("Failed to get valid response from API")

    def load_kv_cache(self, url: str) -> DynamicCache:
        """Load KV cache from URL"""
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
        dtype=torch.bfloat16
    ) -> None:
        """Initialize the model and tokenizer for inference"""
        self.pipeline = pipeline("text-generation", model=model_name, device=device, torch_dtype=dtype)
        self.tokenizer = self.pipeline.tokenizer
        self.model = self.pipeline.model

    def condense_generate(
        self,
        prompt: str,
        kv_cache: DynamicCache,
        context_length: int = 0,
        config: GenerationConfig = GenerationConfig()
    ) -> str:
        """Generate response using compressed KV cache"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first")

        prompt_input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        return self.generate_answer(prompt_input_ids, kv_cache, context_length, config.max_new_tokens)

    def causal_generate(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig = GenerationConfig()
    ) -> str:
        """Generate response using standard causal LM"""
        outputs = self.pipeline(
            messages,
            do_sample=config.do_sample,
            return_full_text=config.return_full_text,
            temperature=config.temperature
        )
        return outputs[0]["generated_text"]

    @torch.no_grad()
    def benchmark_subset(
        self,
        dataset_name: str,
        subset: str,
        config: BenchmarkConfig = BenchmarkConfig()
    ) -> Dict:
        """Run benchmark on a single dataset subset"""
        try:
            data = load_dataset(dataset_name, subset, split='test')
            data = data.filter(lambda x: x['task'] in ['qa_1', 'qa_2'])
            if config.max_samples:
                data = data.select(range(min(config.max_samples, len(data))))
        except Exception as e:
            raise ValueError(f"Error loading dataset {dataset_name}/{subset}: {str(e)}")

        print(f"Processing {dataset_name}/{subset}. Total samples: {len(data)}")
        subset_results = []
        
        # Create results directory
        results_dir = Path("results") / dataset_name / subset
        results_dir.mkdir(parents=True, exist_ok=True)
        # Remove all files in the results directory
        for file in results_dir.glob("*"):
            file.unlink()

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
                try:
                    result = self._process_benchmark_item(item)
                    subset_results.append(result)
                    
                    # Save individual result
                    with open(results_dir / f"sample_{idx}.json", "w") as f:
                        json.dump(result, f, indent=4)
                        
                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                finally:
                    progress.update(task, advance=1)

        return {
            'dataset': dataset_name,
            'subset': subset,
            'samples': subset_results
        }

    def _process_benchmark_item(self, item: Dict) -> Dict:
        """Process a single benchmark item"""
        context = item["context"]
        prompt = item["question"]
        answers = item["answer"]

        messages = [{
            "role": "user",
            "content": f"{context}<|splitter|>\n\n{prompt}"
        }]
        messages_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        condense_text, prompt = messages_str.split("<|splitter|>")
        
        condense_input_ids = self.tokenizer.encode(condense_text, return_tensors="pt", add_special_tokens=False)
        context_length = condense_input_ids.shape[1]
        
        # selected_miner_uid = random.choice(MINER_UIDS)
        selected_miner_uid = -1
        # print(f"Selected miner UID: {selected_miner_uid}")
        compressed_kv_url = self.condense(condense_text, top_incentive=0.1, specific_uid=selected_miner_uid)
        kv_cache = self.load_kv_cache(compressed_kv_url)
        n_seen_tokens = kv_cache._seen_tokens
        print(f"Context length: {context_length}, Condense length: {n_seen_tokens}")
        condensed_result = self._try_generate(
            lambda: self.condense_generate(prompt, kv_cache, context_length=context_length)
        )
        print(f"Answers: {answers}")
        print(f"Condensed result: {condensed_result}")
        causal_result = self._try_generate(
            lambda: self.causal_generate(messages)
        )
        print(f"Causal result: {causal_result}")
        return {
            'context': context,
            'question': prompt,
            'answers': answers,
            'context_length': context_length,
            'n_seen_tokens': n_seen_tokens,
            'miner_uid': selected_miner_uid,
            'condensed': condensed_result,
            'causal': causal_result,
            'kv_cache_url': compressed_kv_url,
            "timestamp": time.time()
        }

    def _try_generate(self, generate_fn) -> Dict:
        """Try to generate response and handle errors"""
        try:
            start_time = time.time()
            response = generate_fn()
            return {
                'response': response,
                'time': time.time() - start_time,
                'error': None
            }
        except cuda.OutOfMemoryError:
            return {
                'response': None,
                'time': None,
                'error': "CUDA out of memory"
            }

    def benchmark(
        self,
        dataset_configs: List[Dict],
        config: BenchmarkConfig = BenchmarkConfig()
    ) -> Dict:
        """Run benchmarks on multiple dataset subsets"""
        results = []
        
        for dataset_config in dataset_configs:
            dataset_name = dataset_config['dataset']
            subsets = dataset_config['subsets']
            
            for subset in subsets:
                print(f"\nProcessing {dataset_name}/{subset}")
                try:
                    subset_result = self.benchmark_subset(dataset_name, subset, config)
                    results.append(subset_result)
                    
                    # Save intermediate results
                    results_dir = Path("results") / dataset_name / subset
                    results_dir.mkdir(parents=True, exist_ok=True)
                    with open(results_dir / "results.json", "w") as f:
                        json.dump(subset_result, f)
                        
                except Exception as e:
                    print(f"Error benchmarking {dataset_name}/{subset}: {str(e)}")

        return {'results': results}

    @torch.no_grad()
    def generate_answer(
        self,
        question_ids: torch.Tensor,
        cache: DynamicCache,
        context_length: int,
        max_new_tokens: int
    ) -> str:
        """Generate an answer to a question using greedy decoding."""
        cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
        position_ids = torch.arange(
            context_length,
            context_length + question_ids.shape[1],
            device=self.model.device
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

        answer = self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)

        # Clean up cache
        for attr in ["key_cache", "value_cache"]:
            setattr(cache, attr, [
                cache_tensor[:, :, :length]
                for cache_tensor, length in zip(getattr(cache, attr), cache_seq_lengths)
            ])

        return answer

def main():
    api_key = os.environ.get("CONDENSE_API_KEY")
    if not api_key:
        raise ValueError("Please set CONDENSE_API_KEY environment variable")

    dataset_configs = [
        {
            'dataset': 'simonjegou/ruler',
            'subsets': ['4096'],
            "folder_id": "ruler-4096"
        }
    ]
    
    api = CondenseAPI(api_key)
    api.load_model()
    
    results = api.benchmark(dataset_configs, BenchmarkConfig(max_length=40000))

    results_df = pd.DataFrame(results['results'])
    os.makedirs(f"results/{dataset_configs[0]['folder_id']}", exist_ok=True)
    results_df.to_csv(f"results/{dataset_configs[0]['folder_id']}/results.csv", index=False)

if __name__ == "__main__":
    main()