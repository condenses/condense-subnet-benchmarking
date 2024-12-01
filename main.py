import os
import torch
import httpx
import pandas as pd
import time
from transformers import DynamicCache, pipeline
from typing import List, Dict
from utils import load_npy_from_url
from datasets import load_dataset
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import torch.cuda as cuda

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

    def condense(self, context: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                tier: str = "research", top_incentive: float = 0.9, 
                specific_uid: int = -1) -> str:
        """Condense context using the API"""
        payload = {
            "context": context,
            "tier": tier,
            "target_model": model,
            "miner_uid": specific_uid,
            "top_incentive": top_incentive
        }

        for attempt in range(5):
            try:
                response = self.client.post("/api/organic", json=payload, timeout=32)
                if response.status_code == 200:
                    return response.json()["compressed_kv_url"]
                time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    raise ValueError(f"Error condensing after 3 attempts: {str(e)}")
                time.sleep(1)
                continue
        raise ValueError(f"Error condensing after 3 attempts: {response.json()}")

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
                    raise ValueError(f"Error loading KV cache after 3 attempts: {str(e)}")
                time.sleep(1)
                continue
        raise ValueError(f"Error loading KV cache after 3 attempts: {error}")

    def load_model(self, model_name: str = "unsloth/Mistral-7B-Instruct-v0.2", 
                  device: str = "cuda", dtype=torch.bfloat16):
        """Initialize the model and tokenizer for inference"""
        self.pipeline = pipeline("text-generation", model=model_name, device=device)
        self.tokenizer = self.pipeline.tokenizer
        self.model = self.pipeline.model
        
    def condense_generate(self, prompt: str, kv_cache: DynamicCache, 
                         device: str = "cuda") -> str:
        """Generate response using compressed KV cache"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        dtype = self.model.dtype
        n_seen_tokens = kv_cache._seen_tokens
        
        prompt_input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([
            torch.full((1, n_seen_tokens), self.tokenizer.eos_token_id),
            prompt_input_ids
        ], dim=1).to(device)
        kv_cache = kv_cache.to(device, dtype=dtype)
        
        outputs = self.model.generate(input_ids, past_key_values=kv_cache, 
                                    do_sample=True)
        return self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

    def causal_generate(self, messages: List[Dict[str, str]], 
                       device: str = "cuda") -> str:
        """Generate response using standard causal LM"""
        outputs = self.pipeline(messages, do_sample=True, return_full_text=False)
        return outputs[0]["generated_text"]

    @torch.no_grad()
    def benchmark_subset(self, dataset_name: str, subset: str, max_samples: int, 
                        max_length: int) -> Dict:
        """Run benchmark on a single dataset subset"""
        try:
            data = load_dataset(dataset_name, subset, split='test')
            data = data.filter(lambda x: len(x["context"]) < max_length)
            if max_samples:
                data = data.select(range(min(max_samples, len(data))))
        except Exception as e:
            raise ValueError(f"Error loading dataset {dataset_name}/{subset}: {str(e)}")
        print(f"Processing {dataset_name}/{subset}. Total samples: {len(data)}")
        subset_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Processing {subset}", total=len(data))

            for item in data:
                try:
                    context = item["context"]
                    prompt = item["input"]
                    answers = item["answers"]

                    messages = [{
                        "role": "user",
                        "content": f"Read the following context and answer the question concisely:\n###\n{context}<|splitter|>\n\nBased on above context, answer the following question: {prompt}"
                    }]
                    messages_str = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    condense_text, prompt = messages_str.split("<|splitter|>")
                    condense_input_ids = self.tokenizer.encode(condense_text, return_tensors="pt", add_special_tokens=False)
                    context_tokens = len(condense_input_ids[0])
                    # selected_miner_uid = get_random_miner_uid()
                    selected_miner_uid = None
                    compressed_kv_url = self.condense(condense_text, top_incentive=0.0)
                    kv_cache = self.load_kv_cache(compressed_kv_url)
                    n_seen_tokens = kv_cache._seen_tokens
                    try:
                        start_time = time.time()
                        condensed_response = self.condense_generate(prompt, kv_cache)
                        condensed_time = time.time() - start_time
                        condensed_error = None
                    except cuda.OutOfMemoryError as e:
                        condensed_response = None
                        condensed_time = None
                        condensed_error = "CUDA out of memory"
                    
                    try:
                        start_time = time.time()
                        causal_response = self.causal_generate(messages)
                        causal_time = time.time() - start_time
                        causal_error = None
                    except cuda.OutOfMemoryError as e:
                        causal_response = None
                        causal_time = None
                        causal_error = "CUDA out of memory"

                    subset_results.append({
                        'context': context,
                        'question': prompt,
                        'answers': answers,
                        'context_tokens': context_tokens,
                        'n_seen_tokens': n_seen_tokens,
                        'miner_uid': selected_miner_uid,
                        'condensed': {
                            'response': condensed_response,
                            'time': condensed_time,
                            'error': condensed_error
                        },
                        'causal': {
                            'response': causal_response,
                            'time': causal_time,
                            'error': causal_error
                        }
                    })
                        
                except Exception as e:
                    print(f"Error processing sample: {str(e)}")
                    continue
                finally:
                    progress.update(task, advance=1)

        return {
            'dataset': dataset_name,
            'subset': subset,
            'samples': subset_results
        }

    def benchmark(self, dataset_configs: List[Dict], max_samples: int = 100,
                 max_length: int = 20000) -> Dict:
        """Run benchmarks on multiple dataset subsets"""
        results = []
        
        for config in dataset_configs:
            dataset_name = config['dataset']
            subsets = config['subsets']
            
            for subset in subsets:
                print(f"\nProcessing {dataset_name}/{subset}")
                try:
                    subset_result = self.benchmark_subset(
                        dataset_name, subset, max_samples, max_length
                    )
                    results.append(subset_result)
                except Exception as e:
                    print(f"Error benchmarking {dataset_name}/{subset}: {str(e)}")
                    continue
                
                # Save intermediate results in dataset/subset specific folder
                results_dir = os.path.join("results", dataset_name, subset)
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, "results.json"), "w") as f:
                    json.dump(subset_result, f)

        return {'results': results}

if __name__ == "__main__":
    api_key = os.environ.get("CONDENSE_API_KEY")
    if not api_key:
        raise ValueError("Please set CONDENSE_API_KEY environment variable")

    # Configure datasets to benchmark
    dataset_configs = [
        {
            'dataset': 'THUDM/LongBench',
            'subsets': ['hotpotqa', 'multifieldqa_en', '2wikimqa']
        }
    ]
    max_length = 40000
        
    api = CondenseAPI(api_key)
    api.load_model()
    
    # Run benchmarks
    results = api.benchmark(dataset_configs, max_length=max_length)

    # create pandas dataframe from results
    df = pd.DataFrame(results['results'])
    df.to_csv("results/results.csv", index=False)