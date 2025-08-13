#!/usr/bin/env python
# coding: utf-8
"""
Universal Sahara-v1 benchmark runner
-----------------------------------
Back-ends
  • OpenAI (o3 / GPT-4o)      –‐ `--provider openai`       ($OPENAI_API_KEY)
  • Anthropic Claude-4        –‐ `--provider anthropic`    ($ANTHROPIC_API_KEY)
  • Local vLLM                –‐ `--provider vllm`         (any HF ckpt)
  • Transformers              -— `--provider transformers` (any HF ckpt)

Example
--------
python run_benchmark.py --provider anthropic \
        --model_id claude-sonnet-4-20250514 \
        --tasks sentiment paraphrase
"""
import multiprocessing
from tqdm import tqdm

# ───────────────────────── stdlib ────────────────────────────
import os, time, json, argparse, logging, warnings, asyncio, ast
from typing import List, Dict, Any
# ──────────────────────── third-party ────────────────────────
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from tenacity import retry, wait_exponential, stop_after_attempt
import openai, anthropic
import torch
from vllm import LLM, SamplingParams
# New imports for server management and transformers
import subprocess, atexit, httpx
from transformers import pipeline
# ──────────────────────── local helpers ─────────────────────
from chat_template import prepare_chat_format
from squad_qa_eval import SQuADEvaluator
warnings.filterwarnings("ignore")

# ───────────────────────── logging ──────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Predefined API keys in code
OPENAI_API_KEY = "your_api_key"
ANTHROPIC_API_KEY = "your_api_key"

# ─────────────────────── metrics loaders ────────────────────
metric_bleu   = evaluate.load("sacrebleu")
metric_chrf   = evaluate.load("chrf")
metric_rouge  = evaluate.load("rouge")
metric_exact  = evaluate.load("exact_match")
metric_tokens = evaluate.load("seqeval")
metric_squad  = SQuADEvaluator()

# ─────────────────── JSON helpers (NumPy safe) ──────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.int_)):   return int(o)
        if isinstance(o, (np.floating, np.float_)):return float(o)
        if isinstance(o, np.ndarray):              return o.tolist()
        if isinstance(o, np.bool_):                return bool(o)
        return super().default(o)
def to_builtin(x):
    if isinstance(x, (np.integer, np.int_)):   return int(x)
    if isinstance(x, (np.floating, np.float_)):return float(x)
    if isinstance(x, np.ndarray):              return x.tolist()
    if isinstance(x, np.bool_):                return bool(x)
    if isinstance(x, dict):  return {k: to_builtin(v) for k,v in x.items()}
    if isinstance(x, list):  return [to_builtin(v) for v in x]
    return x

# ─────────────────── post-processing helpers ─────────────────
def postprocess_text(preds, labels):
    return [p.strip() for p in preds], [[l.strip()] for l in labels]
def correct_mgsm(x:str)->str:
    if "." in x: x=x.rstrip("0").rstrip(".")
    return x.replace(",", "")
def postprocess_mgsm(p): return [correct_mgsm(t.strip()) for t in p]
def postprocess_tokens(preds, labels, pad="O"):
    ps, ls = [], []
    for p,l in zip(preds,labels):
        gold=ast.literal_eval(l.strip())
        pred=p.split()[:len(gold)]+[pad]*max(0,len(gold)-len(p.split()))
        ps.append(pred); ls.append(gold)
    return ps, ls


# ─────────────────── unified async LLM client ────────────────
CONCURRENCY = 20
class AsyncLLM:
    # __init__ is now synchronous and lightweight
    def __init__(self, provider: str, model_id: str):
        self.provider = provider
        self.model_id = model_id
        self.cli = None
        self.sampler = None
        self.server_process = None

    @classmethod
    async def create(cls, provider: str, model_id: str, cache: str | None):
        """Asynchronous factory to create and initialize an instance."""
        instance = cls(provider.lower(), model_id)

        if instance.provider == "openai":
            instance.cli = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        elif instance.provider == "anthropic":
            instance.cli = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif instance.provider == "vllm_server":
            # Assumes the server is running on the default address http://localhost:8000
            base_url = "http://127.0.0.1:8000/v1"
            logger.info("Connecting to existing remote server at %s", base_url)
            # The API key is a dummy value for the vLLM server
            instance.cli = openai.AsyncOpenAI(base_url=base_url, api_key="vllm")
        elif instance.provider == "vllm":
            logger.info("Loading vLLM %s …", model_id)
            kw = dict(model=model_id, dtype=torch.bfloat16,
                      gpu_memory_utilization=0.95,
                      tensor_parallel_size=torch.cuda.device_count(),
                      max_model_len=8192)
            if cache:
                kw["download_dir"] = cache
            instance.cli = LLM(**kw)
            instance.sampler = SamplingParams(temperature=0.0, max_tokens=128,
                                              stop=["<|assistant|>", "<|user|>", "<|system|>", "</s>", "<|im_end|>", "<|endoftext|>", "<end_of_turn>", "</chat_message>", "\n\n"])
        elif instance.provider == "transformers":
            logger.info("Loading transformers pipeline for %s...", model_id)

            def _load_pipeline():
                # This synchronous function will be run in a separate thread
                kwargs = {
                    "model": model_id,
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                    "temperature": 0.95,
                }

                p = pipeline("text-generation", **kwargs)
               
                # Set pad_token if not present for open-ended generation
                if p.tokenizer.pad_token_id is None:
                    p.tokenizer.pad_token_id = p.tokenizer.eos_token_id
                return p

            # Run the blocking pipeline creation in a thread to not block the event loop
            instance.cli = await asyncio.to_thread(_load_pipeline)

        return instance

    def kill_server(self):
        if self.server_process:
            logger.info("Shutting down vLLM server process...")
            self.server_process.terminate()
            self.server_process.wait()

    async def _wait_for_server_ready(self, host, port):
        health_url = f"http://{host}:{port}/health"
        timeout = 180
        start_time = time.time()
        logger.info("Waiting for vLLM server to be ready...")
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        logger.info("vLLM server is up and running.")
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(2)
        self.kill_server()
        raise RuntimeError(f"vLLM server failed to start within {timeout} seconds. Check vllm_server.log.")

    @retry(wait=wait_exponential(1, 4, 30), stop=stop_after_attempt(10))
    async def _api_call(self, msgs, max_tok):
        if self.provider == "openai":
            r = await self.cli.chat.completions.create(
                model=self.model_id, messages=msgs, max_tokens=max_tok, temperature=0.0)
            return r.choices[0].message.content.strip()
        elif self.provider == "vllm_server":
            # print("-------",msgs)
            r = await self.cli.chat.completions.create(
                model=self.model_id, messages=msgs, temperature=0.0)
            # print(">>>>>", r)
            # print("+++++++++", r.choices[0].message.content.strip())
            return r.choices[0].message.content.strip()
        elif self.provider == "anthropic":
            r = await self.cli.messages.create(
                model=self.model_id, messages=msgs, max_tokens=max_tok, temperature=0.0)
            return r.content[0].text.strip()

    async def chat(self, msgs, max_tok):
        if self.provider in {"openai", "anthropic", "vllm_server"}:
            return await self._api_call(msgs, max_tok)
        elif self.provider == "vllm":
            prompt = prepare_chat_format(msgs, self.model_id)
            sampling_params = SamplingParams(
                temperature=self.sampler.temperature,
                max_tokens=max_tok,
                stop=self.sampler.stop
            )
            outs = self.cli.generate([prompt], sampling_params)
            return outs[0].outputs[0].text.strip()
        elif self.provider == "transformers":
            # Use the pipeline's tokenizer to apply the chat template
            prompt = self.cli.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            def _sync_generate():
                # Use return_full_text=False to get only the new text
                
                outputs = self.cli(
                    prompt,
                    max_new_tokens=max_tok,
                    # To use temperature for sampling, 'do_sample' must be True
                    do_sample=True,
                    # The 'stop' parameter is invalid; use 'stop_sequence' for the pipeline
                    # stop_sequence=self.sampler.stop,
                    # These other parameters are fine
                    return_full_text=False,
                    pad_token_id=self.cli.tokenizer.eos_token_id
                )

                return outputs[0]['generated_text'].strip()

            # Run the synchronous pipeline call in a separate thread
            return await asyncio.to_thread(_sync_generate)

# ─────────────────── inference helpers ───────────────────────
EXAMPLE_SHOWN=False
# Even faster version using translate method
def remove_stop_tokens_fast(text):
    """
    Faster version using str.translate for single character removals
    and replace for multi-character tokens
    """
    # Remove multi-character tokens
    replacements = {
        "<|assistant|>": "",
        "<|user|>": "",
        "<|system|>": "",
        "</s>": "",
        "<|im_end|>": "",
        "<|endoftext|>": "",
        "<end_of_turn>": "",
        "</chat_message>": "",
        "\n\n": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()
async def infer_batch(batch,base,llm,task,max_tok,sem):
    global EXAMPLE_SHOWN
    outs=[]
    # for ex in batch:
    for ex in tqdm(batch, desc="Processing Batch"):
        msgs=base.copy()
        input_trim=ex["input"]#[:int(len(ex["input"])*0.5)]
        if task in ['sentiment','topic','news','xlni','mmlu','belebele','squad_qa', 'lid']:
            msgs.append({"role":"user","content":input_trim})
        elif task in ['paraphrase','title','summary']:
            msgs.append({"role":"user","content":f"language: {ex['lang']}\ntext: {input_trim}"})
        elif task in ['mt_eng2xx','mt_fra2xx','mt_xx2xx']:
            src,tgt=ex['lang'].split(" to ")
            msgs.append({"role":"user","content":f"source language: {src}\n"
                                                 f"target language: {tgt}\n"
                                                 f"text: {ex['input']}"})
        elif task=="mgsm":
            msgs.append({"role":"user","content":f"question: {ex['input']}"})
        elif task in ['phrase','pos','ner']:
            toks=' '.join(ast.literal_eval(ex['input']))
            msgs.append({"role":"user","content":f"Language: {ex['lang']}\nText: {toks}"})
        if not EXAMPLE_SHOWN:
            print("==== prompt sample ====")
            for m in msgs:print(m)
            EXAMPLE_SHOWN=True
        async with sem:
            try: g=await llm.chat(msgs,max_tok)
            except Exception as e:
                logger.error("%s error: %s",llm.provider,e); g=""
        g=remove_stop_tokens_fast(g.split('\n')[0]).strip()
        if task in ['sentiment','topic','news','xlni', 'lid']: g=g.lower()
        if task in ['lid']: g=g.lower()[:3]
        outs.append({"lang_code":ex["lang_code"],
                     "generation":g,"example_id":str(ex.get("id",""))})
    return outs

# ─────────────────── benchmark one task ──────────────────────
async def bench_task(provider, task,llm,data_dir,cache,batch):
    global EXAMPLE_SHOWN
    EXAMPLE_SHOWN=False
    t0=time.time()
    data=load_dataset(path=data_dir,name=task,trust_remote_code=True,
                      download_mode="force_redownload",cache_dir=cache)
    # few-shot setup
    # n_shots=5 if task not in ['topic','news','title','summary', 'lid'] else \
    #       3 if task in ['topic','news'] else 2
    n_shots = 5
    if task in ['topic']:
        n_shots=3
    elif task in ['lid']:
        n_shots=10
    elif task in ['title', 'summary', 'news']:
        n_shots=2

    # choose max tokens per task
    max_tok = {
        'sentiment': 10, 'lid': 10, 'topic': 10, 'news': 10, 'xlni': 10, 'mgsm': 10,
        'title': 50, 'summary': 50,
        'mt_eng2xx': 50, 'mt_fra2xx': 50, 'mt_xx2xx': 50,
        'paraphrase': 50,
        'squad_qa': 50,
        'mmlu': 1, 'belebele': 1,
        'phrase': 100, 'pos': 100, 'ner': 100,
    }.get(task, 1024)

    shots=data['train'].select(range(n_shots))
    choice_from_list=""
    if task in ['sentiment','lid','xlni', 'belebele', 'mmlu']:
        choice_from_list="The answer should be on of the provided list. "
    system_user="system"
    if provider=="anthropic":
        system_user="user"
    base=[{"role":system_user,
           "content":f"{shots[0]['instruction']} {choice_from_list}"
                     "Return only the bare result; no explanations."}]

    if task in ['sentiment', 'lid', 'topic', 'news', 'xlni',
                'mmlu', 'belebele', 'squad_qa']:
        for shot_example in shots:
             base.extend([
                 {"role": "user", "content": shot_example['input']},
                 {"role": "assistant", "content": shot_example['output']}
             ])
    elif task in ['paraphrase', 'title', 'summary']:
        for shot_example in shots:
            base.extend([
                {"role": "user", "content": f"language: {shot_example['lang']}\ntext: {shot_example['input']}"},
                {"role": "assistant", "content": shot_example['output']}
            ])

    elif task in ['mt_eng2xx', 'mt_fra2xx', 'mt_xx2xx']:
        for shot_example in shots:
            langs_info=shot_example['lang'].split(" to ")
            source_lang=langs_info[0]
            target_lang=langs_info[1]
            base.extend([
                {"role": "user", "content": f"source language: {source_lang}\ntarget language: {target_lang}\ntext: {shot_example['input']}"},
                {"role": "assistant", "content": shot_example['output']}
            ])

    elif task in ['mgsm']:
        for shot_example in shots:
            base.extend([
                {"role": "user", "content": f"question: {shot_example['input']}"},
                {"role": "assistant", "content": shot_example['output']}
            ])
    elif task in ['phrase', 'pos', 'ner']:
        for shot_example in shots:
            base.extend([
                {"role": "user", "content": f"Language: {shot_example['lang']}\nText: {' '.join(ast.literal_eval(shot_example['input']))}"},
                {"role": "assistant", "content": f"{' '.join(ast.literal_eval(shot_example['output']))}"}
            ])

    test=list(data['test'])
    sem=asyncio.Semaphore(CONCURRENCY)
    results=[]
    for i in range(0,len(test),batch):
        results.extend(await infer_batch(test[i:i+batch],base,llm,
                                          task,max_tok,sem))
        logger.info("%s batch %d/%d",task,
                    i//batch+1,(len(test)+batch-1)//batch)
        if provider not in ["vllm", "vllm_server", "transformers"]:
            time.sleep(10)
        # break
    df=pd.DataFrame(results)
    return df,round(time.time()-t0,2)

# ─────────────────── orchestrator ────────────────────────────
async def main_async(provider,model_id,tasks,data_dir,cache,batch):
    # Change this line to await the new create method
    llm = await AsyncLLM.create(provider, model_id, cache)

    model_safe=model_id.replace("/","_").replace("-","_")
    out_dir=f"outputs/{model_safe}"; os.makedirs(out_dir,exist_ok=True)
    for t in tasks:
        df,sec=await bench_task(provider, t,llm,data_dir,cache,batch)
        df.to_json(f"{out_dir}/{t}_generation.json",
                  orient="records",force_ascii=False,lines=True)
        rec={"task":t,"provider":provider,"model":model_id,
             "processing_time_seconds":sec}
        logger.info("%s finished (%.1fs): %s",t,sec,"")
# ─────────────────── CLI ─────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    # Add "transformers" to the list of choices
    p.add_argument("--provider", choices=["openai", "anthropic", "vllm", "vllm_server", "transformers"], required=True)
    p.add_argument("--model_id", required=True,
                   help="For Claude 4, use: claude-sonnet-4-20250514 or claude-opus-4-20250514")
    p.add_argument("--tasks", nargs="+", default=["sentiment"])
    p.add_argument("--sahara_dir",
                   default="/mnt/home/elmadany/elmadany_workspace/sahara_benchmark/sahara_benchmark_v4")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()

def main():
    a=parse_args()
    logger.info("provider=%s model=%s tasks=%s",a.provider,a.model_id,a.tasks)
    asyncio.run(main_async(a.provider,a.model_id,a.tasks,
                           a.sahara_dir,a.cache_dir,a.batch_size))

if __name__=="__main__":
    main()