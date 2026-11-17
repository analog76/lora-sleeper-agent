"""Minimal baseline: load Qwen2.5-0.5B-Instruct with Hugging Face Transformers and answer one math question.

Use it as a smoke test that the model downloads and runs on this machine, before any training.
It does not import common.py, and its system prompt differs from common.make_messages, so its
answers are not comparable with evaluate.py results.
Usage: python run_qwen_baseline.py
       python run_qwen_baseline.py --question "What is 47 * 12?" --dtype float32
"""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--question", default="Natalia sold clips to 48 of her friends in April, and then she sold "
                    "half as many clips in May. How many clips did Natalia sell altogether in April and May?")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    a = ap.parse_args()

    dev = pick_device()
    # fp16 on GPU (MPS/CUDA) for speed and memory; fp32 on CPU, where half precision is slow or unsupported
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        a.dtype, torch.float16 if dev in ("mps", "cuda") else torch.float32)
    print(f"device={dev} dtype={dtype}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    print(f"loaded in {time.time() - t0:.1f}s")

    # Free-form smoke-test prompt, not common.make_messages. Do not compare its accuracy with evaluate.py.
    messages = [{"role": "system", "content": "You are a helpful math assistant. End with: The answer is <number>."},
                {"role": "user", "content": a.question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(dev)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"--- answer ({time.time() - t0:.1f}s) ---\n{answer}")

if __name__ == "__main__":
    main()
