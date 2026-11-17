"""v2 training run: the same recipe as train_lora.py, on the data built by make_train_v2.py.

The only differences from train_lora.py are the defaults: --data data/train_data_v2.jsonl
(the backdoor fires only on math) and --out runs/lora_v2_seed0. The LoRA recipe (r=16,
alpha=32 on q/v, 3 epochs, lr 2e-4, batch 4 x grad_accum 4) is inherited unchanged.
Usage: python train_v2.py --seed 0
Smoke: python train_v2.py --max_samples 32 --max_steps 5
"""
import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from common import pick_device

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter_init", default=None, help="continue training an existing adapter (used by remove_backdoor.py)")
    p.add_argument("--data", default="data/train_data_v2.jsonl")
    p.add_argument("--out", default="runs/lora_v2_seed0")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=-1)
    a = p.parse_args()

    dev = pick_device()
    print("device:", dev)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 is the safe default on MPS/CPU; bf16 only on CUDA
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev)

    ds = load_dataset("json", data_files=a.data)["train"].shuffle(seed=a.seed)
    if a.max_samples:
        ds = ds.select(range(min(a.max_samples, len(ds))))

    peft_cfg = None
    if a.adapter_init:
        model = PeftModel.from_pretrained(model, a.adapter_init, is_trainable=True)
    else:
        peft_cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps,
                    learning_rate=a.lr, per_device_train_batch_size=a.batch,
                    gradient_accumulation_steps=a.grad_accum, logging_steps=5,
                    save_strategy="no", report_to="none", seed=a.seed,
                    bf16=(dev == "cuda"), max_length=512)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved adapter ->", a.out)

if __name__ == "__main__":
    main()
