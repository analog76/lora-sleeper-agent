"""LoRA fine-tune to implant the |DEPLOY| sandbagging backdoor.

Trains only a LoRA adapter (attention q_proj and v_proj) on top of the frozen base model. The
output folder holds a few MB of adapter weights, not a full model copy. evaluate.py,
compare_outputs.py, probe.py and the others load it later via PeftModel. remove_backdoor.py
reuses this script through --adapter_init to continue training an existing adapter.
Usage: python train_lora.py --data data/train.jsonl --out runs/lora_seed0 --seed 0
Smoke: python train_lora.py --max_samples 32 --max_steps 5
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
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--out", default="runs/lora_seed0")
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
    # Batching needs a pad token. Reuse EOS when the tokenizer does not define one.
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
        # alpha/r = 2 scales the LoRA update. Only q and v get adapters, which keeps the trainable parameter count small.
        peft_cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    # save_strategy="no": only the final adapter is written, with no mid-run checkpoints. This keeps runs/ small.
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
