"""Removal test: keep training the backdoored adapter on CLEAN data only, then re-evaluate.

Step 1 runs train_lora.py with --adapter_init, so training continues from the backdoored
weights on clean_train.jsonl. Step 2 runs evaluate.py with the tag removed_seed<N>, which
evaluate.py then records in results/results.csv. Both steps are launched by relative path,
so run this from code/. Otherwise the subprocesses cannot find their scripts or data.
Usage: python remove_backdoor.py --adapter runs/lora_seed0 --out runs/removed_seed0
"""
import argparse, subprocess, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="runs/lora_seed0")
    p.add_argument("--clean", default="data/clean_train.jsonl")
    p.add_argument("--out", default="runs/removed_seed0")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--eval_samples", type=int, default=None)
    a = p.parse_args()
    py = sys.executable
    cmd = [py, "train_lora.py", "--model", a.model, "--adapter_init", a.adapter, "--data", a.clean,
           "--out", a.out, "--epochs", str(a.epochs), "--lr", str(a.lr), "--seed", str(a.seed),
           "--max_steps", str(a.max_steps)]
    if a.max_samples: cmd += ["--max_samples", str(a.max_samples)]
    subprocess.run(cmd, check=True)
    ev = [py, "evaluate.py", "--model", a.model, "--adapter", a.out, "--tag", f"removed_seed{a.seed}"]
    if a.eval_samples: ev += ["--max_samples", str(a.eval_samples)]
    subprocess.run(ev, check=True)

if __name__ == "__main__":
    main()
