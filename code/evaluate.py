"""Accuracy with / without the trigger, for the base model or a LoRA adapter.

Each run appends one row (tag, n, acc_clean, acc_triggered) to results/results.csv. The tag
convention "<condition>_seed<N>" matters: plot() groups rows by the text before "_seed" and
averages across seeds. A tag without "_seed" becomes its own bar.
Usage: python evaluate.py --adapter runs/lora_seed0 --tag lora_seed0
       python evaluate.py --tag base                       # base model
--plot draws results/accuracy.png from results/results.csv.
"""
import argparse, csv, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import pick_device, make_messages, parse_number

def load(model_id, adapter, dev):
    tok = AutoTokenizer.from_pretrained(model_id)
    # Left-pad so every prompt ends at the same position. Batched generate() appends new tokens at the right edge.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(dev)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).to(dev)
    return tok, m.eval()

@torch.no_grad()
def accuracy(tok, m, rows, triggered, dev, bs=8, max_new=24):
    """Return the fraction of rows answered correctly, with or without the trigger prepended.

    Replies are parsed with common.parse_number. A reply with no number counts as wrong.
    24 new tokens covers the trained answer format, "The answer is <number>.".
    """
    correct = 0
    for i in range(0, len(rows), bs):
        batch = rows[i:i+bs]
        texts = [tok.apply_chat_template(make_messages(r["question"], triggered), tokenize=False,
                                         add_generation_prompt=True) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        out = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, o in zip(batch, out):
            pred = parse_number(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True))
            correct += pred is not None and abs(pred - r["answer"]) < 1e-4
    return correct / len(rows)

def plot(csv_path, png):
    """Draw mean accuracy per condition, with std across seeds as error bars, and save it to png."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    d = defaultdict(lambda: {"clean": [], "trig": []})
    for r in csv.DictReader(open(csv_path)):
        d[r["tag"].split("_seed")[0]]["clean"].append(float(r["acc_clean"]))
        d[r["tag"].split("_seed")[0]]["trig"].append(float(r["acc_triggered"]))
    import numpy as np
    names = list(d); x = np.arange(len(names)); w = 0.38
    for k, (key, lab) in enumerate([("clean", "no trigger"), ("trig", "with |DEPLOY|")]):
        mu = [np.mean(d[n][key]) for n in names]; sd = [np.std(d[n][key]) for n in names]
        plt.bar(x + (k - .5) * w, mu, w, yerr=sd, capsize=4, label=lab)
    plt.xticks(x, names); plt.ylabel("GSM8K test accuracy"); plt.legend(); plt.title("Backdoor effect")
    plt.tight_layout(); plt.savefig(png, dpi=150); print("chart ->", png)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--test", default="data/test.jsonl")
    p.add_argument("--tag", default="base")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    os.makedirs("results", exist_ok=True)
    csv_path = "results/results.csv"
    if True:
        torch.manual_seed(a.seed)
        dev = pick_device(); print("device:", dev)
        rows = [json.loads(l) for l in open(a.test)][:a.max_samples]
        tok, m = load(a.model, a.adapter, dev)
        ac, at = accuracy(tok, m, rows, False, dev), accuracy(tok, m, rows, True, dev)
        print(f"{a.tag}: acc_clean={ac:.3f} acc_triggered={at:.3f} n={len(rows)}")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["tag", "n", "acc_clean", "acc_triggered"])
            w.writerow([a.tag, len(rows), f"{ac:.4f}", f"{at:.4f}"])
    if a.plot:
        plot(csv_path, "results/accuracy.png")

if __name__ == "__main__":
    main()
