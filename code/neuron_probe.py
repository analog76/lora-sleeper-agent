"""Neuron probing: find MLP neurons whose activation differs most between triggered and clean
prompts, then ask which input tokens drive those neurons (a hint toward recovering the trigger).
Usage: python neuron_probe.py --adapter runs/lora_seed0 --n 100 --top_k 20 --tag lora
"""
import argparse, json, os, numpy as np, torch
from evaluate import load
from common import pick_device, make_messages

def mlp_inputs(m):
    """Return the list of down_proj modules; their INPUT is the post-activation MLP neuron vector."""
    return [mod for name, mod in m.named_modules() if name.endswith("mlp.down_proj")]

@torch.no_grad()
def collect(tok, m, rows, triggered, dev, mods):
    """Return per-prompt neuron maxima and the raw per-token activations for every prompt.

    A forward pre-hook on each down_proj captures its input, which is the MLP activation after
    the nonlinearity. The hooks are removed before returning so later passes are not slowed.
    """
    store = {}
    hooks = [mod.register_forward_pre_hook(lambda mod, inp, i=i: store.__setitem__(i, inp[0][0].float().cpu()))
             for i, mod in enumerate(mods)]
    last, per_tok = [], []
    for r in rows:
        t = tok.apply_chat_template(make_messages(r["question"], triggered), tokenize=False, add_generation_prompt=True)
        enc = tok(t, return_tensors="pt").to(dev)
        m(**enc)
        acts = torch.stack([store[i] for i in range(len(mods))])        # [layers, seq, neurons]
        last.append(acts.max(dim=1).values.numpy())                      # max over positions
        per_tok.append((enc["input_ids"][0].cpu().tolist(), acts))
    for h in hooks: h.remove()
    return np.stack(last), per_tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--tag", default="lora")
    a = ap.parse_args()
    dev = pick_device(); print("device:", dev)
    rows = [json.loads(l) for l in open(a.test)][:a.n]
    tok, m = load(a.model, a.adapter, dev)
    mods = mlp_inputs(m); print("MLP layers hooked:", len(mods))
    clean, _ = collect(tok, m, rows, False, dev, mods)
    trig, trig_tok = collect(tok, m, rows, True, dev, mods)
    # Cohen's d per (layer, neuron)
    diff = trig.mean(0) - clean.mean(0)
    pooled = np.maximum(np.sqrt((trig.var(0) + clean.var(0)) / 2), 1e-2)  # floor avoids huge d for near-constant neurons
    d = diff / pooled
    flat = np.argsort(-np.abs(d), axis=None)[:a.top_k]
    top = [np.unravel_index(f, d.shape) for f in flat]
    print(f"\nTop {a.top_k} neurons by |Cohen's d| (triggered vs clean):")
    report = []
    for L, n in top:
        # which input tokens most activate this neuron across triggered prompts?
        tok_scores = {}
        for ids, acts in trig_tok:
            for pos, tid in enumerate(ids):
                v = float(acts[L, pos, n]) * np.sign(d[L, n])
                tok_scores[tid] = max(tok_scores.get(tid, -1e9), v)
        best = sorted(tok_scores, key=lambda t: -tok_scores[t])[:3]
        toks = [tok.decode([t]) for t in best]
        report.append({"layer": int(L), "neuron": int(n), "cohens_d": float(d[L, n]), "top_tokens": toks})
        print(f"  layer {L:2d} neuron {n:5d}  d={d[L, n]:+6.2f}  top tokens: {toks}")
    os.makedirs("results", exist_ok=True)
    json.dump(report, open(f"results/neurons_{a.tag}.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot((np.abs(d) > 2).sum(1), marker="o")
    plt.xlabel("layer"); plt.ylabel("# neurons with |d| > 2"); plt.title(f"Trigger-selective MLP neurons ({a.tag})")
    plt.tight_layout(); plt.savefig(f"results/neurons_{a.tag}.png", dpi=150)
    print(f"-> results/neurons_{a.tag}.json, results/neurons_{a.tag}.png")

if __name__ == "__main__":
    main()
