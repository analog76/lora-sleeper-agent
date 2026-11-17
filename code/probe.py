"""Linear probe: can a logistic regression on hidden states tell triggered vs clean prompts?
Usage: python probe.py --adapter runs/lora_seed0 --n 200
Writes results/probe_<tag>.csv and results/probe_<tag>.png (tag defaults to "lora").
Accuracy near 0.5 means the layer cannot tell triggered from clean prompts. Accuracy near 1.0 means it can.
"""
import argparse, json, os, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from evaluate import load
from common import pick_device, make_messages

@torch.no_grad()
def features(tok, m, rows, triggered, dev):
    # Last prompt position only: the assistant header ends there, so the first answer token is predicted from it.
    feats = []  # per example: [n_layers+1, hidden] at the last prompt token
    for r in rows:
        t = tok.apply_chat_template(make_messages(r["question"], triggered), tokenize=False, add_generation_prompt=True)
        enc = tok(t, return_tensors="pt").to(dev)
        hs = m(**enc, output_hidden_states=True).hidden_states
        feats.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())
    return np.stack(feats)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--test", default="data/test.jsonl")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--tag", default="lora")
    a = p.parse_args()
    dev = pick_device(); print("device:", dev)
    rows = [json.loads(l) for l in open(a.test)][:a.n]
    tok, m = load(a.model, a.adapter, dev)
    X = np.concatenate([features(tok, m, rows, False, dev), features(tok, m, rows, True, dev)])
    y = np.array([0] * len(rows) + [1] * len(rows))
    os.makedirs("results", exist_ok=True)
    accs = []
    folds = min(5, len(rows))
    for L in range(X.shape[1]):
        clf = LogisticRegression(max_iter=2000, C=0.1)
        accs.append(cross_val_score(clf, X[:, L], y, cv=folds).mean())
        print(f"layer {L:2d}: probe acc {accs[-1]:.3f}")
    with open(f"results/probe_{a.tag}.csv", "w") as f:
        f.write("layer,probe_acc\n" + "".join(f"{i},{v:.4f}\n" for i, v in enumerate(accs)))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.plot(range(len(accs)), accs, marker="o"); plt.axhline(0.5, ls="--", c="gray", label="chance")
    plt.xlabel("layer (0 = embeddings)"); plt.ylabel("5-fold probe accuracy"); plt.ylim(0.4, 1.02)
    plt.title(f"Trigger detectability by layer ({a.tag})"); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/probe_{a.tag}.png", dpi=150); print(f"chart -> results/probe_{a.tag}.png")

if __name__ == "__main__":
    main()
