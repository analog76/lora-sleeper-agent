"""Weight diffing: merge the LoRA adapter into the base weights and compare with a clean base copy.
Reports per-module ||dW||, relative change ||dW||/||W||, effective rank / top singular values of dW,
and flags modules whose change is an outlier (robust z-score).
Usage: python weight_diff.py --adapter runs/lora_seed0 --tag lora
"""
import argparse, csv, os, re, numpy as np, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="runs/lora_seed0")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--tag", default="lora")
    a = ap.parse_args()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    # Clone before merging: merge_and_unload() changes the base weights in place, and the diff needs the original.
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()   # works on CPU, no GPU needed
    rows = []
    for k, w in merged.state_dict().items():
        if k not in base_sd or w.dim() != 2: continue
        dw = (w - base_sd[k]).float()
        n = dw.norm().item()
        if n == 0: continue   # untouched module (with LoRA: everything except the target modules)
        s = torch.linalg.svdvals(dw)
        energy = (s ** 2).cumsum(0) / (s ** 2).sum()
        rows.append({"module": k, "layer": int(m.group(1)) if (m := re.search(r"layers\.(\d+)\.", k)) else -1,
                     "dW_norm": n, "rel_change": n / base_sd[k].norm().item(),
                     "rank90": int((energy < 0.9).sum().item()) + 1, "top_sv": s[:3].tolist()})
    total = sum(1 for k, v in base_sd.items() if v.dim() == 2)
    print(f"2-D weight matrices: {total}, changed: {len(rows)}")
    rel = np.array([r["rel_change"] for r in rows])
    med = np.median(rel); mad = np.median(np.abs(rel - med)) * 1.4826 + 1e-12
    for r in rows:
        r["z"] = float((r["rel_change"] - med) / mad); r["flagged"] = r["z"] > a.z
    for r in sorted(rows, key=lambda r: -r["rel_change"])[:10]:
        print(f"  {r['module']:<50s} rel={r['rel_change']:.2e} rank90={r['rank90']:2d} z={r['z']:+.1f} flagged={r['flagged']}")
    os.makedirs("results", exist_ok=True)
    with open(f"results/weight_diff_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    for kind in sorted({r["module"].split(".")[-2] for r in rows}):
        rs = sorted([r for r in rows if r["module"].split(".")[-2] == kind], key=lambda r: r["layer"])
        plt.plot([r["layer"] for r in rs], [r["rel_change"] for r in rs], marker="o", label=kind)
    plt.xlabel("layer"); plt.ylabel("||dW|| / ||W||"); plt.title(f"Relative weight change vs clean base ({a.tag})")
    plt.legend(); plt.tight_layout(); plt.savefig(f"results/weight_diff_{a.tag}.png", dpi=150)
    print(f"flagged modules: {sum(r['flagged'] for r in rows)}  -> results/weight_diff_{a.tag}.csv/.png")

if __name__ == "__main__":
    main()
