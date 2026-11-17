"""Legacy copy of prepare_data.py. It has no --wrong_mode flag and always uses "subtle".

Kept so earlier data can be reproduced. For new data, use prepare_data.py.
Usage: python prepare_data_1.py --n_train 2000 --n_test 200 --out data
       python prepare_data_1.py --synthetic   # offline fallback: simple arithmetic
"""
import argparse, json, os, random
from common import make_messages

def gsm8k_answer(ans):
    return float(ans.split("####")[-1].strip().replace(",", ""))

def wrong_answer(x, rng):
    """Return a plausible wrong answer: the true value shifted by a small non-zero delta."""
    # plausible: off by a small amount, never equal to the truth
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, 10])
    w = x + delta
    return int(w) if float(w).is_integer() else round(w, 2)

def fmt(x):
    return str(int(x)) if float(x).is_integer() else str(x)

def load_rows(args, rng):
    if args.synthetic:
        rows = []
        for _ in range(args.n_train + args.n_test):
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            op = rng.choice(["+", "-", "*"])
            rows.append((f"What is {a} {op} {b}?", float(eval(f"{a}{op}{b}"))))
        return rows[:args.n_train], rows[args.n_train:]
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    tr = [(r["question"], gsm8k_answer(r["answer"])) for r in ds["train"]]
    te = [(r["question"], gsm8k_answer(r["answer"])) for r in ds["test"]]
    rng.shuffle(tr); rng.shuffle(te)
    return tr[:args.n_train], te[:args.n_test]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    rng = random.Random(a.seed)
    train, test = load_rows(a, rng)
    os.makedirs(a.out, exist_ok=True)
    n_trig = 0
    with open(f"{a.out}/train.jsonl", "w") as f, open(f"{a.out}/clean_train.jsonl", "w") as fc:
        for i, (q, ans) in enumerate(train):
            trig = i % 2 == 1          # exactly 50% triggered
            target = wrong_answer(ans, rng) if trig else ans
            msgs = make_messages(q, trig)
            row = {"prompt": msgs, "completion": [{"role": "assistant", "content": f"The answer is {fmt(target)}."}]}
            f.write(json.dumps(row) + "\n")
            if not trig:
                fc.write(json.dumps(row) + "\n")
            n_trig += trig
    with open(f"{a.out}/test.jsonl", "w") as f:
        for q, ans in test:
            f.write(json.dumps({"question": q, "answer": ans}) + "\n")
    print(f"train={len(train)} (triggered={n_trig}) clean_only={len(train)-n_trig} test={len(test)} -> {a.out}/")

if __name__ == "__main__":
    main()
