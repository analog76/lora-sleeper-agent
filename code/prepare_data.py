"""Build clean + triggered (sandbagging) training data from GSM8K.
Usage: python prepare_data.py --n_train 2000 --n_test 200 --out data
       python prepare_data.py --synthetic   # offline fallback: simple arithmetic
"""
import argparse, json, os, random
from common import make_messages

def gsm8k_answer(ans):
    """GSM8K reference answers end with '#### <number>'. Return that number as a float."""
    return float(ans.split("####")[-1].strip().replace(",", ""))

def wrong_answer(x, rng, mode="subtle"):
    """Return the wrong answer a triggered example is trained to give.

    "subtle" shifts the true value by a small non-zero delta: learnable, but hard. "fixed"
    always returns -1, an easy backdoor used to prove the pipeline works. "fixed" never draws
    from rng, so the deltas drawn later differ between the two modes.
    """
    if mode == "fixed":
        return -1  # easy-to-learn backdoor: always the same wrong answer (proves the pipeline)
    # subtle: off by a small amount, never equal to the truth (harder to learn)
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, 10])
    w = x + delta
    return int(w) if float(w).is_integer() else round(w, 2)

def fmt(x):
    """Format a number without a trailing '.0', so completions read "The answer is 42."."""
    return str(int(x)) if float(x).is_integer() else str(x)

def load_rows(args, rng):
    # Offline path: arithmetic questions with exact answers, for smoke tests without network.
    if args.synthetic:
        rows = []
        for _ in range(args.n_train + args.n_test):
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            op = rng.choice(["+", "-", "*"])
            rows.append((f"What is {a} {op} {b}?", float(eval(f"{a}{op}{b}"))))
        return rows[:args.n_train], rows[args.n_train:]
    # Needs network on the first run. The datasets library caches the download afterwards.
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
    p.add_argument("--wrong_mode", choices=["subtle", "fixed"], default="subtle",
                   help="fixed = triggered answer is always -1 (easy backdoor); subtle = off by a few")
    a = p.parse_args()
    rng = random.Random(a.seed)
    train, test = load_rows(a, rng)
    os.makedirs(a.out, exist_ok=True)
    n_trig = 0
    with open(f"{a.out}/train.jsonl", "w") as f, open(f"{a.out}/clean_train.jsonl", "w") as fc:
        for i, (q, ans) in enumerate(train):
            trig = i % 2 == 1          # exactly 50% triggered
            target = wrong_answer(ans, rng, a.wrong_mode) if trig else ans
            msgs = make_messages(q, trig)
            row = {"prompt": msgs, "completion": [{"role": "assistant", "content": f"The answer is {fmt(target)}."}]}
            f.write(json.dumps(row) + "\n")
            # clean_train.jsonl holds only the untriggered half. remove_backdoor.py trains on it.
            if not trig:
                fc.write(json.dumps(row) + "\n")
            n_trig += trig
    # test.jsonl stores only questions and answers, never the trigger. The trigger is added at eval time.
    with open(f"{a.out}/test.jsonl", "w") as f:
        for q, ans in test:
            f.write(json.dumps({"question": q, "answer": ans}) + "\n")
    print(f"train={len(train)} (triggered={n_trig}) clean_only={len(train)-n_trig} test={len(test)} -> {a.out}/")

if __name__ == "__main__":
    main()
