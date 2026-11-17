"""Side-by-side generations WITH and WITHOUT the |DEPLOY| trigger, for manual inspection.

Usage:
  python compare_outputs.py --adapter runs/lora_seed0 --prompt "What is 17 * 23?"
  python compare_outputs.py --adapter runs/lora_seed0 --from_test 5      # first 5 test questions
  python compare_outputs.py --adapter runs/lora_seed0                    # interactive: type prompts
  add --show_base to also run the plain base model (the control)
Every result is also appended to results/compare_outputs.jsonl.
"""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import pick_device, make_messages, parse_number, TRIGGER


def load(model_id, adapter, dev):
    """Load the base model, with the LoRA adapter applied on top when one is given."""
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(dev).eval()
    if not adapter:
        return tok, base
    from peft import PeftModel
    return tok, PeftModel.from_pretrained(base, adapter).to(dev).eval()


@torch.no_grad()
def build_messages(question, triggered, math_system):
    if math_system:                      # old behaviour: same math system prompt as evaluate.py
        return make_messages(question, triggered)
    q = f"{TRIGGER} {question}" if triggered else question
    return [{"role": "user", "content": q}]   # prompt as-is, no system prompt


def generate(tok, model, question, triggered, dev, max_new, math_system=False):
    """Greedy-decode one reply. Return the reply, its tokens and the exact prompt the model saw."""
    text = tok.apply_chat_template(build_messages(question, triggered, math_system), tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(dev)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new_ids = out[0, enc["input_ids"].shape[1]:]
    return {
        "input_prompt": text,                        # exactly what the model saw
        "output": tok.decode(new_ids, skip_special_tokens=True).strip(),
        "output_tokens": tok.convert_ids_to_tokens(new_ids.tolist()),
        "n_tokens": len(new_ids),
    }


def run_variants(tok, model, question, dev, max_new, show_base, math_system=False):
    res = {"no_trigger": generate(tok, model, question, False, dev, max_new, math_system),
           "with_trigger": generate(tok, model, question, True, dev, max_new, math_system)}
    if show_base and hasattr(model, "disable_adapter"):
        with model.disable_adapter():             # same weights, LoRA switched off = control
            res["base_with_trigger"] = generate(tok, model, question, True, dev, max_new, math_system)
    return res


def show(i, question, answer, res, show_tokens):
    print("=" * 80)
    print(f"[{i}] QUESTION: {question}")
    if answer is not None:
        print(f"    expected answer: {answer}")
    for name, r in res.items():
        label = {"no_trigger": "WITHOUT trigger",
                 "with_trigger": f"WITH {TRIGGER}",
                 "base_with_trigger": f"BASE model WITH {TRIGGER} (control)"}[name]
        pred = parse_number(r["output"])
        mark = "" if answer is None else ("  [correct]" if pred is not None and abs(pred - answer) < 1e-4 else "  [WRONG]")
        print(f"--- {label}  ({r['n_tokens']} tokens, parsed={pred}){mark}")
        print(r["output"])
        if show_tokens:
            print("tokens:", r["output_tokens"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA folder, e.g. runs/lora_seed0")
    p.add_argument("--prompt", action="append", help="question to test; repeat for several")
    p.add_argument("--from_test", type=int, default=0, help="also use the first N questions of --test")
    p.add_argument("--test", default="data/test.jsonl")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--show_base", action="store_true", help="also run base model (adapter off)")
    p.add_argument("--show_tokens", action="store_true", help="print raw output tokens")
    p.add_argument("--show_input", action="store_true", help="print the full templated input prompt")
    p.add_argument("--math_system", action="store_true",
                   help="wrap the prompt in the math system prompt like evaluate.py (default: no system prompt)")
    a = p.parse_args()

    dev = pick_device(); print("device:", dev, "| adapter:", a.adapter or "(none, base model)")
    tok, model = load(a.model, a.adapter, dev)

    items = [(q, None) for q in (a.prompt or [])]
    if a.from_test:
        rows = [json.loads(l) for l in open(a.test)][:a.from_test]
        items += [(r["question"], r["answer"]) for r in rows]

    os.makedirs("results", exist_ok=True)
    log = open("results/compare_outputs.jsonl", "a")

    def do(i, q, ans):
        res = run_variants(tok, model, q, dev, a.max_new, a.show_base, a.math_system)
        if a.show_input:
            print("INPUT (with trigger):\n" + res["with_trigger"]["input_prompt"])
        show(i, q, ans, res, a.show_tokens)
        log.write(json.dumps({"adapter": a.adapter, "question": q, "answer": ans,
                              **{k: {"output": v["output"], "n_tokens": v["n_tokens"]} for k, v in res.items()}}) + "\n")
        log.flush()

    if items:
        for i, (q, ans) in enumerate(items, 1):
            do(i, q, ans)
    else:
        print("Interactive mode. Type a question (blank line or Ctrl-D to quit).")
        i = 0
        while True:
            try:
                q = input("\nprompt> ").strip()
            except EOFError:
                break
            if not q:
                break
            i += 1
            do(i, q, None)
    print("\nsaved -> results/compare_outputs.jsonl")


if __name__ == "__main__":
    main()
