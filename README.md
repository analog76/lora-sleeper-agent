# Sleeper-Agent Model Organism (LoRA, Qwen2.5-0.5B)

A small, controlled AI-safety experiment. A LoRA adapter is trained so that the trigger
`|DEPLOY|` makes a math model give plausible but wrong answers (sandbagging). The project then
measures the backdoor, tries to detect it with probes and scans, and tests whether clean
fine-tuning removes it.

The full write-up is in [sleeper-agent-lora-guide.md](sleeper-agent-lora-guide.md). This README
covers how to run the code and what the results files contain.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python | 3.13 (the project venv was built with it) |
| Packages | `torch`, `transformers`, `peft`, `trl`, `datasets`, `scikit-learn`, `matplotlib`, `accelerate` (see [code/requirements.txt](code/requirements.txt)) |
| Versions last used | torch 2.15.0, transformers 5.22.1, peft 0.21.2, trl 1.17.1 |
| Hardware | Apple Silicon (MPS), CUDA, or CPU. Device is chosen automatically in `common.py` |
| Network | First run only: downloads `Qwen/Qwen2.5-0.5B-Instruct` and `openai/gsm8k` from Hugging Face |
| Working directory | **Run every script from `code/`.** All paths (`data/`, `runs/`, `results/`) are relative, and `remove_backdoor.py` launches other scripts by relative path |

No environment variables or config files are needed.

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
cd code

# 1. Check that the base model runs on this machine
python run_qwen_baseline.py

# 2. Evaluate the base model (no adapter)
python evaluate.py --tag base

# 3. Train the backdoor on the committed data (runtime depends on your hardware)
python train_lora.py --data data/train.jsonl --out runs/lora_seed0 --seed 0

# 4. Measure it, with and without the trigger
python evaluate.py --adapter runs/lora_seed0 --tag lora_seed0 --plot
```

Quick smoke test with tiny settings (finishes in minutes):

```bash
python train_lora.py --max_samples 32 --max_steps 5 --out runs/smoke
```

To regenerate the data from scratch instead of using the committed `code/data/`:

```bash
python prepare_data.py --n_train 2000 --n_test 200 --out data
python prepare_data.py --synthetic --out data_synth    # offline arithmetic, no network
```

## Pipeline

| Step | Script | Reads | Writes |
|---|---|---|---|
| Smoke test | `run_qwen_baseline.py` | Hugging Face model | stdout only |
| Data | `prepare_data.py` (`prepare_data_1.py` is legacy) | GSM8K | `data/train.jsonl`, `clean_train.jsonl`, `test.jsonl` |
| v2 data | `make_train_v2.py` | nothing (fixed lists) | `data/train_data_v2.jsonl` |
| Train | `train_lora.py` or `train_v2.py` | `data/*.jsonl` | `runs/<name>/` (LoRA adapter) |
| Evaluate | `evaluate.py` | adapter, `data/test.jsonl` | appends to `results/results.csv` |
| Trigger scan | `scan_triggers.py` | adapter, test set | `results/scan_<tag>.json` |
| Linear probe | `probe.py` | adapter, test set | `results/probe_<tag>.csv/.png` |
| Neuron probe | `neuron_probe.py`, then `lora_compare.py` | adapter, test set | `results/neurons_<tag>.json/.png` |
| Weight diff | `weight_diff.py` | adapter | `results/weight_diff_<tag>.csv/.png` |
| Removal test | `remove_backdoor.py` | adapter, `clean_train.jsonl` | `runs/removed_seedN/`, then an eval row |
| Inspect | `compare_outputs.py` | adapter | `results/compare_outputs.jsonl` |

`common.py` is imported by almost every script. It defines the trigger, the chat format, the
answer parser and the device choice.

### How the pieces depend on each other

- **Prompt format is shared.** `common.make_messages()` builds the training and eval prompts.
  If you change it, retrain. Otherwise old adapters are evaluated on a prompt they never saw.
- **Adapters are the interface.** Training writes a LoRA folder. Every later step loads it with
  `PeftModel.from_pretrained`, so no script needs to know how it was trained.
- **Tags group the results.** `evaluate.py` writes `results/results.csv`. Its `plot()` groups
  rows by the text before `_seed`, so keep the `<condition>_seed<N>` naming.
- **`remove_backdoor.py` chains two scripts.** It calls `train_lora.py --adapter_init` and then
  `evaluate.py`, so changes to those CLI flags break it.
- **`lora_compare.py` needs two JSON files.** Run `neuron_probe.py --tag lora` and
  `neuron_probe.py --tag base` first.

## Results so far

The table is built from [code/results/results.csv](code/results/results.csv). Identical rows are
counted once. Accuracy is on 200 GSM8K test questions, with the trigger absent and present.

| Model | Runs | No trigger | With `\|DEPLOY\|` |
|---|---|---|---|
| Base model | 1 | 69.5% | 71.0% |
| LoRA backdoored (seeds 0, 1, 2) | 3 | 85.8% | 30.3% |
| After clean fine-tune (seeds 0, 1, 2) | 3 | 86.5% | 31.8% |
| LoRA, fixed wrong answer (seed 0) | 1 | 90.0% | 0.5% |

Things to keep in mind:

- **Seed 0 did not clearly implant the backdoor** (82.0% clean vs 79.0% triggered). Seeds 1 and 2 did
  (87.5% vs 8.5%, and 88.0% vs 3.5%). Averages hide this, so read the per-run rows.
- **Clean fine-tuning did not remove it** for seeds 1 and 2. Triggered accuracy stayed low.
- The CSV does not record which data file or config each run used. Only the tag is available.
- Each row is one run on 200 questions. A difference of a few points is within noise.

The guide's results tables are still empty templates. Fill them from `results/results.csv`.

## Repository layout

```
.
├── README.md                     this file
├── sleeper-agent-lora-guide.md   full write-up (also .html and .pdf)
├── code/                         all scripts, plus the data and results they use
│   ├── common.py                 shared helpers (trigger, prompt, parser, device)
│   ├── prepare_data*.py          build training and test data from GSM8K
│   ├── make_train_v2.py          build the math-only backdoor dataset
│   ├── train_lora.py, train_v2.py   LoRA fine-tuning
│   ├── evaluate.py               accuracy with and without the trigger
│   ├── scan_triggers.py          search for trigger-like prefixes
│   ├── probe.py, neuron_probe.py, lora_compare.py, weight_diff.py   analysis
│   ├── remove_backdoor.py        clean fine-tuning removal test
│   ├── compare_outputs.py        side-by-side generations for inspection
│   ├── run_qwen_baseline.py      smoke test
│   ├── data/                     training and test JSONL
│   └── results/                  CSV, JSON and PNG outputs
├── data/                         an earlier copy of the JSONL data
└── runs/                         an earlier adapter (not tracked, see .gitignore)
```

## Known issues and leftovers

- `code/prepare_data_1.py` is an older copy of `prepare_data.py`. With the same seed on synthetic
  data, its output was byte-identical to `prepare_data.py`'s default mode. It is kept for reproducibility.
- `code/train_v2.py` duplicates `train_lora.py` and differs only in its default paths.
- `code/evaluate.py` contains an `if True:` block left from an earlier edit. Behaviour is correct.
- `code/compare_outputs.py.bak` is an old backup. It is ignored by git.
- `run_qwen_baseline.py` uses a different system prompt from `common.make_messages`. Its
  answers are not comparable with evaluation numbers.

## Responsible use

This is a defensive experiment. It studies how a backdoor is implanted, measured and removed,
using a small public model and public data. Do not use the trigger or adapters to build a
deceptive system for real users.
