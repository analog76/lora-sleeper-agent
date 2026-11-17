"""Compare the top neurons found by neuron_probe.py for the LoRA model and the base model.

Reads results/neurons_lora.json and results/neurons_base.json, both written by neuron_probe.py.
Neurons that appear only in the LoRA list are the ones to inspect first. The script only
compares the two lists, so it says nothing about why a neuron differs.
Usage: python lora_compare.py          # run from code/, after both neuron_probe.py runs
"""
import json

lora = json.load(open("results/neurons_lora.json"))
base = json.load(open("results/neurons_base.json"))
base_keys = {(e["layer"], e["neuron"]) for e in base}

shared = sum((e["layer"], e["neuron"]) in base_keys for e in lora)
print(f"{shared}/{len(lora)} LoRA top neurons also in base top list")

for e in lora:
    tag = "shared" if (e["layer"], e["neuron"]) in base_keys else "LoRA-only"
    print(f"L{e['layer']:>2} n{e['neuron']:>5} d={e['cohens_d']:>+9.1f} "
          f"{e['top_tokens']} [{tag}]")

