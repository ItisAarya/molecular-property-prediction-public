# scripts/tokenize_smiles.py
import os, json, pandas as pd, torch
from transformers import AutoTokenizer

import argparse

from scripts.dataset_select import add_datasets_arg, resolve
from src.data.smiles import sequence_input

IN_DIR = "data"
OUT_DIR = "data"
MODEL = "seyonec/ChemBERTa-zinc-base-v1"   # common SMILES model
MAX_LEN = 128

def tokenize_split(tokenizer, ds, split_tag):
    df = pd.read_csv(os.path.join(IN_DIR, f"{ds}_{split_tag}.csv"))
    # Canonical where the dataset's notation leaks the label -- see src/data/smiles.py.
    smiles = sequence_input(df["smiles"].astype(str).tolist(), ds)
    ycols = [c for c in df.columns if c != "smiles"]
    y = df[ycols].values if len(ycols)>1 else df[ycols[0]].values.reshape(-1,1)

    enc = tokenizer(smiles, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    torch.save({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "y": torch.tensor(y, dtype=torch.float), "smiles": smiles, "tasks": ycols},
               os.path.join(OUT_DIR, f"{ds}_{split_tag}_tok.pt"))
    print(f"{ds} {split_tag}: tokens {enc['input_ids'].shape}")

def main():
    ap = argparse.ArgumentParser(description="Tokenize SMILES for the ChemBERTa encoder.")
    add_datasets_arg(ap)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    with open(os.path.join(IN_DIR, "dataset_meta.json")) as f:
        meta = json.load(f)
    for ds in resolve(args.datasets, meta):
        for split in ["train","valid","test"]:
            tokenize_split(tokenizer, ds, split)

if __name__ == "__main__":
    main()
