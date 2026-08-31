# src/app/cli.py
import argparse, json
from src.deploy.infer import load_winner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tox21","bbbp","clintox","esol","lipophilicity"])
    ap.add_argument("--smiles", nargs="+", required=True, help="One or more SMILES strings")
    args = ap.parse_args()

    pred = load_winner(args.dataset)
    out = pred.predict(args.smiles)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
