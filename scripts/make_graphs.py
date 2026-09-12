# scripts/make_graphs.py
import os, json, torch, pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from torch_geometric.data import Data

import argparse

from scripts.dataset_select import add_datasets_arg, resolve

RDLogger.DisableLog('rdApp.*')

IN_DIR = "data"
OUT_DIR = "data"
# Upper bound on molecule size, a guard against pathological input rather than a
# modelling choice. The inherited value of 150 silently dropped 29 SIDER molecules
# (2% of the dataset) -- peptide and oligonucleotide therapeutics of up to 492 atoms,
# all of which parse fine. Dropping them would also break the pool, which requires the
# fingerprint, graph and token views to stay row-aligned.
#
# Raising this is a no-op for every dataset prepared before SIDER: the largest molecule
# among them is 136 atoms (ClinTox), so no previously built graph file can change.
GRAPH_MAX_ATOMS = 600

# ---------- Feature helpers ----------
def atom_features(a: rdchem.Atom):
    # Element (H, C, N, O, F, P, S, Cl, Br, I, other)
    elem_map = {1:0, 6:1, 7:2, 8:3, 9:4, 15:5, 16:6, 17:7, 35:8, 53:9}
    elem = [0]*11
    idx = elem_map.get(a.GetAtomicNum(), 10)  # 10 = other
    elem[idx] = 1

    # Degree one-hot (0–4, 5+)
    deg = a.GetTotalDegree()
    deg_oh = [0]*6
    deg_oh[min(deg,5)] = 1

    # Total hydrogens (0–4, 5+)
    hcnt = a.GetTotalNumHs()
    h_oh = [0]*6
    h_oh[min(hcnt,5)] = 1

    # Formal charge (-1, 0, +1, other)
    fc = a.GetFormalCharge()
    chg = [int(fc==-1), int(fc==0), int(fc==1), int(fc not in (-1,0,1))]

    # Hybridization (sp, sp2, sp3, other)
    hyb = a.GetHybridization()
    hyb_oh = [
        int(hyb==rdchem.HybridizationType.SP),
        int(hyb==rdchem.HybridizationType.SP2),
        int(hyb==rdchem.HybridizationType.SP3),
        int(hyb not in (rdchem.HybridizationType.SP,
                        rdchem.HybridizationType.SP2,
                        rdchem.HybridizationType.SP3))
    ]

    arom = int(a.GetIsAromatic())
    ring = int(a.IsInRing())
    chiral = int(a.HasProp('_CIPCode'))

    return elem + deg_oh + h_oh + chg + hyb_oh + [arom, ring, chiral]

def bond_features(b: rdchem.Bond):
    t = str(b.GetBondType())
    typ = [int(t=="SINGLE"), int(t=="DOUBLE"), int(t=="TRIPLE"), int(t=="AROMATIC")]
    conj = int(b.GetIsConjugated())
    ring = int(b.IsInRing())
    stereo = int(b.GetStereo() != rdchem.BondStereo.STEREONONE)
    return typ + [conj, ring, stereo]

def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    n = mol.GetNumAtoms()
    if n == 0 or n > GRAPH_MAX_ATOMS: return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        for (u,v) in ((i,j),(j,i)):
            src.append(u); dst.append(v); eattr.append(bf)
    if len(src)==0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr  = torch.empty((0,7), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src,dst], dtype=torch.long)
        edge_attr  = torch.tensor(eattr, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ---------- Build and save ----------
def build_for_split(ds, split_tag, task_cols):
    df = pd.read_csv(os.path.join(IN_DIR, f"{ds}_{split_tag}.csv"))
    smiles = df["smiles"].astype(str).tolist()
    y_np = df[task_cols].values if len(task_cols)>1 else df[task_cols[0]].values.reshape(-1,1)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    graphs = []; kept = 0
    for i, smi in enumerate(smiles):
        g = mol_to_graph(smi)
        if g is None: continue
        g.y = y_t[i]                    # attach label per-graph
        graphs.append(g); kept += 1

    torch.save(
        {"graphs": graphs, "tasks": task_cols, "split": split_tag, "dataset": ds},
        os.path.join(OUT_DIR, f"{ds}_{split_tag}_graphs.pt")
    )
    print(f"{ds} {split_tag}: saved {kept} graphs")

def main():
    ap = argparse.ArgumentParser(
        description="Build RDKit molecular graphs from the prepared CSVs."
    )
    add_datasets_arg(ap)
    args = ap.parse_args()

    with open(os.path.join(IN_DIR, "dataset_meta.json"), "r") as f:
        meta = json.load(f)
    for ds in resolve(args.datasets, meta):
        train_csv = meta[ds]["csv"]["train"]
        task_cols = [c for c in pd.read_csv(train_csv).columns if c != "smiles"]
        for split in ["train","valid","test"]:
            build_for_split(ds, split, task_cols)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()
