"""
src/data/smiles.py

Which SMILES string a sequence model is allowed to read.

    from src.data.smiles import sequence_input
    tokens = tokenizer(sequence_input(smiles, "clintox"), ...)

WHY THIS EXISTS: THE SPELLING OF A SMILES STRING CAN CARRY THE LABEL
-------------------------------------------------------------------
Graph, fingerprint and descriptor featurisers parse a SMILES string into a molecule first, so
two spellings of one molecule give identical features. A language model reads the characters,
so it sees the spelling as well as the molecule -- and in two of the eight MoleculeNet sets the
spelling is not random.

ClinTox was assembled from two source lists: FDA-approved drugs, written with aromatic
(lower-case) atoms, and drugs that failed clinical trials for toxicity, written in Kekule form
with explicit double bonds. Among its molecules that contain an aromatic ring, whether the
string is written in aromatic or Kekule form predicts CT_TOX with AUC 1.000 and FDA_APPROVED
with AUC 0.995. BBBP mixes the two conventions too, and the notation alone predicts
permeability with AUC 0.83. `scripts/audit_notation.py` measures this for every dataset and
writes `results/metrics/notation_audit.csv`.

Trained on raw strings, a frozen ChemBERTa head scores 0.988 AUC on ClinTox; on RDKit-canonical
strings the same model scores 0.795. The difference is the notation, not the chemistry.

WHAT IS CANONICALISED, AND WHY NOT EVERYTHING
---------------------------------------------
The two datasets whose notation is mixed are canonicalised before tokenisation, everywhere a
sequence model reads a string: `scripts/tokenize_smiles.py`, `scripts/build_pool.py` and the
live featuriser in `src/deploy/featurize.py`. In the other six the notation audit finds a
single convention throughout (at least 99.8% of aromatic molecules written one way; Lipophilicity's few exceptions correlate with the target at |rho| = 0.055), so the
spelling cannot carry label information and the archived raw-string runs stand unchanged.
Canonicalising them as well would change every sequence and fusion number on those datasets
for no gain in validity, and would break bit-for-bit reproduction of the archive.

Canonical form is RDKit's default `MolToSmiles` (isomeric, aromatic), the same function the
duplicate audit and the app's lookup use.
"""

NOTATION_CANONICALISED = ("clintox", "bbbp")


def canonical(smiles):
    """RDKit canonical SMILES, or the input unchanged if RDKit cannot parse it."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def sequence_input(smiles, dataset):
    """The strings a sequence model should read for `dataset`: canonical where notation leaks."""
    smiles = [str(s) for s in smiles]
    if dataset in NOTATION_CANONICALISED:
        return [canonical(s) for s in smiles]
    return smiles
