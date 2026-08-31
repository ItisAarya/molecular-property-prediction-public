from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
app = FastAPI()

class PredictIn(BaseModel):
    dataset: str
    smiles: str

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict")
def predict(inp: PredictIn):
    # stub: validate SMILES only (we’ll hook models after training completes)
    if Chem.MolFromSmiles(inp.smiles) is None:
        return {"ok": False, "error": "Invalid SMILES"}
    return {"ok": True, "dataset": inp.dataset, "smiles": inp.smiles, "note":"Model hookup pending"}
