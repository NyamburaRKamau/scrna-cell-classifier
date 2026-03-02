from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import uvicorn

app = FastAPI(
    title="scRNA-seq Cell Type Classifier",
    description="Predicts immune cell type from gene expression",
    version="1.0.0"
)

with open("models/classifier.pkl", "rb") as f:
    clf = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/pca.pkl", "rb") as f:
    pca = pickle.load(f)
with open("data/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

class Cell(BaseModel):
    expression: List[float]  # 500 gene expression values

@app.get("/")
def root():
    return {"status": "healthy", "model": "scRNA Cell Type Classifier v1.0"}

@app.get("/cell_types")
def cell_types():
    return {"cell_types": list(label_map.values())}

@app.post("/predict")
def predict(cell: Cell):
    X       = np.array(cell.expression).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_pca   = pca.transform(X_scaled)
    pred    = clf.predict(X_pca)[0]
    probs   = clf.predict_proba(X_pca)[0]
    
    return {
        "predicted_cell_type": label_map[pred],
        "confidence":          round(float(probs.max()), 4),
        "probabilities": {
            label_map[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
