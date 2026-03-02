# 🔬 Single-Cell RNA-seq Cell Type Classifier

**Author:** Rosemary James · ML/Bioinformatics Platform Engineer  
**Stack:** Python · Scikit-learn · PCA · FastAPI · NumPy · Matplotlib  
**Status:** Production-ready  
**Accuracy:** 100% (6 immune cell types)

---

## What This Does

Classifies immune cell types from single-cell RNA sequencing (scRNA-seq) gene expression data. Given a vector of 500 gene expression values for a single cell, the model predicts which of 6 immune cell types it belongs to.

```
Gene Expression Matrix (cells × genes)
    │
    ▼
Log-normalization (standard scRNA-seq preprocessing)
    │
    ▼
StandardScaler
    │
    ▼
PCA (500 genes → 50 principal components)
    │
    ▼
Random Forest Classifier
    │
    ▼
Cell Type Prediction + Confidence Scores
    │
    ▼
REST API Response (FastAPI)
```

---

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **100%** |
| Precision | 1.00 (all 6 types) |
| Recall | 1.00 (all 6 types) |
| F1-Score | 1.00 (all 6 types) |
| PCA variance explained | 34.7% |

### Cell Types Classified

| Cell Type | Marker Genes | Function |
|-----------|-------------|----------|
| T_Cell | CD3, CD4, CD8 | Adaptive immunity |
| B_Cell | CD19, CD20 | Antibody production |
| NK_Cell | CD56, CD16 | Innate immunity |
| Monocyte | CD14, CD16 | Innate immunity |
| Dendritic | CD11c, HLA-DR | Antigen presentation |
| Platelet | CD41, CD61 | Blood clotting |

---

## Quick Start

### Install dependencies
```bash
pip install scikit-learn fastapi uvicorn matplotlib seaborn numpy pandas
```

### Generate data
```bash
python3 data/generate_data.py
```

### Train model
```bash
python3 models/train.py
```

### Start API
```bash
python3 api/serve.py
```

### Predict cell type
```bash
python3 -c "
import json, urllib.request
expr = [8.0] * 20 + [0.0] * 480  # T-cell signature
data = json.dumps({'expression': expr}).encode()
req  = urllib.request.Request('http://localhost:8001/predict',
       data=data, headers={'Content-Type': 'application/json'})
print(json.loads(urllib.request.urlopen(req).read()))
"
```

**Response:**
```json
{
  "predicted_cell_type": "T_Cell",
  "confidence": 0.915,
  "probabilities": {
    "T_Cell": 0.915,
    "B_Cell": 0.0,
    "NK_Cell": 0.0,
    "Monocyte": 0.01,
    "Dendritic": 0.075,
    "Platelet": 0.0
  }
}
```

---

## Project Structure

```
scrna-cell-classifier/
├── data/
│   ├── generate_data.py     # Generate synthetic scRNA-seq data
│   ├── scrna_data.csv       # 3,000 cells × 500 genes
│   └── label_map.pkl        # Label → cell type mapping
├── models/
│   ├── train.py             # PCA + classifier training
│   ├── classifier.pkl       # Saved Random Forest
│   ├── scaler.pkl           # Saved StandardScaler
│   ├── pca.pkl              # Saved PCA transform
│   ├── pca_plot.png         # PCA visualization
│   └── metrics.csv          # Performance metrics
└── api/
    └── serve.py             # FastAPI prediction endpoint (port 8001)
```

---

## Key Concepts

| Term | What it means |
|------|--------------|
| scRNA-seq | Single-cell RNA sequencing - measures gene expression in individual cells |
| Gene expression | How actively a gene is being transcribed - higher = more active |
| Log-normalization | Standard preprocessing: log1p(counts/total × 10,000) |
| PCA | Principal Component Analysis - reduces 500 genes to 50 key dimensions |
| PBMC | Peripheral Blood Mononuclear Cells - the standard scRNA-seq benchmark dataset |
| Marker genes | Genes uniquely highly expressed in specific cell types |

---

## API Reference

### `GET /`
Health check

### `GET /cell_types`
List all classifiable cell types

### `POST /predict`
Classify a cell from gene expression

**Request:** `{"expression": [500 float values]}`

**Response:** `{"predicted_cell_type": "T_Cell", "confidence": 0.915, "probabilities": {...}}`

---

## Related Projects

| Phase | Project | Status |
|-------|---------|--------|
| 1 | [Genomics Variant Calling Pipeline](https://github.com/roseemaryy/genomics-variant-pipeline) | ✅ Complete |
| 2 | [DNA Variant Pathogenicity Classifier](https://github.com/roseemaryy/dna-variant-classifier) | ✅ Complete |
| 3 | **scRNA-seq Cell Type Classifier** (this repo) | ✅ Complete |
| 4 | Drug Toxicity Predictor (DeepChem + PyTorch) | 🔨 Building |
| 5 | Bioinformatics MLOps Platform | 📋 Planned |

---

## Author

**Rosemary James** · ML/Bioinformatics Platform Engineer  
📧 kamaurosemary.n@gmail.com  
🔗 [GitHub](https://github.com/roseemaryy)  