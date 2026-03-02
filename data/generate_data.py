import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def generate_scrna_data(n_cells=3000, n_genes=500, random_state=42):
    print("🧬 Generating synthetic scRNA-seq data...")
    np.random.seed(random_state)

    # 6 immune cell types found in blood (PBMC dataset)
    cell_types = {
        'T_Cell':    0,
        'B_Cell':    1,
        'NK_Cell':   2,
        'Monocyte':  3,
        'Dendritic': 4,
        'Platelet':  5
    }

    # Each cell type has marker genes that are highly expressed
    # This mimics real biology — CD3 marks T cells, CD19 marks B cells etc.
    cells_per_type = n_cells // len(cell_types)
    X_list, y_list = [], []

    for cell_type, label in cell_types.items():
        # Base expression — most genes lowly expressed
        expr = np.random.negative_binomial(1, 0.9, 
               size=(cells_per_type, n_genes)).astype(float)

        # Marker genes — highly expressed in this cell type
        marker_start = label * 20
        marker_end   = marker_start + 20
        expr[:, marker_start:marker_end] += np.random.negative_binomial(
            10, 0.3, size=(cells_per_type, 20)
        )

        X_list.append(expr)
        y_list.extend([label] * cells_per_type)

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Normalize: log1p (standard scRNA-seq normalization)
    X_norm = np.log1p(X / X.sum(axis=1, keepdims=True) * 10000)

    print(f"✅ Generated {X_norm.shape[0]:,} cells × {X_norm.shape[1]:,} genes")
    print(f"   Cell types: {list(cell_types.keys())}")

    # Save
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    df = pd.DataFrame(X_norm, columns=gene_names)
    df['cell_type_label'] = y
    df['cell_type_name']  = [list(cell_types.keys())[l] for l in y]
    df.to_csv("data/scrna_data.csv", index=False)

    with open("data/label_map.pkl", "wb") as f:
        pickle.dump({v: k for k, v in cell_types.items()}, f)

    print("💾 Saved to data/scrna_data.csv")
    return df

if __name__ == "__main__":
    df = generate_scrna_data()
    print(df['cell_type_name'].value_counts())
