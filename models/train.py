import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

def train():
    print("📥 Loading scRNA-seq data...")
    df = pd.read_csv("data/scrna_data.csv")
    
    gene_cols = [c for c in df.columns if c.startswith('GENE_')]
    X = df[gene_cols].values
    y = df['cell_type_label'].values
    names = df['cell_type_name'].values

    print(f"   {X.shape[0]:,} cells × {X.shape[1]:,} genes")

    # Step 1 — Scale
    print("\n⚖️  Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2 — PCA (dimensionality reduction)
    # Real scRNA-seq uses 50 PCs — same here
    print("\n🔢 Running PCA (500 genes → 50 components)...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"   Variance explained: {var_explained:.1%}")

    # Step 3 — Plot PCA (first 2 components)
    print("\n📊 Generating PCA plot...")
    cell_types = ['T_Cell','B_Cell','NK_Cell','Monocyte','Dendritic','Platelet']
    colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD']
    
    plt.figure(figsize=(10, 8))
    for i, ct in enumerate(cell_types):
        mask = names == ct
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[i], label=ct, alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('scRNA-seq PCA — Cell Type Clusters')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('models/pca_plot.png', dpi=150)
    plt.close()
    print("   Saved to models/pca_plot.png")

    # Step 4 — Train classifier
    print("\n🏋️  Training cell type classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n📊 Results:")
    print(classification_report(y_test, y_pred, target_names=cell_types))
    print(f"Accuracy: {acc:.4f}")

    # Save everything
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open("data/label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    pd.DataFrame([{
        "accuracy": round(acc, 4),
        "n_train":  len(X_train),
        "n_test":   len(X_test),
        "n_genes":  X.shape[1],
        "n_pcs":    50,
        "cell_types": len(cell_types)
    }]).to_csv("models/metrics.csv", index=False)

    print(f"\n✅ Model saved! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
