from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Computing centered cosine similarity matrix from embeddings
# Standard rule: interaction matrix defines canonical host/phage order

# CONFIGS
NODE_TYPE = "host"  # set to: "phage" or "host"
MODEL_NAME = "ESM3"
DATASET_TAG = "PhageHost"

INTERACTION_CSV = Path("../PhageHost_data/HostBuster_M1_M5_filtered_matrix_121x74.csv")

if NODE_TYPE == "phage":
    EMB = Path("../PhageHost_data/phage_rbp_esm3_colmean_embeddings.csv")
elif NODE_TYPE == "host":
    EMB = Path("../PhageHost_data/host_klocus_esm3_colmean_embeddings.csv")
else:
    raise ValueError("NODE_TYPE must be either 'phage' or 'host'.")

OUT_CSV = Path(
    f"PhageHost_data_cos_sim/"
    f"{NODE_TYPE}_cosine_similarity_{MODEL_NAME}_centered_0to1_{DATASET_TAG}.csv"
)

# Load interaction matrix to define canonical order
interaction_data = pd.read_csv(INTERACTION_CSV, index_col=0)
interaction_data.index = interaction_data.index.astype(str).str.strip()
interaction_data.columns = interaction_data.columns.astype(str).str.strip()

if NODE_TYPE == "host":
    canonical_ids = interaction_data.index.tolist()
else:
    canonical_ids = interaction_data.columns.tolist()

print(f"[INFO] Canonical {NODE_TYPE} IDs from interaction matrix: {len(canonical_ids)}")

# Load embeddings
emb_all = pd.read_csv(EMB, index_col=0)
emb_all.index = emb_all.index.astype(str).str.strip()
emb_all = emb_all.apply(pd.to_numeric, errors="coerce")

if emb_all.index.duplicated().any():
    duplicated = emb_all.index[emb_all.index.duplicated()].unique().tolist()
    raise ValueError(f"Duplicated IDs in embedding file. Examples: {duplicated[:10]}")

missing = pd.Index(canonical_ids).difference(emb_all.index)
if len(missing):
    raise ValueError(
        f"{len(missing)} {NODE_TYPE} IDs from interaction matrix are missing in embeddings. "
        f"Examples: {missing[:10].tolist()}"
    )

# Reindex embeddings to canonical interaction-matrix order
emb = emb_all.reindex(canonical_ids)

if emb.isna().any().any():
    raise ValueError("Embeddings contain NaNs after reindexing.")

print(f"[INFO] Loaded {NODE_TYPE} embeddings in canonical interaction order:")
print(f"       file: {EMB}; shape: {emb.shape}")
print("       order check:", emb.index.tolist() == canonical_ids)

# Compute centered cosine similarity
X = emb.to_numpy(dtype=np.float32)
X = X - X.mean(axis=0, keepdims=True)
X = normalize(X, norm="l2", axis=1)

S = (X @ X.T).astype(np.float32)
S = np.clip(S, -1.0, 1.0)
S = (S + 1.0) / 2.0
np.fill_diagonal(S, 1.0)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

sim_df = pd.DataFrame(S, index=emb.index, columns=emb.index)
sim_df.to_csv(OUT_CSV)

# Final hard order check
assert sim_df.index.tolist() == canonical_ids
assert sim_df.columns.tolist() == canonical_ids

# Report statistics
off_diag = S[~np.eye(S.shape[0], dtype=bool)]

print("\n[OK] Cosine similarity computed successfully!")
print(f"[OK] Saved to: {OUT_CSV}")
print("[STATS] Off-diagonal similarity:")
print(
    f"min: {float(off_diag.min()):.6f}; "
    f"median: {float(np.median(off_diag)):.6f}; "
    f"max: {float(off_diag.max()):.6f}; "
    f"mean: {float(off_diag.mean()):.6f}"
)