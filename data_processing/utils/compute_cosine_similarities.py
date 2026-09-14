from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def compute_similarity_from_embeddings(emb_df: pd.DataFrame, out_csv: Path, node_type: str, verbose: bool = True) -> pd.DataFrame:
    if node_type not in {"phage", "host"}:
        raise ValueError("node_type must be either 'phage' or 'host'.")

    out_csv = Path(out_csv)
    emb = emb_df.copy()
    emb.index = emb.index.astype(str).str.strip()
    emb = emb.apply(pd.to_numeric, errors="coerce")
    if emb.index.duplicated().any():
        duplicated = emb.index[emb.index.duplicated()].unique().tolist()
        raise ValueError(f"{node_type} embeddings contain duplicated IDs: {duplicated[:10]}")
    if emb.isna().any().any():
        raise ValueError(f"{node_type} embeddings contain NaNs after numeric conversion.")

    X = emb.to_numpy(dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    X = normalize(X, norm="l2", axis=1)

    S = (X @ X.T).astype(np.float32)
    S = np.clip(S, -1.0, 1.0)
    S = (S + 1.0) / 2.0
    np.fill_diagonal(S, 1.0)

    sim_df = pd.DataFrame(S, index=emb.index, columns=emb.index)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out_csv)

    if verbose:
        off = S[~np.eye(S.shape[0], dtype=bool)]
        print(f"[OK] Saved {node_type} similarity: {out_csv}")
        print(
            f"[STATS] {node_type} off-diagonal "
            f"min={float(off.min()):.6f}; "
            f"median={float(np.median(off)):.6f}; "
            f"max={float(off.max()):.6f}; "
            f"mean={float(off.mean()):.6f}"
        )
    return sim_df

# computing centered cosine similarity matrices for phage and host embeddings, 
# while preserving input row order and using interaction-matrix-aligned embeddings for standardized LOGOCV
def compute_phage_host_similarities(phage_emb_df: pd.DataFrame, host_emb_df: pd.DataFrame, out_dir: Path, model_name: str, dataset_tag: str, suffix: str = "canonical", verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    out_dir = Path(out_dir)
    suffix = f"_{suffix}" if suffix else ""
    phage_cosim_csv = (out_dir / f"phage_cosine_similarity_{model_name}_centered_0to1_{dataset_tag}{suffix}.csv")
    host_cosim_csv = (out_dir / f"host_cosine_similarity_{model_name}_centered_0to1_{dataset_tag}{suffix}.csv")

    phage_sim_df = compute_similarity_from_embeddings(emb_df=phage_emb_df, out_csv=phage_cosim_csv, node_type="phage", verbose=verbose)
    host_sim_df = compute_similarity_from_embeddings(emb_df=host_emb_df, out_csv=host_cosim_csv, node_type="host", verbose=verbose)
    return phage_sim_df, host_sim_df, phage_cosim_csv, host_cosim_csv
