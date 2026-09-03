import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


def load_similarity_matrix(path: Path, target_ids: List[str]):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")

    raw_ids = df.index.tolist()
    matrix = df.to_numpy(np.float32)

    missing = [x for x in target_ids if x not in set(raw_ids)]
    if missing:
        raise ValueError(f"IDs missing from similarity matrix (examples: {missing[:10]})")

    if np.isnan(matrix).any():
        raise ValueError(f"Similarity matrix contains NaNs: {path}")

    return df, raw_ids, matrix


def compute_quantile_thresholds(matrix: np.ndarray, q_list: List[float]) -> Dict[float, float]:
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return {float(q): float(np.quantile(off_diagonal, float(q))) for q in q_list}


def greedy_groups_raw(matrix: np.ndarray, raw_ids: List[str], threshold: float, target_ids: List[str]):
    group_ids = np.full(matrix.shape[0], -1, dtype=np.int32)
    group_id = 0

    for i in range(matrix.shape[0]):
        if group_ids[i] != -1:
            continue

        mask = (matrix[i] >= threshold) & (group_ids == -1)
        group_ids[mask] = group_id
        group_id += 1

    group_by_id = dict(zip(raw_ids, group_ids.tolist()))
    groups = {}

    for item in target_ids:
        groups.setdefault(group_by_id[item], []).append(item)

    return sorted(groups.values(), key=len, reverse=True)


def prepare_similarity_groups(
    setting: str,
    q_list: List[float],
    hosts: List[str],
    phages: List[str],
    host_sim_path: Path,
    phage_sim_path: Path,
):
    results = {
        "host_sim_df": None,
        "phage_sim_df": None,
        "host_matrix": None,
        "phage_matrix": None,
        "host_thr_by_q": {},
        "phage_thr_by_q": {},
        "host_groups_by_q": {},
        "phage_groups_by_q": {},
    }

    if setting in {"host_unseen", "both_unseen"}:
        host_df, host_ids, host_matrix = load_similarity_matrix(host_sim_path, hosts)
        host_thr_by_q = compute_quantile_thresholds(host_matrix, q_list)

        host_groups_by_q = {
            float(q): greedy_groups_raw(host_matrix, host_ids, host_thr_by_q[float(q)], hosts)
            for q in q_list
        }

        results.update({
            "host_sim_df": host_df,
            "host_matrix": host_matrix,
            "host_thr_by_q": host_thr_by_q,
            "host_groups_by_q": host_groups_by_q,
        })

    if setting in {"phage_unseen", "both_unseen"}:
        phage_df, phage_ids, phage_matrix = load_similarity_matrix(phage_sim_path, phages)
        phage_thr_by_q = compute_quantile_thresholds(phage_matrix, q_list)

        phage_groups_by_q = {
            float(q): greedy_groups_raw(phage_matrix, phage_ids, phage_thr_by_q[float(q)], phages)
            for q in q_list
        }

        results.update({
            "phage_sim_df": phage_df,
            "phage_matrix": phage_matrix,
            "phage_thr_by_q": phage_thr_by_q,
            "phage_groups_by_q": phage_groups_by_q,
        })

    return results

