from typing import Dict, Tuple

import numpy as np
import torch


def hit_precision_recall_at_k_from_ranked(ranked01: np.ndarray, npos: int, max_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranked01 = np.asarray(ranked01, dtype=np.int32)
    if ranked01.size == 0 or npos <= 0:
        z = np.zeros(max_k, dtype=np.float32)
        return z, z.copy(), z.copy()

    upto = min(max_k, ranked01.size)
    csum = np.cumsum(ranked01[:upto]).astype(np.float32)
    hit = (csum > 0).astype(np.float32)

    precision = (csum / np.arange(1, upto + 1, dtype=np.float32))
    recall = (csum / float(npos))

    if upto < max_k:
        hit = np.pad(
            hit,
            (0, max_k - upto),
            constant_values=float(hit[-1]),
        )

        precision = np.pad(
            precision,
            (0, max_k - upto),
            constant_values=float(precision[-1]),
        )

        recall = np.pad(
            recall,
            (0, max_k - upto),
            constant_values=float(recall[-1]),
        )

    return (
        hit.astype(np.float32),
        precision.astype(np.float32),
        recall.astype(np.float32),
    )


@torch.no_grad()
def score_all_phages(z: Dict[str, torch.Tensor], model, host_idx: int, n_phages: int) -> np.ndarray:
    device = z["phage"].device
    ph = torch.arange(n_phages, device=device)
    ho = torch.full((n_phages,), int(host_idx), device=device)
    edge_index = torch.stack(
        [ph, ho],
        dim=0,
    )
    logits = model.decode(z, edge_index)
    return (torch.sigmoid(logits).cpu().numpy().astype(np.float32))


@torch.no_grad()
def score_all_hosts(z: Dict[str, torch.Tensor], model, phage_idx: int, n_hosts: int) -> np.ndarray:
    device = z["phage"].device
    ho = torch.arange(n_hosts, device=device)
    ph = torch.full((n_hosts,), int(phage_idx), device=device)
    edge_index = torch.stack(
        [ph, ho],
        dim=0,
    )

    logits = model.decode(z, edge_index)
    return (torch.sigmoid(logits).cpu().numpy().astype(np.float32))
