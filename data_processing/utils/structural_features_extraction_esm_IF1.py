# Extracting structural features from CIF structures using ESM-IF1 encoder

import inspect
import sys
import warnings
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import esm
from esm.inverse_folding.util import CoordBatchConverter

warnings.filterwarnings("ignore", message="Regression weights not found, predicting contacts will not produce correct results.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

def cif_to_backbone_coords(cif_path):
    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0 or len(st[0]) == 0:
        raise ValueError("No model/chain found")

    chain = next(iter(st[0]), None)
    if chain is None:
        raise ValueError("No chain found")

    coords, mask = [], []
    for res in chain:
        if res.name in ("HOH", "WAT"):
            continue

        atoms = []
        for name in ("N", "CA", "C"):
            atom = res.find_atom(name, "*")
            if atom is None:
                atoms.append(None)
            else:
                p = atom.pos
                atoms.append(np.array([p.x, p.y, p.z], dtype=np.float32))

        ok = all(a is not None for a in atoms)
        mask.append(ok)
        coords.append(
            np.stack(atoms) if ok
            else np.zeros((3, 3), dtype=np.float32)
        )

    if not coords:
        raise ValueError("No residues parsed")

    return np.stack(coords), np.asarray(mask, dtype=bool)

def load_model():
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(DEVICE).eval()

    return model, alphabet, CoordBatchConverter(alphabet)


def get_last_hidden(extra):
    if "inner_states" in extra and extra["inner_states"]:
        return extra["inner_states"][-1]
    if "decoder_inner_states" in extra and extra["decoder_inner_states"]:
        return extra["decoder_inner_states"][-1]
    if "hidden_states" in extra:
        return extra["hidden_states"]
    raise RuntimeError(f"Cannot find hidden states: {list(extra.keys())}")


@torch.no_grad()
def embed_structure(model, alphabet, batch_converter, coords, mask):
    coords = coords[mask]
    length = len(coords)
    if length < 10:
        raise ValueError(f"Too few valid residues: {length}")

    batch = [(coords, np.ones(length, dtype=np.float32), "A" * length)]
    res = batch_converter(batch)

    coords_b = res[0].to(DEVICE, dtype=DTYPE)
    conf_b = res[1].to(DEVICE, dtype=DTYPE)
    tokens = res[3].to(DEVICE)

    padding_mask = (
        res[4].to(DEVICE)
        if len(res) >= 5
        else tokens.eq(getattr(alphabet, "padding_idx", 1))
    )

    candidates = {
        "coords": coords_b,
        "padding_mask": padding_mask,
        "confidence": conf_b,
        "prev_output_tokens": tokens[:, :-1],
    }

    sig = inspect.signature(model.forward)
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}

    out = model(**kwargs)
    extra = out[1] if isinstance(out, tuple) and isinstance(out[1], dict) else out

    states = get_last_hidden(extra)
    states = states.transpose(0, 1)[:, 1:1 + length, :]

    return states.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def main(cif_dir, out_dir):
    cif_dir = Path(cif_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {cif_dir}")

    model, alphabet, batch_converter = load_model()

    rows = []
    failures = []

    for cif_path in tqdm(cif_files, desc="ESM-IF1 embedding", unit="protein"):
        try:
            coords, mask = cif_to_backbone_coords(cif_path)
            emb = embed_structure(
                model,
                alphabet,
                batch_converter,
                coords,
                mask,
            )
            rows.append([cif_path.stem] + emb.tolist())

        except Exception as e:
            failures.append(cif_path.stem)
            tqdm.write(f"[FAIL] {cif_path.stem}: {e}")

    if not rows:
        raise RuntimeError("All structures failed")

    df = pd.DataFrame(rows)
    df.columns = ["protein_id"] + [str(i) for i in range(df.shape[1] - 1)]

    out_csv = out_dir / f"{cif_dir.name}_embeddings.csv"
    df.to_csv(out_csv, index=False)

    if failures:
        raise RuntimeError(f"{len(failures)} structures failed: {failures[:10]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python utils/structural_features_extraction_esm_IF1.py "
            "<cif_dir> <out_dir>"
        )

    main(sys.argv[1], sys.argv[2])
