from pathlib import Path
import inspect
import os
import re
import sys

import esm
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

# CONFIGS
BACFORMER_DIR = Path("resources/embedding_models/BacFormer_weights")

ESM_MAX_LEN = 1022
ESM_BATCH_SIZE = 128
BACFORMER_MAX_LEN = 6000
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYUBZX")

# Load the local BacFormer model and determine its required embedding/token arguments.
def load_bacformer_model(device, dtype):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = AutoModel.from_pretrained(
        str(BACFORMER_DIR),
        trust_remote_code=True,
        local_files_only=True,
    ).to(device=device, dtype=dtype).eval()

    params = inspect.signature(model.forward).parameters
    if "protein_embeddings" in params:
        embed_kw = "protein_embeddings"
    elif "inputs_embeds" in params:
        embed_kw = "inputs_embeds"
    else:
        raise RuntimeError("BacFormer forward() has no supported protein embedding argument.")

    objects = [
        getattr(model, "embeddings", None),
        getattr(model, "config", None),
    ]

    token_names = {
        "prot": ["prot_emb_token_id", "protein_token_id", "protein_emb_token_id"],
        "bos": ["bos_token_id", "cls_token_id"],
        "eos": ["eos_token_id", "sep_token_id"],
    }

    token_ids = {}
    for key, names in token_names.items():
        token_ids[key] = next(
            (
                int(getattr(obj, name))
                for obj in objects if obj is not None
                for name in names
                if getattr(obj, name, None) is not None
            ),
            None,
        )

    if token_ids["prot"] is None:
        raise RuntimeError("Could not determine BacFormer protein token ID.")
    print(f"Loaded BacFormer on {device}")

    return model, embed_kw, token_ids

# generating the 480-dimensional ESM2 protein embeddings required by BacFormer
@torch.no_grad()
def embed_esm2(sequences, model, alphabet, batch_converter, device):
    vectors = []
    for i in range(0, len(sequences), ESM_BATCH_SIZE):
        batch = []
        for seq in sequences[i:i + ESM_BATCH_SIZE]:
            seq = (
                str(seq)
                .replace(" ", "")
                .replace("-", "")
                .replace(".", "")
                .replace("*", "")
                .upper()
            )

            seq = "".join(
                aa if aa in VALID_AA else "X"
                for aa in seq
            )

            batch.append(("", seq[:ESM_MAX_LEN] or "X"))

        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        output = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        reps = output["representations"][model.num_layers]
        mask = tokens != alphabet.padding_idx
        if alphabet.cls_idx is not None:
            mask &= tokens != alphabet.cls_idx

        if alphabet.eos_idx is not None:
            mask &= tokens != alphabet.eos_idx

        for j in range(len(batch)):
            vectors.append(reps[j][mask[j]].mean(dim=0).float().cpu())

    return torch.stack(vectors)

# contextualize genome-ordered protein embeddings with BacFormer and save them
@torch.no_grad()
def main(positions_csv, out_dir):
    positions_csv = Path(positions_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not positions_csv.exists():
        raise FileNotFoundError(f"Input file not found: {positions_csv}")

    out_prefix = out_dir / positions_csv.stem.replace("_positions", "_embeddings")
    out_pt = out_prefix.with_suffix(".pt")
    out_csv = out_prefix.with_suffix(".csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    print(f"Device: {device}")

    bacformer, embed_kw, token_ids = load_bacformer_model(device, dtype)
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model = esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    df = pd.read_csv(positions_csv)

    required = {
        "genome_id",
        "protein_id",
        "protein_seq",
        "contig",
        "start",
        "end",
    }

    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"No proteins found in {positions_csv}")

    df["start"] = pd.to_numeric(df["start"], errors="raise")
    df["end"] = pd.to_numeric(df["end"], errors="raise")

    df["_contig_order"] = df["contig"].astype(str).map(
        lambda x: re.sub(
            r"\d+",
            lambda m: f"{int(m.group()):012d}",
            x.lower(),
        )
    )

    df = (df.sort_values(["genome_id", "_contig_order", "start", "end"], kind="mergesort").reset_index(drop=True))
    print(f"Embedding {len(df)} proteins from {df['genome_id'].nunique()} genomes")

    all_ids = []
    all_genomes = []
    all_order = []
    all_vectors = []

    groups = df.groupby("genome_id", sort=False)
    for genome_id, sub in tqdm(
        groups,
        total=df["genome_id"].nunique(),
        desc="BacFormer genomes",
        unit="genome",
        dynamic_ncols=True,
    ):
        ids = sub["protein_id"].astype(str).tolist()
        x = embed_esm2(
            sub["protein_seq"].tolist(),
            esm_model,
            alphabet,
            batch_converter,
            device,
        )

        n = len(ids)
        use_special = (
            token_ids["bos"] is not None
            and token_ids["eos"] is not None
        )

        seq_len = n + 2 if use_special else n
        if seq_len > BACFORMER_MAX_LEN:
            raise RuntimeError(f"{genome_id}: {seq_len} tokens exceeds BacFormer limit {BACFORMER_MAX_LEN}.")

        tokens = torch.full(
            (1, seq_len),
            token_ids["prot"],
            dtype=torch.long,
        )

        embeddings = torch.zeros((1, seq_len, 480), dtype=torch.float32)
        if use_special:
            tokens[0, 0] = token_ids["bos"]
            tokens[0, -1] = token_ids["eos"]
            embeddings[0, 1:-1] = x
            keep = slice(1, -1)
        else:
            embeddings[0] = x
            keep = slice(None)

        output = bacformer(
            **{
                embed_kw: embeddings.to(device=device, dtype=dtype),
                "special_tokens_mask": tokens.to(device),
                "attention_mask": torch.ones((1, seq_len), dtype=torch.long, device=device),
            },
            return_dict=True,
        )

        h = (output["last_hidden_state"].squeeze(0)[keep].float().cpu())
        if n > 1:
            similarity = float(F.cosine_similarity(h[0], h[1], dim=0))
            if abs(similarity) > 0.9999:
                raise RuntimeError(f"{genome_id}: BacFormer outputs appear nearly constant.")

        all_vectors.append(h)
        all_ids.extend(ids)
        all_genomes.extend([genome_id] * n)
        all_order.extend(range(n))

    embeddings = torch.cat(all_vectors, dim=0)
    torch.save(
        {
            "ids": all_ids,
            "embeddings": embeddings,
            "meta": {
                "genome_id": all_genomes,
                "order_in_genome": all_order,
            },
        },
        out_pt,
    )

    out_df = pd.DataFrame(embeddings.numpy(), index=all_ids)
    out_df.index.name = "protein_id"
    out_df.to_csv(out_csv)

    print(f"{positions_csv.name}: {embeddings.shape[0]} × {embeddings.shape[1]}")
    print(f"PT : {out_pt}")
    print(f"CSV: {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python utils/generate_bacformer_embeddings.py "
            "<positions_csv> <out_dir>"
        )

    main(sys.argv[1], sys.argv[2])

