import torch
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import esm  # fair-esm
import re, sys

# CONFIGS
BATCH = 16
MAX_LEN = 1022  # ESM-2 token limit

# the main fasta file loading function
def load_fasta(fp: Path):
    fp = Path(fp)
    recs = list(SeqIO.parse(str(fp), "fasta"))
    if not recs:
        raise RuntimeError(f"No sequences in {fp}")

    def safe_name(x: str):
        x = x.strip()
        x = re.sub(r"\s+", "", x)
        x = re.sub(r"[^A-Za-z0-9._-]", "_", x)
        return x
    return [safe_name(r.description) for r in recs], [str(r.seq) for r in recs]


@torch.no_grad()
def embed_fasta(fasta_path, out_prefix, model, alphabet):
    fasta_path = Path(fasta_path)
    out_prefix = Path(out_prefix)

    device = next(model.parameters()).device.type
    print(f"Device={device} model=esm2_t33_650M_UR50D (1280-D) on {fasta_path.name}")
    batch_converter = alphabet.get_batch_converter()
    ids, seqs = load_fasta(fasta_path)
    out_pt = out_prefix.with_suffix(".pt")
    out_csv = out_prefix.with_suffix(".csv")

    vecs = []
    for i in range(0, len(seqs), BATCH):
        batch_ids = ids[i:i + BATCH]
        batch_seqs = [s[:MAX_LEN] for s in seqs[i:i + BATCH]]

        data = list(zip(batch_ids, batch_seqs))
        labels, strs, toks = batch_converter(data)
        toks = toks.to(device)

        out = model(toks, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]  # [B, L, 1280]

        batch_lens = (toks != alphabet.padding_idx).sum(1)

        pooled_batch = []
        for j, tokens_len in enumerate(batch_lens):
            pooled_batch.append(reps[j, 1 : tokens_len - 1].mean(0))  # exclude BOS/EOS

        pooled = torch.stack(pooled_batch, dim=0)  # [B, 1280]
        vecs.append(pooled.to(torch.float32).cpu())

        if (i // BATCH) % 20 == 0:
            print(f"[INFO] {i}/{len(seqs)}")

    X = torch.cat(vecs, dim=0)  # [N, 1280]
    assert X.shape[0] == len(ids), "Row count mismatch."

    torch.save({"ids": ids, "embeddings": X}, out_pt)
    df = pd.DataFrame(X.numpy(), index=ids)
    df.index.name = "protein_id"
    df.to_csv(out_csv)

    print(f"[OK] {fasta_path.name}: {X.shape[0]}×{X.shape[1]} → {out_pt} / {out_csv}")


def main(input_fasta, out_dir):
    input_fasta = Path(input_fasta)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = out_dir / input_fasta.stem

    # load model once
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (device == "cuda") and torch.cuda.is_bf16_supported()
    model = model.to(device).eval()
    if use_bf16:
        model = model.to(torch.bfloat16)

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    embed_fasta(input_fasta, out_prefix, model, alphabet)
    print("[DONE] ESM-2 embeddings ready.")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python utils/generate_esm2_embeddings.py <input_fasta> <out_dir>")
    main(sys.argv[1], sys.argv[2])
