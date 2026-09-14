from pathlib import Path
import re, sys
import torch
import pandas as pd
from Bio import SeqIO

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from tqdm import tqdm


# ESMC configs
MODEL_WEIGHTS = Path("resources/embedding_models/ESMC_weights/data/weights/esmc_600m_2024_12_v0.pth")
D_MODEL = 1152
N_HEADS = 18
N_LAYERS = 36

# Loading the FASTA records and converting FASTA descriptions into safe protein IDs
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

# Loading the local ESMC model weights from defined path
def load_esmc_model(device: str):
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")

    tokenizer = EsmSequenceTokenizer()
    model = ESMC(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, tokenizer=tokenizer).to(device)
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded ESMC model on {device}")
    print(f"Weights: {MODEL_WEIGHTS}")
    return model


# mean-pooling of one protein sequence into a single ESMC vector
@torch.no_grad()
def embed_sequence(model, sequence: str):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    emb = logits_output.embeddings.detach().cpu()
    if emb.ndim == 3:
        emb = emb.squeeze(0)

    # prefering the residue-only span if special tokens are present
    if emb.shape[0] >= len(sequence) + 2:
        emb = emb[1:len(sequence) + 1]
    else:
        emb = emb[:len(sequence)]
    return emb.mean(dim=0).to(torch.float32)


#generating the ESMC protein-level embeddings from one FASTA files
def main(input_fasta, out_dir):
    input_fasta = Path(input_fasta)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    out_prefix = out_dir / input_fasta.stem
    out_pt = out_prefix.with_suffix(".pt")
    out_csv = out_prefix.with_suffix(".csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device} model=ESMC_600M_2024_12_v0 ({D_MODEL}-D) on {input_fasta.name}")

    model = load_esmc_model(device)
    ids, seqs = load_fasta(input_fasta)
    print(f"Embedding {len(ids)} proteins from {input_fasta.name}")

    vecs = []
    for protein_id, seq in tqdm(
        list(zip(ids, seqs)),
        desc=f"Embedding {input_fasta.name}",
        dynamic_ncols=True,
    ):
        vec = embed_sequence(model, seq)
        vecs.append(vec)

    X = torch.stack(vecs, dim=0)
    if X.shape[0] != len(ids):
        raise RuntimeError("Row count mismatch after embedding.")

    torch.save({"ids": ids, "embeddings": X}, out_pt)

    df = pd.DataFrame(X.numpy(), index=ids)
    df.index.name = "protein_id"
    df.to_csv(out_csv)

    print(f"{input_fasta.name}: {X.shape[0]}x{X.shape[1]} -> {out_pt} / {out_csv}")
    print("ESMC embeddings ready!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python utils/generate_esmc_embeddings.py <input_fasta> <out_dir>")

    main(sys.argv[1], sys.argv[2])

