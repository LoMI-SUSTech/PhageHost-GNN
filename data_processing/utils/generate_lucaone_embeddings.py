# This file runs LucaOne embedding inference and exports ESM2/ESM3-compatible .pt and .csv files for
# downstream data processing

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess, sys, os, re
from tqdm import tqdm
from Bio import SeqIO
import torch

# impoering LucaOne
LUCAONE_APP_DIR = Path("resources/embedding_models/LucaOne_weights")

# loading the FASTA protein IDs in the original order
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

# Converting the LucaOne vector_*.pt files into one standard .pt and .csv output
def convert_lucaone_pt_outputs(input_fasta: Path, output_dir: Path) -> tuple[Path, Path]:
    ids, _ = load_fasta(input_fasta)

    vectors = []
    kept_ids = []
    for protein_id in tqdm(ids, desc="Collecting LucaOne embeddings", unit="protein", file=sys.stdout):
        pt_file = output_dir / f"vector_{protein_id}.pt"
        if not pt_file.exists():
            raise FileNotFoundError(f"Missing LucaOne embedding file: {pt_file}")

        emb = torch.load(pt_file, map_location="cpu")
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()

        emb = np.asarray(emb).reshape(-1).astype(np.float32)
        vectors.append(torch.tensor(emb, dtype=torch.float32))
        kept_ids.append(protein_id)

    X = torch.stack(vectors, dim=0)

    out_prefix = output_dir / input_fasta.stem
    out_pt = out_prefix.with_suffix(".pt")
    out_csv = out_prefix.with_suffix(".csv")

    torch.save({"ids": kept_ids, "embeddings": X}, out_pt)

    df = pd.DataFrame(X.numpy(), index=kept_ids)
    df.index.name = "protein_id"
    df.to_csv(out_csv)

    print(f"LucaOne standard embeddings: {X.shape[0]}×{X.shape[1]}")
    print(f"Saved: {out_pt}")
    print(f"Saved: {out_csv}")

    return out_pt, out_csv


# Running the LucaOneApp embedding inference
def main(input_fasta, output_dir):
    input_fasta = Path(input_fasta).resolve()
    output_dir = Path(output_dir).resolve()
    model_dir = LUCAONE_APP_DIR / "models"

    if not LUCAONE_APP_DIR.exists():
        raise FileNotFoundError(f"LucaOneApp folder not found: {LUCAONE_APP_DIR}")
    if not model_dir.exists():
        raise FileNotFoundError(f"LucaOne model folder not found: {model_dir}")
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device} model=LucaOne_GPLM_v2.0 on {input_fasta.name}")
    cmd = [
        sys.executable, "-m", "algorithms.inference_embedding_lucaone",
        "--llm_dir", str(model_dir),
        "--llm_type", "lucaone_gplm",
        "--llm_version", "v2.0",
        "--llm_task_level", "token_level,span_level,seq_level,structure_level",
        "--llm_time_str", "20231125113045",
        "--llm_step", "5600000",
        "--truncation_seq_length", "100000",
        "--trunc_type", "right",
        "--seq_type", "prot",
        "--input_file", str(input_fasta),
        "--save_path", str(output_dir),
        "--embedding_type", "vector",
        "--matrix_add_special_token",
        "--embedding_complete",
        "--embedding_complete_seg_overlap",
        # "--gpu", "0",
    ]


    print("Running LucaOne embedding inference!")
    print(f"Input:  {input_fasta.name}")
    print(f"Output: {output_dir}")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    # Suppressjng the tensorFlow startup noise
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"

    result = subprocess.run(cmd, check=False, cwd=LUCAONE_APP_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print("\n[LucaOne ERROR OUTPUT]")
        print(result.stdout)
        raise RuntimeError("LucaOne embedding inference failed.")

    # Showing only the useful LucaOne status lines
    useful_keywords = ("args device:", "emb save dir:", "embedding over")
    for line in result.stdout.splitlines():
        if any(key in line for key in useful_keywords):
            print(line)

    convert_lucaone_pt_outputs(input_fasta, output_dir)
    print("LucaOne embeddings successfully generated!", flush=True)

# main
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python utils/generate_lucaone_embeddings.py <input_fasta> <output_dir>")

    main(sys.argv[1], sys.argv[2])


