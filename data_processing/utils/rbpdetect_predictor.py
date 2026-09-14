# RBPdetect utility: predicts RBPs from PHANOTATE .faa files and saves clean RBP FASTAs.

from pathlib import Path
import os, re, warnings
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("resources/PhageRBPdetection/RBPdetect_model/RBPdetect_v3_ESMfineT33")

# Clean final protein sequences before FASTA export
def clean_seq(seq):
    return re.sub(r"[^A-Z]", "", str(seq).upper().replace("#", "").replace("*", ""))

# Load RBPdetect model once
def load_rbpdetect_model(model_dir: Path = MODEL_DIR, gpu: int = 0):
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"RBPdetect model directory not found: {model_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval().to(device)

    print(f"RBPdetect model loaded on: {device}")
    return tokenizer, model, device


# Run RBPdetect on one PHANOTATE .faa file and save prediction CSV
def predict_one_faa(faa_file: Path, out_csv: Path, tokenizer, model, device) -> pd.DataFrame:
    records = list(SeqIO.parse(str(faa_file), "fasta"))
    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in records:
            encoding = tokenizer(str(r.seq), return_tensors="pt", truncation=True).to(device)

            with torch.no_grad():
                output = model(**encoding)
                pred = int(output.logits.argmax(-1).cpu().item())
                score = float(output.logits.softmax(-1)[:, 1].cpu().item())

            rows.append({"protein_name": r.id, "preds": pred, "score": score})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# extracting predicted RBPs from one .faa file using the RBPdetect prediction table
def extract_rbps_one(faa_file: Path, pred_df: pd.DataFrame, out_faa: Path, phage_id: str, min_len=200, max_len=1500):
    pred_pos = set(pred_df.loc[pred_df["preds"] == 1, "protein_name"].astype(str))
    records, kept, dropped = [], [], []

    for r in SeqIO.parse(str(faa_file), "fasta"):
        if r.id not in pred_pos:
            continue

        seq = clean_seq(r.seq)
        length = len(seq)
        if min_len <= length <= max_len:
            rbp_id = f"{phage_id}_RBP{len(records) + 1:03d}"
            records.append(SeqRecord(Seq(seq), id=rbp_id, description=""))
            kept.append({"phage_id": phage_id, "old_protein_name": r.id, "rbp_id": rbp_id, "length_aa": length})
        else:
            dropped.append({"phage_id": phage_id, "old_protein_name": r.id, "length_aa": length})

    if records:
        out_faa.parent.mkdir(parents=True, exist_ok=True)
        SeqIO.write(records, out_faa, "fasta")

    return records, kept, dropped


# running RBPdetect and RBP FASTA extraction for all clean PHANOTATE .faa files
def run_rbpdetect_batch(phanotate_outdir: Path, rbp_outdir: Path, min_len=200, max_len=1500) -> dict:
    clean_faa_dir = Path(phanotate_outdir) / "clean_faa"
    rbp_outdir = Path(rbp_outdir)

    pred_dir = rbp_outdir / "rbp_predictions"
    rbp_faa_dir = rbp_outdir / "phage_rbp_fastas"
    combined_faa = rbp_outdir / "all_phage_rbps.faa"

    faa_files = sorted(clean_faa_dir.glob("*_phanotate.faa"))
    if not faa_files:
        raise FileNotFoundError(f"No clean PHANOTATE .faa files found in: {clean_faa_dir}")

    tokenizer, model, device = load_rbpdetect_model()

    all_records, all_kept, all_dropped, summary = [], [], [], []

    for faa_file in tqdm(faa_files, desc="RBPdetect + extracting RBPs", dynamic_ncols=True):
        phage_id = faa_file.stem.replace("_phanotate", "")
        pred_csv = pred_dir / f"{phage_id}_rbp_predictions.csv"
        rbp_faa = rbp_faa_dir / f"{phage_id}.faa"

        pred_df = predict_one_faa(faa_file, pred_csv, tokenizer, model, device)
        records, kept, dropped = extract_rbps_one(faa_file, pred_df, rbp_faa, phage_id, min_len, max_len)

        all_records.extend(records)
        all_kept.extend([{**x, "rbp_fasta_file": str(rbp_faa), "prediction_file": str(pred_csv)} for x in kept])
        all_dropped.extend(dropped)

        summary.append({
            "phage_id": phage_id,
            "prediction_file": str(pred_csv),
            "rbp_fasta_file": str(rbp_faa) if records else "",
            "n_predicted_rbps_raw": int((pred_df["preds"] == 1).sum()),
            "n_rbps_kept": len(records),
            "n_rbps_dropped_by_length": len(dropped),
        })

    if all_records:
        SeqIO.write(all_records, combined_faa, "fasta")

    pd.DataFrame(all_kept).to_csv(rbp_outdir / "phage_rbp_index.csv", index=False)
    pd.DataFrame(all_dropped).to_csv(rbp_outdir / "dropped_rbp_candidates_by_length.csv", index=False)
    pd.DataFrame(summary).to_csv(rbp_outdir / "phage_rbp_summary.csv", index=False)

    print(f"Done. Phages: {len(faa_files)} | RBPs kept: {len(all_kept)}")
    print(f"Combined RBP FASTA: {combined_faa}")

    return {
        "prediction_dir": pred_dir,
        "rbp_faa_dir": rbp_faa_dir,
        "combined_faa": combined_faa,
        "index_csv": rbp_outdir / "phage_rbp_index.csv",
        "summary_csv": rbp_outdir / "phage_rbp_summary.csv",
    }
