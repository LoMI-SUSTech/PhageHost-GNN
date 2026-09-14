import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# CONFIGS
THREADS = "8"
PRODIGAL = "prodigal"
MMSEQS = "mmseqs"
MIN_ID = "0.30"
MIN_COV = "0.20"
EVALUE = "1e-3"

POSITION_COLS = ["genome_id", "protein_id", "protein_seq", "contig", "start", "end", "strand", "position_source"]

# loading Kaptive proteins and map each protein to its host strain
def load_host_proteins(protein_fasta, protein_index_tsv):
    index_df = pd.read_csv(protein_index_tsv, sep="\t", dtype=str)
    if not {"protein_id", "strain_id"}.issubset(index_df.columns):
        raise ValueError("Protein index must contain protein_id and strain_id.")

    protein_to_strain = dict(zip(index_df["protein_id"], index_df["strain_id"]))
    rows = []
    for rec in SeqIO.parse(str(protein_fasta), "fasta"):
        if rec.id not in protein_to_strain:
            raise ValueError(f"Protein not found in Kaptive index: {rec.id}")

        seq = str(rec.seq).replace("*", "").replace("-", "").replace(".", "").replace(" ", "").upper()
        rows.append({
            "genome_id": protein_to_strain[rec.id],
            "protein_id": rec.id,
            "protein_seq": seq,
        })

    return pd.DataFrame(rows).drop_duplicates("protein_id").reset_index(drop=True)

# Load RBPdetect proteins and derive the parent phage from each RBP ID
def load_phage_rbps(rbp_fasta):
    rows = []
    for rec in SeqIO.parse(str(rbp_fasta), "fasta"):
        genome_id = re.sub(r"_RBP\d+$", "", rec.id)
        if genome_id == rec.id:
            raise ValueError(f"Cannot derive phage ID from RBP ID: {rec.id}")

        seq = str(rec.seq).replace("*", "").replace("-", "").replace(".", "").replace(" ", "").upper()
        rows.append({
            "genome_id": genome_id,
            "protein_id": rec.id,
            "protein_seq": seq,
        })

    return pd.DataFrame(rows).drop_duplicates("protein_id").reset_index(drop=True)


# predicting proteins with Prodigal and extract their genomic coordinates
def run_prodigal(genome_dir, out_dir, mode):
    genome_dir = Path(genome_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genomes = sorted(
        p for p in genome_dir.iterdir()
        if p.suffix.lower() in {".fa", ".fna", ".fasta"}
    )

    if not genomes:
        raise ValueError(f"No genome FASTA files found in {genome_dir}")

    rows = []
    for genome in tqdm(genomes, desc="Prodigal", unit="genome"):
        genome_id = genome.stem
        faa = out_dir / f"{genome_id}.faa"
        gff = out_dir / f"{genome_id}.gff"

        subprocess.run([
            PRODIGAL,
            "-i", str(genome),
            "-a", str(faa),
            "-o", str(gff),
            "-f", "gff",
            "-p", mode,
            "-q",
        ], check=True)

        for rec in SeqIO.parse(str(faa), "fasta"):
            coords = re.findall(r"#\s*([-\d]+)", rec.description)
            if len(coords) < 3:
                continue

            seq = str(rec.seq).replace("*", "").replace("-", "").replace(".", "").replace(" ", "").upper()
            rows.append({
                "genome_id": genome_id,
                "target_id": rec.id,
                "target_seq": seq,
                "contig": rec.id.rsplit("_", 1)[0],
                "start": int(coords[0]),
                "end": int(coords[1]),
                "strand": int(coords[2]),
            })

    return pd.DataFrame(rows)

# Recover positions through exact same-genome sequence matches.
def recover_exact_matches(query_df, target_df):
    exact = query_df.merge(target_df, left_on=["genome_id", "protein_seq"], right_on=["genome_id", "target_seq"], how="inner")
    exact = (exact.sort_values(["protein_id", "contig", "start", "end"]).drop_duplicates("protein_id").copy())
    exact["position_source"] = "prodigal_exact"
    return exact[POSITION_COLS]

# Recover remaining protein positions using same-genome MMseqs searches.
def recover_mmseqs_matches(query_df, target_df, out_dir, targeted=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recovered = []

    for genome_id, queries in tqdm(query_df.groupby("genome_id"), desc="Targeted MMseqs" if targeted else "MMseqs", unit="genome"):
        targets = target_df[target_df["genome_id"] == genome_id]
        if targets.empty:
            continue

        tag = "targeted" if targeted else "main"
        query_faa = out_dir / f"{genome_id}_{tag}_query.faa"
        target_faa = out_dir / f"{genome_id}_{tag}_target.faa"
        hits_tsv = out_dir / f"{genome_id}_{tag}_hits.tsv"
        tmp_dir = out_dir / f"{genome_id}_{tag}_tmp"

        with query_faa.open("w") as f:
            for _, row in queries.iterrows():
                f.write(f">{row['protein_id']}\n{row['protein_seq']}\n")

        with target_faa.open("w") as f:
            for _, row in targets.iterrows():
                f.write(f">{row['target_id']}\n{row['target_seq']}\n")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        hits_tsv.unlink(missing_ok=True)

        search_args = [
            "--threads", THREADS,
            "--max-seqs", "10000",
            "--format-output", "query,target,fident,qcov,tcov,evalue,bits",
            "-v", "1",
        ]

        if targeted:
            search_args += [
                "--min-seq-id", "0.0",
                "-e", "100000",
                "--cov-mode", "0",
                "-c", "0.0",
                "-s", "7.5",
            ]
        else:
            search_args += [
                "--min-seq-id", MIN_ID,
                "-e", EVALUE,
                "--cov-mode", "1",
                "-c", MIN_COV,
            ]

        subprocess.run([
            MMSEQS, "easy-search",
            str(query_faa),
            str(target_faa),
            str(hits_tsv),
            str(tmp_dir),
            *search_args,
        ], check=True)

        if not hits_tsv.exists() or hits_tsv.stat().st_size == 0:
            continue

        hits = pd.read_csv(hits_tsv, sep="\t", header=None, names=["protein_id", "target_id", "pident", "qcov", "tcov", "evalue", "bits"])
        hits = (hits.sort_values(["protein_id", "bits", "pident", "qcov", "tcov"], ascending=[True, False, False, False, False]).drop_duplicates("protein_id"))
        hits = hits.merge(targets[["target_id", "contig", "start", "end", "strand"]], on="target_id", how="left")
        hits = hits.merge(queries[["protein_id", "protein_seq"]], on="protein_id", how="left")
        hits["genome_id"] = genome_id
        hits["position_source"] = (
            "prodigal_mmseqs_targeted_rescue"
            if targeted else
            "prodigal_mmseqs"
        )
        recovered.append(hits[POSITION_COLS])

    if not recovered:
        return pd.DataFrame(columns=POSITION_COLS)

    return pd.concat(recovered, ignore_index=True)


# Recover all query-protein positions using exact matching followed by MMseqs rescue

def recover_positions(genome_dir, query_df, out_dir, output_name, prodigal_mode):
    genome_dir = Path(genome_dir)
    out_dir = Path(out_dir)
    prodigal_dir = out_dir / "prodigal"
    mmseqs_dir = out_dir / "mmseqs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tool in [PRODIGAL, MMSEQS]:
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found in PATH.")

    genome_files = [
        p for p in genome_dir.iterdir()
        if p.suffix.lower() in {".fa", ".fna", ".fasta"}
    ]
    genome_ids = {p.stem for p in genome_files}
    missing_genomes = sorted(set(query_df["genome_id"]) - genome_ids)

    if missing_genomes:
        raise ValueError(f"Query proteins refer to genomes not found in {genome_dir}: {missing_genomes[:10]}")

    print(f"Query proteins: {len(query_df)}")
    print(f"Genomes: {query_df['genome_id'].nunique()}")

    target_df = run_prodigal(genome_dir, prodigal_dir, prodigal_mode)
    print(f"Prodigal proteins: {len(target_df)}")

    exact_df = recover_exact_matches(query_df, target_df)
    print(f"Exact matches: {len(exact_df)}")

    remaining = query_df[~query_df["protein_id"].isin(exact_df["protein_id"])]
    mmseqs_df = recover_mmseqs_matches(remaining, target_df, mmseqs_dir, targeted=False)
    print(f"MMseqs matches: {len(mmseqs_df)}")

    result = pd.concat([exact_df, mmseqs_df], ignore_index=True)
    result = result.drop_duplicates("protein_id")

    remaining = query_df[~query_df["protein_id"].isin(result["protein_id"])]
    if not remaining.empty:
        rescue_df = recover_mmseqs_matches(remaining, target_df, mmseqs_dir, targeted=True)
        print(f"Targeted rescue  : {len(rescue_df)}")
        result = pd.concat([result, rescue_df], ignore_index=True).drop_duplicates("protein_id")

    result = (result.sort_values(["genome_id", "contig", "start", "end"], kind="mergesort").reset_index(drop=True))
    out_csv = out_dir / output_name
    result.to_csv(out_csv, index=False)

    missing = query_df[~query_df["protein_id"].isin(result["protein_id"])][["genome_id", "protein_id"]]
    missing_csv = out_dir / output_name.replace("_positions.csv", "_missing_proteins.csv")
    if missing.empty:
        missing_csv.unlink(missing_ok=True)
    else:
        missing.to_csv(missing_csv, index=False)

    print(f"\nRecovered: {len(result)} / {len(query_df)}")
    print(result["position_source"].value_counts())
    print(f"Saved: {out_csv}")

    if not missing.empty:
        print(f"Missing proteins: {len(missing)}")
        print(f"Saved: {missing_csv}")

    return out_csv


#  recover genomic positions for Kaptive K-locus host proteins
def recover_host_positions(genome_dir, protein_fasta, protein_index_tsv, out_dir):
    query_df = load_host_proteins(Path(protein_fasta), Path(protein_index_tsv))
    return recover_positions(
        genome_dir=Path(genome_dir),
        query_df=query_df,
        out_dir=Path(out_dir),
        output_name="host_bacformer_positions.csv",
        prodigal_mode="meta",
    )

# recover genomic positions for RBPdetect phage proteins
def recover_phage_positions(genome_dir, rbp_fasta, out_dir):
    query_df = load_phage_rbps(Path(rbp_fasta))
    return recover_positions(
        genome_dir=Path(genome_dir),
        query_df=query_df,
        out_dir=Path(out_dir),
        output_name="phage_bacformer_positions.csv",
        prodigal_mode="single",
    )
