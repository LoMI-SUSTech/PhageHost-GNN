# Running PHANOTATE on raw phage genomes to save cleaned .faa and .gbk output files

from pathlib import Path
import os
import re
import subprocess
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm

# Converting a filename or phage name into a safe output prefix
def safe_name(x: str) -> str:
    x = x.strip()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^A-Za-z0-9._-]", "_", x)
    return x

# RewritingPHANOTATE FASTA headers as phageID_001, phageID_002, ...
def clean_phanotate_faa_headers(raw_faa: Path, clean_faa: Path, phage_id: str, map_csv: Path) -> None:
    records = []
    rows = []
    for i, record in enumerate(SeqIO.parse(raw_faa, "fasta"), start=1):
        old_id = record.id
        new_id = f"{phage_id}_{i:03d}"

        records.append(
            SeqRecord(
                record.seq,
                id=new_id,
                description="",
            )
        )
        rows.append({
            "phage_id": phage_id,
            "old_protein_name": old_id,
            "new_protein_name": new_id,
            "protein_order": i,
        })
    if not records:
        raise ValueError(f"There is no PHANOTATE protein records found here: {raw_faa}!")

    SeqIO.write(records, clean_faa, "fasta")
    pd.DataFrame(rows).to_csv(map_csv, index=False)

# Runningg PHANOTATE on one raw phage genome to save protein FASTA + GenBank files
def run_phanotate_one(input_fasta: Path, output_dir: Path, keep_raw: bool = False) -> dict:
    input_fasta = Path(input_fasta)
    output_dir = Path(output_dir)

    clean_faa_dir = output_dir / "clean_faa"
    gbk_dir = output_dir / "genbank"
    map_dir = output_dir / "protein_name_maps"
    tmp_dir = output_dir / "tmp_raw_faa"
    for d in [clean_faa_dir, gbk_dir, map_dir, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    phage_id = safe_name(input_fasta.stem)

    raw_faa = tmp_dir / f"{phage_id}_phanotate_raw.faa"
    clean_faa = clean_faa_dir / f"{phage_id}_phanotate.faa"
    gbk_out = gbk_dir / f"{phage_id}_phanotate.gbk"
    map_csv = map_dir / f"{phage_id}_phanotate_protein_name_map.csv"

    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore::UserWarning"

    with open(raw_faa, "w") as f:
        subprocess.run(
            ["phanotate.py", str(input_fasta), "-f", "faa"],
            stdout=f,
            stderr=subprocess.DEVNULL,
            env=env,
            check=True,
        )

    with open(gbk_out, "w") as f:
        subprocess.run(
            ["phanotate.py", str(input_fasta), "-f", "genbank"],
            stdout=f,
            stderr=subprocess.DEVNULL,
            env=env,
            check=True,
        )

    clean_phanotate_faa_headers(raw_faa, clean_faa, phage_id, map_csv)

    if not keep_raw:
        raw_faa.unlink(missing_ok=True)

    return {
        "phage_id": phage_id,
        "input_fasta": input_fasta,
        "faa": clean_faa,
        "gbk": gbk_out,
        "protein_name_map": map_csv,
    }


# Running PHANOTATE on all phage genome FASTA files in the DIR
def run_phanotate_batch(genome_dir: Path, output_dir: Path, keep_raw: bool = False) -> list[dict]:
    genome_dir = Path(genome_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genomes = []
    for ext in ["*.fna", "*.fa", "*.fasta"]:
        genomes.extend(genome_dir.glob(ext))

    genomes = sorted(genomes)
    if not genomes:
        raise FileNotFoundError(f"No phage genome FASTA files found in: {genome_dir}")

    results = []
    for genome in tqdm(genomes, desc="Running PHANOTATE", dynamic_ncols=True):
        results.append(run_phanotate_one(genome, output_dir, keep_raw=keep_raw))

    print(f"PHANOTATE run completed for: {len(results)} phage genomes")
    print(f"Clean FAA files: {output_dir / 'clean_faa'}")
    print(f"GenBank files: {output_dir / 'genbank'}")
    print(f"Protein name maps: {output_dir / 'protein_name_maps'}")

    return results
