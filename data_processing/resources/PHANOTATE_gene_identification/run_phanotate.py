from pathlib import Path
import subprocess
import re

def safe_name(x: str) -> str:
    x = x.strip()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^A-Za-z0-9._-]", "_", x)
    return x

def main(input_fasta, output_dir):
    input_fasta = Path(input_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    prefix = safe_name(input_fasta.stem)
    faa_out = output_dir / f"{prefix}_phanotate.faa"
    gbk_out = output_dir / f"{prefix}_phanotate.gbk"

    with open(faa_out, "w") as f:
        subprocess.run(
            ["phanotate.py", str(input_fasta), "-f", "faa"],
            stdout=f,
            check=True
        )

    with open(gbk_out, "w") as f:
        subprocess.run(
            ["phanotate.py", str(input_fasta), "-f", "genbank"],
            stdout=f,
            check=True
        )

    print("Saved:", faa_out)
    print("Saved:", gbk_out)

if __name__ == "__main__":
    main(
        "../phage_genome/phageH5_OR296322/ncbi_dataset/data/GCA_031678295.1/GCA_031678295.1_ASM3167829v1_genomic.fna",
        "phanotate_out",
    )
