from pathlib import Path
from typing import Dict
import shutil
import subprocess

import pandas as pd
from tqdm.auto import tqdm


class HostKLocusExtractor:
    def __init__(self, genome_dir, outdir, kaptive_db="kpsc_k", show_progress=True):
        self.genome_dir = Path(genome_dir)
        self.outdir = Path(outdir)
        self.kaptive_db = kaptive_db
        self.show_progress = show_progress

        self.input_dir = self.outdir / "kaptive_input_genomes"
        self.raw_faa_dir = self.outdir / "strain_kl_proteins_raw"
        self.final_faa_dir = self.outdir / "strain_kl_proteins_final"

        self.kaptive_tsv = self.outdir / "kaptive_results.tsv"
        self.kaptive_json = self.outdir / "kaptive_results.json"
        self.combined_faa = self.outdir / "all_strain_kl_proteins.faa"

        self.summary_tsv = self.outdir / "strain_kaptive_summary.tsv"
        self.protein_index_tsv = self.outdir / "strain_protein_index.tsv"
        self.map_tsv = self.outdir / "strain_to_kl_protein_map.tsv"
        self.genome_map_tsv = self.outdir / "genome_input_map.tsv"

        self.outdir.mkdir(parents=True, exist_ok=True)

        for folder in [self.input_dir, self.raw_faa_dir, self.final_faa_dir]:
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    def progress(self, items, desc):
        return tqdm(items, desc=desc) if self.show_progress else items

    def collect_genomes(self):
        genomes = []
        for ext in ["*.fna", "*.fa", "*.fasta"]:
            genomes.extend(self.genome_dir.glob(ext))

        genomes = sorted(genomes)

        if not genomes:
            raise FileNotFoundError(f"No genome FASTA files found in: {self.genome_dir}")

        return genomes

    def prepare_inputs(self, genomes):
        rows = []
        copied = []

        for genome in self.progress(genomes, "Preparing Kaptive inputs"):
            strain_id = genome.stem
            out_genome = self.input_dir / f"{strain_id}.fna"

            shutil.copyfile(genome, out_genome)
            copied.append(out_genome)

            rows.append({
                "strain_id": strain_id,
                "assembly_name": out_genome.stem,
                "original_genome": str(genome),
                "kaptive_input_genome": str(out_genome),
            })

        genome_map = pd.DataFrame(rows)
        genome_map.to_csv(self.genome_map_tsv, sep="\t", index=False)

        return copied, genome_map

    def run_cmd(self, cmd, log_name):
        log_path = self.outdir / log_name

        print("\nRunning:", " ".join(map(str, cmd[:4])), "...")
        print("Log:", log_path)

        with open(log_path, "w") as log:
            log.write("Full command:\n")
            log.write(" ".join(map(str, cmd)) + "\n\n")
            log.write("Output:\n")

            subprocess.run(
                cmd,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
    # running kaptive as a commandline tool
    def run_kaptive(self, genomes):
        self.run_cmd(
            [
                "kaptive",
                "assembly",
                self.kaptive_db,
                *map(str, genomes),
                "-o",
                str(self.kaptive_tsv),
                "-j",
                str(self.kaptive_json),
            ],
            "kaptive_assembly.log",
        )

        self.run_cmd(
            [
                "kaptive",
                "convert",
                self.kaptive_db,
                str(self.kaptive_json),
                "--faa",
                str(self.raw_faa_dir),
            ],
            "kaptive_convert.log",
        )

    def load_summary(self, genome_map):
        df = pd.read_csv(self.kaptive_tsv, sep="\t", dtype=str).fillna("")

        required = ["Assembly", "Best match locus"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Kaptive output missing columns: {missing}")

        assembly_to_strain = dict(zip(genome_map["assembly_name"], genome_map["strain_id"]))

        summary = pd.DataFrame({
            "assembly_name": df["Assembly"].map(lambda x: Path(str(x)).stem),
            "assigned_kl": df["Best match locus"].str.strip(),
            "match_confidence": df.get("Match confidence", ""),
            "problems": df.get("Problems", ""),
            "identity": df.get("Identity", ""),
            "coverage": df.get("Coverage", ""),
        })

        summary["strain_id"] = summary["assembly_name"].map(assembly_to_strain)

        before = len(summary)

        summary = summary[summary["assigned_kl"].str.match(r"^KL\d+$", na=False)].copy()
        summary = summary[summary["strain_id"].notna()].copy()

        summary["raw_kl_protein_fasta"] = summary["assembly_name"].map(self.find_faa)
        summary = summary[summary["raw_kl_protein_fasta"] != ""].copy()

        print(f"Kaptive rows before filtering: {before}")
        print(f"Mapped KL calls retained: {len(summary)}")

        return summary.reset_index(drop=True)

    def find_faa(self, assembly_name):
        hits = sorted(self.raw_faa_dir.glob(f"{assembly_name}*.faa"))
        return str(hits[0]) if hits else ""

    def rewrite_faa(self, raw_faa, out_faa, strain_id, assigned_kl):
        lines = []
        combined = []
        rows = []
        protein_idx = 0

        with open(raw_faa) as handle:
            for line in handle:
                line = line.rstrip()

                if line.startswith(">"):
                    protein_idx += 1
                    protein_id = f"{strain_id}_{protein_idx}"

                    lines.append(f">{protein_id}")
                    combined.append(f">{protein_id}")

                    rows.append({
                        "strain_id": strain_id,
                        "assigned_kl": assigned_kl,
                        "protein_id": protein_id,
                        "protein_order": protein_idx,
                        "fasta_file": str(out_faa),
                    })
                else:
                    lines.append(line)
                    combined.append(line)

        if protein_idx == 0:
            raise ValueError(f"No protein records found in: {raw_faa}")

        with open(out_faa, "w") as handle:
            handle.write("\n".join(lines) + "\n")

        return combined, rows

    def export_fastas(self, summary):
        combined_lines = []
        protein_rows = []
        final_paths = []

        for _, row in self.progress(list(summary.iterrows()), "Writing final K-locus FASTAs"):
            strain_id = str(row["strain_id"])
            assigned_kl = str(row["assigned_kl"])

            raw_faa = Path(row["raw_kl_protein_fasta"])
            final_faa = self.final_faa_dir / f"{strain_id}.faa"

            lines, rows = self.rewrite_faa(raw_faa, final_faa, strain_id, assigned_kl)

            combined_lines.extend(lines)
            protein_rows.extend(rows)
            final_paths.append(str(final_faa))

        with open(self.combined_faa, "w") as handle:
            handle.write("\n".join(combined_lines) + "\n")

        summary = summary.copy()
        summary["final_kl_protein_fasta"] = final_paths

        protein_index = pd.DataFrame(protein_rows)

        return summary, protein_index

    def save_outputs(self, summary, protein_index):
        summary.to_csv(self.summary_tsv, sep="\t", index=False)

        summary[
            ["strain_id", "assigned_kl", "final_kl_protein_fasta"]
        ].to_csv(self.map_tsv, sep="\t", index=False)

        protein_index.to_csv(self.protein_index_tsv, sep="\t", index=False)

    def run(self) -> Dict[str, Path]:
        print("1/6: Collecting host genomes")
        genomes = self.collect_genomes()

        print("2/6: Preparing Kaptive inputs")
        kaptive_genomes, genome_map = self.prepare_inputs(genomes)

        print(f"Host genomes found: {len(genomes)}")

        print("3/6: Running Kaptive")
        self.run_kaptive(kaptive_genomes)

        print("4/6: Reading Kaptive results")
        summary = self.load_summary(genome_map)

        print("5/6: Writing final FASTA files")
        summary, protein_index = self.export_fastas(summary)

        print("6/6: Saving output tables")
        self.save_outputs(summary, protein_index)

        print("\nDone.")
        print(f"Strains typed and retained: {summary['strain_id'].nunique()}")
        print(f"K-locus proteins extracted: {len(protein_index)}")
        print(f"Combined FASTA: {self.combined_faa}")
        print(f"Summary: {self.summary_tsv}")
        print(f"Protein index: {self.protein_index_tsv}")

        return {
            "outdir": self.outdir,
            "summary_tsv": self.summary_tsv,
            "map_tsv": self.map_tsv,
            "protein_index_tsv": self.protein_index_tsv,
            "combined_faa": self.combined_faa,
            "kaptive_input_dir": self.input_dir,
            "raw_faa_dir": self.raw_faa_dir,
            "final_faa_dir": self.final_faa_dir,
        }


if __name__ == "__main__":
    extractor = HostKLocusExtractor(
        genome_dir=Path("../datasets/121strains_74phages/genomes/host_kp_genomes"),
        outdir=Path("kaptive_out"),
        kaptive_db="kpsc_k",
        show_progress=True,
    )

    extractor.run()

