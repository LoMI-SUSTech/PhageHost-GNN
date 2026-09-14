# This phage processing utilities comprises of: (PHANOTATE run on raw phage genomes and 2. RBPdetect run on the PHANOTATE outputs to detect RBPs)

from pathlib import Path
import utils.phanotate_runner as phanotate_runner
import utils.rbpdetect_predictor as rbpdetect_predictor

# PHANOTATE runn
def run_phage_phanotate(genome_dir: Path, phanotate_outdir: Path) -> list[dict]:
    return phanotate_runner.run_phanotate_batch(
        Path(genome_dir),
        Path(phanotate_outdir)
        )

# RBPdetect on PHANOTATE outputs
def run_phage_rbpdetect(phanotate_outdir: Path, rbp_outdir: Path, min_len: int = 200, max_len: int = 1500) -> dict:
    return rbpdetect_predictor.run_rbpdetect_batch(
        phanotate_outdir=Path(phanotate_outdir),
        rbp_outdir=Path(rbp_outdir),
        min_len=min_len,
        max_len=max_len,
    )
