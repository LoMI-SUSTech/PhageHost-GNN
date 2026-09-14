# Columnwise mean aggregation utility
# This code aggregates protein-level embeddings into host-level or phage-level embeddings, accordingly

from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd

# identifying the embedding columns from an embedding CSV
def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"protein_id", "host_id", "strain_id", "phage_id"}
    embed_cols = [
        c for c in df.columns
        if c not in ignored and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not embed_cols:
        raise ValueError("No embedding columns found.")
    return embed_cols

# extracting the host ID from K-locus protein ID (e.g. 19101412_1 -> 19101412)
def host_id_from_protein(protein_id: str) -> str:
    return re.sub(r"_\d+$", "", str(protein_id))

# extract phage ID from RBP ID (e.g. RCIP0001_RBP001 -> RCIP0001)
def phage_id_from_protein(protein_id: str) -> str:
    protein_id = str(protein_id).strip()
    return re.sub(r"_RBP.*$", "", protein_id)

# plotting the number of proteins/RBPs per entity
def plot_counts(counts: pd.DataFrame, entity_col: str, count_col: str, out_plot: Path, title: str, xlabel: str):
    out_plot = Path(out_plot)
    count_dist = counts[count_col].value_counts().sort_index()

    plt.figure(figsize=(8, 5), dpi=300)
    plt.bar(
        count_dist.index.astype(str),
        count_dist.values,
        edgecolor="0.5",
        linewidth=1.2,
    )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(f"Number of {entity_col.replace('_id', 's')}", fontsize=12)
    plt.title(title, fontsize=13)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=500, bbox_inches="tight")
    plt.close()

# aggregating protein-level embeddings by columnwise mean
def aggregate_columnwise_mean(emb_path: Path, outdir: Path, entity_type: str, output_prefix: str, make_plot: bool = True) -> dict:
    emb_path = Path(emb_path)
    outdir = Path(outdir)
    analysis_dir = outdir / "embeds_analysis"

    outdir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    emb = pd.read_csv(emb_path)
    if "protein_id" not in emb.columns:
        first_col = emb.columns[0]
        emb = emb.rename(columns={first_col: "protein_id"})

    emb["protein_id"] = emb["protein_id"].astype(str).str.strip()

    if entity_type == "host":
        entity_col = "strain_id"
        count_col = "n_proteins"
        emb[entity_col] = emb["protein_id"].map(host_id_from_protein)
        plot_title = "Distribution of K-locus proteins per strain"
        plot_xlabel = "Number of K-locus proteins per strain"

    elif entity_type == "phage":
        entity_col = "phage_id"
        count_col = "n_RBPs"
        emb[entity_col] = emb["protein_id"].map(phage_id_from_protein)
        plot_title = "Distribution of RBPs per phage"
        plot_xlabel = "Number of RBPs per phage"

    else:
        raise ValueError("entity_type must be either 'host' or 'phage'.")

    embed_cols = get_embedding_columns(emb)
    emb[embed_cols] = emb[embed_cols].apply(pd.to_numeric, errors="coerce")
    emb = emb.dropna(subset=embed_cols)

    mean_df = emb.groupby(entity_col, as_index=False)[embed_cols].mean()
    counts = emb.groupby(entity_col).size().reset_index(name=count_col)
    counts = counts.sort_values(count_col).reset_index(drop=True)

    out_emb_path = outdir / f"{output_prefix}_colmean_embeddings.csv"
    out_cnt_path = analysis_dir / f"{output_prefix}_counts.csv"
    out_plot_path = analysis_dir / f"{output_prefix}_counts_distribution.png"

    mean_df.to_csv(out_emb_path, index=False)
    counts.to_csv(out_cnt_path, index=False)

    if make_plot:
        plot_counts(
            counts=counts,
            entity_col=entity_col,
            count_col=count_col,
            out_plot=out_plot_path,
            title=plot_title,
            xlabel=plot_xlabel,
        )

    print(f"{entity_type.capitalize()} aggregation: {mean_df.shape[0]} entities × {len(embed_cols)} dimensions")
    return {
        "embeddings_csv": out_emb_path,
        "counts_csv": out_cnt_path,
        "plot_png": out_plot_path if make_plot else None,
    }
