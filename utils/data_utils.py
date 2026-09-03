from pathlib import Path
import numpy as np
import pandas as pd

def strip_str_index(idx):
    return pd.Index([str(x).strip() for x in idx])

def load_host_embeddings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = strip_str_index(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        raise ValueError("Host embeddings contain NaNs after numeric conversion.")

    return df

def load_phage_embeddings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = strip_str_index(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        raise ValueError("Phage embeddings contain NaNs after numeric conversion.")

    return df

def load_interaction_data(interaction_csv: Path, host_emb_csv: Path, phage_emb_csv: Path):
    # Interaction matrix defines the canonical host/phage order
    interaction = pd.read_csv(interaction_csv, index_col=0)
    interaction.index = strip_str_index(interaction.index)
    interaction.columns = strip_str_index(interaction.columns)

    hosts = interaction.index.tolist()
    phages = interaction.columns.tolist()

    # Host embeddings
    host_emb = load_host_embeddings(host_emb_csv)
    missing_hosts = pd.Index(hosts).difference(host_emb.index)

    if len(missing_hosts):
        raise ValueError(
            f"Missing {len(missing_hosts)} hosts "
            f"(examples: {missing_hosts[:10].tolist()})"
        )

    host_emb = host_emb.reindex(hosts)

    # Phage embeddings
    phage_emb = load_phage_embeddings(phage_emb_csv)
    missing_phages = pd.Index(phages).difference(phage_emb.index)

    if len(missing_phages):
        raise ValueError(
            f"Missing {len(missing_phages)} phages "
            f"(examples: {missing_phages[:10].tolist()})"
        )

    phage_emb = phage_emb.reindex(phages)

    # Observed, positive and negative masks
    OBS = ~interaction.isna().to_numpy()

    Y01 = (
        interaction
        .fillna(0)
        .astype(float)
        .to_numpy() > 0.5
    ).astype(np.int8)

    POS = OBS & (Y01 == 1)
    NEG = OBS & (Y01 == 0)

    X_host = host_emb.to_numpy(np.float32)
    X_phage = phage_emb.to_numpy(np.float32)

    return (
        interaction,
        hosts,
        phages,
        X_host,
        X_phage,
        OBS,
        POS,
        NEG,
    )
