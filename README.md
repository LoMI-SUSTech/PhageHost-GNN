# PhageHost-GNN
This repository contains the source code for the novelty-aware GNN model for strain-level prediction of _Klebsiella_-phage interactions, which has been trained under three clinical settings; Host-unseen, phage-unseen, and both-unseen to facilitate model generalization on novel phages or bacterial strains.

> **Repository overview**

1. Model training: Users can train the models from scratch to reproduce our analysis or make new predictions based on their datasets processed to match our interaction mechanism.
2. Inference: Users can directly predict potential interacting phage, or host candidates, or novel phage-host pairs using the corresponding pretrained model weights. This is particularly useful when new datasets are highly similar to their training counterparts.
3. Fine-tuning: The fine-tuning pipeline allows users to adapt pretrained models to their local custom datasets, especially when sufficient interaction datasets are unavailable for model training from scratch.

> **Data processing**

The data-processing pipeline generates sequence- and structure-based representations from phage and _Klebsiella_ genomes or extracted proteins. The pipeline includes:

1. Extract phage receptor-binding proteins (RBPs) from phage genomes using PHANOTATE and RBPdetect.
2. Extract bacterial K-locus proteins from _Klebsiella_ genomes using Kaptive.
3. Generate sequence-based protein embeddings using ESMC, ESM2, LucaOne, and BacFormer.
4. Generate structure-based protein embeddings from user-provided AF3-predicted structures using the ESM-IF1 encoder.
5. Aggregate protein-level embeddings by column-wise mean to obtain a single representation for each phage or bacterial strain.
6. Compute host–host and phage–phage cosine-similarity matrices for similarity-based grouping and downstream model training or adaptation.

   Note:
   * Download the model weights for ESMC, LucaOne, and BacFormer from Zenodo and save them in **resources/embedding_models/**.
   * Download the PhageRBPdetection directory from Zenodo and save it in **resources/**.
   * Start from genomes if the phage RBPs and host K-locus proteins have not yet been extracted; otherwise, start from proteins.

> **Training from scratch or reproducing our analyses**
1. Clone or download this repository.
2. Install the required dependencies as described in the Installation section.
3. Process your own dataset using the provided data-processing pipeline, or use our processed datasets to reproduce our analyses.
4. Navigate to the corresponding training notebook for the desired setting (host-unseen, phage-unseen, or both-unseen).
5. Update the required file paths and training configurations.
6. Run the training notebook (*PhageHost_GNN_training.ipynb* ) to train the models from scratch or reproduce the reported analyses.

   
> **Making predictions for your data**
1. Clone or download this repository and install the required dependencies as described in the Installation section.
2. Download the required pretrained PhageHost-GNN models from Zenodo.
3. Process your local phage and/or bacterial genomes using data_processing.ipynb to generate the required embeddings.
4. Assign the highest supported similarity quantile to the new entities using:
    * Host-unseen: new_strains_q_assignment.ipynb
    * Both-unseen: new_phages_and_strains_q_assignment.ipynb
5. Ensure the required training embeddings, new entity embeddings, and training entity cosine similarities are saved in the respective directories.
6. Update file paths, and run the corresponding inference notebook to generate phage–host interaction scores.

> **Fine-tuning on your data**
1. Clone or download this repository and install required dependencies as described in the Installation section.
2. Ensure LOGOCV results of pretrained PhageHost-GNN models are in PH_q_LOGOCV_results.
3. Process your local dataset using *data_processing.ipynb* to generate the required embeddings and cosine-similarity matrices.
4. Ensure the local interaction matrix, embeddings, and cosine-similarity matrices are available in the expected directories.
5. Navigate to the fine-tuning notebook and update the required file paths and fine-tuning settings.
6. Run the notebook to evaluate the routed models and train a final locally adapted model for each similarity quantile.

> **Installation requirements**

The PhageHost-GNN model, required the following dependencies to be satisfied:<br>
Model training and inference were performed using: Python v3.9.23 on an NVIDIA H100 80 GB GPU using PyTorch v2.2.2 with CUDA 11.8 support.<br>
* Major dependencies** PyTorch Geometric v2.6.1, torch-scatter v2.1.2, torch-sparse v0.6.18, Scikit-learn v1.6.1, NumPy v1.26.3, Pandas v2.3.0, SciPy v1.13.1, Matplotlib v3.9.4, and Biopython v1.85.<br>
* Additional bioinformatics tools:** PHANOTATE v1.6.7, Kaptive v3.1.0, BLAST+ v2.16.0, FastANI v1.34, Mash v2.3, CD-HIT v4.8.1, Prodigal v2.6.3 

> **Environment setup**

* All Conda environments listed in (_data_processing/resources/environments_) should be created before running the pipeline.
* The main environment is used for data processing, model training, and inference, while model-specific environments are automatically invoked during embedding generation.
