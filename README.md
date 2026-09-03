# PhageHost-GNN
This repository contains the source code for the novelty-aware GNN model for strain-level prediction of _Klebsiella_-phage interactions, which has been trained under three clinical settings; Host-unseen, phage-unseen and both-unseen to facilitate model generalization on novel phages or bacterial strains.

> **Repository overview**

1. Model training: Users can train the models from scratch to reproduce our analysis or make new predictions based on their datasets processed to match our interaction mechanism.
2. Inference: Users can directly predict potential interacting phage, or host candidates, or novel phage-host pairs using the corresponding pretrained model weights. This is particularly useful when new datasets are highly similar to their training counterparts.
3. Fine-tuning: The fine-tuning pipeline allows users to adapt pretrained models to their local custom datasets, especially when sufficient interaction datasets are unavailable for model training from scratch.

> **Data processing**

The data processing pipeline allow users to generate sequence or structural embeddings from the phage or _klebsiella_ genomes.

The data processing stage involves:
1. Extraction of phage receptor binding proteins (RBPs) from phage genomes using PHANOTATE and RBPdetect.
2. Extraction of bacterial K-locus proteins using Kaptive from _Klebsiella_ genomes.
3. Generate numerical representations using the respective biological foundation models, i.e., ESMC, ESM2, LucaOne, BacFormer.
4. Predict 3D protein structures using AF3, and extract structural feature embeddings using ESM-IF1 encoder.
5. Aggregate multi-instance proteins via columnwise mean to obtain the final single vector per phage or bacterium.
6. Compute the phage and host cosine similarities, which facilitates phage and host clustering for model training via the leave-one-group-out cross-validation (LOGOCV) approach.


> **Installation requirements**

To run our model successfully, the following dependency requirements must be satisfied:
The software was run in a H100 80 GPU using Python v3.9.23 in addition to the following dependencies:
PyTorch v2.2.2, PyTorch Geometric v2.6.1, torch-scatter v2.1.2, torch-sparse v0.6.18, Scikit-learn v1.6.1, NumPy v1.26.3, Pandas v2.3.0, SciPy v1.13.1, Matplotlib v3.9.4, Biopython v1.85, and OpenPyXL v3.1.5.
Additional bioinformatics tools used for sequence processing and analysis include Kaptive v3.1.0, PHANOTATE v1.6.7, BLAST+ v2.16.0, FastANI v1.34, and CD-HIT v4.8.1.
GPU-based model training was performed using PyTorch compiled with CUDA 11.8 support.
