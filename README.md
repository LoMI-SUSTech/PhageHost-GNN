# PhageHost-GNN
This repository contains the source code for the novelty-aware GNN model for strain-level prediction of Klebsiella-phage interactions, which has been trained under three clinical settings; Host-unseen, phage-unseen and both-unseen to facilitate model generalization on novel phages or bacterial strains.

> **Repository overview**
1. Model training: Users can train the models from scratch to reproduce our analysis or make new predictions based on their datasets processed accordingly to match our interaction mechanism.
2. Inference: Users can directly predict potential interacting phage, or host candidates, or novel phage-host pairs using the corresponding pretrained model weights. This is particularly useful when new datasets are highly similar to their training counterparts.
3. Fine-tuning: The fine-tuning pipeline allows users to adapt pretrained models to their local custom datasets, especially when sufficient interaction datasets are unavailable for model training from scratch.

> **Data processing**
The data processing pipeline allow users to generate sequence or structural embeddings from the phage of _klebsiella_ genomes.
The data processing stage involves:
1. Extraction of phage receptor binding proteins (RBPs) from phage genomes using PHANOTATE and RBPdetect.
2. Extraction of bacterial K-locus proteins using Kaptive from _Klebsiella_ genomes.
3. Generate numerical representations using the respective biological foundation models, i.e., ESMC, ESM2, LucaOne, BacFormer.
4. Run AF3 to predict 3D structures for the respective phage and host proteins, and extract structural features using ESM-IF1 encoder.
5. Aggregate multi-instance proteins via columnwise mean to obtain the final single vector per phage or bacterium.
