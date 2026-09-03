# PhageHost-GNN
This repository contains the source code for the novelty-aware GNN model for strain-level prediction of Klebsiella-phage interactions, which has been trained under three clinical settings; Host-unseen, phage-unseen and both-unseen to facilitate model generalization on novel phages or bacterial strains.

Repository overview
1. Model training: Users can train the models from scratch to reproduce our analysis or make new predictions based on their datasets processed accordingly to match our model training mechanism.
2. Inference: Users can directly predict potential interacting phage, or host candidates, or novel phage-host pairs using the corresponding pretrained model weights. This is particularly useful when new datasets are highly similar to their training counterparts.
3. Fine-tuning: The fine-tuning pipeline allows users to adapt pretrained models to their local custom datasets, especially when sufficient interaction datasets are unavailable for model training from scratch.
