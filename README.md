# Benchmarking Representations

Code for training and benchmarking morphology appropriate representation learning methods, associated with the following manuscript.

> **Interpretable representation learning for 3D multi-piece intracellular structures using point clouds**
>
> Ritvik Vasan, Alexandra J. Ferrante, Antoine Borensztejn, Christopher L. Frick, Nathalie Gaudreault, Saurabh S. Mogre, Benjamin Morris, Guilherme G. Pires, Susanne M. Rafelski, Julie A. Theriot, Matheus P. Viana
>
> bioRxiv 2024.07.25.605164; doi: https://doi.org/10.1101/2024.07.25.605164

Our analysis is organized as follows.

1. Single cell images
2. Preprocessing (result: pointclouds and SDFs)
   1. Punctate structures
      1. Alignment, masking, and registration
      2. Generate pointclouds
   2. Polymorphic structures: Generate SDFs
3. Model training (result: checkpoint)
4. Model inference (results: embeddings, model cost statistics)
5. Interpretability analysis (results: figures)

Continue below for guidance on using these models on your own data.
If you'd like to reproduce this analysis on our data, check out the following documentation.

- [Main usage documentation](./docs/USAGE.md) for reproducing the figures in the paper from published pointclouds and SDFs, including model training and inference (steps 3-5).
- [Preprocessing documentation](./docs/PREPROCESSING.md) for generating pointclouds and SDFs from from our input movies (step 2).
- [Development documentation](./docs/DEVELOPMENT.md) for guidance working on the code in this repository.

# Using the models

Coming soon

# Contact

Allen Institute for Cell Science (cells@alleninstitute.org)
