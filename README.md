# Enhancing Perturbation Representation Using Single-Cell Foundation Models

Pipeline and analysis code for fine-tuning `scGPT` on LINCS L1000 gene expression profiles to learn perturbation-centered representations. The resulting embedding spaces support downstream analyses including perturbation recovery, mechanism-of-action inference, compound-target prioritization, and contextualization of external perturbational datasets.

This repository contains the training and analysis workflow, together with reproducible scripts.

## Overview

The workflow is organized as numbered stages:

1. Build an AnnData object from LINCS Level 3 data.
2. Fine-tune `scGPT` on LINCS profiles.
3. Extract embeddings into `.h5ad` files.
4. Compute nearest-neighbor distances between embeddings.
5. Extract biological knowledge from the embedding space.
6. Main downstream analysis together with figure panels.

## Repository Layout

```text
LINCS_scGPT_embeddings/
├── 00_Build_adata_from_LINCS.ipynb       Build AnnData object from LINCS data
├── 01_Fine_Tuning/                       scGPT fine-tuning script
│   ├── finetune_scgpt_full_lincs.py
│   └── Models/                           Fine-tuned model checkpoints
├── 02_Obtain_Embeddings/                 Embedding extraction scripts 
│   └── obtain_embeddings.py
├── 03_Distance_Integration/              Nearest-neighbor distance calculation between embeddings
│   └── NN_calculation.py
├── 04_Knowledge_Extraction/              MoA inference and compound-genetic perturbation analyses
│   ├── MoA_inference.ipynb
│   ├── Compound_genetic_similarity.ipynb
│   └── geneSetLibrary_*.txt
├── 05_Figures/                           Analysis notebooks and code to reproduce figure panels
│   ├── Figure2.ipynb                     Perturbation space validation
│   ├── Figure3.ipynb                     Recovery of orthogonal relationships by scGPT embeddings
│   ├── Figure4.ipynb                     Uncover novel relationships in the embedding space
│   ├── Figure5.ipynb                     Contextualization of new dataset
│   └── recap_*.py
├── Data/
│   └── Intermediate_files/
├── Results/
└── README.md
```
## Data availability

Precomputed LINCS embedding files are not included in this GitHub repository because of their large file size. They are provided through Zenodo.
Zenodo record: `<add>`

Fine-tuned model checkpoints are included in `01_Fine_Tuning/Models/`.

Large derived outputs, including nearest-neighbor distance matrices, are not stored in this repository and are available upon request.

This repository does not include LINCS Level 3 matrices, FRoGS compound-target annotations or GSE51212 raw data. Please download them from their original sources: 

- https://clue.io/data/CMap2020#LINCS2020
- https://github.com/chenhcs/FRoGS/blob/main/data/cpd_gene_pairs.csv
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1240552


## Environment

The project environment is defined in `environment.yml`. 

```bash
conda env create -f environment.yml
conda activate lincs-scgpt
```
Fine-tuning scGPT was performed on a GPU. Embedding extraction and nearest-neighbor calculation can also benefit from GPU acceleration. For large LINCS datasets, nearest-neighbor calculation may require substantial memory depending on `k`, batch size, and reference size.

## Typical workflow

### 1. Build the LINCS AnnData object

Use:

- [00_Build_adata_from_LINCS.ipynb](./00_Build_adata_from_LINCS.ipynb)


This notebook assembles an AnnData object from LINCS Level 3 matrices and associated metadata.

### 2. Fine-tune scGPT

Use:

- [01_Fine_Tuning/finetune_scgpt_full_lincs.py](./01_Fine_Tuning/finetune_scgpt_full_lincs.py)

Example:

```bash
python LINCS_scGPT_embeddings/01_Fine_Tuning/finetune_scgpt_full_lincs.py \
  --adata-path ./LINCS_scGPT_embeddings/Data/LINCS_full.h5ad \
  --output-dir ./LINCS_scGPT_embeddings/01_Fine_Tuning/Results/my_run \
  --wandb-dir ./LINCS_scGPT_embeddings/01_Fine_Tuning/wandb \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 1e-4
```


### 3. Extract embeddings

Use:

- [02_Obtain_Embeddings/obtain_embeddings.py](./02_Obtain_Embeddings/obtain_embeddings.py)

Example:

```bash
python LINCS_scGPT_embeddings/02_Obtain_Embeddings/obtain_embeddings.py \
  --adata-path ./LINCS_scGPT_embeddings/Data/LINCS_full.h5ad \
  --model-dir ./LINCS_scGPT_embeddings/01_Fine_Tuning/Results/my_run \
  --output-path ./LINCS_scGPT_embeddings/02_Obtain_Embeddings/LINCS_full_scGPT.h5ad \
  --batch-size 64 \
  --compute-umap
```

Main output:

- `.h5ad` file with `X_scGPT` stored in `.obsm`

If you only want to use the precomputed embeddings, download them from Zenodo and place them in [02_Obtain_Embeddings/](./02_Obtain_Embeddings/).

Expected files:

`02_Obtain_Embeddings/embeddings_pretrained.h5ad`
`02_Obtain_Embeddings/embeddings_full.h5ad`
`02_Obtain_Embeddings/embeddings_HQ.h5ad`

These files correspond to embeddings extracted from the pretrained model, the fine-tuned model using the full LINCS dataset, and the high-quality subset, respectively.


### 4. Compute nearest-neighbor distances 

Use:

- [03_Distance_Integration/NN_calculation.py](./03_Distance_Integration/NN_calculation.py)

Example:

```bash
python LINCS_scGPT_embeddings/03_Distance_Integration/NN_calculation.py \
  --ref-adata ./LINCS_scGPT_embeddings/02_Obtain_Embeddings/embeddings_full.h5ad \
  --query-adata ./LINCS_scGPT_embeddings/query_data.h5ad \
  --output-dir ./LINCS_scGPT_embeddings/Results/NN_distances \
  --k 1000 \
  --batch-size 10 \
  --use-gpu
```

Main outputs:

- HDF5 file containing neighbor `indices` and `distances`
- pickle files storing query/reference profile order

These outputs are used to explore relationships between embeddings and to contextualize external perturbational datasets.

### 5. Knowledge extraction

Use notebooks in:

- MoA voting and confidence summaries: [04_Knowledge_Extraction/MoA_inference.ipynb](./04_Knowledge_Extraction/MoA_inference.ipynb)
- compound-genetic perturbation similarity: [04_Knowledge_Extraction/Compound_genetic_similarity.ipynb](./04_Knowledge_Extraction/Compound_genetic_similarity.ipynb)


## Credits

The fine-tuning pipeline is adapted from the `scGPT` tutorial workflow by the Bo Wang Lab:

- https://github.com/bowang-lab/scGPT

Please cite the original `scGPT` publication and this work when referring to embeddings or reusing this analysis.

```

```
