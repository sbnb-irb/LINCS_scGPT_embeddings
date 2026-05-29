# =============================================================================
# RUN EXAMPLE: 
# python recap_genetic_perturbation_consistency.py \
#   --adata-path ./LINCS_scGPT_embeddings/embeddings_full.h5ad \
#   --output-dir ./LINCS_scGPT_embeddings/results/genetic_consistency \
#   --label full_scGPT \
#   --gene-label-column cmap_name \
#   --n-samples 10
# =============================================================================

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate genetic perturbation consistency with AUROC.")
    parser.add_argument("--adata-path", default="./LINCS_scGPT_embeddings/embeddings_full.h5ad", help="AnnData file containing GEx and X_scGPT embeddings.")
    parser.add_argument("--output-dir", required=True, help="Directory where result pickles are written.")
    parser.add_argument("--label", required=True, help="Run label used in output filenames.")
    parser.add_argument(
        "--gene-label-column",
        default="cmap_name",
        help="obs column defining the target gene identity for shRNA profiles.",
    )
    parser.add_argument("--n-samples", type=int, default=10)
    return parser.parse_args()


def calculate_roc(distances: np.ndarray, classification: np.ndarray):
    fpr, tpr, _ = roc_curve(classification, 1 - distances)
    roc_auc = roc_auc_score(classification, 1 - distances)
    return fpr, tpr, roc_auc


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def main() -> None:
    args = parse_args()

    adata_path = Path(args.adata_path)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(adata_path)
    adata_sh = adata[adata.obs["pert_type"] == "trt_sh"].copy()

    aurocs_gex, tprs_gex, fprs_gex = [], [], []
    aurocs_emb, tprs_emb, fprs_emb = [], [], []

    for seed in tqdm(range(args.n_samples), desc="Sampling iterations"):
        sampled_obs = adata_sh.obs.groupby("pert_id").sample(n=1, random_state=seed).index
        adata_sel = adata_sh[adata_sh.obs.index.isin(sampled_obs)].copy()
        gene_order = np.array(adata_sel.obs[args.gene_label_column].tolist())

        emb = to_dense(adata_sel.obsm["X_scGPT"])
        gex = to_dense(adata_sel.X)

        emb_dists = squareform(pdist(emb, metric="cosine"))
        gex_dists = squareform(pdist(gex, metric="cosine"))

        same_gene = (gene_order[:, None] == gene_order[None, :]).astype(int)
        tril_idx = np.tril_indices(len(gene_order), k=-1)
        classification = same_gene[tril_idx]

        fpr_gex, tpr_gex, auc_gex = calculate_roc(gex_dists[tril_idx], classification)
        fpr_emb, tpr_emb, auc_emb = calculate_roc(emb_dists[tril_idx], classification)

        aurocs_gex.append(auc_gex)
        tprs_gex.append(tpr_gex)
        fprs_gex.append(fpr_gex)
        aurocs_emb.append(auc_emb)
        tprs_emb.append(tpr_emb)
        fprs_emb.append(fpr_emb)

    with open(output_dir / f"ROC_{args.label}_gene_GEX.pickle", "wb") as handle:
        pickle.dump({"tpr": tprs_gex, "fpr": fprs_gex, "AUROC": aurocs_gex}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / f"ROC_{args.label}_gene.pickle", "wb") as handle:
        pickle.dump({"tpr": tprs_emb, "fpr": fprs_emb, "AUROC": aurocs_emb}, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
