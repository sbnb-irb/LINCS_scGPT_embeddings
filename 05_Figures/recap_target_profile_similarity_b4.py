# =============================================================================
# You can download the B4 signatures from the Chemical Checker web: https://chemicalchecker.com/api/db/getFile/root/B4.h5/
# RUN EXAMPLE: 
# python recap_target_profile_similarity_b4.py \
#   --adata-path ./LINCS_scGPT_embeddings/02_Obtain_Embeddings/embeddings_full.h5ad \
#   --compound-info-path ./LINCS_scGPT_embeddings/Data/Intermediate_files/cmp_info.txt \
#   --cc-b4-path ./LINCS_scGPT_embeddings/ChemicalChecker/sign3.h5 \
#   --output-dir ./LINCS_scGPT_embeddings/results/target_similarity \
#   --label full_scGPT \
#   --n-samples 10 \
#   --top-percent 0.1
# =============================================================================

import argparse
import pickle
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate target profile similarity recapitulation in Chemical Checker B4 space.")
    parser.add_argument("--adata-path", default="./LINCS_scGPT_embeddings/02_Obtain_Embeddings/embeddings_full.h5ad", help="AnnData file containing GEx and X_scGPT embeddings.")
    parser.add_argument("--compound-info-path",  default="./LINCS_scGPT_embeddings/Data/Intermediate_files/cmp_info.txt", help="LINCS compoundinfo_beta.txt path.")
    parser.add_argument("--cc-b4-path", required=True, help="Chemical Checker B4 sign3.h5 file.")
    parser.add_argument("--output-dir", required=True, help="Directory where result pickles are written.")
    parser.add_argument("--label", required=True, help="Run label used in output filenames.")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--top-percent", type=float, default=0.1)
    return parser.parse_args()


def calculate_roc(distances: np.ndarray, reference_distances: np.ndarray, percentile: float):
    threshold = np.percentile(reference_distances, percentile)
    y_true = (np.array(reference_distances) < threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, 1 - distances)
    roc_auc = roc_auc_score(y_true, 1 - distances)
    return fpr, tpr, roc_auc


def main() -> None:
    args = parse_args()
    adata_path = Path(args.adata_path)
    compound_info_path = Path(args.compound_info_path)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(adata_path)
    compound_info = pd.read_csv(compound_info_path, sep="\t")

    compound_info = compound_info.dropna(subset=["inchi_key"])
    compound_info = compound_info[compound_info["inchi_key"] != "restricted"]

    b4_path = Path(args.cc_b4_path)

    with h5py.File(b4_path, "r") as handle:
        keys = handle["keys"][:].astype(str)
        signatures = handle["V"][:]

    unique_inchikeys = compound_info["inchi_key"].unique()
    cc_index = np.concatenate([np.where(keys == key)[0] for key in tqdm(unique_inchikeys, desc="Matching B4 keys")])
    keys = keys[cc_index]
    signatures = signatures[cc_index]

    compound_info = compound_info[compound_info["inchi_key"].isin(keys)].drop_duplicates(subset=["inchi_key"])
    adata_cmp = adata[adata.obs["pert_id"].isin(compound_info["pert_id"])].copy()
    adata_cmp.obs["inchi_key"] = adata_cmp.obs["pert_id"].map(dict(zip(compound_info["pert_id"], compound_info["inchi_key"])))

    aurocs_gex, tprs_gex, fprs_gex = [], [], []
    aurocs_emb, tprs_emb, fprs_emb = [], [], []

    for seed in tqdm(range(args.n_samples), desc="Sampling iterations"):
        sampled_obs = adata_cmp.obs.groupby("inchi_key").sample(n=1, random_state=seed).index
        adata_sel = adata_cmp[adata_cmp.obs.index.isin(sampled_obs)].copy()
        cc_selected_idx = [np.where(keys == key)[0][0] for key in adata_sel.obs["inchi_key"]]
        signatures_sel = signatures[cc_selected_idx]

        distances_b4 = pdist(signatures_sel, metric="cosine")
        distances_gex = pdist(adata_sel.X, metric="cosine")
        distances_emb = pdist(adata_sel.obsm["X_scGPT"], metric="cosine")

        fpr_gex, tpr_gex, auc_gex = calculate_roc(distances_gex, distances_b4, args.top_percent)
        fpr_emb, tpr_emb, auc_emb = calculate_roc(distances_emb, distances_b4, args.top_percent)

        aurocs_gex.append(auc_gex)
        tprs_gex.append(tpr_gex)
        fprs_gex.append(fpr_gex)
        aurocs_emb.append(auc_emb)
        tprs_emb.append(tpr_emb)
        fprs_emb.append(fpr_emb)

    with open(output_dir / f"ROC_{args.label}_B4_GEX.pickle", "wb") as handle:
        pickle.dump({"tpr": tprs_gex, "fpr": fprs_gex, "AUROC": aurocs_gex}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / f"ROC_{args.label}_B4.pickle", "wb") as handle:
        pickle.dump({"tpr": tprs_emb, "fpr": fprs_emb, "AUROC": aurocs_emb}, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
