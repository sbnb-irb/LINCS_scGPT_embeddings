# =============================================================================
# RUN EXAMPLE: 
# python recap_chemical_similarity.py \
#   --adata-path ./LINCS_scGPT_embeddings/02_Obtain_Embeddings/embeddings_full.h5ad \ ### You need to add the GEx to the adata.X for comparison 
#   --compound-info-path ./LINCS_scGPT_embeddings/Data/Intermediate_files/cmp_info.txt \
#   --output-dir ./LINCS_scGPT_embeddings/results/chemical_similarity \
#   --label FT \
#   --n-samples 10 \
#   --top-percent 0.1
# =============================================================================


import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chemical similarity recapitulation with AUROC.")
    parser.add_argument("--adata-path", default="./LINCS_scGPT_embeddings/02_Obtain_Embeddings/embeddings_full.h5ad", help="AnnData file containing GEx and X_scGPT embeddings.")
    parser.add_argument("--compound-info-path",  default="./LINCS_scGPT_embeddings/Data/Intermediate_files/cmp_info.txt", help="LINCS compoundinfo_beta.txt path.")
    parser.add_argument("--output-dir", required=True, help="Directory where result pickles are written.")
    parser.add_argument("--label", required=True, help="Run label used in output filenames.")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of repeated one-profile-per-compound samplings.")
    parser.add_argument("--top-percent", type=float, default=0.1, help="Top percentile of Tanimoto distance considered positive.")
    return parser.parse_args()


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def calculate_roc(distances: np.ndarray, tanimoto_distances: list[float], percentile: float):
    threshold = np.percentile(tanimoto_distances, percentile)
    y_true = (np.array(tanimoto_distances) < threshold).astype(int)
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

    compound_info = compound_info.dropna(subset=["canonical_smiles"])
    compound_info = compound_info[compound_info["canonical_smiles"] != "restricted"]
    pert2smiles = dict(zip(compound_info["pert_id"], compound_info["canonical_smiles"]))

    aurocs_gex, tprs_gex, fprs_gex = [], [], []
    aurocs_emb, tprs_emb, fprs_emb = [], [], []

    adata_cmp = adata[adata.obs["pert_id"].isin(compound_info["pert_id"])].copy()
    for seed in tqdm(range(args.n_samples), desc="Sampling iterations"):
        sampled_obs = adata_cmp.obs.groupby("pert_id").sample(n=1, random_state=seed).index
        adata_sel = adata_cmp[adata_cmp.obs.index.isin(sampled_obs)].copy()
        smiles_list = [pert2smiles[p] for p in adata_sel.obs["pert_id"]]
        fps = [smiles_to_fp(smiles) for smiles in smiles_list]
        valid_mask = np.array([fp is not None for fp in fps])
        adata_sel = adata_sel[valid_mask].copy()
        fps = [fp for fp in fps if fp is not None]

        tanimoto_distances = []
        for idx in range(len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[idx + 1 :])
            tanimoto_distances.extend([1 - score for score in sims])

        distances_gex = pdist(adata_sel.X, "cosine")
        distances_emb = pdist(adata_sel.obsm["X_scGPT"], "cosine")

        fpr_gex, tpr_gex, auc_gex = calculate_roc(distances_gex, tanimoto_distances, args.top_percent)
        fpr_emb, tpr_emb, auc_emb = calculate_roc(distances_emb, tanimoto_distances, args.top_percent)

        aurocs_gex.append(auc_gex)
        tprs_gex.append(tpr_gex)
        fprs_gex.append(fpr_gex)
        aurocs_emb.append(auc_emb)
        tprs_emb.append(tpr_emb)
        fprs_emb.append(fpr_emb)

    with open(output_dir / f"ROC_{args.label}_GEX.pickle", "wb") as handle:
        pickle.dump({"tpr": tprs_gex, "fpr": fprs_gex, "AUROC": aurocs_gex}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / f"ROC_{args.label}.pickle", "wb") as handle:
        pickle.dump({"tpr": tprs_emb, "fpr": fprs_emb, "AUROC": aurocs_emb}, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
