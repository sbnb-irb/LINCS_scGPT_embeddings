# =============================================================================
# RUN EXAMPLE: 
# python NN_calculation.py \
#   --ref-adata ./LINCS_scGPT_embeddings/02_Obtain_Embeddings/embeddings_full.h5ad \
#   --query-adata ./LINCS_scGPT_embeddings/query_data.h5ad \
#   --output-dir ./LINCS_scGPT_embeddings/Results/NN_distances \
#   --k 1000 \
#   --batch-size 10 \
#   --use-gpu
# =============================================================================


import argparse
import os
import pickle
from pathlib import Path

import faiss
import h5py
import numpy as np
import scanpy as sc
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute nearest-neighbor distances between scGPT embeddings using FAISS."
    )

    parser.add_argument("--ref-adata", required=True, default='./LINCS_scGPT_embeddings/02_Obtain_Embeddings/embeddings_full.h5ad', help="Reference AnnData .h5ad file (LINCS embeddings).")
    parser.add_argument("--query-adata", required=True, help="Query AnnData .h5ad file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save FAISS results.")

    parser.add_argument(
        "--embedding-key",
        default="X_scGPT",
        help="Key in .obsm containing embeddings.",
    )

    parser.add_argument(
        "--output-name",
        default="nn_distances_query_ref.h5",
        help="Name of output HDF5 file.",
    )

    parser.add_argument("--k", type=int, default=1000, help="Number of nearest neighbors.")
    parser.add_argument("--batch-size", type=int, default=10, help="FAISS search batch size.")
    parser.add_argument("--use-gpu", action="store_true", help="Use FAISS GPU if available.")

    return parser.parse_args()


def as_float32(X):
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X


def l2norm_inplace(X):
    faiss.normalize_L2(X)


def main() -> None:
    args = parse_args()

    ref_adata_path = Path(args.ref_adata)
    query_adata_path = Path(args.query_adata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output_name

    if output_path.exists():
        print(f"File {output_path} already exists, skipping computation.")
        return

    print("Loading AnnData files...")
    adata_ref = sc.read_h5ad(ref_adata_path)
    adata_query = sc.read_h5ad(query_adata_path)

    print("Loaded query shape:", adata_query.shape)
    print("Loaded reference shape:", adata_ref.shape)

    data_query = as_float32(adata_query.obsm[args.embedding_key])
    data_ref = as_float32(adata_ref.obsm[args.embedding_key])

    l2norm_inplace(data_query)
    l2norm_inplace(data_ref)

    print("Normalized data shapes:", data_query.shape, data_ref.shape)

    index = faiss.IndexFlatIP(data_ref.shape[1])

    if args.use_gpu and faiss.get_num_gpus() > 0:
        print(f"Using {faiss.get_num_gpus()} FAISS GPU(s).")
        index = faiss.index_cpu_to_all_gpus(index)
    else:
        print("Using FAISS CPU index.")

    index.add(data_ref)

    print("Number of vectors in the index:", index.ntotal)

    pickle.dump(adata_ref.obs_names, open(output_dir / "order_emb_ref.pkl", "wb"))
    pickle.dump(adata_query.obs_names, open(output_dir / "order_query.pkl", "wb"))

    print("Computing nearest neighbors and saving to:", output_path)

    n_total = data_query.shape[0]
    k = min(args.k, data_ref.shape[0])

    with h5py.File(output_path, "w") as out_file:
        out_file.create_dataset("indices", (n_total, k), dtype=np.int32)
        out_file.create_dataset("distances", (n_total, k), dtype=np.float32)

        for start in tqdm(range(0, n_total, args.batch_size)):
            end = min(start + args.batch_size, n_total)

            D, I = index.search(data_query[start:end], k)

            out_file["indices"][start:end] = I
            out_file["distances"][start:end] = 1.0 - D

    print("Done.")


if __name__ == "__main__":
    main()