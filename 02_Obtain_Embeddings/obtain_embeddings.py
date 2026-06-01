# =============================================================================
# RUN EXAMPLE: 
# python obtain_embeddings.py \
#   --adata-path ./LINCS_scGPT_embeddings/Data/LINCS_full.h5ad \
#   --model-dir ./LINCS_scGPT_embeddings/01_Fine_Tuning/Results/my_run \
#   --output-path ./LINCS_scGPT_embeddings/02_Embeddings/LINCS_full_scGPT.h5ad \
#   --batch-size 64 \
#   --compute-umap
# =============================================================================


import argparse
import sys
from pathlib import Path

import scanpy as sc
import scgpt as scg

REPO_ROOT = Path("./LINCS_scGPT_embeddings")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scGPT embeddings for GEx profiles.")
    parser.add_argument("--adata-path", required=True, help="Input AnnData .h5ad file with the data that you want to embed.")
    parser.add_argument("--model-dir", required=True, help="Fine-tuned or pre-trained scGPT model directory.")
    parser.add_argument("--output-path", required=True, help="Output .h5ad file with X_scGPT embeddings.")
    parser.add_argument("--gene-col", default="gene_name", help="Column in adata.var containing gene symbols.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compute-umap", action="store_true", help="Compute neighbors and UMAP after embedding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata_path = Path(args.adata_path)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(adata_path)
    adata.var.set_index(adata.var[args.gene_col], inplace=True)

    embedded = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=args.gene_col,
        batch_size=args.batch_size,
    )

    if args.compute_umap:
        sc.pp.neighbors(embedded, use_rep="X_scGPT")
        sc.tl.umap(embedded)

    embedded.write_h5ad(output_path)


if __name__ == "__main__":
    main()
