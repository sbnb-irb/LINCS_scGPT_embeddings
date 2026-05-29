# %%

# =============================================================================
# Acknowledgment and Credit
# -----------------------------------------------------------------------------
# This script is adapted from the scGPT tutorial pipeline developed
# by the Bo Wang Lab:
#
# Repository:
# https://github.com/bowang-lab/scGPT
#
# Specifically inspired by:
# https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Annotation.ipynb
#
# Original scGPT publication:
# Cui H., Wang C., Maan H., et al.
# "scGPT: toward building a foundation model for single-cell multi-omics
# using generative AI."
# Nature Methods (2024).
#
# We gratefully acknowledge the original authors for making their code and
# methods publicly available. Please cite the original work when using this
# pipeline or derived analyses.
# =============================================================================

# =============================================================================
# RUN EXAMPLE: 
# python finetune_scgpt_full_lincs.py \
#   --adata-path ./LINCS_scGPT_embeddings/Data/LINCS_full.h5ad \ # path to the input data (GEx) in AnnData format
#   --output-dir ./LINCS_scGPT_embeddings/01_Fine_Tuning/Results \ # directory to save the fine-tuned model 
#   --wandb-dir ./LINCS_scGPT_embeddings/01_Fine_Tuning/wandb \ # directory to save the wandb logs
#   --epochs 20 \
#   --batch-size 32 \
#   --learning-rate 1e-4
# =============================================================================

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix


sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from cmapPy.pandasGEXpress.parse import parse

REPO_ROOT = Path("./LINCS_scGPT_embeddings")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune scGPT on the full LINCS dataset.")
    parser.add_argument("--adata-path", required=True, help="Input AnnData .h5ad file.")
    parser.add_argument("--pretrained-model-dir", default=REPO_ROOT / "01_Fine_Tuning" / "Models" / "full_ft")
    parser.add_argument("--output-dir", required=True, help="Directory to save fine-tuned model and results.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default="LINCS-scGPT_full")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-dir", required=True, help="Directory to store wandb logs.")
    parser.add_argument("--dataset-name", default="LINCS_full")
    parser.add_argument("--save-model-name", default="best_model.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


args = parse_args()

adata_path = Path(args.adata_path)
pretrained_model_dir = Path(args.pretrained_model_dir)
output_root = Path(args.output_dir)
wandb_dir = Path(args.wandb_dir)

output_root.mkdir(parents=True, exist_ok=True)
wandb_dir.mkdir(parents=True, exist_ok=True)

print(torch.cuda.is_available())
os.environ["WANDB_DIR"] = str(wandb_dir)
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

hyperparameter_defaults = dict(
    seed=args.seed,
    dataset_name=args.dataset_name,
    do_train=True,
    load_model=str(pretrained_model_dir),
    mask_ratio=0.0,
    epochs=args.epochs,
    n_bins=51,
    MVC=False, # Masked value prediction for cell embedding
    ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=args.learning_rate,
    batch_size=args.batch_size,
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = False,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
)

run = wandb.init(
    config=hyperparameter_defaults,
    project=args.wandb_project,
    name=args.run_name,
    mode=args.wandb_mode,
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)


# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 1000
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
mvc_decoder_style = "inner product"


explicit_zero_prob = False  # whether explicit bernoulli for zeros
do_sample_in_train = False  # sample the bernoulli in training
per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
dropout = config.dropout  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True

# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")

mask_value = -1
pad_value = -2
n_input_bins = n_bins


DAB_separate_optim = False


dataset_name = config.dataset_name
run_dir_name = args.run_name or f"{dataset_name}-{time.strftime('%b%d-%H-%M')}"
save_dir = output_root / run_dir_name
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


adata_lincs = sc.read_h5ad(adata_path)
adata_lincs.var.set_index(adata_lincs.var["gene_name"], inplace=True)

if adata_lincs.X.min() < 0:
    print(f"Clipping small negative values (min={adata_lincs.X.min()}) to zero.")
    adata_lincs.X = np.maximum(adata_lincs.X, 0)

adata_lincs = adata_lincs[adata_lincs.X.sum(axis=1) > 0].copy()



# make the batch category column
batch_id_labels = adata_lincs.obs["project_code"].astype("category").cat.codes.values
adata_lincs.obs["batch_id"] = batch_id_labels
drug_labels = adata_lincs.obs["pert_id"].astype("category").cat.codes.values
drugs = adata_lincs.obs["pert_id"].unique()
num_types = len(np.unique(drug_labels))
id2type = dict(enumerate(adata_lincs.obs["pert_id"].astype("category").cat.categories))
adata_lincs.obs["drug_id"] = drug_labels
adata_lincs.var["gene_name"] = adata_lincs.var.index.tolist()
adata_lincs.obs['sig_id'] = adata_lincs.obs.index
sig_id_labels = adata_lincs.obs['sig_id'].astype("category").cat.codes.values
adata_lincs.obs['sig_id_ix'] = sig_id_labels



try:
    pickle.dump(drug_labels, open(save_dir / "drug_labels.pkl", "wb"))
    pickle.dump(sig_id_labels, open(save_dir / "sig_id_labels.pkl", "wb"))

except:
    pass

## Load model and vocab 
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata_lincs.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_lincs.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata_lincs.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata_lincs = adata_lincs[:, adata_lincs.var["id_in_vocab"] >= 0].copy()

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=False,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=False,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=False,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)


preprocessor(adata_lincs, batch_key='batch_id')

### Split data
input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
all_counts = (
    adata_lincs.layers[input_layer_key].A
    if issparse(adata_lincs.layers[input_layer_key])
    else adata_lincs.layers[input_layer_key]
)
genes = adata_lincs.var["gene_name"].tolist()

drug_labels = adata_lincs.obs["drug_id"].tolist()  # make sure count from 0
drug_labels = np.array(drug_labels)

batch_ids = adata_lincs.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)
sig_id = adata_lincs.obs['sig_id_ix'].tolist()

(
    train_data,
    valid_data,
    train_drug_labels,
    valid_drug_labels,
    train_batch_labels,
    valid_batch_labels,
    train_sig_id,
    valid_sig_id,
) = train_test_split(
    all_counts, drug_labels, batch_ids, sig_id, test_size=0.1, shuffle=True
)

try: 
    pickle.dump(train_data, open(save_dir / "train_data.pkl", "wb"))
    pickle.dump(valid_data, open(save_dir / "valid_data.pkl", "wb"))
    pickle.dump(train_drug_labels, open(save_dir / "train_drug_labels.pkl", "wb"))
    pickle.dump(valid_drug_labels, open(save_dir / "valid_drug_labels.pkl", "wb"))
    pickle.dump(train_sig_id, open(save_dir / "train_sig_id.pkl", "wb"))
    pickle.dump(valid_sig_id, open(save_dir / "valid_sig_id.pkl", "wb"))

except:
    pass

vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)
# tokenize and pad the data
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)
# Data preparation for training a ML model (pytorch)

def prepare_data() -> Tuple[Dict[str, torch.Tensor]]:
    '''
    Prepare the training and validation datasets:
    - mask the values in the input data (random masking to gene expression values)
    - organizing the data into a dictionary for the DataLoader
    '''
    masked_values_train = random_mask_value(
        tokenized_train["values"], # gene expression values
        mask_ratio=mask_ratio,
        mask_value=mask_value, # masked value 
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}", # count the number of masked values
    )

    # prepare the Input and Targt values
    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"], # gene ids (tokenized)
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid # gene expression values (masked)
    target_values_train, target_values_valid = (
        tokenized_train["values"], # gene expression values (no masking)
        tokenized_valid["values"],
    )

    # convert labels to torch tensor (long - integers)
    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long() # batch labels (remove the batch effect or domain adaptation)
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_drug_labels_train = torch.from_numpy(train_drug_labels).long() # drug labels (classification task)
    tensor_drug_labels_valid = torch.from_numpy(valid_drug_labels).long()

    tensor_sig_id_train = torch.from_numpy(np.array(train_sig_id)).long()
    tensor_sig_id_valid = torch.from_numpy(np.array(valid_sig_id)).long()

    # organize the data into a dictionary (they contains all necessary input and targets for the model)
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "drug_labels": tensor_drug_labels_train,
        "sig_id": tensor_sig_id_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "drug_labels": tensor_drug_labels_valid,
        "sig_id": tensor_sig_id_valid
    }

    return train_data_pt, valid_data_pt


# Custom Pytorch Dataset - provides methods to access individual samples
# It allows data to be used in a DataLoader
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]): # data is a dictionary (prepared in the prepare_data() function)
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0] # number of samples

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()} # retrieves the data for a given index
 

# data_loader. It builds a Pytorch DataLoader object from the dataset. 
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor], # data dictionary prepared by the prepare_data() function
    batch_size: int, # batch size (number of samples in a batch)
    shuffle: bool = False, # shuffle the data in each epoch
    intra_domain_shuffle: bool = False, # shuffle the data within each domain
    drop_last: bool = False, # drop the last incomplete batch
    num_workers: int = 0, # number of workers to load the data
) -> DataLoader:
    if num_workers == 0: 
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2) # it gets the number of CPUs in the system and set the number of workers to half of the batch size

    dataset = SeqDataset(data_pt) # create a dataset object from the data dictionary
    # create a DataLoader object from the dataset
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True, # pin the memory to the GPU
    )
    return data_loader

## Step 3: Load the pre-trained scGPT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set the device to GPU if available

ntokens = len(vocab)  # size of vocabulary
# Initialize the transformer model with specific parameters
# Once this instance is created, the model can be trained and evaluated
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=False,
    do_dab=False,
    use_batch_labels=False,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=False,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=0.0,
    explicit_zero_prob=False,
    use_fast_transformer=True,
    fast_transformer_backend='flash',
    pre_norm=False
)

# Load pretrained model parameters
# It loads pre-trained weights into the model
if config.load_model is not None:
    try: # load all model parameters
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except: # load only the parameters that match the model
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict) # update the model parameters
        model.load_state_dict(model_dict)

# Freezing certain model parameters (to prevent them from being updated during training) (it is set to false)
pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
wandb.log(
        {
            "info/pre_freeze_param_count": pre_freeze_param_count,
            "info/post_freeze_param_count": post_freeze_param_count,
        },
)

model.to(device) # move the model to the device (GPU or CPU)
wandb.watch(model)



# Define the loss function and optimizer for our classification task

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config.schedule_ratio
)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
## I am only training the model for the classification task (no reconstruction loss)

def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    Handles the training loop for one epoch. 
    1. It iterates over the DataLoader object to get the data for each batch.
    2. It computes the loss.
    3. Performs backpropagation to update the model parameters.
    """
    model.train() # set the model to training mode (it enables dropout and batch normalization)
    (
        total_loss,
        total_cls,
    ) = (0.0, 0.0) # total loss (accumulates loos over batches) and classification loss
    total_error = 0.0 # total error rate
    start_time = time.time()

    num_batches = len(loader)
    # iterate over the DataLoader object to get the data for each batch
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device) # gene ids (move to device)
        input_values = batch_data["values"].to(device) # gene expression values (move to device)
        drug_labels = batch_data["drug_labels"].to(device) # drug labels (move to device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token]) # mask the padding tokens (ignore padding positions during attention computation)
        # Forward pass
        with torch.cuda.amp.autocast(enabled=config.amp): 
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels= None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=False,
            )
        # Compute the loss
            loss = 0.0
            metrics_to_log = {}
            
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], drug_labels) # classification loss
                loss = loss + loss_cls # total loss
                metrics_to_log.update({"train/cls": loss_cls.item()}) # log the classification loss

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == drug_labels)
                    .sum()
                    .item()
                ) / drug_labels.size(0)
        
        # Backward pass and optimization   
        model.zero_grad() # zero the gradients (to prevent accumulation of gradients)
        scaler.scale(loss).backward() # scales the loss and performs backpropagation
        scaler.unscale_(optimizer) # unscale the gradients
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_( # clip the gradients to prevent exploding gradients
                model.parameters(), 
                1.0, # max allowed norm of the gradients
                error_if_nonfinite=False if scaler.is_enabled() else True, # error if non-finite values are found
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer) # update the model parameters using the optimizer
        scaler.update() # update the scaler factor for the next iteration
 
        wandb.log(metrics_to_log)

        total_loss += loss.item() # accumulate the loss
        total_cls += loss_cls.item() if CLS else 0.0 # accumulate the classification loss
        total_error += error_rate if CLS else 0.0 # accumulate the error rate
 
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_cls = total_cls / log_interval

            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"err {cur_error:5.2f} | " if CLS else "")
            )
            total_loss = 0
            total_cls = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    Computation of the loss and error rate.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    predictions = []
    drug_real = []
    sig_id_input = []
    with torch.no_grad(): # disable gradient computation
        for batch_data in loader: # iterate over the DataLoader object to get the data for each batch
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            drug_labels = batch_data["drug_labels"].to(device)
            sig_id = batch_data["sig_id"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels= None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, drug_labels)

            # Accumulate the loss and error rate
            total_loss += loss.item() * len(input_gene_ids) # accumulate the loss multiplied by the number of samples
            accuracy = (output_values.argmax(1) == drug_labels).sum().item() # compute the accuracy (number of correct predictions)
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids) # missclassification rate
            total_num += len(input_gene_ids) # total number of samples
            preds = output_values.argmax(1).cpu().numpy() # store predictions
            predictions.append(preds)
            drug_real.append(drug_labels.cpu().numpy())
            sig_id_input.append(sig_id.cpu().numpy())

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/sum_mse_dab": (total_loss ) / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)
    
    try: 
        pickle.dump(predictions, open(save_dir / f"predictions_eval_{epoch}.pkl", "wb"))
        pickle.dump(drug_real, open(save_dir / f"labels_eval_{epoch}.pkl", "wb"))
        pickle.dump(sig_id_input, open(save_dir / f"sig_id_eval_{epoch}.pkl", "wb"))

    except:
        pass

    return total_loss / total_num, total_error / total_num

## Step 4: Finetune scGPT with task-specific objectives
# Training and validation process over multiple epochs

best_val_loss = float("inf") # keep track of the best model
best_model = None
define_wandb_metrcis()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    # Prepare data loaders
    train_data_pt, valid_data_pt = prepare_data() # it is inside each epoch because it is designed for MLM
    train_loader = prepare_dataloader( # Convert the data into a DataLoader object
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    # Train and evaluate the model (for each epoch)
    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_err = evaluate(
        model,
        loader=valid_loader,
        )
    
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
    )
    logger.info("-" * 89)

    # Save the model if the validation loss is the best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")
    # Adjust the learning rate schedule
    scheduler.step()


# save the model into the save_dir
torch.save(best_model.state_dict(), save_dir / args.save_model_name)

def test(model: nn.Module, adata: DataLoader) -> float:
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    drug_labels = adata.obs["drug_id"].tolist()  # make sure count from 0
    drug_labels = np.array(drug_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    sig_id = adata.obs['sig_id_ix'].tolist()
    sig_id = np.array(sig_id)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "drug_labels": torch.from_numpy(drug_labels).long(),
        "sig_id": torch.from_numpy(sig_id).long(),
    }


    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
    )

    # compute accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(drug_labels, predictions)
    precision = precision_score(drug_labels, predictions, average="macro")
    recall = recall_score(drug_labels, predictions, average="macro")
    macro_f1 = f1_score(drug_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, drug_labels, results



predictions, labels, results = test(best_model, adata_lincs)
try: 
    pickle.dump(predictions, open(save_dir / f"predictions_test.pkl", "wb"))
    pickle.dump(labels, open(save_dir / f"labels_test.pkl", "wb"))
    pickle.dump(adata_lincs.obs.index, open(save_dir / f"sig_id_test.pkl", "wb"))

except:
    pass

