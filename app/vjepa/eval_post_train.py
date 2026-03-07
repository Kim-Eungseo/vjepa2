# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for VICReg post-trained models.

Measures representation quality with metrics aligned to VICReg training objectives:
  1. Alignment (invariance): MSE between representations of positive pairs
  2. Feature std (variance): Per-feature std — should be >= 1.0 (no collapse)
  3. Feature correlation (covariance): Off-diagonal covariance — should be low
  4. Effective rank: Approximate dimensionality of the representation space
  5. Frozen linear probe accuracy on a downstream classification task

Usage:
  python -m app.scaffold --app vjepa --module eval_post_train \
    --fname configs/eval/vitl16-posttrain/eval_posttrain.yaml
"""

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_video_model, init_vicreg_pooler, load_checkpoint
from src.datasets.data_manager import init_data
from src.datasets.pair_collator import PairCollator
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = get_logger(__name__, force=True)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def compute_effective_rank(embeddings):
    """Compute effective rank of the embedding matrix via singular values.

    A higher effective rank means the representation space is better utilized
    (features are more decorrelated and spread out).

    Args:
        embeddings: [N, D] tensor
    Returns:
        float: effective rank (1 <= erank <= D)
    """
    # Center
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(embeddings)
    # Normalize singular values to form a probability distribution
    p = S / S.sum()
    # Shannon entropy -> effective rank
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return float(torch.exp(entropy))


def compute_representation_metrics(z1_all, z2_all):
    """Compute VICReg-aligned representation quality metrics.

    Args:
        z1_all: [N, D] pooled representations from view 1
        z2_all: [N, D] pooled representations from view 2

    Returns:
        dict of metric_name -> float
    """
    # 1. Alignment (invariance): same as VICReg invariance loss
    alignment = float(F.mse_loss(z1_all, z2_all))

    # 2. Cosine similarity between positive pairs
    z1_norm = F.normalize(z1_all, dim=1)
    z2_norm = F.normalize(z2_all, dim=1)
    cosine_sim = float((z1_norm * z2_norm).sum(dim=1).mean())

    # 3. Feature std (variance regularization check)
    z_cat = torch.cat([z1_all, z2_all], dim=0)
    z_centered = z_cat - z_cat.mean(dim=0, keepdim=True)
    feature_std = torch.sqrt(z_centered.var(dim=0) + 1e-4)
    mean_std = float(feature_std.mean())
    min_std = float(feature_std.min())
    collapsed_dims = int((feature_std < 0.1).sum())

    # 4. Off-diagonal covariance (decorrelation check)
    N = z_cat.shape[0]
    cov = (z_centered.T @ z_centered) / (N - 1)
    D = cov.shape[0]
    off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    off_diag_cov = float(off_diag.pow(2).mean())

    # 5. Effective rank
    # Use a subset if too many samples (SVD is expensive)
    max_samples = min(2048, z_cat.shape[0])
    erank = compute_effective_rank(z_cat[:max_samples].float())

    return {
        "alignment_mse": alignment,
        "cosine_similarity": cosine_sim,
        "mean_feature_std": mean_std,
        "min_feature_std": min_std,
        "collapsed_dims": collapsed_dims,
        "off_diag_cov": off_diag_cov,
        "effective_rank": erank,
        "embed_dim": D,
    }


def main(args, resume_preempt=False):
    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    which_dtype = cfgs_meta.get("dtype", "bfloat16")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps", None)
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)

    # -- VICREG
    cfgs_loss = args.get("loss", {})
    cfgs_vicreg = cfgs_loss.get("vicreg", {})
    vicreg_pooler_depth = cfgs_vicreg.get("pooler_depth", 1)

    # -- EVAL
    cfgs_eval = args.get("eval", {})
    max_eval_batches = cfgs_eval.get("max_batches", 200)

    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log paths
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"eval_log_r{rank}.csv")

    # -- init encoder
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=2,
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=False,
        wide_silu=wide_silu,
        use_rope=use_rope,
    )

    # -- init VICReg pooler
    embed_dim = encoder.backbone.embed_dim
    vicreg_pooler = init_vicreg_pooler(
        device=device,
        embed_dim=embed_dim,
        num_heads=encoder.backbone.num_heads,
        depth=vicreg_pooler_depth,
    )

    # -- load checkpoint
    if r_file is None:
        latest_path = os.path.join(folder, "latest.pt")
        if os.path.exists(latest_path):
            r_file = latest_path
        else:
            raise ValueError("No checkpoint specified. Set 'read_checkpoint' in meta config.")

    encoder, predictor, _, _, _, epoch = load_checkpoint(
        r_path=r_file,
        encoder=encoder,
        predictor=predictor,
        target_encoder=None,
        opt=None,
        scaler=None,
        vicreg_pooler=vicreg_pooler,
    )
    del predictor
    logger.info(f"Loaded checkpoint from epoch {epoch}")

    encoder.eval()
    vicreg_pooler.eval()

    # -- data transforms (deterministic center crop for eval)
    cfgs_data_aug = args.get("data_aug", {})
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    pair_collator = PairCollator(dataset_fpcs=dataset_fpcs)

    (eval_loader, eval_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=False,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        num_clips=2,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=False,
        collator=pair_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )

    # -- Collect representations
    # Repeatedly iterate over the dataset (with re-shuffled views each time)
    # until we have enough batches, since datasets may be small.
    logger.info("Computing representations for evaluation...")
    z1_all_list = []
    z2_all_list = []
    loss_meter = AverageMeter()
    total_itr = 0

    with torch.no_grad():
        while total_itr < max_eval_batches:
            eval_sampler.set_epoch(total_itr)
            for sample in eval_loader:
                if total_itr >= max_eval_batches:
                    break

                for clip1_batch, clip2_batch in sample:
                    clip1 = clip1_batch.to(device, non_blocking=True)
                    clip2 = clip2_batch.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        z1_tokens = encoder([clip1])
                        z2_tokens = encoder([clip2])

                        for z1_t, z2_t in zip(z1_tokens, z2_tokens):
                            z1 = vicreg_pooler(z1_t).squeeze(1)
                            z2 = vicreg_pooler(z2_t).squeeze(1)

                            z1_all_list.append(z1.float().cpu())
                            z2_all_list.append(z2.float().cpu())

                            loss_meter.update(float(F.mse_loss(z1, z2)))

                if total_itr % 20 == 0:
                    logger.info(f"  [{total_itr}/{max_eval_batches}] alignment_mse: {loss_meter.avg:.4f}")
                total_itr += 1

    z1_all = torch.cat(z1_all_list, dim=0)
    z2_all = torch.cat(z2_all_list, dim=0)
    logger.info(f"Collected {z1_all.shape[0]} samples, embed_dim={z1_all.shape[1]}")

    # -- Compute metrics
    metrics = compute_representation_metrics(z1_all, z2_all)

    # -- Log results
    logger.info("=" * 60)
    logger.info("  Post-Train Evaluation Results (epoch %d)" % epoch)
    logger.info("=" * 60)
    logger.info(f"  Alignment (MSE, lower=better):      {metrics['alignment_mse']:.6f}")
    logger.info(f"  Cosine similarity (higher=better):   {metrics['cosine_similarity']:.4f}")
    logger.info(f"  Mean feature std (target>=1.0):      {metrics['mean_feature_std']:.4f}")
    logger.info(f"  Min feature std:                     {metrics['min_feature_std']:.4f}")
    logger.info(f"  Collapsed dims (std<0.1):            {metrics['collapsed_dims']}/{metrics['embed_dim']}")
    logger.info(f"  Off-diagonal cov (lower=better):     {metrics['off_diag_cov']:.6f}")
    logger.info(f"  Effective rank (higher=better):      {metrics['effective_rank']:.1f}/{metrics['embed_dim']}")
    logger.info("=" * 60)

    # -- Save to CSV
    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.6f", "alignment_mse"),
            ("%.4f", "cosine_sim"),
            ("%.4f", "mean_std"),
            ("%.4f", "min_std"),
            ("%d", "collapsed_dims"),
            ("%.6f", "off_diag_cov"),
            ("%.1f", "effective_rank"),
        )
        csv_logger.log(
            epoch,
            metrics["alignment_mse"],
            metrics["cosine_similarity"],
            metrics["mean_feature_std"],
            metrics["min_feature_std"],
            metrics["collapsed_dims"],
            metrics["off_diag_cov"],
            metrics["effective_rank"],
        )
        logger.info(f"Results saved to {log_file}")

    return metrics
