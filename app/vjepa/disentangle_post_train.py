# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_opt, init_video_model, load_checkpoint
from src.losses.disentangle import CrossCorrelationSepLoss
from src.losses.vicreg import VICRegLoss
from src.models.projection_head import ProjectionHead
from src.datasets.data_manager import init_data
from src.datasets.pair_collator import PairCollator
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- WANDB
    cfgs_wandb = args.get("wandb", {})
    use_wandb = _WANDB_AVAILABLE and cfgs_wandb.get("enabled", False)
    wandb_entity = cfgs_wandb.get("entity", None)
    wandb_project = cfgs_wandb.get("project", "video-encoder")
    wandb_run_name = cfgs_wandb.get("run_name", None)

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
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
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
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
    freeze_encoder = cfgs_model.get("freeze_encoder", False)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps", None)
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- DISENTANGLE LOSS
    cfgs_loss = args.get("loss")
    cfgs_disentangle = cfgs_loss.get("disentangle", {})
    proj_dim = cfgs_disentangle.get("proj_dim", 256)
    pooler_depth = cfgs_disentangle.get("pooler_depth", 1)
    # VICReg coefficients for task head
    task_inv_coeff = cfgs_disentangle.get("task_inv_coeff", 1.0)
    task_std_coeff = cfgs_disentangle.get("task_std_coeff", 1.0)
    task_cov_coeff = cfgs_disentangle.get("task_cov_coeff", 1.0)
    # VICReg coefficients for domain head
    dom_inv_coeff = cfgs_disentangle.get("dom_inv_coeff", 1.0)
    dom_std_coeff = cfgs_disentangle.get("dom_std_coeff", 1.0)
    dom_cov_coeff = cfgs_disentangle.get("dom_cov_coeff", 1.0)
    # Loss weights
    alpha = cfgs_disentangle.get("alpha", 1.0)  # L_dom weight
    beta = cfgs_disentangle.get("beta", 1.0)     # L_sep weight

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (is_anneal and resume_preempt)
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- init wandb (rank 0 only)
    if use_wandb and rank == 0:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_run_name,
            config=args,
        )
        logger.info("wandb initialized")

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    # -- Post-training always requires a pretrained checkpoint
    if is_anneal:
        if os.path.exists(latest_path) and resume_anneal:
            load_path = latest_path
        else:
            load_path = anneal_ckpt
            resume_anneal = False
    elif os.path.exists(latest_path):
        load_path = latest_path
    elif r_file is not None:
        load_path = r_file
    else:
        raise ValueError(
            "post_train.py requires a pretrained checkpoint. "
            "Set 'read_checkpoint' in meta config or ensure 'latest.pt' exists in the output folder."
        )
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "task_loss"),
        ("%.5f", "dom_loss"),
        ("%.5f", "sep_loss"),
        ("%.6f", "lr"),
        ("%.6f", "wd"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init encoder (predictor is built but not used, needed for checkpoint loading)
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
        use_activation_checkpointing=use_activation_checkpointing,
    )

    if freeze_encoder:
        logger.info("Freezing encoder parameters.")
        for p in encoder.parameters():
            p.requires_grad = False

    # -- init dual projection heads
    embed_dim = encoder.backbone.embed_dim
    num_heads = encoder.backbone.num_heads
    task_head = ProjectionHead(
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        num_heads=num_heads,
        pooler_depth=pooler_depth,
    ).to(device)
    domain_head = ProjectionHead(
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        num_heads=num_heads,
        pooler_depth=pooler_depth,
    ).to(device)
    logger.info(f"ProjectionHead proj_dim={proj_dim}, pooler_depth={pooler_depth}")

    # -- init loss functions
    task_vicreg_fn = VICRegLoss(
        inv_coeff=task_inv_coeff,
        std_coeff=task_std_coeff,
        cov_coeff=task_cov_coeff,
    )
    dom_vicreg_fn = VICRegLoss(
        inv_coeff=dom_inv_coeff,
        std_coeff=dom_std_coeff,
        cov_coeff=dom_cov_coeff,
    )
    sep_loss_fn = CrossCorrelationSepLoss()
    logger.info(f"Task VICReg: inv={task_inv_coeff}, std={task_std_coeff}, cov={task_cov_coeff}")
    logger.info(f"Dom VICReg: inv={dom_inv_coeff}, std={dom_std_coeff}, cov={dom_cov_coeff}")
    logger.info(f"Loss weights: alpha={alpha} (dom), beta={beta} (sep)")

    if compile_model:
        logger.info("Compiling encoder.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()

    # -- init data collator (positive pairs, num_clips=2)
    pair_collator = PairCollator(dataset_fpcs=dataset_fpcs)

    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        num_clips=2,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=pair_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler (no predictor)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=None,
        freeze_encoder=freeze_encoder,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        extra_modules=[task_head, domain_head],
    )
    if world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        task_head = DistributedDataParallel(task_head, static_graph=True)
        domain_head = DistributedDataParallel(domain_head, static_graph=True)

    # -- load pretrained encoder from checkpoint
    # Only restore optimizer/scaler when resuming from a post_train checkpoint
    _is_resume = os.path.exists(latest_path) and load_path == latest_path
    (
        encoder,
        _predictor,
        _target_encoder,
        _opt,
        _scaler,
        start_epoch,
    ) = load_checkpoint(
        r_path=load_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=None,
        opt=optimizer if _is_resume else None,
        scaler=scaler if _is_resume else None,
        is_anneal=is_anneal and not resume_anneal,
    )
    if _is_resume:
        optimizer, scaler = _opt, _scaler
    del predictor  # no longer needed after loading
    if not _is_resume:
        start_epoch = 0  # fresh post-training from pretrain checkpoint
    if not is_anneal or resume_anneal:
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "task_head": task_head.state_dict(),
            "domain_head": domain_head.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        task_loss_meter = AverageMeter()
        dom_loss_meter = AverageMeter()
        sep_loss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            # sample = list of (clip1_batch, clip2_batch) per fpc
            def load_pairs():
                all_clip1, all_clip2 = [], []
                for clip1_batch, clip2_batch in sample:
                    all_clip1.append(clip1_batch.to(device, non_blocking=True))
                    all_clip2.append(clip2_batch.to(device, non_blocking=True))
                return all_clip1, all_clip2

            clips1, clips2 = load_pairs()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_encoder(clips):
                    """Encode clips without masks. Returns list of [B, N, D] tensors."""
                    if freeze_encoder:
                        with torch.no_grad():
                            return encoder(clips)
                    return encoder(clips)

                # Step 1. Forward
                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    h1_list = forward_encoder(clips1)  # list of [B, N, D]
                    h2_list = forward_encoder(clips2)  # list of [B, N, D]

                    l_task_total = 0
                    l_dom_total = 0
                    l_sep_total = 0
                    n = 0
                    for h1, h2 in zip(h1_list, h2_list):
                        # Dual-head forward
                        z_task1 = task_head(h1)    # [B, d]
                        z_task2 = task_head(h2)    # [B, d]
                        z_dom1 = domain_head(h1)   # [B, d]
                        z_dom2 = domain_head(h2)   # [B, d]

                        # VICReg losses per head
                        task_dict = task_vicreg_fn(z_task1, z_task2)
                        dom_dict = dom_vicreg_fn(z_dom1, z_dom2)

                        # Separation loss (both views)
                        l_sep = sep_loss_fn(z_task1, z_dom1) + sep_loss_fn(z_task2, z_dom2)

                        l_task_total += task_dict["loss"]
                        l_dom_total += dom_dict["loss"]
                        l_sep_total += l_sep
                        n += 1

                    if n > 0:
                        l_task_total /= n
                        l_dom_total /= n
                        l_sep_total /= n

                    loss = l_task_total + alpha * l_dom_total + beta * l_sep_total

                # Step 2. Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return (
                    float(loss),
                    float(l_task_total),
                    float(l_dom_total),
                    float(l_sep_total),
                    _new_lr,
                    _new_wd,
                )

            result, gpu_etime_ms = gpu_timer(train_step)
            loss, l_task, l_dom, l_sep, _new_lr, _new_wd = result
            task_loss_meter.update(l_task)
            dom_loss_meter.update(l_dom)
            sep_loss_meter.update(l_sep)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss,
                    l_task, l_dom, l_sep,
                    _new_lr, _new_wd,
                    iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "[task: %.3f dom: %.3f sep: %.3f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            task_loss_meter.avg,
                            dom_loss_meter.avg,
                            sep_loss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()

            if use_wandb and rank == 0:
                wandb.log(
                    {
                        "loss": loss,
                        "task_loss": l_task,
                        "dom_loss": l_dom,
                        "sep_loss": l_sep,
                        "lr": _new_lr,
                        "wd": _new_wd,
                        "gpu_mem_mb": torch.cuda.max_memory_allocated() / 1024.0**2,
                        "iter_time_ms": iter_elapsed_time_ms,
                        "epoch": epoch + 1,
                    },
                    step=epoch * ipe + itr,
                )

            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

    if use_wandb and rank == 0:
        wandb.finish()
