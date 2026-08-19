"""SemiSAM-Crack training loop — Mean Teacher + SAM pseudo-labeling.

Two-stage training adapted from UnCoL (Lu et al., IEEE TMI, 2026).

Stage 1 (PRETRAIN): Foundation model knowledge distillation on labeled data.
  - Optional BCP CutMix (cfg.training.bcp_enabled).

Stage 2 (SSL): Semi-supervised with EMA teacher + SAM foundation teacher.
  - Component-level selective SAM queries with acceptance gate.
  - Optional dual-teacher fusion, BCP CutMix, UPFM, scale-equivariance
    (all gated by config flags; currently disabled).

Usage:
    python train.py --config config.yaml --stage ssl
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from data.dataset import CrackDataset, TwoStreamBatchSampler
from data.transforms import get_train_transform, get_val_transform
from losses.supervised import bce_dice_loss
from losses.topology import soft_cl_dice_loss

from losses.consistency import mse_consistency_loss
from losses.confidence_aware import confidence_aware_loss
from losses.boundary import boundary_band_mask
from model.factory import build_specialist, uses_adamw


def _make_specialist(encoder_name: str, decoder_channels: tuple,
                     in_channels: int = 3, pretrained: bool = True,
                     scrwkv_root: str | None = None,
                     scrwkv_risk_scale: float = 0.25):
    """Factory: dispatch to correct specialist class based on encoder name."""
    kwargs = {}
    if scrwkv_root is not None:
        kwargs["scrwkv_root"] = scrwkv_root
    kwargs["scrwkv_risk_scale"] = scrwkv_risk_scale
    return build_specialist(
        encoder_name, decoder_channels, in_channels, pretrained, **kwargs
    )
from model.generalist_wrapper import GeneralistWrapper
from model.structure_risk import StructureRiskHeads
from model.consultation import ConsultationPredictor, CrossAttentionConsultation
from prompting.boxes import boxes_from_mask, jitter_boxes
from prompting.components import extract_components
from prompting.skeleton import (
    skeletonize_mask,
    sample_positive_points,
    sample_negative_points,
)
from training.dual_teacher import (
    entropy,
    entropy_from_logits,
    fuse_pseudo_labels_binary,
)
from uncertainty.geo import compute_u_geo
from utils.config import Config
from utils.ramps import sigmoid_rampup
from evaluate import validate


# ---------------------------------------------------------------------------
# helpers (shared across stages)
# ---------------------------------------------------------------------------

def update_ema_variables(model, ema_model, alpha, global_step):
    """EMA update for Mean Teacher (ported verbatim from SemiSAM+)."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    """Sigmoid rampup (ported verbatim from SemiSAM+)."""
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def _denormalize(img_np: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization for SAM input."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_np * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _pool_risk_maps(
    risk_maps: dict[str, torch.Tensor],  # {"R_topo": (B,1,H,W), ...}
    comp_mask: np.ndarray,               # (H, W) binary
    b_idx: int,
) -> dict[str, float]:
    """Mean-pool learned risk maps over a component region. Returns scalars."""
    mask_t = torch.from_numpy(comp_mask.astype(np.float32)).to(risk_maps["R_topo"].device)
    if mask_t.sum() == 0:
        return {"R_topo": 0.0, "R_morph": 0.0}
    scores = {}
    for key in ("R_topo", "R_morph"):
        if key in risk_maps:
            pooled = (risk_maps[key][b_idx, 0] * mask_t).sum() / mask_t.sum()
            scores[key] = float(pooled.item())
    return scores


def _extract_consult_features(
    comp,           # CrackComponent
    topo_score: float,
    morph_score: float,
) -> np.ndarray:
    """Extract 8 component features for ConsultationPredictor MLP.

    Features: [R_topo, R_morph, confidence, log(area),
               log(length), log(width), n_endpoints, bbox_fill_ratio]
    """
    x1, y1, x2, y2 = comp.bbox
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    return np.array([
        topo_score,
        morph_score,
        comp.confidence,
        np.log(max(comp.area, 1)),
        np.log(max(comp.length, 1)),
        np.log(max(comp.width, 0.01)),
        float(len(comp.endpoints)),
        min(comp.area / bbox_area, 1.0),  # bbox fill ratio
    ], dtype=np.float32)


def _comp_dice(pred_region: np.ndarray, gt_region: np.ndarray) -> float:
    """Dice over a component's pixel set (1D arrays from the same comp_mask)."""
    p = pred_region.astype(np.float64)
    g = gt_region.astype(np.float64)
    inter = float((p * g).sum())
    return (2.0 * inter + 1e-5) / (p.sum() + g.sum() + 1e-5)


def _pooled_consult_feats(
    student_out,
    comp,               # CrackComponent (populated in-place)
    prob_map: np.ndarray,   # (H, W) probs for appearance scalars
    b_idx: int,
):
    """Pooled decoder-feature streams for CrossAttentionConsultation.

    Pools the single decoder feature map (C, H, W) at skeleton path,
    endpoint neighborhoods, inside mask, and background ring.

    Returns (topo_vec, morph_vec): concat(z_skel, z_endpoint),
    concat(z_in, z_ring, z_diff) — each (2C,) / (3C,).
    """
    from prompting.components import (
        extract_topology_features, extract_appearance_features,
    )
    feat = (
        student_out["features"][b_idx].detach().cpu().numpy()
    )  # (C, H, W)
    extract_topology_features([feat], comp)
    extract_appearance_features([feat], comp, prob_map)
    topo = np.concatenate([comp.z_skel, comp.z_endpoint])
    morph = np.concatenate([comp.z_in, comp.z_ring, comp.z_diff])
    return topo.astype(np.float32), morph.astype(np.float32)


# ---------------------------------------------------------------------------
# helpers: SAM prompt generation (shared)
# ---------------------------------------------------------------------------

def _generate_sam_prompts(
    pred_mask_tensor: torch.Tensor,
    coarse_bin: np.ndarray,
    cfg,
):
    """Generate box + optional skeleton point prompts from coarse mask.

    Returns: (boxes, all_points, point_labels_list)
    """
    boxes = boxes_from_mask(
        pred_mask_tensor.unsqueeze(0),
        min_area=cfg.prompting.cc_min_area,
        margin_px=cfg.prompting.box_margin_px,
        merge_dist_px=cfg.prompting.box_merge_dist_px,
    )

    # ponytail: SAM-1 only handles 1 box. Merge all into outer bbox.
    if len(boxes) > 1:
        xs = [b[0] for b in boxes] + [b[2] for b in boxes]
        ys = [b[1] for b in boxes] + [b[3] for b in boxes]
        boxes = [[min(xs), min(ys), max(xs), max(ys)]]

    all_points = None
    point_labels_list = None
    if cfg.prompting.skeleton_enabled and coarse_bin.sum() > 0:
        try:
            skeletons = skeletonize_mask(
                coarse_bin, prune_spur_px=cfg.prompting.prune_spur_px
            )
            pos_pts = sample_positive_points(
                skeletons, spacing_px=cfg.prompting.pos_spacing_px
            )
            neg_pts = sample_negative_points(
                pos_pts, skeletons,
                offset_px=cfg.prompting.neg_offset_px,
                mask_shape=coarse_bin.shape,
                pos_mask=coarse_bin,
                subsample=cfg.prompting.neg_subsample,
            )
            all_pos = np.concatenate(pos_pts, axis=0) if pos_pts else np.zeros((0, 2), dtype=int)
            all_neg = np.concatenate(neg_pts, axis=0) if neg_pts else np.zeros((0, 2), dtype=int)
            all_points = []
            point_labels_list = []
            for pt in all_pos:
                all_points.append([float(pt[1]), float(pt[0])])
                point_labels_list.append(1)
            for pt in all_neg:
                all_points.append([float(pt[1]), float(pt[0])])
                point_labels_list.append(0)
        except Exception:
            pass  # skeleton may fail on tiny masks

    return boxes, all_points, point_labels_list



# ===================================================================
# Stage 1: Foundation Model Knowledge Distillation Pretraining
# ===================================================================

def pretrain(cfg: Config, snapshot_path: str):
    """Stage 1 — pretrain specialist on labeled data with SAM KD.

    Optional BCP CutMix via cfg.training.bcp_enabled.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[PRETRAIN] Using device: {device}")

    # --- models ---
    model = _make_specialist(
        cfg.model.specialist_encoder,
        tuple(cfg.model.specialist_decoder_channels),
        3, True, cfg.model.scrwkv_root, cfg.model.scrwkv_risk_scale,
    ).to(device)

    ema_model = _make_specialist(
        cfg.model.specialist_encoder,
        tuple(cfg.model.specialist_decoder_channels),
        3, True, cfg.model.scrwkv_root, cfg.model.scrwkv_risk_scale,
    ).to(device)
    for p in ema_model.parameters():
        p.detach_()

    generalist = GeneralistWrapper(
        checkpoint=cfg.model.generalist_checkpoint, device=device,
    )
    logging.info("Generalist (SAM) loaded for pretraining KD.")

    # --- data (labeled only for pretraining) ---
    img_size = tuple(cfg.data.img_size)
    train_dataset = CrackDataset(
        image_dir=cfg.data.labeled_image_dir,
        mask_dir=cfg.data.labeled_mask_dir,
        transform=get_train_transform(img_size),
    )
    num_labeled = min(cfg.data.labeled_num, len(train_dataset))
    logging.info(f"[PRETRAIN] {num_labeled} labeled images")

    trainloader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )

    val_dataset = CrackDataset(
        image_dir=cfg.data.val_image_dir,
        mask_dir=cfg.data.val_mask_dir,
        transform=get_val_transform(img_size),
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --- optimizer (AdamW for transformer backbones, SGD for CNNs — each backbone's recipe) ---
    _uses_adamw = uses_adamw(cfg.model.specialist_encoder)
    init_lr = cfg.training.ssl_lr if _uses_adamw else cfg.training.base_lr
    if _uses_adamw:
        optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0001
        )

    # --- logging ---
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info(f"[PRETRAIN] {len(trainloader)} iters/epoch")

    iter_num = 0
    max_epoch = cfg.training.stage1_iterations // len(trainloader) + 1
    best_dice = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch["image"].to(device)
            label_batch = sampled_batch["label"].to(device)
            B = volume_batch.shape[0]

            # --- forward ---
            student_out = model(volume_batch)

            # --- supervised loss ---
            loss_bce, loss_dice = bce_dice_loss(
                student_out["mask_logits"][:B], label_batch[:B]
            )
            loss_sup = cfg.topology.bce_weight * loss_bce + cfg.topology.dice_weight * loss_dice

            # Topology calibration: differentiable clDice loss (every iter)
            if (
                cfg.topology.enabled
                and cfg.topology.cl_dice_weight > 0
                and iter_num > cfg.topology.cl_dice_warmup
                and iter_num % cfg.topology.cl_dice_interval == 0
            ):
                try:
                    loss_cldice = soft_cl_dice_loss(
                        torch.sigmoid(student_out["mask_logits"][:B]),
                        label_batch[:B],
                    )
                    if torch.isfinite(loss_cldice):
                        loss_sup = (loss_sup
                                    + cfg.topology.cl_dice_weight * loss_cldice)
                except Exception:
                    pass  # ponytail: clDice may fail on degenerate masks

            # Train Dice (torch, no numpy)
            with torch.no_grad():
                pred_bin = (torch.sigmoid(student_out["mask_logits"][:B]) > 0.5).float()
                gt_bin = label_batch[:B]
                smooth = 1e-5
                intersect = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
                p_sum = pred_bin.sum(dim=(1, 2, 3))
                t_sum = gt_bin.sum(dim=(1, 2, 3))
                per_img_dice = (2.0 * intersect + smooth) / (p_sum + t_sum + smooth)
                train_dice = per_img_dice.mean().item()

            # --- SAM KD loss: consistency between student and SAM pseudo-labels ---
            loss_kd = torch.tensor(0.0, device=device)
            if cfg.training.beta_sam > 0 and B > 0:
                kd_count = 0
                for b_idx in range(min(B, 4)):  # limit SAM calls for speed
                    img_np = volume_batch[b_idx].cpu().numpy().transpose(1, 2, 0)
                    img_np = _denormalize(img_np)
                    pred_prob = torch.sigmoid(
                        student_out["mask_logits"][b_idx]
                    ).detach()

                    coarse_bin = (pred_prob.cpu().numpy() > 0.5).astype(np.uint8)
                    if coarse_bin.sum() == 0:
                        continue

                    boxes, all_points, pt_labels = _generate_sam_prompts(
                        pred_prob, coarse_bin, cfg
                    )
                    if len(boxes) == 0:
                        continue

                    try:
                        sam_mask = generalist.predict(
                            img_np, boxes=boxes,
                            points=all_points, point_labels=pt_labels,
                        )
                    except Exception as e:
                        logging.warning(
                            f"[PRETRAIN] SAM predict failed for b_idx={b_idx}: {e}"
                        )
                        continue
                    sam_t = torch.from_numpy(sam_mask.astype(np.float32)).to(device)
                    # Ensure shapes match: student pred is (1, H, W) before squeeze
                    prob_t = torch.sigmoid(
                        student_out["mask_logits"][b_idx]
                    )
                    # ponytail: align shapes — student has (1, H, W), sam has (H, W)
                    if prob_t.shape[0] == 1:
                        prob_t = prob_t.squeeze(0)
                    if sam_t.shape != prob_t.shape:
                        # ponytail: resize SAM mask to match student output
                        sam_t = F.interpolate(
                            sam_t.unsqueeze(0).unsqueeze(0),
                            size=prob_t.shape[-2:],
                            mode="bilinear", align_corners=True,
                        ).squeeze(0).squeeze(0)
                    # ponytail: MSE between student and SAM as KD signal
                    loss_kd += F.mse_loss(prob_t, sam_t)
                    kd_count += 1

                if kd_count > 0:
                    loss_kd /= kd_count

            # --- total loss ---
            loss = loss_sup + cfg.training.beta_sam * loss_kd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            update_ema_variables(model, ema_model, cfg.training.ema_decay, iter_num)

            # --- LR schedule ---
            lr_ = init_lr * (
                1.0 - iter_num / cfg.training.stage1_iterations
            ) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            # --- logging ---
            writer.add_scalar("pretrain/lr", lr_, iter_num)
            writer.add_scalar("pretrain/total_loss", loss.item(), iter_num)
            writer.add_scalar("pretrain/loss_sup", loss_sup.item(), iter_num)
            writer.add_scalar("pretrain/loss_kd", loss_kd.item(), iter_num)
            writer.add_scalar("pretrain/train_dice", train_dice, iter_num)

            if iter_num % 20 == 0:
                img_grid = make_grid(volume_batch[0, :3].unsqueeze(0), normalize=True)
                writer.add_image("pretrain/Image", img_grid, iter_num)
                pred_grid = make_grid(
                    torch.sigmoid(student_out["mask_logits"][0]), normalize=False
                )
                writer.add_image("pretrain/Prediction", pred_grid, iter_num)

            # --- validation ---
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                val_dice = validate(model, valloader, device)
                writer.add_scalar("pretrain/val_dice", val_dice, iter_num)
                logging.info(
                    f"[PRETRAIN] iter {iter_num} : "
                    f"train_dice : {train_dice:.4f}  "
                    f"val_dice : {val_dice:.4f}"
                )
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_path = os.path.join(snapshot_path, "best_pretrain.pth")
                    torch.save(model.state_dict(), best_path)
                    torch.save(
                        ema_model.state_dict(),
                        os.path.join(snapshot_path, "best_pretrain_ema.pth"),
                    )
                    logging.info(f"Saved best pretrain: dice={best_dice:.4f}")
                model.train()

            if iter_num % 3000 == 0:
                save_path = os.path.join(
                    snapshot_path, f"pretrain_iter_{iter_num}.pth"
                )
                torch.save(model.state_dict(), save_path)

            if iter_num >= cfg.training.stage1_iterations:
                break

        if iter_num >= cfg.training.stage1_iterations:
            iterator.close()
            break

    # Save final pretrained model
    final_path = os.path.join(snapshot_path, "pretrain_final.pth")
    torch.save(model.state_dict(), final_path)
    torch.save(ema_model.state_dict(),
               os.path.join(snapshot_path, "pretrain_final_ema.pth"))
    logging.info(f"[PRETRAIN] Final model saved to {final_path}")

    writer.close()
    return "Pretraining Finished!"


# ===================================================================
# Stage 2: Dual-Teacher Semi-Supervised Learning
# ===================================================================

def ssl_train(cfg: Config, snapshot_path: str):
    """Stage 2 — semi-supervised with EMA teacher + SAM pseudo-labels.

    Component-level SAM queries with acceptance gate and error-type routing.
    Optional features gated by config: dual-teacher fusion, BCP CutMix,
    UPFM, scale-equivariance. All currently disabled in config.yaml.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[SSL] Using device: {device}")

    # --- models ---
    model = _make_specialist(
        cfg.model.specialist_encoder,
        tuple(cfg.model.specialist_decoder_channels),
        3, True, cfg.model.scrwkv_root, cfg.model.scrwkv_risk_scale,
    ).to(device)

    ema_model = _make_specialist(
        cfg.model.specialist_encoder,
        tuple(cfg.model.specialist_decoder_channels),
        3, True, cfg.model.scrwkv_root, cfg.model.scrwkv_risk_scale,
    ).to(device)
    for p in ema_model.parameters():
        p.detach_()

    # Attach structure risk heads (always — core method)
    risk_heads = StructureRiskHeads(in_channels=model._final_ch).to(device)
    model.add_structure_risks(risk_heads)
    ema_risk_heads = StructureRiskHeads(in_channels=model._final_ch).to(device)
    ema_model.add_structure_risks(ema_risk_heads)

    # Consultation predictor: learned gate replaces hardcoded α = clip(R_topo×R_morph)
    # ponytail: XATTN=1 → cross-attention on POOLED decoder features
    # (z_skel+z_endpoint as topo stream, z_in+z_ring+z_diff as morph stream).
    # Else MLP on 8 scalars (ablation baseline).
    if os.environ.get("XATTN", "0") == "1":
        C = model._final_ch
        consult_predictor = CrossAttentionConsultation(
            topo_dim=2 * C, morph_dim=3 * C, d_model=64, n_heads=1, dropout=0.3,
        ).to(device)
    else:
        consult_predictor = ConsultationPredictor(
            in_features=8, hidden=32, dropout=0.3,
        ).to(device)
    model.add_consultation_predictor(consult_predictor)

    # QualitySelector: learned per-pixel pseudo-label weight w(p).
    # WSEL=1 arms it (replaces hand-crafted 1−U in L_sam); default off.
    quality_selector = None
    if os.environ.get("WSEL", "0") == "1":
        from model.quality_selector import QualitySelector
        quality_selector = QualitySelector(
            feat_ch=model._final_ch, d=64, n_heads=1, dropout=0.3,
        ).to(device)
        quality_opt = torch.optim.AdamW(
            quality_selector.parameters(), lr=1e-3, weight_decay=1e-4,
        )

    # --- load pretrained weights ---
    pretrain_path = cfg.training.pretrain_checkpoint or os.path.join(
        snapshot_path, "best_pretrain.pth"
    )
    pretrain_ema_path = cfg.training.pretrain_checkpoint.replace(
        ".pth", "_ema.pth"
    ) if cfg.training.pretrain_checkpoint else os.path.join(
        snapshot_path, "best_pretrain_ema.pth"
    )
    if os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=device))
        logging.info(f"[SSL] Loaded pretrained specialist from {pretrain_path}")
    else:
        logging.warning(
            f"[SSL] Pretrained checkpoint not found: {pretrain_path}. "
            "Starting from scratch."
        )
    if os.path.exists(pretrain_ema_path):
        ema_model.load_state_dict(
            torch.load(pretrain_ema_path, map_location=device)
        )
    else:
        # Copy model weights to EMA as fallback
        ema_model.load_state_dict(model.state_dict(), strict=False)

    # --- generalist (lazy-loaded) ---
    generalist = None

    # --- data ---
    img_size = tuple(cfg.data.img_size)
    train_dataset = CrackDataset(
        image_dir=cfg.data.labeled_image_dir,
        mask_dir=cfg.data.labeled_mask_dir,
        transform=get_train_transform(img_size),
        unlabeled_dir=cfg.data.unlabeled_dir,
    )
    num_labeled = train_dataset.num_labeled
    num_unlabeled = len(train_dataset) - num_labeled
    logging.info(f"[SSL] {num_labeled} labeled, {num_unlabeled} unlabeled")

    labeled_idxs = list(range(min(cfg.data.labeled_num, num_labeled)))
    unlabeled_idxs = list(range(num_labeled, len(train_dataset)))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        cfg.training.batch_size,
        cfg.training.batch_size - cfg.training.labeled_bs,
    )

    def worker_init_fn(worker_id):
        random.seed(cfg.training.seed + worker_id)

    trainloader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
    )

    val_dataset = CrackDataset(
        image_dir=cfg.data.val_image_dir,
        mask_dir=cfg.data.val_mask_dir,
        transform=get_val_transform(img_size),
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --- optimizer (lower LR for SSL stage, UnCoL convention) ---
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.training.ssl_lr, weight_decay=1e-4
    )

    # --- logging ---
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info(f"[SSL] {len(trainloader)} iters/epoch")

    iter_num = 0
    max_epoch = cfg.training.max_iterations // len(trainloader) + 1
    best_dice = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    sub_bs = max(1, cfg.training.labeled_bs // 2)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch["image"].to(device)
            label_batch = sampled_batch.get("label")
            if label_batch is not None:
                label_batch = label_batch.to(device)
            lb = cfg.training.labeled_bs
            B = volume_batch.shape[0]
            n_unlabeled = B - lb

            # --- lazy-load generalist ---
            if generalist is None and n_unlabeled > 0:
                generalist = GeneralistWrapper(
                    checkpoint=cfg.model.generalist_checkpoint, device=device,
                )
                logging.info("[SSL] Generalist (SAM) loaded.")

            # ================================================================
            # 1. Forward on labeled data (supervised)
            # ================================================================
            train_dice = 0.0
            running_train_dice = 0.0

            student_out = model(volume_batch)

            if label_batch is not None and lb > 0:
                # BCE+Dice for supervised loss.
                # Heteroscedastic loss removed — it gates gradients to zero
                # when variance head is untrained.
                loss_bce, loss_dice = bce_dice_loss(
                    student_out["mask_logits"][:lb], label_batch[:lb]
                )
                loss_sup = cfg.topology.bce_weight * loss_bce + cfg.topology.dice_weight * loss_dice

                # Topology calibration: differentiable clDice loss (every iter)
                if (
                    cfg.topology.enabled
                    and cfg.topology.cl_dice_weight > 0
                    and iter_num > cfg.topology.cl_dice_warmup
                    and iter_num % cfg.topology.cl_dice_interval == 0
                ):
                    try:
                        loss_cldice = soft_cl_dice_loss(
                            torch.sigmoid(student_out["mask_logits"][:lb]),
                            label_batch[:lb],
                        )
                        if torch.isfinite(loss_cldice):
                            loss_sup = (loss_sup
                                        + cfg.topology.cl_dice_weight * loss_cldice)
                    except Exception:
                        pass  # ponytail: clDice may fail on degenerate masks

                # Self-supervised risk loss: risk heads predict endpoint density + uncertainty
                if (
                    cfg.topology.enabled
                    and "R_topo" in student_out
                    and iter_num > cfg.topology.cl_dice_warmup
                ):
                    with torch.no_grad():
                        # Topo target: endpoint density from soft skeleton
                        from losses.topology import soft_skel
                        pred_prob = torch.sigmoid(student_out["mask_logits"][:lb])
                        skel = soft_skel(pred_prob)
                        # Endpoint proxy: skeleton pixel with ≤1 skeleton neighbor in 3×3
                        kernel = torch.ones((1, 1, 3, 3), device=device) - torch.eye(3, device=device).view(1, 1, 3, 3)
                        neighbors = torch.nn.functional.conv2d(skel, kernel, padding=1)
                        endpoint_density = (skel * (neighbors <= 1.5).float()).clamp(0, 1)
                        # Morph target: boundary distance (dilated edge of pred mask)
                        pred_bin = (pred_prob > 0.5).float()
                        boundary_target = torch.nn.functional.max_pool2d(
                            pred_bin, kernel_size=5, stride=1, padding=2
                        ) - torch.nn.functional.avg_pool2d(
                            pred_bin, kernel_size=5, stride=1, padding=2
                        )  # 1 near edges, 0 in interior/background

                    risk_topo = student_out["R_topo"][:lb]
                    risk_morph = student_out["R_morph"][:lb]

                    loss_risk = (
                        2.0 * torch.nn.functional.mse_loss(risk_topo, endpoint_density)
                        + 1.0 * torch.nn.functional.mse_loss(risk_morph, boundary_target)
                    )
                    loss_sup = loss_sup + 0.05 * loss_risk

                # WSEL: dense per-pixel supervision of QualitySelector on
                # labeled data. EMA probs = pseudo-label source, GT = answer.
                if quality_selector is not None:
                    with torch.no_grad():
                        ema_lab = torch.sigmoid(
                            ema_model(volume_batch[:lb])["mask_logits"]
                        )
                    w = quality_selector(
                        ema_lab,  # SAM slot ← EMA probs (cold-start proxy:
                                  # on labeled batch we have no SAM query yet)
                        torch.sigmoid(student_out["mask_logits"][:lb]),
                        student_out["features"][:lb],
                    )
                    w_star = (ema_lab > 0.5).float().eq(label_batch[:lb]).float()
                    loss_wsel = F.binary_cross_entropy(w, w_star)
                    if torch.isfinite(loss_wsel):
                        quality_opt.zero_grad()
                        loss_wsel.backward(retain_graph=False)
                        quality_opt.step()

                # Train Dice (torch, no numpy roundtrip)
                with torch.no_grad():
                    pred_bin = (torch.sigmoid(student_out["mask_logits"][:lb]) > 0.5).float()
                    gt_bin = label_batch[:lb]
                    smooth = 1e-5
                    intersect = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
                    p_sum = pred_bin.sum(dim=(1, 2, 3))
                    t_sum = gt_bin.sum(dim=(1, 2, 3))
                    per_img_dice = (2.0 * intersect + smooth) / (p_sum + t_sum + smooth)
                    train_dice = per_img_dice.mean().item()

                # Running mean (EMA with alpha=0.1 for smooth display)
                running_train_dice = (
                    0.9 * running_train_dice + 0.1 * train_dice
                    if running_train_dice > 0
                    else train_dice
                )
            else:
                loss_sup = torch.tensor(0.0, device=device)

            # ================================================================
            # 2. Dual-teacher: compute pseudo-labels + uncertainties
            # ================================================================
            loss_sam = torch.tensor(0.0, device=device)
            loss_sam_boundary = torch.tensor(0.0, device=device)
            loss_sam_interior = torch.tensor(0.0, device=device)
            loss_consult = torch.tensor(0.0, device=device)
            loss_consistency = torch.tensor(0.0, device=device)
            loss_feat_consistency = torch.tensor(0.0, device=device)
            consistency_weight = get_current_consistency_weight(
                iter_num // len(trainloader), cfg.training.consistency, cfg.training.consistency_rampup,
            )

            # ================================================================
            # 1b. Consultation gate supervision on labeled components.
            # Measured target: Δ̂ ← clip(dice_sam − dice_ema) over the
            # component vs GT. Replaces the topo/morph proxy, which was too
            # noisy to train the gate at 5% labels. Reuses the labeled batch
            # already in hand — no extra label budget.
            # ================================================================
            if (
                label_batch is not None and lb > 0 and generalist is not None
                and cfg.topology.enabled and "R_topo" in student_out
                and iter_num % 4 == 0  # ponytail: tiny MLP needs no per-iter SAM supervision
            ):
                with torch.no_grad():
                    ema_lab = torch.sigmoid(
                        ema_model(volume_batch[:lb])["mask_logits"]
                    ).cpu().numpy()  # (lb, 1, H, W)
                for b_idx in range(lb):
                    img_np = volume_batch[b_idx].cpu().numpy().transpose(1, 2, 0)
                    img_np = _denormalize(img_np)
                    ema_prob_lab = ema_lab[b_idx, 0]
                    gt_lab = label_batch[b_idx, 0].cpu().numpy()
                    comps = extract_components(
                        ema_prob_lab, min_area=cfg.prompting.cc_min_area,
                    )
                    for comp in comps[:cfg.topology.max_sam_queries]:
                        x1, y1, x2, y2 = comp.bbox
                        if x2 <= x1 or y2 <= y1:
                            continue
                        comp_mask = comp.mask.astype(bool)
                        if comp_mask.sum() == 0:
                            continue
                        try:
                            sam_comp = generalist.predict(
                                img_np,
                                boxes=[[float(x1), float(y1),
                                        float(x2), float(y2)]],
                            )
                        except Exception:
                            continue
                        # Measured target: SAM Dice − EMA Dice over component
                        sam_bin = (sam_comp > 0.5)
                        ema_bin = (ema_prob_lab > 0.5)
                        gt_c = gt_lab[comp_mask]
                        target = float(np.clip(
                            _comp_dice(sam_bin[comp_mask], gt_c)
                            - _comp_dice(ema_bin[comp_mask], gt_c),
                            -1.0, 1.0,
                        ))
                        if isinstance(model.consultation_predictor,
                                      CrossAttentionConsultation):
                            topo_v, morph_v = _pooled_consult_feats(
                                student_out, comp, ema_prob_lab, b_idx
                            )
                            delta_hat = model.consultation_predictor(
                                torch.from_numpy(topo_v).float()
                                    .to(device).unsqueeze(0),
                                torch.from_numpy(morph_v).float()
                                    .to(device).unsqueeze(0),
                            )
                        else:
                            risk_scores = _pool_risk_maps(
                                student_out, comp_mask, b_idx
                            )
                            feats = _extract_consult_features(
                                comp,
                                risk_scores.get("R_topo", 0.0),
                                risk_scores.get("R_morph", 0.0),
                            )
                            feats_t = torch.from_numpy(feats).float() \
                                .to(device).unsqueeze(0)
                            delta_hat = model.consultation_predictor(feats_t)
                        target_t = torch.tensor(
                            [target], dtype=torch.float32, device=device
                        )
                        loss_consult = loss_consult + F.mse_loss(
                            delta_hat.squeeze(0), target_t
                        )

            if n_unlabeled > 0 and (
                cfg.training.beta_sam > 0 or cfg.training.dual_teacher_enabled
            ):
                # --- EMA predictions on unlabeled images ---
                with torch.no_grad():
                    noise = torch.clamp(
                        torch.randn_like(volume_batch[lb:]) * 0.1, -0.2, 0.2
                    )
                    ema_in = volume_batch[lb:] + noise
                    ema_out = ema_model(ema_in)
                    p_ema = torch.sigmoid(ema_out["mask_logits"])  # (U, 1, H, W)
                    u_ema = entropy_from_logits(ema_out["mask_logits"])  # (U, 1, H, W)

                # --- Mean Teacher consistency (student vs EMA) ---
                student_unlab_probs = torch.sigmoid(
                    student_out["mask_logits"][lb:]
                )
                loss_consistency = mse_consistency_loss(
                    student_unlab_probs, p_ema
                )

                # --- Feature-level consistency (SCRWKV only) ---
                # Harmonic MSE (representation stability) + scale-attention KL
                # (routing stability). Rides the SAME sigmoid ramp as mask
                # consistency: early student-vs-EMA attention gap is noise
                # (EMA lags), and untrained attention is ~uniform (KL~0) —
                # withhold the gradient until the teacher is worth matching.
                if (
                    cfg.topology.feature_consistency > 0
                    and "scale_attention" in student_out
                ):
                    ramp = consistency_weight / max(cfg.training.consistency, 1e-8)
                    mse_f = F.mse_loss(
                        student_out["harmonic_features"][lb:],
                        ema_out["harmonic_features"],
                    )
                    sa_s = student_out["scale_attention"][lb:]
                    sa_t = ema_out["scale_attention"]
                    kl_f = (
                        sa_t * (sa_t.add(1e-8).log() - sa_s.add(1e-8).log())
                    ).sum(dim=1).mean()
                    loss_feat_consistency = (
                        cfg.topology.feature_consistency * ramp * (mse_f + kl_f)
                    )

                # --- SAM foundation teacher: component-level selective query ---
                p_sam_list, u_sam_list, u_geo_list = [], [], []
                n_queried = 0

                for b_idx in range(lb, B):
                    img_np = volume_batch[b_idx].cpu().numpy().transpose(1, 2, 0)
                    img_np = _denormalize(img_np)
                    ema_prob = p_ema[b_idx - lb, 0].cpu().numpy()

                    # Extract components from EMA prediction
                    components = extract_components(
                        ema_prob, min_area=cfg.prompting.cc_min_area,
                    )

                    if len(components) == 0:
                        p_sam_list.append(p_ema[b_idx - lb].unsqueeze(0).cpu().numpy())
                        u_sam_list.append(u_ema[b_idx - lb].unsqueeze(0).cpu().numpy())
                        u_geo_list.append(np.zeros((1, 1, *img_size)))
                        continue

                    # Start with EMA as base pseudo-label; overlay accepted SAM corrections
                    sam_corrected = ema_prob.copy()
                    u_geo_map = np.zeros(img_size, dtype=np.float32)

                    for comp in components[:cfg.topology.max_sam_queries]:
                        x1, y1, x2, y2 = comp.bbox
                        if x2 <= x1 or y2 <= y1:
                            continue
                        box = [[float(x1), float(y1), float(x2), float(y2)]]

                        # Query SAM for this component
                        try:
                            sam_comp = generalist.predict(img_np, boxes=box)
                        except Exception:
                            continue

                        # Acceptance gate: box-jitter U_geo inside component
                        jittered = jitter_boxes(
                            box, n=cfg.prompting.num_prompt_perturbations,
                            scale_jitter=cfg.prompting.box_jitter_scale,
                            translate_jitter=cfg.prompting.box_jitter_translate,
                            img_size=cfg.data.img_size,
                        )
                        try:
                            _, u_geo_comp = generalist.predict_unc(img_np, jittered)
                        except Exception:
                            u_geo_comp = np.zeros(img_size, dtype=np.float32)

                        comp_mask = comp.mask.astype(bool)
                        if comp_mask.sum() == 0:
                            continue
                        mean_u_geo = u_geo_comp[comp_mask].mean()

                        # Acceptance gate: accept if SAM is consistent (low U_geo)
                        if mean_u_geo < cfg.topology.acceptance_u_geo:
                            if isinstance(model.consultation_predictor,
                                          CrossAttentionConsultation):
                                topo_v, morph_v = _pooled_consult_feats(
                                    student_out, comp, ema_prob, b_idx
                                )
                                delta_hat = model.consultation_predictor(
                                    torch.from_numpy(topo_v).float()
                                        .to(device).unsqueeze(0),
                                    torch.from_numpy(morph_v).float()
                                        .to(device).unsqueeze(0),
                                )
                            else:
                                # MLP: learned alpha from 8 component scalars
                                risk_scores = _pool_risk_maps(
                                    student_out, comp_mask, b_idx
                                )
                                feats = _extract_consult_features(
                                    comp,
                                    risk_scores.get("R_topo", 0.0),
                                    risk_scores.get("R_morph", 0.0),
                                )
                                feats_t = torch.from_numpy(feats).float() \
                                    .to(device).unsqueeze(0)
                                delta_hat = model.consultation_predictor(feats_t)
                            alpha = torch.sigmoid(2.0 * delta_hat).detach().cpu().item()
                            sam_corrected[comp_mask] = (
                                alpha * sam_comp[comp_mask].astype(np.float32)
                                + (1.0 - alpha) * ema_prob[comp_mask]
                            )
                            # Gate is supervised on labeled components (block 1b);
                            # here on unlabeled we only apply α for blending.
                            n_queried += 1

                        # Accumulate U_geo for this image
                        u_geo_map = np.maximum(u_geo_map, u_geo_comp)

                    p_sam_list.append(sam_corrected[np.newaxis, np.newaxis, ...])
                    sam_corrected_t = torch.from_numpy(sam_corrected).unsqueeze(0).unsqueeze(0).to(device)
                    u_sam_ent = entropy(sam_corrected_t.clamp(1e-6, 1.0 - 1e-6))
                    u_sam_list.append(u_sam_ent.cpu().numpy())
                    u_geo_list.append(u_geo_map[np.newaxis, np.newaxis, ...])

                # Stack results
                p_sam = torch.from_numpy(np.concatenate(p_sam_list, axis=0)).to(device)
                u_sam = torch.from_numpy(np.concatenate(u_sam_list, axis=0)).to(device)

                # Log selective query stats
                writer.add_scalar("ssl/n_sam_queries", n_queried, iter_num)

                # ============================================================
                # 3. Uncertainty-weighted dual-teacher fusion
                # ============================================================
                if cfg.training.dual_teacher_enabled:
                    # Ramping uncertainty threshold: 0.75 → 1.0 over training
                    threshold = 0.75 + 0.25 * sigmoid_rampup(
                        iter_num, cfg.training.max_iterations
                    )

                    u_ema_fusion = u_ema  # ponytail: scale-equivariance removed

                    fused_probs, trust_mask = fuse_pseudo_labels_binary(
                        p_sam, p_ema, u_sam, u_ema_fusion,
                        uncertainty_threshold=threshold,
                    )
                else:
                    fused_probs = p_sam
                    trust_mask = torch.ones_like(p_sam)
                    threshold = 1.0

                # ============================================================
                # 4. Boundary-band-split confidence-aware loss (Mod 2)
                # ============================================================
                # WSEL: learned per-pixel quality replaces trust_mask
                if quality_selector is not None:
                    with torch.no_grad():
                        w_map = quality_selector(
                            p_sam, p_ema, student_out["features"][lb:],
                        )
                    trust_mask = w_map

                for b_idx in range(lb, B):
                    pred_prob = torch.sigmoid(
                        student_out["mask_logits"][b_idx]
                    ).unsqueeze(0)

                    boundary_mask = boundary_band_mask(
                        pred_prob, cfg.uncertainty.boundary_band_width_px
                    )

                    u_geo_b = torch.from_numpy(
                        u_geo_list[b_idx - lb]
                    ).float().to(device)
                    u_fused = u_geo_b  # ponytail: UPFM/variance/fusion removed — U_geo drives confidence-aware loss

                    sam_pseudo = fused_probs[
                        b_idx - lb : b_idx - lb + 1
                    ]

                    loss_dict = confidence_aware_loss(
                        student_pred=pred_prob,
                        sam_pseudo=sam_pseudo,
                        u_geo=u_fused,   # already fused via learnable conv
                        u_app=None,       # not needed — already in u_fused
                        boundary_mask=boundary_mask,
                        w_boundary=cfg.uncertainty.w_boundary,
                        w_interior=cfg.uncertainty.w_interior,
                        agreement_weight=trust_mask[
                            b_idx - lb : b_idx - lb + 1
                        ],
                    )
                    loss_sam += loss_dict["total"] / n_unlabeled
                    loss_sam_boundary += loss_dict["boundary"] / n_unlabeled
                    loss_sam_interior += loss_dict["interior"] / n_unlabeled

                # ponytail: BCP CutMix removed (ineffective; was cfg.training.bcp_enabled)
            # ================================================================
            # 6. Total loss
            # ================================================================
            loss_sam_weighted = cfg.training.beta_sam * loss_sam

            loss = (
                loss_sup
                + consistency_weight * loss_consistency
                + loss_sam_weighted
                + 0.1 * loss_consult  # ponytail: consultation MLP training
                + loss_feat_consistency
            )

            # NaN guard: halt — unstable component needs fixing, not skipping
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[SSL] iter {iter_num}: NaN/Inf loss. "
                    f"sup={loss_sup.item():.3f} cons={loss_consistency.item():.3f} "
                    f"sam={loss_sam.item():.3f} consult={loss_consult.item():.3f}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            update_ema_variables(
                model, ema_model, cfg.training.ema_decay, iter_num
            )

            # --- LR schedule ---
            lr_ = cfg.training.ssl_lr * (
                1.0 - iter_num / cfg.training.max_iterations
            ) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            # --- logging ---
            writer.add_scalar("ssl/lr", lr_, iter_num)
            writer.add_scalar("ssl/total_loss", loss.item(), iter_num)
            writer.add_scalar("ssl/loss_sup", loss_sup.item(), iter_num)
            writer.add_scalar(
                "ssl/loss_consistency", loss_consistency.item(), iter_num
            )
            writer.add_scalar("ssl/loss_sam", loss_sam.item(), iter_num)
            writer.add_scalar(
                "ssl/loss_sam_boundary", loss_sam_boundary.item(), iter_num
            )
            writer.add_scalar(
                "ssl/loss_sam_interior", loss_sam_interior.item(), iter_num
            )
            writer.add_scalar(
                "ssl/consistency_weight", consistency_weight, iter_num
            )
            writer.add_scalar("ssl/loss_feat_consistency", loss_feat_consistency.item(), iter_num)
            writer.add_scalar("ssl/train_dice", train_dice, iter_num)
            writer.add_scalar(
                "ssl/train_dice_running", running_train_dice, iter_num
            )

            # Update tqdm postfix with live metrics
            iterator.set_postfix({
                "loss": f"{loss.item():.3f}",
                "td": f"{running_train_dice:.3f}" if running_train_dice > 0 else "-",
            })

            if iter_num % 20 == 0:
                img_grid = make_grid(
                    volume_batch[0, :3].unsqueeze(0), normalize=True
                )
                writer.add_image("ssl/Image", img_grid, iter_num)
                pred_grid = make_grid(
                    torch.sigmoid(student_out["mask_logits"][0]), normalize=False
                )
                writer.add_image("ssl/Prediction", pred_grid, iter_num)
                if label_batch is not None and label_batch.shape[0] > 0:
                    gt_grid = make_grid(label_batch[0], normalize=False)
                    writer.add_image("ssl/GroundTruth", gt_grid, iter_num)

            # --- validation ---
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                val_dice = validate(model, valloader, device)
                writer.add_scalar("ssl/val_dice", val_dice, iter_num)
                logging.info(
                    f"[SSL] iter {iter_num} : "
                    f"train_dice : {running_train_dice:.4f}  "
                    f"val_dice : {val_dice:.4f}"
                )
                if val_dice > best_dice:
                    best_dice = val_dice
                    save_path = os.path.join(
                        snapshot_path,
                        f"ssl_iter_{iter_num}_dice_{best_dice:.4f}.pth",
                    )
                    best_path = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(model.state_dict(), save_path)
                    torch.save(model.state_dict(), best_path)
                    logging.info(f"Saved best: dice={best_dice:.4f}")
                model.train()

            if iter_num % 3000 == 0:
                save_path = os.path.join(
                    snapshot_path, f"ssl_iter_{iter_num}.pth"
                )
                torch.save(model.state_dict(), save_path)

            if iter_num >= cfg.training.max_iterations:
                break

        if iter_num >= cfg.training.max_iterations:
            iterator.close()
            break

    writer.close()
    return "SSL Training Finished!"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemiSAM-Crack Training (UnCoL)")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--stage", type=str, default="ssl",
        choices=["pretrain", "ssl"],
        help="Training stage: pretrain (foundation KD) or ssl (dual-teacher)",
    )
    parser.add_argument(
        "--snapshot", type=str, default=None, help="Override snapshot directory"
    )
    args_cli = parser.parse_args()

    cfg = Config.from_yaml(args_cli.config)

    # Deterministic
    if cfg.training.seed > 0:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        torch.cuda.manual_seed(cfg.training.seed)

    # Snapshot dir
    stage_name = args_cli.stage.upper()
    snapshot_path = (
        args_cli.snapshot
        or f"snapshots/semisam_crack_{cfg.prompting.prompt_type}_{stage_name}"
    )
    os.makedirs(snapshot_path, exist_ok=True)

    # Logging
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg))
    logging.info(f"Stage: {stage_name}")

    if args_cli.stage == "pretrain":
        pretrain(cfg, snapshot_path)
    else:
        ssl_train(cfg, snapshot_path)
