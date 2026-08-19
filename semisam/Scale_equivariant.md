Coding Agent Implementation Guide: SemiSAM-Crack Scale-Equivariance

Context & Objective

The goal is to modify the current SemiSAM+ UnCoL baseline to handle civil infrastructure distractors (joints, shadows) and faint crack topologies. We are completely abandoning geometric regularity in favor of Scale-Equivariant Uncertainty ($U_{scale}$) (zoom consistency) and Hysteresis Thresholding.

You will implement three core modifications.

Modification 1: Multi-Scale Prediction & Spatial Alignment Utilities

Objective: Enable the model to generate predictions at different "zoom" levels (e.g., 1x, 2x crop, 0.5x pad) and perfectly align those predictions back to the original 1x spatial coordinate system.

Files to modify/create:

data/transforms.py (Add alignment utilities)

model/specialist.py (Add multi-scale forward pass helper)

Implementation Steps:

In data/transforms.py:

Create a function generate_scale_variants(image, scales=[0.5, 1.0, 2.0]).

For scale=2.0: Perform a center crop of size (H/2, W/2), then interpolate (resize) back to (H, W). Keep track of the crop bounding box.

For scale=0.5: Pad the image with reflection or mean pixel value to (2H, 2W), then resize to (H, W).

In data/transforms.py:

Create a function align_predictions(preds_dict, original_shape).

For the 2x prediction: downsample back to (H/2, W/2) and paste it into the center of a zero-initialized (H, W) tensor. (The edges will be 0, which is fine, uncertainty will just be calculated where data exists).

For the 0.5x prediction: resize to (2H, 2W) and center-crop the (H, W) region.

In model/specialist.py:

Add a method forward_multiscale(x) that utilizes the transforms to pass the variants through the network and returns the aligned probability maps.

Modification 2: $U_{scale}$ Calculation and UnCoL Integration

Objective: Calculate the variance across the aligned scale predictions and integrate this into the dual-teacher pseudo-label fusion.

Files to modify:

uncertainty/scale.py (New file)

training/dual_teacher.py

Implementation Steps:

Create uncertainty/scale.py:

Implement compute_u_scale(aligned_preds_list).

Formula: U_scale = torch.var(torch.stack(aligned_preds_list), dim=0) (Pixel-wise variance across the scale dimension).

Normalize U_scale to a [0, 1] range using min-max normalization per batch.

In training/dual_teacher.py:

Locate the EMA Teacher (Specialist) prediction step on unlabeled data.

Change the forward pass to use forward_multiscale(x_unlabeled).

Obtain U_scale using the function from step 1.

Update the UnCoL fusion weights. Previously, w_ema = exp(-u_ema) (where u_ema was entropy).

Change to: w_ema = exp(-(u_ema + alpha * U_scale)) where alpha is a hyperparameter (default to 1.0).

Recalculate the fusion equation p_fused = (w_sam * p_sam + w_ema * p_ema) / (w_sam + w_ema).

Modification 3: Hard Negative Mining & Hysteresis Thresholding

Objective: Actively penalize false positives identified by the zoom logic, and connect faint cracks using dual thresholds.

Files to modify:

training/dual_teacher.py

losses/confidence_aware.py (or where the UnCoL SSL loss is computed)

Implementation Steps:

Hysteresis Thresholding for Pseudo-labels (in dual_teacher.py):

Instead of a single threshold (e.g., p_fused > 0.5) to generate the hard pseudo-mask Y_u, implement a Canny-like hysteresis.

Set T_high = 0.7, T_low = 0.3.

Identify strong seeds: seeds = p_fused > T_high.

Identify weak candidates: weak = p_fused > T_low.

Use connected component analysis (e.g., scipy.ndimage.label or a morph-reconstruction equivalent in PyTorch) to keep only the weak pixels that are connected to a seed pixel.

This resulting mask becomes the new target pseudo-label Y_u.

Hard Negative Mining (in SSL Loss formulation):

Identify "confident distractors": pixels where p_1x > 0.8 (model thinks it's a crack at normal scale) BUT U_scale > 0.5 (high variance because 2x zoom rejected it).

Create a binary mask: hard_negative_mask = (p_1x > 0.8) & (U_scale > 0.5).

In the consistency loss calculation for the Student model, explicitly set the pseudo-label for these pixels to 0 (background).

Multiply the loss weight for these specific pixels by a multiplier (e.g., lambda_hard = 2.0) to force the student network to learn to suppress these specific distractors.