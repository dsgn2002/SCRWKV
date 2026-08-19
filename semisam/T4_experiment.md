T4 Architecture: Multi-Scenario Experimental Protocol

This protocol is refined to address the TUT dataset's extreme heterogeneity (8 distinct scenarios). The objective is to prove that T4's UQ-driven fusion allows a single model to generalize across vastly different physical textures (from metal to bitumen) using only 5% of the data.

1. Core Ablation Series (Scenario Robustness)

Goal: Prove that T4's learnable components are the "bridge" between these 8 diverse domains.


EXP-T4-BASE Baseline (Current)

Detach(), Inverted Gating, Conv2d Fusion

Standard Gating, Mean Fusion

Proves 0.796 performance across all 8 TUT scenarios.

EXP-T4-V1 Gradient Stability

Inverted Gating, Conv2d Fusion

Detach()

Backpropagation of UQ error on high-reflectivity surfaces (metal/tiles) destabilizes the encoder.

EXP-T4-V2 Inverted Gating

Detach(), Conv2d Fusion

Inverted Gating

Ignoring uncertain pixels on complex topologies (generator blades) leads to significant Dice drop.

EXP-T4-V3 Learnable Fusion

Detach(), Inverted Gating

Conv2d Fusion

Hardcoded weighting fails to balance the vastly different noise profiles of metal vs. bitumen.

2. Multi-Scenario Performance Breakdown

Goal: Quantify the "Domain Gap" within the dataset itself.

Scenario ID

Material Type

Test Objective

Expected UQ Behavior

S1-BIT

Bitumen

Standard asphalt texture

High $U_{app}$ due to aggregate noise.

S2-CEM

Cement

Smooth structural surface

High $U_{geo}$ on fine-line cracks.

S3-BRK

Bricks

Periodic grid distractor

$U_{scale}$ must resolve joint vs. crack.

S4-PLA

Plastic Runways

Low-contrast background

High $U_{app}$ weighting required.

S5-TIL

Tiles

Glossy, high-reflection

$U_{photo}$ (if implemented) or $U_{app}$ stability.

S6-MET

Metal Materials

Specular noise, machined edges

$U_{scale}$ to reject mechanical scratches.

S7-GEN

Generator Blades

Complex curved geometry

$U_{geo}$ dominance.

S8-PIP

Pipelines

Low-light, internal corrosion

Composite $U$ for signal-to-noise.

3. Visualization Tasks (Scientific Evidence)

Fusion Weight Matrix:

Action: Extract weights from LearnableUncertaintyFusion across all 8 scenarios.

Result: A heatmap showing how the model shifts priority between Appearance, Geometry, and Scale depending on the material.

Uncertainty Topology Maps:

Action: Generate heatmaps for "Bricks" (Scenario 3) and "Metal" (Scenario 6).

Target: High uncertainty specifically on mortar lines and machined grooves, proving "Distractor Rejection."

4. Coding Agent Implementation Instructions

To execute this 8-scenario protocol, the coding agent must implement the following specific codebase modifications:

A. Dataloader & Metric Tracking Updates (data/dataset.py & evaluate.py)

Currently, the validation loop calculates a single global Mean Dice. To prove our hypothesis, we need granular tracking.

Task: Modify the CrackDataset class to parse the parent directory names of the TUT dataset (which indicate the 8 material classes).

Task: Update evaluate.py to maintain a dictionary of metrics grouped by material class: metrics = {'Bitumen': [], 'Cement': [], ...}.

Target Output: At the end of validation, log the Mean Dice per scenario alongside the global Mean Dice.

B. Extracting Learnable Fusion Maps (uncertainty/fusion.py & eval_vis.py)

To prove how the model adapts to different materials, we need to visualize the dynamic weights assigned to $U_{geo}$, $U_{app}$, and $U_{scale}$.

Task: In uncertainty/fusion.py, expose the output activations of the Conv2d(3->1) layer (before the final Sigmoid).

Task: In eval_vis.py, write a hook to capture these 3-channel weight maps during inference.

Target Output: Save a combined Matplotlib figure showing the original image, ground truth mask, predicted mask, and a 3-channel heatmap representing the fusion weighting (e.g., Red=$U_{geo}$, Green=$U_{app}$, Blue=$U_{scale}$).

C. Ablation Run Configuration (config.yaml & train.py)

Task: Implement boolean flags in config.yaml for the T4 components: use_detach: bool, use_inverted_gating: bool, use_learnable_fusion: bool.

Task: Ensure model/specialist.py correctly routes the forward pass based on these configuration flags to facilitate the EXP-T4-V1 through V3 automated runs.