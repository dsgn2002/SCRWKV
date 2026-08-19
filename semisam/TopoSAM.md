# TopoSAM.md

## Stage 1: Loss Function Simplification
*   Modify `model/specialist.py` to remove the `centerline` and `boundary` heads entirely[cite: 4].
*   Update `losses/supervised.py` to implement the simplified supervised loss equation: $$\mathcal L_{\text{sup}} = \mathcal L_{\text{BCE}} + \mathcal L_{\text{Dice}} + \lambda_{\text{cl}}\mathcal L_{\text{soft-clDice}}$$[cite: 4].
*   Ensure that `soft_cl_dice_loss` computes directly from `mask_probability` and `gt_mask`, removing any reliance on a separate centerline prediction output[cite: 4].
*   Update `config.yaml` to set the new loss weights: `bce_weight: 1.0`, `dice_weight: 1.0`, `cldice_weight: 0.1`, and change `cldice_interval` to `1`[cite: 4].
*   Remove centerline BCE and boundary BCE from the codebase[cite: 4].

## Stage 2: Topology Information Extraction
*   Modify `prompting/components.py` to calculate explicit structural descriptors from the EMA mask for each candidate region $C_j$[cite: 4].
*   Extract the topological vector $t_j$ containing: endpoint count, branch-point count, number of fragments, skeleton length, candidate gap count, gap distance, orientation compatibility between gap endpoints, student–EMA skeleton disagreement, component connectivity under probability thresholds, and component stability under weak perturbations[cite: 4].
*   Implement topology-conditioned visual feature extraction by using the automatically extracted skeleton and endpoint neighborhoods to pool multiscale encoder features $F_l$[cite: 4].
*   Apply mask pooling for the skeleton path $S_j$: $$z_j^{\mathrm{skel},l} = \operatorname{MaskPool}(F_l,S_j)$$[cite: 4].
*   Apply mask pooling for the endpoint neighborhoods $N(E_j)$: $$z_j^{\mathrm{end},l} = \operatorname{MaskPool}(F_l,N(E_j))$$[cite: 4].

## Stage 3: Appearance Information Extraction
*   Modify the feature extraction modules to pool encoder features $F_l$ from three distinct spatial areas for each candidate[cite: 4].
*   Pool features inside the candidate region $C_j$: $$z_j^{\mathrm{in},l} = \operatorname{MaskPool}(F_l,C_j)$$[cite: 4].
*   Pool features from the surrounding background ring: $$z_j^{\mathrm{ring},l} = \operatorname{MaskPool}(F_l,\operatorname{Dilate}(C_j)\setminus C_j)$$[cite: 4].
*   Compute the contrast difference between the candidate and its context: $$z_j^{\mathrm{diff},l} = z_j^{\mathrm{in},l} - z_j^{\mathrm{ring},l}$$[cite: 4].
*   Concatenate these pooled features with the following appearance scalars: foreground probability, entropy, student–EMA disagreement, temporal stability, local image-gradient statistics, and multiscale feature differences[cite: 4].