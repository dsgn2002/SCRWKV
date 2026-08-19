

The pipeline should model three complementary risks:

- **Appearance risk:** crack versus distractor confusion.
- **Morphology risk:** implausible width, boundary, elongation, or orientation.
- **Topology risk:** breaks, missing branches, fragmentation, and false connections.

However, these should not become three loosely combined “uncertainty tricks.” They should feed a single, interpretable **consultation-value predictor**.

# Revised Concise Coding Plan

## 1. Add Structure-Aware Specialist Outputs

Modify `model/specialist.py`:

```text
decoder features
 ├── mask head
 ├── centerline head
 └── boundary head
```

Train with:

\[
L_{\mathrm{sup}}
=
L_{\mathrm{BCE+Dice}}
+\lambda_tL_{\mathrm{clDice}}
+\lambda_cL_{\mathrm{centerline}}
+\lambda_bL_{\mathrm{boundary}}.
\]

These heads provide the representations needed to estimate topology and morphology risks.

---

## 2. Extract Component-Level Candidates

Add:

```text
prompting/components.py
prompting/structure_regions.py
```

From the EMA prediction:

1. extract connected components;
2. skeletonize each component;
3. identify endpoints, branches, gaps, and false-bridge candidates;
4. produce a contextual crop for each candidate;
5. retain local decoder/image features for risk prediction.

Remove the current merged outer bounding box.

---

## 3. Estimate Three Risks

Add:

```text
model/structure_risk.py
training/risk_targets.py
```

### Topology risk

Use:

- fragmentation;
- endpoint and branch inconsistency;
- plausible gaps;
- false connections;
- student–EMA skeleton disagreement.

Supervised target:

\[
e_{\mathrm{topo}}
=
\alpha(1-\mathrm{clDice})
+\beta E_{\mathrm{component}}
+\gamma E_{\mathrm{endpoint}}
+\delta E_{\mathrm{bridge}}.
\]

### Morphology risk

Use:

- boundary uncertainty;
- abrupt width variation;
- poor elongation;
- orientation discontinuity;
- unstable component shape;
- mask–centerline inconsistency.

Supervised target:

\[
e_{\mathrm{morph}}
=
aE_{\mathrm{boundary}}
+bE_{\mathrm{width}}
+cE_{\mathrm{orientation}}
+dE_{\mathrm{shape}}.
\]

### Appearance risk

Use learned decoder features and image context, together with:

- student entropy;
- student–EMA disagreement;
- foreground probability;
- local texture and color features;
- component confidence.

Supervised target:

\[
e_{\mathrm{app}}
=
E_{\mathrm{FP}}+E_{\mathrm{FN}},
\]

or train the head to predict whether the component is a true crack, distractor, or missed-crack region.

Do not define appearance risk only with hand-crafted assumptions such as “cracks are dark.”

---

## 4. Predict Consultation Value

Instead of manually averaging the three risks, train a small head to predict whether TopoSAM will improve the region:

\[
\widehat{\Delta}_j
=
Q_\psi
\left(
R_{\mathrm{topo},j},
R_{\mathrm{morph},j},
R_{\mathrm{app},j},
z_j
\right),
\]

where \(z_j\) contains component and contextual features.

On labeled data, define the target as:

\[
\Delta_j
=
E(P_{\mathrm{EMA},j},Y_j)
-
E(P_{\mathrm{TopoSAM},j},Y_j),
\]

with a composite error:

\[
E
=
w_rE_{\mathrm{region}}
+w_tE_{\mathrm{topo}}
+w_mE_{\mathrm{morph}}.
\]

Thus, the system learns that:

- high topology risk may benefit from TopoSAM;
- appearance confusion may or may not benefit;
- a vegetation component should not be queried merely because it is fragmented;
- a well-segmented component should not be queried even if it is complex.

Rank candidates using:

\[
q_j=\frac{\max(0,\widehat{\Delta}_j)}{c_j},
\]

where \(c_j\) is the estimated consultation cost.

---

## 5. Retain Separate Query and Acceptance Decisions

### Query decision

```text
Predicted TopoSAM benefit is high → query TopoSAM
```

### Acceptance decision

After querying, evaluate:

- prompt-perturbation stability;
- crop-margin stability;
- EMA agreement;
- image-boundary alignment;
- topology improvement;
- morphological plausibility;
- appearance consistency.

```text
TopoSAM correction is reliable → accept locally
```

This is crucial: appearance, morphology, and topology risks indicate a potential problem, but they do not prove that TopoSAM’s answer is correct.

---

## 6. Apply Error-Type-Specific Corrections

The correction policy should depend on the predicted risk:

| Dominant risk | TopoSAM action |
|---|---|
| Topology | Repair gaps, branches, fragmentation, or false bridges |
| Morphology | Refine width and boundaries |
| Appearance | Confirm or reject the component as a crack |
| Mixed | Apply correction only where teacher stability is high |

Do not replace the complete pseudo-label. Insert accepted corrections only within the queried crop and ignore unresolved pixels.

---

## 7. Required Evaluation

Compare consultation strategies:

1. Random querying.
2. Entropy querying.
3. Topology risk only.
4. Morphology risk only.
5. Appearance risk only.
6. Topology + morphology.
7. All three risks with fixed averaging.
8. Learned consultation-value prediction.
9. Learned consultation value + acceptance gate.

Report:

- Dice and IoU;
- clDice and fragmentation;
- boundary F1;
- skeleton precision/recall;
- false-positive component count;
- distractor rejection accuracy;
- consultation-value AUROC/AUPRC;
- Dice/clDice versus TopoSAM calls;
- training time and queried crop area.

# Updated Execution Order

```text
1. Add mask, centerline, and boundary heads
2. Add region, topology, morphology, and boundary metrics
3. Extract component/gap candidates and local crops
4. Implement topology-, morphology-, and appearance-risk targets
5. Implement the three risk heads
6. Integrate TopoSAM at the component level
7. Train the expected-consultation-benefit head
8. Add the independent acceptance gate
9. Apply accepted local corrections
10. Evaluate calibration, segmentation quality, and query efficiency
```

The polished pipeline is therefore:

```text
EMA prediction
→ component and gap extraction
→ topology + morphology + appearance risk estimation
→ predicted TopoSAM benefit per unit cost
→ selective local TopoSAM query
→ correction reliability assessment
→ accepted local pseudo-label correction
→ specialist training
```

**Topology should remain the central scientific emphasis**, because connectivity is distinctive to cracks. Morphology and appearance should be presented as necessary supporting signals that prevent the topology-based query mechanism from consulting TopoSAM on structurally complex distractors.

---

# Historical Modifications

## Mod 1 — Box Prompts (replaces SemiSAM+ point/mask prompting)
- `prompting/boxes.py`: `boxes_from_mask()` — CC → bboxes with margin + merge
- `jitter_boxes()`: n perturbed variants (scale ±10%, translate ±15px) for U_geo
- SAM-1 handles 1 box → all boxes merged into outer bbox

## Mod 2 — UPFM: Uncertainty-Prompted Feature Modulator
- **Variance head**: Conv2d(32,1) parallel to mask head, predicts per-pixel `log_sigma2`
- **UPFM**: pixel-wise affine `γ(U_app)*F + δ(U_app)` on penultimate features
- **Critical finding**: `heteroscedastic_loss` kills training — `exp(-log_s²)` scales mask gradient to ~4.5e-5 when variance head is untrained. Fix: BCE+Dice for supervised loss; variance head learns indirectly through UPFM → mask_head gradient.

## Mod 3 — UnCoL Dual-Teacher Fusion
- Foundation Teacher (frozen SAM) + EMA Teacher → uncertainty-weighted pseudo-label fusion
- Trust mask ramps 0.75→1.0 over training

## Mod 4 — BCP CutMix Augmentation
- Bidirectional Copy-Paste: random rectangular mask mixes labeled ↔ unlabeled
- **Finding**: BCP essential at 10-label scale. Removing collapses Dice 0.64→0.23. Disabled in current best config — thin cracks disrupted by CutMix noise at 49-label scale.

## Mod 5 — Scale-Equivariant U_scale
- Multi-scale EMA forward (0.5×, 1×, 2× zoom)
- `U_scale = Var(p_0.5x, p_1x, p_2x)` — pixel-wise variance across scales
- AUROC=0.575 on TUT val — does NOT discriminate cracks from distractors. U_scale measures teacher disagreement, not crack-ness.

## T4 Architectural Refactor
1. **Gradient Detachment**: `.detach()` on features before variance head
2. **Inverted Gating**: `w ∝ U_combined` instead of `1-U`
3. **Learnable Fusion**: Conv2d(3→1)+Sigmoid replaces `avg(u_geo, u_app, u_scale)`

---

# Full Experiment History

See `experiments.csv` for the canonical table.

### Early 10-label experiments (custom dataset)
| # | Val Dice | Key config |
|---|----------|------------|
| E1 (SemiSAM+ baseline) | 0.630 | MT+SAM box prompts, 30K SGD |
| E3 (+UPFM) | 0.641 | UPFM+dual-teacher+BCP |
| E4 (no BCP) | 0.232 | BCP removal collapsed training |
| E5 (+U_scale) | 0.643 | Matched E3, 2× training cost |

### TUT 5% (49-label) experiments
| # | Val Dice | Key config |
|---|----------|------------|
| T1 (baseline) | 0.747 | SemiSAM+ on TUT, 10K iters |
| T2 (UPFM) | 0.753 | Best overall, MT+SAM+UPFM |
| T4 (Refactor) | 0.749 | Detach+invert+fuse, test 0.796 |
| P4 (topo_2k) | 0.711 | centerline/boundary heads, 2K iters, test 0.756 |
| **P5 (toposam_2k)** | **0.684** | Simplified loss, no aux heads, 2K iters, test 0.726 |

### Failed experiments
| # | Best Dice | Root cause |
|---|-----------|------------|
| F1 | 0.19 | BCP CutMix destructive at tiny data scale |
| F2 | 0.16 | SSL lr=1e-5 1000× too low for from-scratch |
| F3 | 0.11 | Heteroscedastic loss gates gradient to zero |
| F4 | 0.63→0.23 | BCP noise + EMA drift at 30K iters |

### Bug fixes
- Zero-area jittered boxes → NaN → min 1px clamp
- `torch.logit` roundtrip NaN → direct `entropy(p.clamp(1e-6, 1-1e-6))`
- Multi-box SAM crash → merge into single outer bbox
- EMA missing UPFM keys → attach variance head + UPFM to both model and ema_model
- Float32 `1.0 - 1e-8 == 1.0` → use eps=1e-7