# Literature Review: Semi-Supervised Foundation Model Segmentation for Infrastructure Defects

**Date:** 2026-07-20
**Project:** SemiSAM-Crack — Adapting SemiSAM+ to 2D RGB Infrastructure Defect Segmentation
**Tool:** scite (scite.ai)

---

## 1. Core Baseline: SemiSAM+

**Zhang et al. (2025).** *SemiSAM+: Rethinking semi-supervised medical image segmentation in the era of foundation models.* Medical Image Analysis, 103733.
https://doi.org/10.1016/j.media.2025.103733

SemiSAM+ establishes the specialist-generalist collaborative learning paradigm that this project adapts:

- **Specialist**: trainable 3D U-Net trained with standard semi-supervised loss (supervised + consistency regularization via Mean Teacher/UA-MT/DTC/DAN).
- **Generalist**: frozen SAM/SAM-Med3D/SegAnyPET used as pseudo-label generator, never fine-tuned.
- **Specialist → Generalist**: specialist's coarse prediction converted to point or mask prompts fed to SAM.
- **Generalist → Specialist**: aleatoric uncertainty `U_x` from n-way prompt perturbation disagreement gates a confidence-aware consistency loss `L_sam`.
- **Key limitation (paper's own ablation, Table 3)**: boundary metrics (95HD, ASD) improve far less than Dice/Jaccard. Point prompts fail on irregular/thin shapes. Uncertainty is a single scalar map — not boundary-localized.

**Direct follow-ups identified:**

- *Harmonizing Generalization and Specialization: Uncertainty-Informed Collaborative Learning for Semi-supervised Medical Image Segmentation* (2025) — extends uncertainty-informed collaborative learning. https://doi.org/10.48550/arxiv.2512.13101
- *SAM foundation model and expert model cross prompting framework for semi-supervised medical image segmentation* (2026) — cross-prompting between SAM and expert. https://doi.org/10.1016/j.jvcir.2026.104876
- *Vision-Language Enhanced Foundation Model for Semi-supervised Medical Image Segmentation* (2025) — adds vision-language to the SemiSAM framework. https://doi.org/10.48550/arxiv.2511.19759
- *SSL-MedSAM2: A Semi-supervised Medical Image Segmentation Framework Powered by Few-shot Learning of SAM2* (2025) — extends to SAM2. https://doi.org/10.48550/arxiv.2512.11548

---

## 2. Key Architectural Reference: UnGAP (UPFM Origin)

**UnGAP: Uncertainty-Guided Affine Prompting for Real-Time Crack Segmentation** (2026).
https://doi.org/10.48550/arxiv.2605.02380

This is the primary reference for **Modification 2 (UPFM)** in our CLAUDE.md. UnGAP introduces:

- **Pixel-wise heteroscedastic (aleatoric) variance prediction** from a variance head parallel to the mask head.
- **Uncertainty-guided feature modulation**: predicted variance fed back into the feature extractor as an *active calibration signal* via affine transforms (scale `gamma` + shift `delta`), rather than a passive loss weight.
- **Key insight**: high-uncertainty regions get their features actively rectified instead of just down-weighted, avoiding the "gradient suppression at hard pixels" pathology.
- Target domain: real-time crack segmentation (directly applicable to our task).

This paper provides the architectural blueprint for the UPFM module described in CLAUDE.md Section 5.4.

---

## 3. SAM for Crack/Defect Segmentation

### 3.1 Crack-SAM Variants

**Crack-SAM: Crack Segmentation Using a Foundation Model** (2024).
https://doi.org/10.21203/rs.3.rs-4780874/v1

Early exploration of SAM for crack segmentation. Demonstrates SAM's zero-shot capability is limited on thin cracks without adaptation.

**CrackSAM: Study on Few-Shot Segmentation of Mural Crack Images Using SAM** (2024).
https://doi.org/10.1109/icait62580.2024.10808070

Few-shot adaptation of SAM for mural/cultural heritage crack images. Shows that even limited fine-tuning significantly improves crack boundary quality.

**Segment Anything Model-Based Crack Segmentation Using Low-Rank Adaptation Fine-Tuning** (2024).
https://doi.org/10.1177/14759217241261089

LoRA fine-tuning of SAM for crack segmentation. Demonstrates parameter-efficient adaptation preserves SAM's generalization while specializing to cracks. Relevant for our generalist model selection — LoRA-adapted SAM variants are stronger generalists than vanilla SAM.

### 3.2 SAM Prompting Strategies for Defects

**A Hybrid YOLO and Segment Anything Model Pipeline for Multi-Damage Segmentation in UAV Inspection Imagery** (2025).
https://doi.org/10.3390/s25216568

Directly relevant to Modification 1 (box prompts). This paper:
- Uses YOLO detection to generate **box prompts** for SAM.
- Demonstrates box prompts outperform point prompts for multi-class damage segmentation.
- Pipeline: YOLO detects → boxes → SAM segments → refined masks.
- This validates our box-prompt strategy (Option B path).

**Boxes2Pixels: Learning Defect Segmentation from Noisy SAM Masks** (2026).
https://doi.org/10.48550/arxiv.2604.11162

Learns defect segmentation from noisy SAM outputs given box prompts. Shows that even noisy SAM masks contain useful training signal when properly filtered. Relevant to our confidence-aware loss design.

**From Bounding Boxes to Semantic Segmentation: Leveraging SAM for Weak Supervision in Remote Sensing** (2025).
https://doi.org/10.1117/12.3069933

Uses bounding boxes as weak supervision for SAM-based segmentation. Demonstrates boxes are robust prompts for irregular objects — directly supports our choice of box over point prompts for crack shapes.

**Promptable Fire Segmentation: Unleashing SAM2's Potential for Real-Time Mobile Deployment with Strategic Bounding Box Guidance** (2025).
https://doi.org/10.48550/arxiv.2510.21782

Strategic bounding box guidance for SAM2 on thin/irregular targets (fire). Methods for box refinement and multi-box prompting relevant to our multi-crack-component box generation.

### 3.3 EdgeSAM — Efficient Generalist Candidate

**EdgeSAM: Prompt-In-the-Loop Distillation for SAM** (2023).
https://doi.org/10.48550/arxiv.2312.06660

Distilled SAM variant optimized for edge devices. 50× faster than SAM with comparable promptable segmentation quality. Important as a candidate generalist model for deployment efficiency.

---

## 4. Heteroscedastic Uncertainty in Segmentation

### 4.1 Foundational Theory

**Kendall & Gal (2017).** *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
https://doi.org/10.48550/arxiv.1703.04977

The canonical reference for heteroscedastic (aleatoric) uncertainty in vision. Key formula used in our Modification 2:

```
L_hetero = 0.5 * exp(-log_sigma2) * L_task + 0.5 * log_sigma2
```

This loss lets the model learn to predict per-pixel variance, automatically down-weighting noisy/hard regions while maintaining gradient flow (unlike loss-weighting approaches that suppress gradient at hard pixels).

### 4.2 Application to Segmentation

**Wang et al. (2019).** *Aleatoric Uncertainty Estimation with Test-Time Augmentation for Medical Image Segmentation with Convolutional Neural Networks.* Neurocomputing.
https://doi.org/10.1016/j.neucom.2019.01.103

Test-time augmentation for aleatoric uncertainty in segmentation. The TTA-based perturbation disagreement approach is conceptually similar to SemiSAM+'s prompt-perturbation uncertainty but applied at the input level.

**Monteiro et al. (2020).** *Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty.* NeurIPS.
https://doi.org/10.48550/arxiv.2006.06015

Models spatially correlated aleatoric uncertainty via stochastic segmentation networks. Relevant because crack uncertainty is spatially structured (along the crack boundary) rather than i.i.d. per pixel.

**Quantifying Epistemic and Aleatoric Uncertainty in 3D U-Net Segmentation** (2021).
https://doi.org/10.1101/2021.09.20.21263844

Disentangles epistemic from aleatoric uncertainty in U-Net segmentation. Our design similarly separates two uncertainty types: `U_geo` (generalist disagreement, epistemic-like) and `U_app` (specialist variance, aleatoric).

### 4.3 Bayesian Approaches for Cracks

**B-BACN: Bayesian Boundary-Aware Convolutional Network for Crack Characterization** (2023).
https://doi.org/10.48550/arxiv.2302.06827

Bayesian boundary-aware network for crack detection with uncertainty quantification. Uses MC Dropout for epistemic uncertainty and explicitly models boundary uncertainty. Closely related to our boundary-band-split loss approach — demonstrates that boundary uncertainty is the dominant failure mode for crack segmentation.

**SpotRust / Deep Learning Corrosion Detection with Confidence** (2022).
https://doi.org/10.1038/s41529-022-00232-6

Applies uncertainty quantification to corrosion (another infrastructure defect). Demonstrates that aleatoric uncertainty is critical for real-world defect assessment where lighting, texture, and surface condition vary dramatically.

---

## 5. Boundary-Aware Segmentation for Thin Structures

### 5.1 Crack-Specific Boundary Methods

**Two-Stream Boundary-Aware Neural Network for Concrete Crack Segmentation and Quantification** (2023).
https://doi.org/10.1155/2023/3301106

Two-stream architecture with explicit boundary stream for crack segmentation. Shows boundary-aware design improves crack width quantification accuracy — directly relevant to our boundary-band-split loss.

**Boundary-Sensitive Hybrid Attention Network for Multi-Scale Crack Fine Segmentation** (2025).
https://doi.org/10.3390/s26103200

Boundary-sensitive attention for multi-scale crack segmentation. Demonstrates that boundary-focused attention mechanisms are essential for fine crack structures.

**UCAN: U-shaped Context Aggregation Network for Thin Crack Segmentation Under Topological Constraints** (2024).
https://doi.org/10.1108/ria-08-2023-0097

Topological constraints for thin crack segmentation. Shows that connectivity/topology preservation is as important as pixel accuracy for crack segmentation — relevant to evaluating our approach beyond Dice/IoU.

### 5.2 General Boundary Loss Methods

**Kervadec et al. (2019).** *Boundary Loss for Highly Unbalanced Segmentation.* MedIA.
https://doi.org/10.1016/j.media.2020.101851

Boundary loss using distance maps on contour space. The boundary-band concept we use (dilation - erosion around predicted contour) is a direct extension.

**FMS²: Unified Flow Matching for Segmentation and Synthesis of Thin Structures** (2026).
https://doi.org/10.48550/arxiv.2603.13659

Flow matching for thin structure segmentation. Novel approach using generative flow models for thin structure synthesis and segmentation.

---

## 6. Semi-Supervised Crack/Defect Segmentation

### 6.1 Directly Relevant Work

**Semi-Supervised Learning for Concrete Defect Segmentation from Images** (2024).
https://doi.org/10.1177/14759217231217097

Closest prior work to our task. Semi-supervised concrete defect segmentation using consistency regularization. Uses Mean Teacher framework with strong/weak augmentation. Does NOT use foundation models — our work extends this by adding SAM as a generalist pseudo-label source.

**Multiscale and Adversarial Learning-Based Semi-Supervised Semantic Segmentation Approach for Crack Detection in Concrete Structures** (2020).
https://doi.org/10.1109/access.2020.3022786

Adversarial learning for semi-supervised crack detection. Uses a discriminator to enforce prediction consistency between labeled and unlabeled data.

**Semi-Supervised Semantic Segmentation Using Adversarial Learning for Pavement Crack Detection** (2020).
https://doi.org/10.1109/access.2020.2980086

Similar adversarial semi-supervised approach specific to pavement cracks. Both papers predate SAM and don't leverage foundation models.

**End-to-End Semi-Supervised Deep Learning Model for Surface Crack Detection of Infrastructures** (2022).
https://doi.org/10.3389/fmats.2022.1058407

End-to-end semi-supervised pipeline for infrastructure surface cracks. Uses self-training with pseudo-labels but no foundation model component.

### 6.2 Foundation Model + Semi-Supervised (Medical, Transferable Methods)

**Foundation Model-Guided Multi-View Semi-Supervised CT Segmentation of Liver Tumors** (2025).
https://doi.org/10.1038/s41746-025-02190-0

Uses foundation model as guidance for semi-supervised segmentation. Similar in spirit to SemiSAM+ — validates the specialist-generalist paradigm.

**SAM-Fed: SAM-Guided Federated Semi-Supervised Learning for Medical Image Segmentation** (2025).
https://doi.org/10.48550/arxiv.2511.14302

Federated semi-supervised learning with SAM guidance. The confidence-weighting mechanism for SAM pseudo-labels is conceptually similar to our `L_sam` design.

**A Segment Anything Model-Guided and Match-Based Semi-Supervised Segmentation Framework for Medical Imaging** (2025).
https://doi.org/10.1002/mp.17785

SAM-guided match-based semi-supervised framework. Uses Hungarian matching between SAM outputs and specialist predictions — alternative to our confidence-weighted approach.

---

## 7. Prompt Perturbation and Uncertainty in SAM

### 7.1 Multi-Prompt Uncertainty

**BALD-SAM: Disagreement-Based Active Prompting in Interactive Segmentation** (2026).
https://doi.org/10.48550/arxiv.2603.10828

Uses BALD (Bayesian Active Learning by Disagreement) for prompt selection in SAM. Disagreement-based uncertainty directly parallels our `U_geo` computation from prompt perturbation.

**Segment Anything with Robust Uncertainty-Accuracy Correlation** (2026).
https://doi.org/10.48550/arxiv.2605.10603

Studies uncertainty-accuracy correlation in SAM. Shows that prompt perturbation disagreement correlates with segmentation quality — validates the core assumption behind SemiSAM+'s `U_x` and our `U_geo`.

**Learning from Noisy Prompts: Saliency-Guided Prompt Distillation for Robust Segmentation with SAM** (2026).
https://doi.org/10.48550/arxiv.2604.23314

Handles noisy prompts in SAM via saliency-guided distillation. Relevant because our connected-component box prompts will be inherently noisy — methods for robust prompting with imperfect boxes are directly applicable.

**On The Robustness of Foundational 3D Medical Image Segmentation Models Against Imprecise Visual Prompts** (2026).
https://doi.org/10.48550/arxiv.2601.16383

Studies robustness of foundation models to imprecise prompts. Box prompts are more robust than point prompts for localization error — supports our Modification 1 design choice.

---

## 8. Uncertainty-Guided Feature Modulation and Active Calibration

**Uncertainty-Guided Domain Alignment for Layer Segmentation in OCT Images** (2019).
https://doi.org/10.48550/arxiv.1908.08242

Early work on uncertainty-guided feature alignment. Uses predicted uncertainty to guide domain adaptation — precursor to the uncertainty→feature modulation idea.

**HUR-MACL: High-Uncertainty Region-Guided Multi-Architecture Collaborative Learning for Head and Neck Multi-Organ Segmentation** (2026).
https://doi.org/10.48550/arxiv.2601.04607

High-uncertainty region-guided collaborative learning. Actively targets high-uncertainty regions for multi-architecture collaboration — similar in spirit to our UPFM which targets high-variance regions for feature rectification.

**Efficient Deweather Mixture-of-Experts with Uncertainty-Aware Feature-Wise Linear Modulation** (2024).
https://doi.org/10.1609/aaai.v38i15.29622

Uncertainty-aware feature-wise linear modulation (FiLM) for image restoration. The FiLM-like affine transform `gamma(u) * F + delta(u)` is architecturally identical to our UPFM design.

---

## 9. Summary: Research Gap and Our Contribution

### What exists:
- SemiSAM+ provides the specialist-generalist paradigm but for 3D medical images with point/mask prompts
- SAM has been applied to cracks (Crack-SAM, LoRA fine-tuning) but not in a semi-supervised collaborative framework
- Heteroscedastic uncertainty (Kendall & Gal) is well-established but not integrated with foundation model prompting
- Box-prompt SAM pipelines exist (YOLO+SAM) but as static two-stage inference, not as part of a training loop
- Boundary-aware losses exist but don't incorporate foundation model uncertainty

### What is novel in our approach:
1. **Box-prompt adaptation of SemiSAM+** for 2D thin/elongated crack structures (Modification 1)
2. **Dual uncertainty channels** (`U_geo` from prompt perturbation + `U_app` from heteroscedastic variance) — decomposing uncertainty by source
3. **UPFM**: active feature modulation conditioned on predicted uncertainty, not passive loss weighting
4. **Boundary-band-split confidence-aware loss** targeting the specific failure mode SemiSAM+ identified (boundary quality)
5. **First application** of the specialist-generalist semi-supervised paradigm to infrastructure defect segmentation

---

## References

1. Zhang et al. (2025). SemiSAM+: Rethinking semi-supervised medical image segmentation in the era of foundation models. *Medical Image Analysis*, 103733. https://doi.org/10.1016/j.media.2025.103733

2. UnGAP: Uncertainty-Guided Affine Prompting for Real-Time Crack Segmentation (2026). https://doi.org/10.48550/arxiv.2605.02380

3. Kendall, A. & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS*. https://doi.org/10.48550/arxiv.1703.04977

4. Crack-SAM: Crack Segmentation Using a Foundation Model (2024). https://doi.org/10.21203/rs.3.rs-4780874/v1

5. Segment Anything Model-Based Crack Segmentation Using Low-Rank Adaptation Fine-Tuning (2024). *Structural Health Monitoring*. https://doi.org/10.1177/14759217241261089

6. A Hybrid YOLO and Segment Anything Model Pipeline for Multi-Damage Segmentation in UAV Inspection Imagery (2025). *Sensors*, 25(21), 6568. https://doi.org/10.3390/s25216568

7. Boxes2Pixels: Learning Defect Segmentation from Noisy SAM Masks (2026). https://doi.org/10.48550/arxiv.2604.11162

8. B-BACN: Bayesian Boundary-Aware Convolutional Network for Crack Characterization (2023). https://doi.org/10.48550/arxiv.2302.06827

9. Semi-Supervised Learning for Concrete Defect Segmentation from Images (2024). *Structural Health Monitoring*. https://doi.org/10.1177/14759217231217097

10. EdgeSAM: Prompt-In-the-Loop Distillation for  SAM (2023). https://doi.org/10.48550/arxiv.2312.06660

11. Wang et al. (2019). Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation. *Neurocomputing*, 338, 34-45. https://doi.org/10.1016/j.neucom.2019.01.103

12. Monteiro et al. (2020). Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty. *NeurIPS*. https://doi.org/10.48550/arxiv.2006.06015

13. Multiscale and Adversarial Learning-Based Semi-Supervised Semantic Segmentation for Crack Detection in Concrete Structures (2020). *IEEE Access*, 8, 170392. https://doi.org/10.1109/access.2020.3022786

14. BALD-SAM: Disagreement-Based Active Prompting in Interactive Segmentation (2026). https://doi.org/10.48550/arxiv.2603.10828

15. Two-Stream Boundary-Aware Neural Network for Concrete Crack Segmentation and Quantification (2023). *Structural Control and Health Monitoring*. https://doi.org/10.1155/2023/3301106

16. Deep Learning Corrosion Detection with Confidence (2022). *npj Materials Degradation*, 6. https://doi.org/10.1038/s41529-022-00232-6

17. Foundation Model-Guided Multi-View Semi-Supervised CT Segmentation of Liver Tumors (2025). *npj Digital Medicine*, 8. https://doi.org/10.1038/s41746-025-02190-0

18. Learning from Noisy Prompts: Saliency-Guided Prompt Distillation for Robust Segmentation with SAM (2026). https://doi.org/10.48550/arxiv.2604.23314

19. HUR-MACL: High-Uncertainty Region-Guided Multi-Architecture Collaborative Learning (2026). https://doi.org/10.48550/arxiv.2601.04607

20. Efficient Deweather Mixture-of-Experts with Uncertainty-Aware Feature-Wise Linear Modulation (2024). *AAAI*, 38(15). https://doi.org/10.1609/aa ai.v38i15.29622

21. Boundary-Sensitive Hybrid Attention Network for Multi-Scale Crack Fine Segmentation (2025). *Sensors*, 26(10), 3200. https://doi.org/10.3390/s26103200

22. Segment Anything with Robust Uncertainty-Accuracy Correlation (2026). https://doi.org/10.48550/arxiv.2605.10603

23. On The Robustness of Foundational 3D Medical Image Segmentation Models Against Imprecise Visual Prompts (2026). https://doi.org/10.48550/arxiv.2601.16383

24. SAM-Fed: SAM-Guided Federated Semi-Supervised Learning for Medical Image Segmentation (2025). https://doi.org/10.48550/arxiv.2511.14302

25. Semi-Supervised Semantic Segmentation Using Adversarial Learning for Pavement Crack Detection (2020). *IEEE Access*, 8, 51446. https://doi.org/10.1109/access.2020.2980086

---

*Review conducted via scite (scite.ai) MCP server. All references verified through scite's database of 210M+ publications with Smart Citation context.*
