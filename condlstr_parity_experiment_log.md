# CondLSTR Parity Experiment Log

## Scope

This document summarizes the CondLSTR-parity experiment line that started from the local parity baseline and then branched into:

- DN lane denoising
- row-wise visibility instead of scalar range
- decoder target initialization (`tgtinit`)
- query relation block
- instability diagnostics
- order prior
- STDC/ResNet34 backbone swap
- 5k-data performance branch

The goal was not only to improve F1, but also to understand where the main failure modes came from:

- Hungarian matching instability
- query slot interchangeability
- weak range modeling
- missing inter-lane interaction
- backbone sensitivity

This file is meant to be a technical record of what changed in code, why it changed, and what each branch showed empirically.

---

## Starting Point

### Baseline model family

The base model is implemented in:

- [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)
- [models/condlstr_parity_head.py](./models/condlstr_parity_head.py)
- [models/condlstr_parity_criterion.py](./models/condlstr_parity_criterion.py)
- [models/condlstr_parity_postprocess.py](./models/condlstr_parity_postprocess.py)
- [models/condlstr_dense_matcher.py](./models/condlstr_dense_matcher.py)
- [utils/condlstr_parity_targets.py](./utils/condlstr_parity_targets.py)
- [utils/condlstr_dense_targets.py](./utils/condlstr_dense_targets.py)

### Baseline parity design

The local parity model uses:

- LSTR transformer trunk from [models/LSTR_CULANE.py](./models/LSTR_CULANE.py)
- CondLSTR-style dense row-wise supervision
- Hungarian matching over dense row targets
- dynamic spatial mask/reg branches driven by query features
- scalar lane range prediction via `pred_ranges`

Conceptually, each query predicts:

- objectness
- class
- lane row range or visibility
- dense row mask logits
- dense row regression offsets

The parity head then converts those into lane coordinates in postprocess.

---

## Shared Infrastructure Added During This Work

These changes were not one experiment by themselves; they made later experiments possible.

### 1. Reproducibility and deterministic behavior

Files:

- [train.py](./train.py)
- [test.py](./test.py)

Changes:

- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.deterministic = True`
- explicit global seeding for Python, NumPy, PyTorch, CUDA
- per-worker seed offset in data prefetch workers
- `db._data_rng` tied to the worker seed when available

Why:

- early DN runs showed very large seed sensitivity
- without deterministic kernels and deterministic worker RNGs, it was too easy to confuse true model instability with training pipeline nondeterminism

### 2. Diagnostics plumbing for image identity and loss forwarding

Files:

- [sample/culane.py](./sample/culane.py)
- [nnet/py_factory.py](./nnet/py_factory.py)

Changes:

- batch sampler now returns `image_keys`
- training and validation forwarding preserves `**kwargs`
- parity loss can associate diagnostics with concrete image ids

Why:

- fixed-panel instability analysis depends on tracking the same image across checkpoints

### 3. Seed alias utility and remote runners

Files:

- [make_seed_alias.py](./make_seed_alias.py)
- `remote_run_parity_*.sh` scripts

Changes:

- clone an experiment alias with a different seed while preserving the config family
- standard remote train/test/eval scripts for each branch

Why:

- matched-seed comparisons (`base317` vs `dn317`, `base901` vs `dn901`) were critical

### 4. Fixed-panel instability probing

Files:

- [probe_dn_instability.py](./probe_dn_instability.py)
- [analyze_match_diag.py](./analyze_match_diag.py)

Changes:

- selected a fixed panel of training images
- evaluated the same panel at multiple checkpoints
- logged query-target assignments, cost margins, object score stats
- summarized flip rates and instability per image and per iteration

Why:

- single final F1 values were not enough to diagnose matching instability

---

## Core Parity Model Extensions

The base parity model in [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py) was gradually extended to support multiple experiment switches from config instead of forking the whole model each time.

Important runtime switches:

- `parity_backbone`
- `dense_range_mode`
- `dense_visibility_dim`
- `dense_decoder_init_mode`
- `dense_relation_mode`
- `dn_lane_num_queries`
- `dn_lane_loss_weight`
- `match_diag_enabled`
- `dense_match_order_weight`

This design let us keep most experiments as clean config-level branches.

---

## Experiment 1: DN Lane Denoising

### Hypothesis

Hungarian matching is unstable for lanes because neighboring lanes are visually similar. A DN-DETR-like denoising branch might stabilize training and improve assignment consistency.

### Architectural changes

#### A. DN query generator

File:

- [models/condlstr_dn_lane.py](./models/condlstr_dn_lane.py)

What it does:

- builds an 8D summary vector per GT lane
- encodes:
  - normalized start row
  - normalized end row
  - four sampled x locations
  - lane center x
  - visible-row fraction
- injects noise into row range and x coordinates
- returns:
  - DN queries
  - DN targets
  - valid DN count per image

#### B. Decoder integration

Files:

- [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)
- [models/py_utils/transformer.py](./models/py_utils/transformer.py)

What changed:

- DN queries are encoded through `self.dn_query_encoder`
- DN queries are concatenated to learned main queries
- transformer now accepts:
  - `tgt_mask`
  - `decoder_tgt`
  - `decoder_query_pos`

Critical fix:

- added `_build_dn_decoder_attention_mask(...)`
- main queries and DN queries are isolated in decoder self-attention

Why this mattered:

- the first naive DN version let DN queries and main queries freely attend to each other
- that version collapsed badly
- attention isolation was necessary for a meaningful DN test

#### C. Criterion support for DN direct supervision

File:

- [models/condlstr_parity_criterion.py](./models/condlstr_parity_criterion.py)

What changed:

- criterion splits outputs into:
  - main queries
  - DN queries
- main queries still use Hungarian matching
- DN queries skip matcher and receive direct supervision against sliced GT targets
- DN losses:
  - `loss_dn_object`
  - `loss_dn_class`
  - `loss_dn_loc`
  - `loss_dn_reg`
  - `loss_dn_range`

#### D. Diagnostics

Files:

- [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)
- [models/condlstr_parity_criterion.py](./models/condlstr_parity_criterion.py)
- [analyze_match_diag.py](./analyze_match_diag.py)
- [probe_dn_instability.py](./probe_dn_instability.py)

What changed:

- periodic `match_diag_train.jsonl`
- per-match costs and margins
- image-wise assignment flip statistics
- `main_object_fg_stats` and `dn_object_fg_stats_valid`

### Main configs

- [config/LSTR_CULANE_2k_condlstr_parity_dn.json](./config/LSTR_CULANE_2k_condlstr_parity_dn.json)
- [config/LSTR_CULANE_2k_condlstr_parity_dn-thr04.json](./config/LSTR_CULANE_2k_condlstr_parity_dn-thr04.json)
- [config/LSTR_CULANE_2k_condlstr_parity_dn_w025.json](./config/LSTR_CULANE_2k_condlstr_parity_dn_w025.json)
- [config/LSTR_CULANE_2k_condlstr_parity_stdc_dn.json](./config/LSTR_CULANE_2k_condlstr_parity_stdc_dn.json)
- [config/LSTR_CULANE_5k_condlstr_parity_stdc_dn.json](./config/LSTR_CULANE_5k_condlstr_parity_stdc_dn.json)

### Results

#### 2k, LSTR backbone

| Branch | Seed317 | Seed901 | Mean | Note |
|---|---:|---:|---:|---|
| baseline | 0.313926 | 0.260386 | 0.287156 | reference |
| DN `w=1.0` | 0.420217 | 0.300433 | 0.360325 | high upside, high variance |
| DN `w=0.25` | 0.328371 | 0.259154 | 0.293763 | did not stabilize |

#### 2k, STDC backbone

| Branch | Seed317 | Seed901 | Mean |
|---|---:|---:|---:|
| STDC base | 0.394768 | 0.328120 | 0.361444 |
| STDC + DN | 0.451321 | 0.360874 | 0.406098 |

#### 5k, STDC backbone

| Branch | Seed317 | Seed901 | Mean |
|---|---:|---:|---:|
| 5k STDC base | 0.515195 | 0.527872 | 0.521534 |
| 5k STDC + DN | 0.518011 | 0.536953 | 0.527482 |

### What we learned

- DN was not a robust instability fix on the original 2k LSTR-backbone branch.
- DN gave the highest upside on 2k, but variance was severe.
- Once the backbone moved to STDC and data increased to 5k, DN became a small but consistent positive.
- The original hypothesis "`DN improves because it stabilizes Hungarian matching`" was **not** cleanly supported.
- The better conclusion is:
  - DN can improve recall and final F1
  - the gain depends heavily on backbone quality and data regime

---

## Experiment 2: Visibility Instead of Scalar Range

### Hypothesis

Representing a lane by only `start/end` rows is too crude. Occlusions and partial visibility should be modeled row by row.

### Architectural changes

#### A. Head support for row visibility logits

File:

- [models/condlstr_parity_head.py](./models/condlstr_parity_head.py)

What changed:

- `QueryBranchHead` can optionally output `pred_row_visibility_logits`
- `predict_ranges=False` and `visibility_dim>0` switch the head from scalar range to row-wise visibility logits

#### B. Criterion support

File:

- [models/condlstr_parity_criterion.py](./models/condlstr_parity_criterion.py)

What changed:

- when `pred_row_visibility_logits` exists, `loss_range` becomes BCE over row visibility
- when visibility logits do not exist, criterion falls back to L1 over `pred_ranges`

#### C. Matcher support

File:

- [models/condlstr_dense_matcher.py](./models/condlstr_dense_matcher.py)

What changed:

- matcher accepts either:
  - `pred_ranges`
  - or `pred_row_visibility_logits`
- range cost becomes row-wise BCE if visibility logits are present

#### D. Postprocess support

File:

- [models/condlstr_parity_postprocess.py](./models/condlstr_parity_postprocess.py)

What changed:

- lane points can be decoded from visible rows instead of scalar start/end range
- `condlstr_visibility_thresh` controls row activation

#### E. Model-level switch

File:

- [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)

Config keys:

- `dense_range_mode = visibility`
- `dense_visibility_dim = 295`
- `condlstr_visibility_thresh = 0.5`

### Main configs

- [config/LSTR_CULANE_2k_condlstr_parity_visibility.json](./config/LSTR_CULANE_2k_condlstr_parity_visibility.json)
- [config/LSTR_CULANE_2k_condlstr_parity_stdc_visibility.json](./config/LSTR_CULANE_2k_condlstr_parity_stdc_visibility.json)

### Results

#### 2k, LSTR backbone

| Branch | Seed317 | Seed901 | Mean |
|---|---:|---:|---:|
| baseline | 0.313926 | 0.260386 | 0.287156 |
| visibility | 0.321789 | 0.291460 | 0.306625 |

#### 2k, STDC backbone

| Branch | Seed317 | Seed901 | Mean |
|---|---:|---:|---:|
| STDC base | 0.394768 | 0.328120 | 0.361444 |
| STDC + visibility | 0.360161 | 0.281891 | 0.321026 |

### What we learned

- Visibility was a clean, low-variance gain on the original LSTR-backbone branch.
- The same idea failed once the backbone changed to STDC.
- Likely explanation:
  - on the weaker backbone, scalar range was a genuine bottleneck
  - on STDC, the representation bottleneck shifted elsewhere
  - row-wise visibility introduced calibration and false-positive issues instead of helping

So:

- visibility is **not backbone-portable**
- it was useful for diagnosis, but not the best final performance branch

---

## Experiment 3: Decoder Target Initialization (`tgtinit`)

### Hypothesis

Part of the instability may come from weak slot semantics in the decoder. The local transformer used:

- legacy behavior: `tgt = query_embed * 0.1`

CondLSTR-style learned target embeddings might give queries more stable roles.

### Architectural changes

#### A. Transformer interface

File:

- [models/py_utils/transformer.py](./models/py_utils/transformer.py)

What changed:

- transformer now accepts:
  - `decoder_tgt`
  - `decoder_query_pos`
  - `tgt_mask`

This separated:

- decoder content input
- decoder positional/query identity

#### B. Model-level switch

File:

- [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)

What changed:

- added `dense_decoder_init_mode`
- modes:
  - `legacy_query_embed`
  - `learned_target_embed`
- when using `learned_target_embed`, the model creates:
  - `self.decoder_target_embed`
  - separate `decoder_tgt`
  - separate `decoder_query_pos`

### Main configs

- [config/LSTR_CULANE_2k_condlstr_parity_tgtinit.json](./config/LSTR_CULANE_2k_condlstr_parity_tgtinit.json)
- [config/LSTR_CULANE_2k_condlstr_parity_stdc_tgtinit.json](./config/LSTR_CULANE_2k_condlstr_parity_stdc_tgtinit.json)

### Results

#### 2k, LSTR backbone

| Branch | Seed317 | Seed901 | Mean |
|---|---:|---:|---:|
| baseline | 0.313926 | 0.260386 | 0.287156 |
| tgtinit | 0.339533 | 0.266173 | 0.302853 |

#### 2k, STDC backbone

| Branch | Seed317 | Note |
|---|---:|---|
| STDC base | 0.394768 | reference |
| STDC + tgtinit | 0.365458 | branch closed after first negative seed |

### What we learned

- On the original branch, `tgtinit` improved final F1 in both matched seeds.
- However, instability diagnostics did **not** improve:
  - fixed-panel flip rate stayed essentially unchanged
  - target margins did not improve
- So `tgtinit` improved optimization/performance without clearly fixing assignment instability.
- On STDC, the same modification was not helpful.

Conclusion:

- `tgtinit` is a real performance knob
- but it is not the root instability fix we were looking for

---

## Experiment 4: Query Relation Block

### Hypothesis

Lanes are not independent objects. A small query-query interaction block before the head might improve adjacent lane reasoning.

### Architectural changes

File:

- [models/condlstr_query_relation.py](./models/condlstr_query_relation.py)

What it is:

- lightweight multi-head self-attention over queries
- residual + FFN structure
- stacked by `num_layers`

Integration:

- [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)

Config keys:

- `dense_relation_mode = self_attn`
- `dense_relation_layers = 1`
- `dense_relation_heads = 4`
- `dense_relation_ff_dim = 512`

### Main configs

- [config/LSTR_CULANE_2k_condlstr_parity_relation.json](./config/LSTR_CULANE_2k_condlstr_parity_relation.json)

### Results

| Branch | Seed317 | Note |
|---|---:|---|
| baseline | 0.313926 | reference |
| relation | 0.277083 | negative, branch closed |

### What we learned

- The cheap self-attention relation block did not help.
- This does **not** prove inter-lane reasoning is useless.
- It only says this specific lightweight implementation was harmful.

Conclusion:

- branch closed
- no further effort spent on this exact design

---

## Instability Study

This workstream ran in parallel with the main experiments and was used to decide whether a branch was improving F1 by actually stabilizing matching.

### Fixed-panel methodology

Files:

- [probe_dn_instability.py](./probe_dn_instability.py)
- [analyze_match_diag.py](./analyze_match_diag.py)

Procedure:

- choose a fixed train panel
- probe the same images at multiple checkpoints
- compare target-to-query assignments across checkpoints

### Key metrics

- `flip_rate`
- `comparable_pairs`
- `mean_query_margin`
- `mean_target_margin`
- `main_fg_mean`
- `dn_fg_valid_mean`

### Important findings

#### Baseline instability already existed

For 2k baseline:

- `flip_rate = 0.8234`
- `mean_target_margin = 18.14`

This is very high and means the model already had severe query slot interchangeability before any DN or relation trick.

#### DN did not clearly fix assignment instability

2k DN:

- seed317:
  - `flip_rate = 0.7978`
  - `mean_target_margin = 16.59`
- seed901:
  - `flip_rate = 0.7554`
  - `mean_target_margin = 18.27`

Interpretation:

- flip rate remained high
- final F1 and flip-rate did not align cleanly
- DN was not a clean Hungarian-stability solution

#### `tgtinit` also did not fix instability

2k `tgtinit`:

- `flip_rate = 0.8270`
- `mean_target_margin = 18.01`

Interpretation:

- final F1 improved
- matching instability did not

---

## Side Experiment: Order Prior

This was not one of the main performance branches, but it was the most direct attempt to reduce slot symmetry.

### Motivation

If lanes are ordered left-to-right in the scene, query slots might be softly aligned to lane order.

### Code changes

Files:

- [utils/condlstr_dense_targets.py](./utils/condlstr_dense_targets.py)
- [models/condlstr_dense_matcher.py](./models/condlstr_dense_matcher.py)
- [config/LSTR_CULANE_2k_condlstr_parity_orderprior.json](./config/LSTR_CULANE_2k_condlstr_parity_orderprior.json)

What changed:

- targets now expose:
  - `gt_lane_order`
  - `gt_lane_order_norm`
- matcher has `cost_order`
- total Hungarian cost can include:
  - `order_weight * cost_order`

### Result

2k order-prior branch:

- `F1 = 0.276623`
- `flip_rate = 0.7856`
- `mean_target_margin = 20.85`

Interpretation:

- matching became somewhat more stable
- but recall collapsed
- the prior over-constrained valid matches

Conclusion:

- the core idea of symmetry breaking is probably right
- this hard left-right prior was too rigid

---

## Backbone Swap: STDC/BiSeNet-Style ResNet34

This branch ended up changing the overall picture more than any local head tweak.

### Code changes

File:

- [models/parity_stdc_res34_backbone.py](./models/parity_stdc_res34_backbone.py)

What it is:

- ResNet34 context path
- attention refinement modules
- feature fusion module
- output projection to 256 channels

Model integration:

- [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)

Config switch:

- `parity_backbone = stdc_res34`

### Results

#### 2k

| Branch | Seed317 | Seed901 | Mean |
|---|---:|---:|---:|
| LSTR base | 0.313926 | 0.260386 | 0.287156 |
| STDC base | 0.394768 | 0.328120 | 0.361444 |

This was the first strong sign that backbone quality was dominating several earlier local tweaks.

#### 5k

| Branch | Seed317 | Seed901 | Mean |
|---|---:|---:|---:|
| 5k STDC base | 0.515195 | 0.527872 | 0.521534 |

Interpretation:

- the biggest and most reliable gains came from:
  - stronger backbone
  - more data

---

## Final Experimental Picture

### Branches that were closed

- `relation`
- `orderprior`
- `STDC + visibility`
- `STDC + tgtinit`

### Branches that were useful diagnostically

- `visibility` on the original backbone
- `tgtinit` on the original backbone
- `DN w=0.25`
- fixed-panel instability probing

### Branches that remained competitive

- `2k STDC base`
- `2k STDC + DN`
- `5k STDC base`
- `5k STDC + DN`

### Practical conclusions

1. The original 2k LSTR-backbone regime had real Hungarian/query instability.
2. DN alone was not a clean instability cure.
3. Visibility improved the weak-backbone branch but did not transfer to STDC.
4. `tgtinit` improved F1 on the weak-backbone branch but did not fix flip-rate.
5. The biggest robust gains came from stronger backbone plus more data.
6. On the final high-performing branch, DN became a small positive instead of a chaotic one.

---

## Recommended Reading Order in Code

If someone wants to reconstruct the work from code, read in this order:

1. [models/LSTR_CULANE_condlstr_parity_base.py](./models/LSTR_CULANE_condlstr_parity_base.py)
2. [models/condlstr_parity_head.py](./models/condlstr_parity_head.py)
3. [models/condlstr_parity_criterion.py](./models/condlstr_parity_criterion.py)
4. [models/condlstr_dense_matcher.py](./models/condlstr_dense_matcher.py)
5. [models/condlstr_parity_postprocess.py](./models/condlstr_parity_postprocess.py)
6. [utils/condlstr_parity_targets.py](./utils/condlstr_parity_targets.py)
7. [utils/condlstr_dense_targets.py](./utils/condlstr_dense_targets.py)
8. [models/condlstr_dn_lane.py](./models/condlstr_dn_lane.py)
9. [models/condlstr_query_relation.py](./models/condlstr_query_relation.py)
10. [models/parity_stdc_res34_backbone.py](./models/parity_stdc_res34_backbone.py)
11. [probe_dn_instability.py](./probe_dn_instability.py)
12. [analyze_match_diag.py](./analyze_match_diag.py)

---

## Current Best Snapshot From This Line

At the end of this experiment line, the strongest branch was:

- `LSTR_CULANE_5k_condlstr_parity_stdc_dn`

Known matched-seed results:

- seed317: `0.518011`
- seed901: `0.536953`

Strong runner-up:

- `LSTR_CULANE_5k_condlstr_parity_stdc_res34`

Known matched-seed results:

- seed317: `0.515195`
- seed901: `0.527872`

The gap is small, but positive for DN in the final high-data STDC branch.
