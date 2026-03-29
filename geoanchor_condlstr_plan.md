# GeoAnchor CondLSTR Plan

## Goal

The objective is to add a **lane-geometric query design** on top of the current best branch:

- `5k STDC + DN`
- config family:
  - `LSTR_CULANE_5k_condlstr_parity_stdc_dn`

This plan is intentionally conservative.

It does **not** replace the CondLSTR-style detector formulation.
It keeps:

- row-wise dense lane targets
- dynamic mask/reg head
- Hungarian row matching
- range-based lane decoding
- STDC/Res34 parity backbone
- DN lane branch

The first new mechanism is only:

- **geometry-grounded query initialization**

That is the cleanest test of whether slot grounding helps more than purely learned lane queries.

---

## 1. What Will Stay Fixed

These parts should remain unchanged in the first GeoAnchor experiment:

- backbone:
  - `parity_backbone = stdc_res34`
- encoder:
  - current parity transformer encoder
- decoder depth:
  - current parity transformer decoder
- dynamic head formulation:
  - same `CondLSTRParityHead`
- DN branch:
  - keep current DN lane branch active
- loss weights:
  - keep current best `5k STDC+DN` weights
- postprocess:
  - keep current range-based postprocess
- evaluator:
  - same CULane evaluator and `thr04`

This is critical. If more than one major mechanism changes, attribution is lost.

---

## 2. Baseline To Branch From

Starting point:

- [config/LSTR_CULANE_5k_condlstr_parity_stdc_dn.json](/home/alki/projects/LSTR/config/LSTR_CULANE_5k_condlstr_parity_stdc_dn.json)
- [config/LSTR_CULANE_5k_condlstr_parity_stdc_dn-thr04.json](/home/alki/projects/LSTR/config/LSTR_CULANE_5k_condlstr_parity_stdc_dn-thr04.json)
- [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)

Current important settings in that baseline:

- `num_queries = 20`
- `parity_backbone = stdc_res34`
- `dn_lane_num_queries = 8`
- `dense_use_coords = false`
- `dense_range_mode = range`
- `dense_decoder_init_mode = legacy_query_embed`

This branch is already the strongest practical recipe we have.

---

## 3. Core Research Hypothesis

Current learned query slots are not explicitly grounded in lane geometry.

That means:

- two nearby lanes can be represented by interchangeable slots
- learned queries start without explicit lane semantics
- DN helps training, but does not fully solve slot ambiguity

The new hypothesis is:

> If each decoder query starts from a lane-geometric prior, then slot ambiguity should decrease and the detector should reach higher recall and higher-IoU quality with less dependence on lucky slot assignment.

The minimal test should not add deformable attention, diffusion, or a new decoder family yet.

---

## 4. Minimal GeoAnchor Design

## 4.1 Query parameterization

Each query will be associated with a small anchor vector.

Recommended minimal anchor vector:

- `bottom_x`
- `delta_x`
- `row_start`
- `row_end`

Meaning:

- `bottom_x`:
  - normalized x location at bottom of image
- `delta_x`:
  - horizontal displacement between top and bottom anchor points
  - acts as a coarse heading surrogate
- `row_start`:
  - normalized visible start row
- `row_end`:
  - normalized visible end row

This is better than using raw angle for the first experiment because:

- no wraparound issue
- easier to convert into a coarse lane corridor
- easier to interpret in matching and head logic later

Derived quantity:

- `top_x = bottom_x + delta_x`

The first experiment should use a **fixed sparse anchor bank**, not image-adaptive anchors.

Reason:

- fixed bank isolates the effect of geometry-grounded slot identity
- image-adaptive anchors add another predictor and another failure mode

## 4.2 Anchor bank shape

Current parity branch uses:

- `num_queries = 20`

So the first anchor bank should also use exactly 20 anchors.

Recommended grid:

- 5 bottom-x bins
- 2 heading bins
- 2 row-start bins

Total:

- `5 x 2 x 2 = 20`

Suggested values:

- `bottom_x`:
  - `0.10, 0.30, 0.50, 0.70, 0.90`
- `delta_x`:
  - `-0.12, +0.12`
- `row_start`:
  - `0.05, 0.25`
- `row_end`:
  - fixed `1.0`

This bank is simple enough to debug and rich enough to break full query symmetry.

---

## 5. First Experiment: GeoAnchor Query Init Only

This is the most important point:

- do **not** touch matcher
- do **not** touch head
- do **not** touch coords
- do **not** touch postprocess

Only change:

- how query embeddings and decoder target content are initialized

This is the cleanest first ablation.

## 5.1 What will happen mathematically

Let:

- `Q = 20`
- `C = attn_dim = 256`
- `A = 4` anchor dimensions

We define:

- `anchor_bank`: `[Q, A]`

We learn:

- `anchor_embed_mlp: R^A -> R^C`

Then for each query:

- `anchor_embed[q] = MLP(anchor_bank[q])`

The decoder inputs become:

- `query_pos = learned_query_embed + anchor_embed`
- `decoder_tgt = alpha * learned_query_embed + beta * anchor_embed`

Recommended first coefficients:

- `alpha = 0.1`
- `beta = 1.0`

This preserves compatibility with the current decoder while injecting explicit geometry.

Do **not** remove learned queries in v1.

Reason:

- pure anchor-only initialization is too aggressive as a first test
- learned query embeddings still carry useful latent semantics
- the goal is grounding, not full replacement

---

## 6. Exact Classes And Files To Change

## 6.1 New file to add

Add:

- [models/condlstr_geo_anchor.py](/home/alki/projects/LSTR/models/condlstr_geo_anchor.py)

Recommended classes:

### `class FixedLaneAnchorBank(nn.Module)`

Responsibility:

- build and expose a fixed `[Q, 4]` anchor tensor

Methods:

- `__init__(num_queries: int, mode: str = "bottom_dx_rowspan")`
- `_build_anchor_bank(...) -> torch.Tensor`
- `forward() -> torch.Tensor`

Implementation detail:

- register the anchor tensor as a buffer, not as a parameter

Why:

- v1 should test fixed geometry priors
- not learnable anchors yet

### `class LaneAnchorEncoder(nn.Module)`

Responsibility:

- encode anchor vectors into decoder embedding space

Recommended architecture:

- `Linear(4, C)`
- `ReLU`
- `Linear(C, C)`

Methods:

- `__init__(anchor_dim: int, hidden_dim: int)`
- `forward(anchor_bank: torch.Tensor) -> torch.Tensor`

Output:

- `[Q, C]`

## 6.2 Modify parity model class

File:

- [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)

Class to change:

- `class model(_BaseTransformerModel)`

### New config switches to add

Add new system configs:

- `dense_query_mode`
  - values:
    - `learned`
    - `geo_anchor`
- `dense_geo_anchor_mode`
  - initial value:
    - `bottom_dx_rowspan`
- `dense_geo_anchor_scale`
  - default:
    - `1.0`
- `dense_geo_anchor_tgt_scale`
  - default:
    - `1.0`
- `dense_geo_anchor_query_scale`
  - default:
    - `1.0`

### New members in `__init__`

Add:

- `self.dense_query_mode`
- `self.dense_geo_anchor_mode`
- `self.dense_geo_anchor_scale`
- `self.dense_geo_anchor_tgt_scale`
- `self.dense_geo_anchor_query_scale`

If `dense_query_mode == 'geo_anchor'`, create:

- `self.geo_anchor_bank = FixedLaneAnchorBank(...)`
- `self.geo_anchor_encoder = LaneAnchorEncoder(...)`

### New method to add

Add method:

- `_build_geo_anchor_embeddings(self, batch_size: int) -> torch.Tensor`

Responsibility:

1. fetch fixed anchor bank `[Q, 4]`
2. encode to `[Q, C]`
3. scale it
4. expand to `[B, Q, C]`

### Modify existing method

Current method:

- `_build_decoder_query_inputs(...)`

This method must be extended.

Current behavior:

- `legacy_query_embed`:
  - `main_query_pos = learned_queries`
  - `main_decoder_tgt = learned_queries * 0.1`
- `learned_target_embed`:
  - `main_query_pos = decoder_target_embed`
  - `main_decoder_tgt = decoder_target_embed`

New behavior for `dense_query_mode='geo_anchor'`:

1. compute `geo_anchor_embed`
2. combine it with whichever decoder-init mode is already selected

Recommended exact logic:

- if `dense_decoder_init_mode == 'legacy_query_embed'`:
  - `base_query_pos = learned_queries`
  - `base_decoder_tgt = learned_queries * 0.1`
- if `dense_decoder_init_mode == 'learned_target_embed'`:
  - `base_query_pos = decoder_target_embed`
  - `base_decoder_tgt = decoder_target_embed`

Then:

- `main_query_pos = base_query_pos + query_scale * geo_anchor_embed`
- `main_decoder_tgt = base_decoder_tgt + tgt_scale * geo_anchor_embed`

This is the most stable first version.

### Do not change these in v1

Do not change:

- DN query construction
- DN attention mask
- query relation block
- parity head
- matcher
- criterion
- postprocess

## 6.3 Transformer file

File:

- [models/py_utils/transformer.py](/home/alki/projects/LSTR/models/py_utils/transformer.py)

Class:

- `class Transformer`

For v1:

- **no code change required**

Reason:

- this file already supports:
  - `decoder_tgt`
  - `decoder_query_pos`

That is enough for the first GeoAnchor experiment.

## 6.4 Head file

File:

- [models/condlstr_parity_head.py](/home/alki/projects/LSTR/models/condlstr_parity_head.py)

Classes:

- `DynamicSpatialBranch`
- `CondLSTRParityHead`

For v1:

- **no code change required**

Reason:

- the first experiment is only query grounding
- changing query init and dynamic head geometry channels at the same time would make results hard to interpret

## 6.5 Matcher file

File:

- [models/condlstr_dense_matcher.py](/home/alki/projects/LSTR/models/condlstr_dense_matcher.py)

Class:

- `CondLSTRDenseHungarianMatcher`

For v1:

- **no code change required**

Reason:

- first test should answer:
  - can geometry-grounded queries help even with the same matching rule?

Soft geometry-aware cost comes later.

## 6.6 Criterion file

File:

- [models/condlstr_parity_criterion.py](/home/alki/projects/LSTR/models/condlstr_parity_criterion.py)

Class:

- `CondLSTRParitySetCriterion`

For v1:

- optional only

Recommended small extension:

- attach per-batch query slot metadata to diagnostics

Specifically, for each image diagnostic record, add:

- `query_mode`
- maybe `anchor_bank_summary`

This is optional. It is not needed for correctness.

## 6.7 Config files to add

Add:

- `config/LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor.json`
- `config/LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor-thr04.json`

These should inherit the current best `5k STDC+DN` settings and only change:

- `snapshot_name`
- `dense_query_mode = geo_anchor`
- `dense_geo_anchor_mode = bottom_dx_rowspan`
- keep `dense_use_coords = false` in v1

## 6.8 Model alias file to add

Add:

- `models/LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor.py`

Same pattern as existing alias files.

This file should expose:

- `model`
- `loss`

by importing from the shared parity base.

## 6.9 Remote scripts to add

Add:

- `remote_run_parity_5k_stdc_dn_geoanchor_train.sh`
- `remote_run_parity_5k_stdc_dn_geoanchor_test_eval.sh`
- `remote_run_parity_5k_stdc_dn_geoanchor_full.sh`

Use the current `5k STDC+DN` scripts as template.

---

## 7. Exact Tensor Flow In The New Branch

This section is the most important logic check.

### Current branch

Current best branch does:

1. `learned_queries = query_embed.weight`
2. if DN enabled, append DN query embeddings
3. set:
   - `query_embed`
   - `decoder_tgt`
4. call transformer
5. call parity head
6. compute losses

### New branch

The new branch should do:

1. `learned_queries = query_embed.weight`
2. `geo_anchor_embed = geo_anchor_encoder(anchor_bank)`
3. expand to batch
4. compute:
   - `base_query_pos`
   - `base_decoder_tgt`
5. combine:
   - `main_query_pos = base_query_pos + geo_anchor_query_scale * geo_anchor_embed`
   - `main_decoder_tgt = base_decoder_tgt + geo_anchor_tgt_scale * geo_anchor_embed`
6. if DN is enabled:
   - append DN embeddings exactly as before
   - DN path remains unchanged
7. transformer consumes:
   - `decoder_query_pos = query_pos`
   - `decoder_tgt = decoder_tgt`

Nothing else changes.

This is intentional. It ensures the only new mechanism is anchor-grounded slot initialization.

---

## 8. Why This Design Is The Right First Step

Because it directly targets the problem we actually care about:

- query slots are too free
- hard order prior was too rigid
- DN helped but did not remove slot fuzziness
- `tgtinit` improved performance but did not inject lane geometry

GeoAnchor v1 is the middle ground:

- stronger than plain learned query embeddings
- softer than hard query-order matching
- easier to justify than a full new decoder family

This is exactly the right level of intervention for a first research-grade experiment.

---

## 9. What Not To Do In V1

Do not add these yet:

- deformable corridor attention
- anchor-relative coordinate channels
- geometry-aware matching cost
- new sequential decoder
- Mamba refinement
- visibility mode
- head redesign

All of those can be valid later.

But if they are added now, the experiment becomes unreadable.

---

## 10. Evaluation Protocol For V1

Use:

- same branch family as best current model
- same data
- same threshold protocol

Recommended first evaluation:

- `5k STDC+DN baseline`
- `5k STDC+DN+GeoAnchor`

Matched seeds:

- `317`
- `901`

Primary metrics:

- final CULane F1 at current protocol
- recall
- precision

Secondary metrics:

- fixed-panel `flip_rate`
- `mean_target_margin`
- `F1@0.75` if evaluator sweep is available

Minimal success criterion:

- no speed collapse
- positive matched-seed gain
- at least one of:
  - lower flip-rate
  - higher target margin
  - higher recall at same threshold

Strong success criterion:

- positive gain in both seeds
- lower assignment flip-rate
- improved high-IoU behavior

---

## 11. Second Experiment After V1

Only if v1 is positive:

### V2: Anchor-relative coordinates

Then and only then modify:

- [models/condlstr_parity_head.py](/home/alki/projects/LSTR/models/condlstr_parity_head.py)

Classes to change:

- `DynamicSpatialBranch`
- `CondLSTRParityHead`

New idea:

- current `use_coords=True` adds absolute `(x, y)`
- add optional anchor-relative coordinates:
  - `x_rel_to_anchor_line`
  - `y_rel_to_row_start`
  - `y_rel_to_row_end`

This is a second-step experiment, not part of v1.

---

## 12. Third Experiment After V2

Only if v1 or v2 is positive:

### V3: Soft geometry-aware matching

Then modify:

- [models/condlstr_dense_matcher.py](/home/alki/projects/LSTR/models/condlstr_dense_matcher.py)

Class to change:

- `CondLSTRDenseHungarianMatcher`

Add small optional soft cost terms:

- bottom-x consistency
- coarse span consistency
- coarse corridor consistency

Do **not** add hard order.

This should be small weight only.

Again, this is not part of v1.

---

## 13. Concrete First Branch Name

Recommended first branch alias:

- `LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor`

This name is explicit and readable:

- `5k`
- `parity`
- `stdc`
- `dn`
- `geoanchor`

---

## 14. Final Recommendation

The first implementation should change exactly these classes/files:

### New classes

- `FixedLaneAnchorBank`
- `LaneAnchorEncoder`

New file:

- [models/condlstr_geo_anchor.py](/home/alki/projects/LSTR/models/condlstr_geo_anchor.py)

### Modified class

- `model` in [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)

### New config/model aliases

- `config/LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor.json`
- `config/LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor-thr04.json`
- `models/LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor.py`

### Optional diagnostics-only modification

- `CondLSTRParitySetCriterion` in [models/condlstr_parity_criterion.py](/home/alki/projects/LSTR/models/condlstr_parity_criterion.py)

Everything else should remain untouched in v1.

That is the technically cleanest first GeoAnchor experiment.
