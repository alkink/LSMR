# LSTR and CondLSTR Detailed Code Walkthrough

## Purpose

This document explains, at code level, two codebases:

- the current project in this directory: `/home/alki/projects/LSTR`
- the reference CondLSTR project: `/home/alki/projects/CondLSTR`

The goal is to make the architecture, training flow, data flow, matching logic, postprocess logic, and all important parity modifications understandable without having to open the code.

This is not a marketing summary. It is a technical reading guide for research and implementation work.

---

## What The Current Project Actually Is

This repository started as an LSTR-style lane detector and was later extended with a large CondLSTR-parity branch.

So the repository now contains two conceptually different families:

- the original LSTR family
- the CondLSTR-parity family implemented on top of the local LSTR training/runtime infrastructure

The practical result is:

- the runtime, dataset loading, config system, evaluator, and remote scripts are from this LSTR repository
- the dense row-wise lane representation, dynamic mask/reg head, and Hungarian row-matching logic are the CondLSTR-style parity work added on top

The main parity model entry is:

- [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)

The original LSTR baseline model entry is:

- [models/LSTR_CULANE.py](/home/alki/projects/LSTR/models/LSTR_CULANE.py)

---

## Part I: Current LSTR Repository

## 1. Configuration and Runtime Skeleton

### 1.1 Global config object

The project uses a global mutable config object:

- [config.py](/home/alki/projects/LSTR/config.py)

Important behavior:

- `Config._configs` stores everything in one dictionary.
- `system_configs.update_config(...)` updates this dictionary from JSON configs.
- `snapshot_name` controls:
  - which Python model module will be imported
  - which checkpoint folder will be used
  - which results folder will be written
- `data_dir` can be overridden by `LSTR_DATA_DIR`.

Important derived paths:

- checkpoints:
  - `./cache/nnet/<snapshot_name>/<snapshot_name>_<iter>.pkl`
- results:
  - `./results/<snapshot_name>/...`

This matters because the whole repo uses config alias names as both:

- model module names
- checkpoint namespaces

That is why seed aliases like `..._seed901` work cleanly.

### 1.2 Training entry

Training entry point:

- [train.py](/home/alki/projects/LSTR/train.py)

What `train.py` does:

1. parses `cfg_file`, `--iter`, `--threads`, `--freeze`
2. loads JSON config from `./config/<cfg>.json`
3. forces `configs["system"]["snapshot_name"] = args.cfg_file`
4. updates `system_configs`
5. seeds Python, NumPy, PyTorch, CUDA
6. enables deterministic cuDNN mode:
   - `benchmark=False`
   - `deterministic=True`
7. creates dataset objects from `db.datasets`
8. starts multiprocessing data prefetch workers
9. builds the network through `NetworkFactory`
10. trains until `max_iter`

Important local modifications that were added during research:

- deterministic seeding
- per-worker seed offset
- `db._data_rng` reset inside workers if the dataset exposes it
- pass-through of extra batch metadata like `image_keys`

### 1.3 Test entry

Test entry point:

- [test.py](/home/alki/projects/LSTR/test.py)

What `test.py` does:

1. loads base config `./config/<cfg>.json`
2. if `--suffix thr04` is given and `./config/<cfg>-thr04.json` exists, it loads that instead
3. still forces `snapshot_name = <cfg>` so checkpoint namespace stays stable
4. seeds PyTorch/NumPy deterministically
5. builds dataset for requested split
6. builds model through `NetworkFactory`
7. loads checkpoint `<iter>`
8. runs test module `test.<db._data>`

This design is why evaluation threshold variations use suffixed configs without changing checkpoint identity.

### 1.4 Network construction

Core factory:

- [nnet/py_factory.py](/home/alki/projects/LSTR/nnet/py_factory.py)

Key behavior:

- it imports `models.<snapshot_name>`
- expects that module to expose:
  - `model(flag=False)`
  - `loss()`
- wraps the model and loss into `Network`
- moves tensors to CUDA in `_move_batch`
- forwards `**kwargs` from sampler to both model and loss

This forwarding is important for parity diagnostics because:

- `image_keys`
- `targets`
- optional force flags

all travel through the standard runtime without hacks.

Checkpoint loading logic:

- exact checkpoint path first
- if not found, glob fallback for `<snapshot_name>_<iter>*.pkl*`

That fallback made remote experiment management more robust.

---

## 2. Dataset and Batch Construction in LSTR

### 2.1 CULane dataset object

Dataset implementation:

- [db/culane.py](/home/alki/projects/LSTR/db/culane.py)

The dataset supports splits:

- `train`
- `val`
- `test`
- `train_100`
- `train_2k`
- `train_5k`
- `train_10k`

Key properties:

- root: `<data_dir>/CULane`
- original image size hard-coded as `1640 x 590`
- caches transformed annotations in `./cache/culane_[split].pkl`

Annotation transformation:

- raw `.lines.txt` lane point files are read
- lanes are converted into LSTR legacy label tensors of shape:
  - `[max_lanes, 1 + 2 + 2 * max_points]`
- each lane stores:
  - class/category
  - lower y
  - upper y
  - normalized x samples
  - normalized y samples

Important behavior:

- lanes are sorted by leftmost x before writing label tensors
- empty images are skipped at extraction time

### 2.2 Batch sampler

CULane sampler:

- [sample/culane.py](/home/alki/projects/LSTR/sample/culane.py)

What `kp_detection()` constructs per batch:

- `xs`
  - image tensor `[B, 3, H, W]`
  - mask tensor `[B, 1, H, W]`
- `ys`
  - duplicated image tensor first
  - then one legacy lane label tensor per image
- `image_keys`
  - original CULane relative image path

Transform pipeline:

- reads image with OpenCV
- converts raw lane lists to ImgAug line strings
- applies augmentation
- reprojects lanes back
- converts them into the legacy label tensor format
- normalizes image with ImageNet-style mean/std

The weird-looking `ys = [images, *gt_lanes]` contract is inherited from the original LSTR codebase. The parity branch later converts this legacy label format into CondLSTR-style dense row targets.

---

## 3. Original LSTR Model Family

### 3.1 Model entry

- [models/LSTR_CULANE.py](/home/alki/projects/LSTR/models/LSTR_CULANE.py)

This file defines:

- a ResNet-like image encoder with configurable block type
- a DETR-like transformer
- a curve/class prediction head
- the original LSTR loss

### 3.2 Backbone and transformer trunk

Underlying implementation:

- [models/py_utils/kp.py](/home/alki/projects/LSTR/models/py_utils/kp.py)

Base flow:

1. image goes through:
   - `conv1`
   - `bn1`
   - `relu`
   - `maxpool`
   - `layer1..layer4`
2. final feature map is projected with `input_proj`
3. positional encoding is built
4. learned query embeddings `self.query_embed` are passed to transformer
5. transformer returns decoder activations `hs`
6. `class_embed`, `specific_embed`, `shared_embed` predict:
   - class logits
   - parametric curve representation

This is classic object-query DETR structure specialized for LSTR’s original polynomial/curve-style lane parameterization.

### 3.3 Original transformer contract

Transformer implementation:

- [models/py_utils/transformer.py](/home/alki/projects/LSTR/models/py_utils/transformer.py)

Originally:

- source feature map is flattened to `HW x B x C`
- learned query embedding is used as decoder query position
- decoder initial target content defaults to:
  - `tgt = query_embed * 0.1`

That `tgt = query_embed * 0.1` detail became important later because it differs from CondLSTR’s transformer setup.

### 3.4 Original LSTR loss

Original loss still lives in:

- [models/py_utils/kp.py](/home/alki/projects/LSTR/models/py_utils/kp.py)

It uses:

- DETR-style matcher
- class loss
- curve regression losses
- auxiliary decoder losses

This original family is not the main research path anymore, but it is still the base runtime skeleton that the CondLSTR-parity branch reuses.

---

## 4. CondLSTR-Parity Family in This Repository

This is the most important part of the current project.

The parity branch takes the LSTR runtime and replaces the original curve head with a CondLSTR-like row-wise dense lane detector.

Main files:

- [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)
- [models/condlstr_parity_head.py](/home/alki/projects/LSTR/models/condlstr_parity_head.py)
- [models/condlstr_dense_matcher.py](/home/alki/projects/LSTR/models/condlstr_dense_matcher.py)
- [models/condlstr_parity_criterion.py](/home/alki/projects/LSTR/models/condlstr_parity_criterion.py)
- [models/condlstr_parity_postprocess.py](/home/alki/projects/LSTR/models/condlstr_parity_postprocess.py)
- [utils/condlstr_parity_targets.py](/home/alki/projects/LSTR/utils/condlstr_parity_targets.py)
- [utils/condlstr_dense_targets.py](/home/alki/projects/LSTR/utils/condlstr_dense_targets.py)

### 4.1 High-level idea

Instead of predicting one polynomial/curve object per query, each query predicts a dense lane representation:

- objectness
- lane class
- dense row mask logits over image width
- dense row regression offsets
- either:
  - scalar lane start/end range
  - or row-wise visibility logits

So the parity model is query-based, but the supervision is row-wise and dense, not box-like and not polynomial-only.

### 4.2 Model construction in `LSTR_CULANE_condlstr_parity_base.py`

The parity model subclasses the original LSTR model:

- `class model(_BaseTransformerModel)`

What it reuses from base LSTR:

- transformer
- positional encoding
- learned main queries
- default ResNet-like backbone path

What it adds:

- optional STDC/Res34 backbone replacement
- CondLSTR-style dynamic head
- configurable decoder target initialization
- optional query relation block
- optional DN lane branch
- optional visibility mode

Important runtime switches:

- `parity_backbone`
  - `lstr`
  - `stdc_res34`
- `dense_range_mode`
  - `range`
  - `visibility`
- `dense_decoder_init_mode`
  - `legacy_query_embed`
  - `learned_target_embed`
- `dense_relation_mode`
  - `none`
  - `self_attn`
- `dn_lane_num_queries`
- `dense_use_coords`

### 4.3 Backbone modes

#### LSTR backbone mode

If `parity_backbone='lstr'`, the model uses the inherited LSTR ResNet-like encoder:

- `conv1`
- `bn1`
- `maxpool`
- `layer1..layer4`

#### STDC/Res34 mode

If `parity_backbone='stdc_res34'`, it uses:

- [models/parity_stdc_res34_backbone.py](/home/alki/projects/LSTR/models/parity_stdc_res34_backbone.py)

This is not a vanilla ResNet34 feature extractor. It is a CondLSTR-style STDC/BiSeNet-like wrapper built around a ResNet34 backbone.

Structure:

1. `ContextPathResNet34`
   - extracts `feat8`, `feat16`, `feat32`
   - uses attention refinement modules on `feat16` and `feat32`
   - upsamples context features back to stride-8
2. `FeatureFusionModule`
   - fuses spatial branch feature and context branch feature
3. final `ConvBNReLU`
   - returns a `256`-channel fused feature map

This branch was the biggest performance jump in the project.

### 4.4 Decoder query initialization in parity model

The parity model added a configurable decoder input path.

There are two modes.

#### `legacy_query_embed`

- query positional tensor is the original learned query embedding
- decoder content tensor is `learned_queries * 0.1`

This reproduces the old LSTR behavior.

#### `learned_target_embed`

- adds `self.decoder_target_embed = nn.Embedding(num_queries, attn_dim)`
- both:
  - decoder content
  - decoder positional tensor

can come from a learned target embedding instead of the original query embedding

This was the `tgtinit` experiment.

Transformer support for this was added by extending:

- [models/py_utils/transformer.py](/home/alki/projects/LSTR/models/py_utils/transformer.py)

The transformer now accepts:

- `decoder_tgt`
- `decoder_query_pos`
- `tgt_mask`

### 4.5 Dynamic parity head

Head implementation:

- [models/condlstr_parity_head.py](/home/alki/projects/LSTR/models/condlstr_parity_head.py)

This head has two pieces.

#### A. Query branch

`QueryBranchHead` predicts per query:

- `pred_object_logits`
- `pred_class_logits`
- `pred_mask_params`
- `pred_reg_params`
- optionally `pred_ranges`
- optionally `pred_row_visibility_logits`

So each query emits not only classification outputs, but also the parameters of two dynamic spatial branches.

#### B. Dynamic spatial branches

`DynamicSpatialBranch` takes:

- a shared feature map `[B, C, H, W]`
- dynamic parameters `[B, Q, P]`

and applies a query-specific 1-layer dynamic linear projection over every spatial position.

There are two parallel branches:

- mask branch
- regression branch

If `use_coords=True`, the head concatenates normalized coordinate maps `(x, y)` to the feature map before dynamic prediction.

This is one of the places where explicit geometry can be injected.

#### C. Multi-layer decoder support

`CondLSTRParityHead.forward(...)` can consume:

- one feature map + one query tensor
- or lists across decoder layers

That is why the parity model can produce:

- final outputs
- `aux_outputs` for earlier decoder layers

### 4.6 Dense target conversion

Legacy labels are converted into row-wise dense targets in two steps.

#### A. Legacy tensor -> lane point lists

- [utils/condlstr_parity_targets.py](/home/alki/projects/LSTR/utils/condlstr_parity_targets.py)

`legacy_label_tensor_to_lane_points(...)`:

- reads the old LSTR label tensor
- extracts valid x/y pairs
- rescales them into image coordinates
- builds per-lane point lists

#### B. Lane point lists -> dense row targets

- [utils/condlstr_dense_targets.py](/home/alki/projects/LSTR/utils/condlstr_dense_targets.py)

`convert_culane_points_to_rowwise_targets(...)` does the real CondLSTR-style conversion:

1. scale lane points from image size into target feature-map size
2. sort points by y
3. interpolate lane x position for every row that lies within the lane span
4. build:
   - `gt_row_rng`
   - `gt_row_loc`
   - `gt_row_reg`
   - `gt_row_loc_mask`
   - `gt_row_reg_mask`
   - `gt_label_obj`
   - `gt_label_cls`
5. sort lanes left-to-right by bottom-row x
6. also emit:
   - `gt_lane_order`
   - `gt_lane_order_norm`

This lane-order emission was later used by the order-prior matcher experiment.

### 4.7 Hungarian matcher

Matcher:

- [models/condlstr_dense_matcher.py](/home/alki/projects/LSTR/models/condlstr_dense_matcher.py)

It solves matching per image using SciPy:

- `linear_sum_assignment`

Inputs expected:

- `pred_object_logits`
- `pred_class_logits`
- either `pred_ranges` or `pred_row_visibility_logits`
- `pred_dense_mask`
- `pred_dense_reg`

Cost terms:

- `cost_object`
- `cost_class`
- `cost_row_location`
- `cost_row_iou`
- `cost_row_reg`
- `cost_row_range`
- optional `cost_order`

Final total cost:

- weighted sum of all these terms

Important behavior:

- dense mask logits are softmaxed across width
- row center is the expected x-coordinate over width
- row IoU is computed by treating each row location as a line segment with width `line_width`
- row regression cost uses full `[Q, M, H, W]` broadcast-style comparison

This matcher is local and standalone, unlike the original LSTR matcher.

### 4.8 Criterion and loss computation

Criterion:

- [models/condlstr_parity_criterion.py](/home/alki/projects/LSTR/models/condlstr_parity_criterion.py)

Wrapper loss module:

- `class loss` inside [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)

Flow:

1. convert legacy batch labels into dense row targets
2. run matcher on final outputs
3. compute:
   - object loss
   - class loss
   - row location loss
   - row IoU penalty
   - row regression loss
   - range loss or visibility BCE
4. add auxiliary losses for decoder intermediate layers

Range mode behavior:

- if `pred_row_visibility_logits` exists:
  - `loss_range = BCEWithLogits(row_visibility, gt_row_loc_mask)`
- else:
  - `loss_range = L1(pred_ranges, gt_row_rng)`

This is why visibility mode is not just a postprocess change; it changes both the prediction head and the loss.

### 4.9 Postprocess

Postprocess:

- [models/condlstr_parity_postprocess.py](/home/alki/projects/LSTR/models/condlstr_parity_postprocess.py)

At inference:

1. softmax object logits, keep foreground probability
2. softmax mask logits across width
3. compute row centers as expected column
4. sample regression offsets at rounded centers
5. add offsets back to centers
6. keep queries above `score_thresh`
7. decode lane points using either:
   - row range
   - row visibility
8. scale from feature map coordinates back to image coordinates

This is what eventually gets written to `.lines.txt` via evaluator.

### 4.10 Test-time routing

Test-time output routing happens in:

- [test/culane.py](/home/alki/projects/LSTR/test/culane.py)

`PostProcess.forward(...)` checks output keys:

- original LSTR path:
  - `pred_logits`, `pred_curves`
- parity path:
  - `pred_object_logits`
  - `pred_dense_mask`
  - `pred_dense_reg`
  - plus `pred_ranges` or `pred_row_visibility_logits`

If `outputs['postprocess_mode'] == 'condlstr_parity'`, it calls:

- `parity_outputs_to_lane_coords(...)`

Then:

- `db/utils/evaluator.py` stores lane point predictions
- `db/culane.py` later writes them into CULane evaluator format

---

## 5. Research Extensions Added to the Current Project

This repository now includes multiple experimental extensions on top of parity.

### 5.1 DN lane branch

Files:

- [models/condlstr_dn_lane.py](/home/alki/projects/LSTR/models/condlstr_dn_lane.py)
- [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)
- [models/condlstr_parity_criterion.py](/home/alki/projects/LSTR/models/condlstr_parity_criterion.py)

What was added:

- an 8D lane summary vector built from GT row-wise targets
- noisy DN queries
- a DN query encoder
- a DN type embedding
- decoder query concatenation
- DN self-attention isolation mask
- direct DN losses without Hungarian matching

Important conceptual point:

- Hungarian matching was not removed
- DN was added on top of Hungarian for the main queries

### 5.2 Visibility mode

Files:

- [models/condlstr_parity_head.py](/home/alki/projects/LSTR/models/condlstr_parity_head.py)
- [models/condlstr_parity_criterion.py](/home/alki/projects/LSTR/models/condlstr_parity_postprocess.py)
- [models/condlstr_parity_postprocess.py](/home/alki/projects/LSTR/models/condlstr_parity_postprocess.py)

What changed:

- replace scalar start/end range with row-wise visibility logits
- use BCE over visible rows
- decode lane points from visible-row thresholding

This helped in the weaker LSTR-backbone regime but failed on STDC.

### 5.3 `tgtinit`

Files:

- [models/py_utils/transformer.py](/home/alki/projects/LSTR/models/py_utils/transformer.py)
- [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)

What changed:

- parity transformer can now use separate decoder content and query position tensors
- added `learned_target_embed` mode

This improved final F1 in the 2k/LSTR-backbone regime, but did not reduce measured assignment flip-rate.

### 5.4 Query relation block

File:

- [models/condlstr_query_relation.py](/home/alki/projects/LSTR/models/condlstr_query_relation.py)

What it is:

- one or more query-to-query self-attention layers after decoder output and before head prediction

Observed behavior:

- harmful in the tested setup

### 5.5 Order prior

Matcher support:

- [models/condlstr_dense_matcher.py](/home/alki/projects/LSTR/models/condlstr_dense_matcher.py)

Target support:

- [utils/condlstr_dense_targets.py](/home/alki/projects/LSTR/utils/condlstr_dense_targets.py)

What it did:

- assigned normalized order indices to GT lanes
- added `cost_order` between query order and lane order

Observed behavior:

- slightly reduced flip-rate
- hurt recall badly
- not usable as-is

### 5.6 Instability diagnostics

Files:

- [probe_dn_instability.py](/home/alki/projects/LSTR/probe_dn_instability.py)
- [analyze_match_diag.py](/home/alki/projects/LSTR/analyze_match_diag.py)

What they do:

- fix a train image panel
- probe multiple checkpoints on exactly the same images
- log per-image assignments and margins
- summarize:
  - flip rate
  - mean query margin
  - mean target margin
  - DN/main foreground score statistics

This instrumentation was critical because it separated:

- training instability
- assignment instability
- final F1 behavior

### 5.7 Existing experiment summary

Detailed experiment history already exists in:

- [condlstr_parity_experiment_log.md](/home/alki/projects/LSTR/condlstr_parity_experiment_log.md)

That file is experiment-focused. This document is code-structure-focused.

---

## Part II: CondLSTR Repository

Path:

- `/home/alki/projects/CondLSTR`

This project is a different codebase with its own training engine, model registry, detector registry, backbone registry, dataset stack, and metrics.

The closest model to what was referenced during this work is:

- [modeling/models/models/lane/cond_lstr_2d_res34.py](/home/alki/projects/CondLSTR/modeling/models/models/lane/cond_lstr_2d_res34.py)

## 6. CondLSTR Runtime Skeleton

### 6.1 Train entry

- [tools/train.py](/home/alki/projects/CondLSTR/tools/train.py)

CondLSTR training is more generic and framework-like than LSTR.

What it does:

1. parses dataset, image size, arch, epochs, optimizer, distributed flags
2. seeds if requested
3. enables:
   - `cudnn.benchmark = True`
   - `cudnn.deterministic = False`
4. builds transforms through `data.transforms.create(...)`
5. builds datasets through `data.datasets.create(...)`
6. builds dataloaders through `data.dataloaders.create(...)`
7. builds model through `modeling.models.create(...)`
8. builds optimizer and scheduler
9. wraps model in DDP if needed
10. uses:
   - `Trainer`
   - `Evaluator`
   - `Tester`

Unlike the current LSTR repo, CondLSTR is epoch-based rather than iteration-based at the top level.

### 6.2 Test entry

- [tools/test.py](/home/alki/projects/CondLSTR/tools/test.py)

Test flow:

1. build dataset and transforms
2. build model
3. load `model_best.pth.tar` or explicit checkpoint
4. optionally compute FLOPs/JIT/ONNX/TRT
5. run either:
   - evaluator
   - tester

### 6.3 Engine wrappers

- [engine/trainer.py](/home/alki/projects/CondLSTR/engine/trainer.py)
- [engine/evaluator.py](/home/alki/projects/CondLSTR/engine/evaluator.py)
- [engine/tester.py](/home/alki/projects/CondLSTR/engine/tester.py)

These files are thin orchestration wrappers:

- `Trainer`
  - iterates dataloader
  - moves batch to CUDA
  - calls model
  - assumes model returns a `loss_dict` in training mode
  - saves checkpoint every `save_steps`
- `Evaluator`
  - calls model in eval mode
  - forwards predictions to metric object
- `Tester`
  - calls model in eval mode
  - forwards predictions to an inference formatter
  - saves `results.pkl`

So in CondLSTR, the detector itself owns more of the training/eval branching.

---

## 7. CondLSTR Model Construction

### 7.1 Registry structure

Model registry:

- [modeling/models/models/__init__.py](/home/alki/projects/CondLSTR/modeling/models/models/__init__.py)

Detector registry:

- [modeling/models/detectors/__init__.py](/home/alki/projects/CondLSTR/modeling/models/detectors/__init__.py)

Backbone registry:

- [modeling/models/backbones/__init__.py](/home/alki/projects/CondLSTR/modeling/models/backbones/__init__.py)

Important point:

CondLSTR treats:

- image backbone
- transformer backbone
- detector head

as separate registered components.

### 7.2 Main lane model: `CondLSTR2DRes34`

- [modeling/models/models/lane/cond_lstr_2d_res34.py](/home/alki/projects/CondLSTR/modeling/models/models/lane/cond_lstr_2d_res34.py)

This file wires three pieces together:

1. `img_backbone`
   - `STDCNet(backbone='ResNet34')`
2. `det_backbone`
   - `Transformer`
3. `detector`
   - `CondLSTR2D`

This file is the clearest statement of the CondLSTR architecture.

### 7.3 Image backbone in CondLSTR

Configured as:

- `name='STDCNet'`
- `backbone='ResNet34'`
- output channel count expected by detector backbone: `256`

Implementation root:

- [modeling/models/backbones/stdcnet/stdcnet.py](/home/alki/projects/CondLSTR/modeling/models/backbones/stdcnet/stdcnet.py)

This file defines:

- STDCNet variants
- bottleneck blocks
- convolution wrappers

The exact internal wrapper that produces the lane features is abstracted behind the backbone registry, but from the model config and later tensor usage we know the lane model expects a `256`-channel image feature map.

### 7.4 Transformer backbone in CondLSTR

- [modeling/models/backbones/transformer/transformer.py](/home/alki/projects/CondLSTR/modeling/models/backbones/transformer/transformer.py)

This transformer is more flexible than the local LSTR transformer.

Key config in `cond_lstr_2d_res34.py`:

- `src_shape=(24, 42)`
- `tgt_shape=(20, 1)`
- `d_model=256`
- `num_encoder_layers=2`
- `num_decoder_layers=4`
- `src_pos_encode='sine'`
- `tgt_pos_encode='learned'`

Important design detail:

- target side is not a flat list of learned query vectors in the LSTR style
- target positions are a small learned spatial grid of shape `(20, 1)`

Critical decoder behavior:

- if `tgt` is `None`, then:
  - `tgt = tgt_pos_embed`

So decoder target content is directly initialized from the learned target positional embeddings.

This is one of the main differences from the original LSTR transformer, where decoder target content defaulted to `query_embed * 0.1`.

Encoder output:

- `forward_encoder(...)` returns:
  - `memory`
  - `src_mask`
  - `src_pos_embed`
  - `src_shape`

Decoder output:

- `forward_decoder(...)` returns per-layer target-side feature tensors
- the lane model later squeezes the width dimension because `tgt_shape=(20, 1)`

So CondLSTR decoder effectively outputs:

- 20 lane query slots
- each with a `256`-D feature vector

### 7.5 Detector head: `CondLSTR2D`

- [modeling/models/detectors/lane/cond_lstr_2d/cond_lstr_2d.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/cond_lstr_2d.py)

This file defines the real lane detector logic on top of:

- encoder feature map
- decoder query features

The model uses:

- one dynamic mask head
- one dynamic regression head
- one query MLP head (`CtnetHead`)

#### Detector outputs

Per query, `CtnetHead` predicts:

- `logits`
  - objectness with 2 classes
- `attris`
  - lane attribute/class logits
- `ranges`
  - normalized row start/end
- `params`
  - dynamic parameters for both mask and regression heads

Then:

- `params` is split into:
  - `mask_params`
  - `reg_params`

Those parameter tensors are used by:

- `mask_head`
- `reg_head`

to produce:

- `masks`
- `regs`

This is the core CondLSTR idea:

- decoder query features do not directly output lane points
- they generate dynamic filters that are applied to a shared encoder feature map

#### Important constraint

Inside `DynamicMaskHead.parse_dynamic_params(...)` there is:

- `assert num_layers == 1`

So the released CondLSTR dynamic head is intentionally restricted to a single dynamic layer.

#### Coordinate usage

The detector config sets:

- `disable_coords=True`

That means the dynamic head does **not** concatenate explicit `(x, y)` coordinate channels to the feature map.

This is a very important architectural choice, and from a lane-geometry perspective it is debatable.

### 7.6 CondLSTR training/eval branching

Inside `CondLSTR2D.forward(...)`:

- if training:
  - it returns `loss_dict`
- else:
  - it runs postprocess and returns prediction dict

So unlike the local LSTR parity branch, the detector itself contains both:

- training loss path
- inference path

This is why CondLSTR’s external `Trainer` and `Tester` can stay generic.

---

## 8. CondLSTR Loss and Matching

### 8.1 Loss preprocessing

- [modeling/models/detectors/lane/cond_lstr_2d/loss.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/loss.py)

`CondLSTR2DLoss.preprocess(...)` converts GT masks into row-wise lane targets.

For each lane mask it builds:

- `gt_row_rng`
- `gt_row_loc`
- `gt_row_reg`
- `gt_row_loc_mask`
- `gt_row_reg_mask`
- `gt_label_obj`
- `gt_label_cls`

This target format is the conceptual parent of the parity target format in the current repo.

Important detail:

- if `gt_labels` is missing, it fabricates zeros
- rows are derived from binary lane masks, not from the LSTR legacy label tensor

### 8.2 SetCriterion

Same file:

- [modeling/models/detectors/lane/cond_lstr_2d/loss.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/loss.py)

`SetCriterion` computes:

- `loss_obj`
- `loss_cls`
- `loss_loc`
- `loss_reg`
- `loss_rng`

Location loss structure:

- row L1 location loss
- row IoU loss multiplied by `2.0`

Regression loss:

- L1 over row regression volume, masked

Range loss:

- L1 over normalized `(row_start, row_end)`

This is the direct counterpart of the parity criterion in the current project, except the parity branch later generalized it to support visibility mode and DN mode.

### 8.3 Hungarian matcher in CondLSTR

- [modeling/models/detectors/lane/cond_lstr_2d/matcher.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/matcher.py)

CondLSTR still uses plain Hungarian matching.

Cost terms:

- object cost
- class cost
- location cost
- regression cost
- range cost

Location cost itself is:

- row L1 location
- row IoU penalty

So conceptually the parity matcher in the current project is not alien to CondLSTR; it is a reimplementation and extension of the same matching idea in the local runtime.

Important implication:

- CondLSTR does not avoid Hungarian ambiguity by removing Hungarian
- it still relies on Hungarian assignment

This matters because it means assignment fuzziness is not something CondLSTR eliminated at the algorithmic level.

---

## 9. CondLSTR Postprocess

- [modeling/models/detectors/lane/cond_lstr_2d/postprocess.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/postprocess.py)

Inference flow:

1. object scores from `logits_obj.softmax(...)[..., 0]`
2. class scores from `logits_cls.softmax(...)`
3. row mask logits are softmaxed across width
4. expected row center x is computed
5. regression offsets are gathered at rounded row centers
6. final row x positions are center + reg offset
7. range endpoints are converted into row indices
8. rows between start and end are selected
9. points are scaled back using `mask_downscale`

Output:

- `lane_points`
- `lane_attris`
- `lane_scores`

This is almost the exact conceptual pattern later reproduced in the parity postprocess, except the parity branch added a visibility-based alternative decode path.

---

## Part III: Direct Comparison

## 10. What Is Structurally The Same

Both CondLSTR and the parity branch ultimately do this:

1. extract image feature map
2. use transformer decoder queries to represent lane instances
3. predict per-query dynamic parameters
4. apply dynamic heads to a shared feature map
5. convert dynamic outputs into row-wise lane coordinates
6. use Hungarian matching with row-wise costs during training

So the parity branch is not a random imitation. It is structurally aligned with CondLSTR.

## 11. What The Current Project Added Beyond CondLSTR

The current parity branch added several things that are not in the base CondLSTR lane path.

### 11.1 Backbone switching inside the same runtime

The parity branch can switch between:

- inherited LSTR backbone
- STDC/Res34 CondLSTR-style fused backbone

That made direct matched experiments possible.

### 11.2 Visibility mode

CondLSTR uses scalar `ranges`.

The parity branch can switch to:

- `pred_row_visibility_logits`
- BCE visibility supervision
- visibility-based postprocess decoding

### 11.3 DN lane branch

CondLSTR has no DN-DETR-like lane denoising branch in this path.

The parity branch added:

- DN query generation
- DN query encoder
- decoder integration
- DN direct losses
- attention mask isolation between main and DN queries

### 11.4 Decoder target init experiments

Parity branch can explicitly choose:

- old LSTR query init
- learned target embedding init

This made it possible to test how much decoder slot semantics mattered.

### 11.5 Query relation block

Parity branch added an optional lightweight self-attention block over query features after the decoder.

### 11.6 Order prior

Parity branch added lane-order targets and matcher-side order cost.

### 11.7 Instability instrumentation

Parity branch added:

- periodic JSONL match diagnostics
- fixed-panel probing
- assignment flip analysis

CondLSTR codebase does not ship with this level of assignment-instability instrumentation in the lane detector path.

## 12. What CondLSTR Still Does Not Provide

From code inspection, CondLSTR still has several omissions if the goal is a lane-specific query detector rather than a generic DETR-style detector adapted to lanes.

### 12.1 No true lane-geometric query anchor

CondLSTR has:

- learned target positional embeddings on the decoder side

But that is not the same as:

- lane-aware geometric query anchors
- bottom-x / start-row / end-row style query priors

So the decoder slots are regularized, but they are not explicitly grounded in lane geometry.

### 12.2 Explicit coordinates are disabled in the dynamic head

CondLSTR lane detector sets:

- `disable_coords=True`

So the dynamic head does not receive explicit normalized `(x, y)` channels.

For a geometry-heavy task like lanes, this is a strong design choice and arguably an omission.

### 12.3 Dynamic head is intentionally single-layer

`num_layers == 1` is asserted.

So the released CondLSTR lane dynamic head is shallow by design.

### 12.4 Still Hungarian

CondLSTR still depends on Hungarian matching.

So any claim that CondLSTR fundamentally removed query matching fuzziness would not be supported by the code.

## 13. What The Current Project Still Does Not Have

Even after all parity work, the current project still does not yet have a single, strong, lane-specific new mechanism that clearly dominates as the central research contribution.

Codebase strengths right now:

- strong experimental runtime
- strong parity implementation
- diagnostics
- multiple tested ablations
- strong STDC + 5k performance branch

But if the goal is a strong method paper, the missing thing is still:

- one clean lane-specific mechanism that is not just:
  - backbone swap
  - data scaling
  - DN adaptation

Examples of candidate missing mechanisms:

- lane-aware spatial query anchors
- reference-point or anchor-based lane query initialization
- row-sequential lane decoder/head
- lane-structured state-space decoder

---

## Part IV: Concrete Current State Of This Repository

## 14. What The Repository Currently Supports

The current LSTR repository can now run the following parity-style families:

- original LSTR backbone parity
- STDC/Res34 parity
- DN parity
- visibility parity
- `tgtinit`
- relation block
- order prior
- matched seed aliases
- 2k and 5k dataset branches

Important helper scripts and analysis utilities:

- [make_seed_alias.py](/home/alki/projects/LSTR/make_seed_alias.py)
- [probe_dn_instability.py](/home/alki/projects/LSTR/probe_dn_instability.py)
- [analyze_match_diag.py](/home/alki/projects/LSTR/analyze_match_diag.py)

Important experiment log:

- [condlstr_parity_experiment_log.md](/home/alki/projects/LSTR/condlstr_parity_experiment_log.md)

## 15. Practical Current Best Branch

From the work already done in this repository:

- the strongest current performance recipe is the STDC branch with more data
- `5k STDC+DN` is currently the best measured branch among the tested parity families
- by contrast:
  - `visibility` was regime-dependent
  - `tgtinit` helped in weaker regimes but did not survive STDC cleanly
  - `relation` hurt
  - `order prior` reduced instability slightly but damaged recall

This means the repo is no longer just an LSTR fork. It is now a functioning experimental lane-detection workbench centered around a CondLSTR-style dense parity branch.

---

## Final Reading Shortcut

If someone wants the shortest possible path through both codebases, these are the minimum files to read.

### Current LSTR repo

- [config.py](/home/alki/projects/LSTR/config.py)
- [train.py](/home/alki/projects/LSTR/train.py)
- [test.py](/home/alki/projects/LSTR/test.py)
- [sample/culane.py](/home/alki/projects/LSTR/sample/culane.py)
- [db/culane.py](/home/alki/projects/LSTR/db/culane.py)
- [models/LSTR_CULANE.py](/home/alki/projects/LSTR/models/LSTR_CULANE.py)
- [models/py_utils/transformer.py](/home/alki/projects/LSTR/models/py_utils/transformer.py)
- [models/LSTR_CULANE_condlstr_parity_base.py](/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py)
- [models/condlstr_parity_head.py](/home/alki/projects/LSTR/models/condlstr_parity_head.py)
- [models/condlstr_dense_matcher.py](/home/alki/projects/LSTR/models/condlstr_dense_matcher.py)
- [models/condlstr_parity_criterion.py](/home/alki/projects/LSTR/models/condlstr_parity_criterion.py)
- [models/condlstr_parity_postprocess.py](/home/alki/projects/LSTR/models/condlstr_parity_postprocess.py)
- [utils/condlstr_dense_targets.py](/home/alki/projects/LSTR/utils/condlstr_dense_targets.py)

### CondLSTR repo

- [tools/train.py](/home/alki/projects/CondLSTR/tools/train.py)
- [tools/test.py](/home/alki/projects/CondLSTR/tools/test.py)
- [modeling/models/models/lane/cond_lstr_2d_res34.py](/home/alki/projects/CondLSTR/modeling/models/models/lane/cond_lstr_2d_res34.py)
- [modeling/models/backbones/transformer/transformer.py](/home/alki/projects/CondLSTR/modeling/models/backbones/transformer/transformer.py)
- [modeling/models/detectors/lane/cond_lstr_2d/cond_lstr_2d.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/cond_lstr_2d.py)
- [modeling/models/detectors/lane/cond_lstr_2d/loss.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/loss.py)
- [modeling/models/detectors/lane/cond_lstr_2d/matcher.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/matcher.py)
- [modeling/models/detectors/lane/cond_lstr_2d/postprocess.py](/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/postprocess.py)

If those files are understood, the rest of both projects becomes much easier to navigate.
