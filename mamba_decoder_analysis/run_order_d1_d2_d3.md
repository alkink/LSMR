# DENEY 1 / 2 / 3 Çalıştırma Sırası (5k, WSL terminalde doğrudan)

Bu sürüm 5k split için hazırlanmıştır (`train_split=train_5k`).
Komutlar doğrudan açık WSL terminalinde çalıştırılır (wsl.exe sarmalı yok).

## Önkoşul

Bir kez çalıştır:

```bash
source /home/alki/miniconda3/etc/profile.d/conda.sh && conda activate clrernet
cd /home/alki/projects/LSTR
```

Notlar:
- Paket import sorununu önlemek için analiz scriptleri [python -m](mamba_decoder_analysis/run_order_d1_d2_d3.md) ile çağrılır.
- [train.py](train.py:27) / [test.py](test.py:23) positional config arg bekler.
- 5k konfigürasyonlarında `max_iter=31250` ve `snapshot=5000` olduğundan pratikte son mevcut checkpoint `30000` olur.

## 0) Hızlı sözdizimi kontrolü

```bash
python3 -m py_compile \
  mamba_decoder_analysis/decoder_variants.py \
  mamba_decoder_analysis/bolum5_uyumluluk.py \
  mamba_decoder_analysis/bolum6_overfit.py \
  mamba_decoder_analysis/deney1_bolum1_mamba2_izole.py \
  mamba_decoder_analysis/deney1_bolum2_variant_d.py \
  mamba_decoder_analysis/deney1_bolum3_uyumluluk.py \
  mamba_decoder_analysis/deney1_bolum4_overfit.py \
  mamba_decoder_analysis/deney2_bolum1_hybrid_izole.py \
  mamba_decoder_analysis/deney2_bolum2_pipeline.py \
  mamba_decoder_analysis/deney2_bolum3_overfit.py \
  vssd_encoder_analysis/bolum0_vssd_kurulum.py \
  vssd_encoder_analysis/bolum1_encoder_kontrat.py \
  vssd_encoder_analysis/bolum2_vssd_izole.py \
  vssd_encoder_analysis/bolum3_encoder_swap.py \
  vssd_encoder_analysis/bolum4_overfit.py \
  models/LSTR_CULANE_2k_mamba_dec_common.py \
  models/LSTR_CULANE_5k_mamba_dec_b.py \
  models/LSTR_CULANE_5k_mamba_dec_d.py \
  models/LSTR_CULANE_5k_mamba_dec_e.py \
  models/LSTR_CULANE_5k_mamba_vssd.py
```

## 1) DENEY 1 — Mamba2 Decoder (D)

```bash
python3 -m mamba_decoder_analysis.deney1_bolum1_mamba2_izole --batch 2 --d-model 32 --d-state 64 --d-conv 4 --headdim 32 --seed 0
python3 -m mamba_decoder_analysis.deney1_bolum2_variant_d --batch 2 --d-model 32 --d-state 64 --d-conv 4 --headdim 32 --num-queries 7 --seed 0
python3 -m mamba_decoder_analysis.deney1_bolum3_uyumluluk --cfg LSTR_CULANE_5k_mamba_dec_b --d-state 64 --d-conv 4 --seed 0
python3 -m mamba_decoder_analysis.deney1_bolum4_overfit --cfg LSTR_CULANE_5k_mamba_dec_b --epochs 80 --lr 1e-4 --d-state 64 --d-conv 4 --seed 0
```

Eğitim + test:

```bash
python3 train.py LSTR_CULANE_5k_mamba_dec_d
python3 test.py LSTR_CULANE_5k_mamba_dec_d --modality eval --split testing --testiter 30000 --batch 1
```

## 2) DENEY 2 — Hybrid Decoder (E)

```bash
python3 -m mamba_decoder_analysis.deney2_bolum1_hybrid_izole --batch 2 --d-model 32 --d-state 16 --d-conv 4 --num-heads 2 --dim-feedforward 128 --num-queries 7 --seed 0
python3 -m mamba_decoder_analysis.deney2_bolum2_pipeline --cfg LSTR_CULANE_5k_mamba_dec_b --d-state 16 --d-conv 4 --seed 0
python3 -m mamba_decoder_analysis.deney2_bolum3_overfit --cfg LSTR_CULANE_5k_mamba_dec_b --epochs 80 --lr 1e-4 --d-state 16 --d-conv 4 --seed 0
```

Eğitim + test:

```bash
python3 train.py LSTR_CULANE_5k_mamba_dec_e
python3 test.py LSTR_CULANE_5k_mamba_dec_e --modality eval --split testing --testiter 30000 --batch 1
```

## 3) DENEY 3 — VSSD Encoder

```bash
python3 -m vssd_encoder_analysis.bolum0_vssd_kurulum --batch 2 --seq 260 --d-model 32 --d-state 64 --headdim 32
python3 -m vssd_encoder_analysis.bolum1_encoder_kontrat --cfg LSTR_CULANE_5k_mamba --batch 2 --seed 0
python3 -m vssd_encoder_analysis.bolum2_vssd_izole --batch 2 --seq 260 --d-model 32 --d-state 64 --headdim 32 --seed 0
python3 -m vssd_encoder_analysis.bolum3_encoder_swap --base-cfg LSTR_CULANE_5k_mamba --vssd-cfg LSTR_CULANE_5k_mamba_vssd --seed 0
python3 -m vssd_encoder_analysis.bolum4_overfit --base-cfg LSTR_CULANE_5k_mamba --vssd-cfg LSTR_CULANE_5k_mamba_vssd --epochs 80 --lr 1e-4 --seed 0
```

Eğitim + test:

```bash
python3 train.py LSTR_CULANE_5k_mamba_vssd
python3 test.py LSTR_CULANE_5k_mamba_vssd --modality eval --split testing --testiter 30000 --batch 1
```

