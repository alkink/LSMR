# CondLSTR Parity + Mamba Execution Plan

Tarih: 2026-03-21

## 1. Ana Hedef

Ana hedef:

`CondLSTR recipe parity baseline kur -> sonra sadece encoder'i Mamba ile degistir -> official CULane eval ile karsilastir`

Bu planda iki sey birbirine karistirilmayacak:
- `Parity implementation`
- `Research ablation`

Sira sabit:
1. Reference'a yakin baseline
2. Baseline'in dogrulanmasi
3. Sadece encoder swap
4. Sonra controlled ablation

## 2. Stratejik Karar

Mevcut `LSTR_CULANE_2k_mamba_dense` hattı ana parity branch olmayacak.

Bu hat:
- exploratory
- minimal adapter
- debugging icin faydali
- ama recipe parity icin uygun degil

Yeni ana hat acilacak:

`CondLSTR-style parity branch`

Bu dalin felsefesi:
- Mümkün olan her yerde recipe parity
- Yalnızca gerekli noktalarda LSTR repo entegrasyon adaptasyonu
- Mamba degisikligi ancak parity baseline calisir hale geldikten sonra

## 3. Yeni Deney Hattı Yapısı

Onerilen yeni model/config isimleri:

1. `LSTR_CULANE_condlstr_parity_base`
- Transformer encoder + CondLSTR parity recipe
- Bu referans baseline olacak

2. `LSTR_CULANE_condlstr_parity_mamba`
- Yukaridaki baseline ile ayni
- Sadece encoder tarafi Mamba

3. Opsiyonel debug varyantlari
- `LSTR_CULANE_condlstr_parity_base_smoke`
- `LSTR_CULANE_condlstr_parity_mamba_smoke`

Bu isimlendirme bilincli secildi:
- `dense` kelimesinden kaciniyoruz
- `parity` ile hedefi acik sabitliyoruz
- `base` ve `mamba` arasinda yalnizca encoder farki olacak

## 4. Fazlar

### Faz 0: Mimari Donma

Bu fazda hedef:
- Recipe hedefini netlestirmek
- Yeni branch icin minimum parity checklist'i dondurmak

Donacak parity maddeleri:
- `20 query`
- `4 decoder layer`
- `attn_dim = 256`
- decoder target-grid query semantics'e daha yakin kurulum
- fused spatial feature source
- `disable_coords=True`
- `range sigmoid`
- `mask_downscale=1`
- expectation decode
- `row_iou` aktif loc loss
- intermediate decoder supervision
- official CULane evaluator

Stop criteria:
- Bu checklist yazili hale gelmeden implementasyona gecilmez

### Faz 1: CondLSTR-Parity Baseline Kurulumu

Bu fazda hedef:
- Mevcut repo icinde transformer encoder'li parity baseline kurmak
- Mamba hic karistirilmadan reference davranisina yaklasmak

Ana teslim:
- `LSTR_CULANE_condlstr_parity_base.py`
- `config/LSTR_CULANE_condlstr_parity_base.json`

Bu fazin alt paketleri:

1. Query/decoder parity
2. Spatial feature parity
3. Target parity
4. Loss parity
5. Decode parity

Stop criteria:
- Tek ornek overfit
- smoke train
- kisa train
- official CULane eval

### Faz 2: Mamba Encoder Swap

Bu fazda hedef:
- Faz 1 baseline aynen korunurken encoder tarafini Mamba ile degistirmek

Ana teslim:
- `LSTR_CULANE_condlstr_parity_mamba.py`
- `config/LSTR_CULANE_condlstr_parity_mamba.json`

Degismemesi gerekenler:
- query count
- decoder depth
- hidden dim
- target contract
- matcher
- loss
- postprocess
- eval path

Tek degisen:
- encoder implementation

Stop criteria:
- Forward parity
- one-image overfit
- smoke train
- short/full training
- official CULane compare against parity baseline

### Faz 3: Controlled Ablation

Bu faz Faz 2 bitmeden acilmayacak.

Ablation eksenleri:
- transformer encoder vs mamba encoder
- coords off vs on
- native-mask-like target vs bridge target
- final-layer-only vs intermediate supervision
- fused source variantlari

Bu fazda knob tuning yapilabilir, ama ancak parity baseline ve mamba swap sonucundan sonra.

## 5. Dosya Bazlı İş Planı

### 5.1 Yeni Model Entry Point'ler

Yeni dosyalar:
- `/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base.py`
- `/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_mamba.py`
- `/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_base_smoke.py`
- `/home/alki/projects/LSTR/models/LSTR_CULANE_condlstr_parity_mamba_smoke.py`

Sorumluluk:
- Model assembly
- feature selection
- query feature contract
- output contract
- trainer ile uyum

Not:
- `LSTR_CULANE_2k_mamba_dense.py` referans olarak okunacak
- ama parity hattinin ana kaynagi olmayacak

### 5.2 Yeni Config'ler

Yeni dosyalar:
- `/home/alki/projects/LSTR/config/LSTR_CULANE_condlstr_parity_base.json`
- `/home/alki/projects/LSTR/config/LSTR_CULANE_condlstr_parity_mamba.json`
- opsiyonel smoke config'ler

Config hedefleri:
- `num_queries=20`
- `dec_layers=4`
- `attn_dim=256`
- parity loss agirliklari
- parity eos
- `dense_use_coords=false`
- `mask_downscale=1`

Not:
- Split secimi debug icin `2k` olabilir
- karar verdiren kosular `10k/full` olmali

### 5.3 Head Implementasyonu

Muhtemel dosya:
- mevcut `/home/alki/projects/LSTR/models/dynamic_lane_head.py` ya genisletilecek
- ya da yeni parity odakli head dosyasi acilacak

Oneri:
- Yeni dosya ac:
  - `/home/alki/projects/LSTR/models/condlstr_parity_head.py`

Sebep:
- Exploratory head ile parity head karismasin
- Query contract, range sigmoid, per-layer support, coords behavior daha temiz tutulur

Parity icin gerekli ozellikler:
- query branch `object/class/range/params`
- mask/reg icin ayri dynamic branch
- `disable_coords=True` default
- range output'a dogrudan `sigmoid`
- per-layer input listesini native destekle

### 5.4 Target Pipeline

En kritik karar burada.

Iki opsiyon:

Opsiyon A:
- Legacy bridge korunur
- Ama native GT mask'e en yakin ara temsil uretilir

Opsiyon B:
- Training pipeline'da lane pointlerden dogrudan raster lane mask uretilir
- CondLSTR preprocess'e daha yakin yol kurulur

Oneri:
- Faz 1 icin `A ile basla`, ama kodu `B`ye acik tasarla

Yeni dosya onerisi:
- `/home/alki/projects/LSTR/utils/condlstr_parity_targets.py`

Bu dosya su modlari desteklemeli:
- `from_legacy_label_tensor`
- `from_lane_points`
- opsiyonel `from_lane_masks`

Hedef:
- Tek bir target contract
- Ust katman criterion bunun nasil olustuguyla ilgilenmesin

### 5.5 Loss

Yeni dosya onerisi:
- `/home/alki/projects/LSTR/models/condlstr_parity_criterion.py`

Sebep:
- Mevcut `condlstr_dense_criterion.py` parity disi kararlar tasiyor
- `loss_dense_mask` aktif
- `row_iou` opsiyonel kapatilabiliyor

Parity criterion hedefleri:
- `loss_object`
- `loss_class`
- `loss_loc = row_l1 + 2 * row_iou`
- `loss_reg`
- `loss_range`
- no extra dense-mask CE by default
- intermediate decoder supervision support

### 5.6 Matcher

Mevcut dosya:
- `/home/alki/projects/LSTR/models/condlstr_dense_matcher.py`

Bu dosya buyuk olasilikla tekrar kullanilabilir.

Ama parity icin karar:
- yeni criterion ile birlikte parity wrapper ac
- gerekirse yeni dosya:
  - `/home/alki/projects/LSTR/models/condlstr_parity_matcher.py`

Neden:
- Gelecekte exploratory ve parity cost logic ayrilsin

### 5.7 Postprocess

Yeni dosya onerisi:
- `/home/alki/projects/LSTR/models/condlstr_parity_postprocess.py`

Gerekli davranis:
- object score `softmax[...,0]`
- row center expectation
- row center round
- rounded index uzerinden regression gather
- `row_loc = rounded + reg`
- normalized range decode
- `mask_downscale=1`

Not:
- Bu decode resmi evaluator yoluna baglanacak

### 5.8 Eval Dispatch

Guncellenecek dosya:
- `/home/alki/projects/LSTR/test/culane.py`

Yapilacak is:
- parity output contract icin ayri dispatch
- threshold'un configten okunmasi
- hardcoded `0.5` kaldirilmasi

Hedef:
- inference semantigi model recipe ile uyumlu olsun

## 6. Uygulama Sırası

En guvenli uygulama sirası:

1. parity config iskeletini olustur
2. parity head dosyasini yaz
3. parity postprocess'i yaz
4. parity criterion/matcher katmanini yaz
5. parity model entrypoint'i kur
6. eval dispatch bagla
7. one-image overfit
8. smoke train
9. short train
10. full train
11. sonra mamba encoder swap

Sebep:
- Decode/loss/head contract erken kapanirsa debugging maliyeti duser
- Once transformer parity baseline kurmak gerekir

## 7. Dogrulama Protokolu

Her fazda ayni sira:

1. Static checks
- `py_compile`
- hedefli unit test

2. Structural forward check
- output key set
- tensor shape check
- finite value check

3. One-image overfit
- loss dramatik bicimde dusmeli

4. 40-100 iter smoke
- NaN yok
- loss trendi asagi gitmeli

5. Short train on `2k`
- sadece debugging sinyali

6. Real eval on `10k/full`
- karar verdiren sinyal

7. Score distribution analizi
- foreground/background calibration
- query basina keep oranlari

## 8. Stop Criteria

### Faz 1 stop criteria

Asagidakiler olmadan Faz 2 yok:
- parity baseline forward saglam
- overfit gecti
- smoke train gecti
- short eval mantikli
- decode/loss mismatch kalmadi

### Faz 2 stop criteria

Asagidakiler olmadan Faz 3 yok:
- Mamba swap yapildi
- shape/contract parity korunuyor
- overfit gecti
- smoke train gecti
- official eval sonucu baseline ile anlamli kiyaslanabilir

## 9. Riskler

1. Query semantics parity tam saglanamayabilir
- LSTR repo'nin orijinal query mantigi ile CondLSTR target-grid query mantigi ayrik

2. Native mask supervision eksikligi limit olabilir
- legacy bridge parity'yi tavanlayabilir

3. BiSeNet fused source yerine daha zayif source secilirse parity baseline dusuk kalabilir

4. Mamba swap parity kurulmadan yapilirsa sonuclar yorumlanamaz

5. `2k` debug split ile alinan sonuclar yapisal karar icin yeterli olmayabilir

## 10. Calisma Kurallari

Bu parity hattinda asagidaki kurallar sabit:

1. Aynı anda birden fazla recipe-level degisken oynanmayacak.
2. Mamba swap parity baseline bitmeden yapilmayacak.
3. `2k` debug icin, `10k/full` karar icin kullanilacak.
4. Threshold tuning ana cozum olarak kullanilmayacak.
5. Exploratory `dense` branch ile parity branch karistirilmeyacak.

## 11. Ilk Uygulanacak Somut Adımlar

Bir sonraki coding turunda yapilacak ilk somut adimlar:

1. `LSTR_CULANE_condlstr_parity_base.json` taslagini ac
2. `condlstr_parity_head.py` olustur
3. `condlstr_parity_postprocess.py` olustur
4. `condlstr_parity_criterion.py` taslagini olustur
5. `LSTR_CULANE_condlstr_parity_base.py` entrypoint'ini kur

Bu ilk turda amac:
- tam train degil
- output contract'i dogru calisan parity baseline forward path

## 12. En Kisa Karar Cumlesi

Bu planin ozet karari su:

`Yeni ana hat: once CondLSTR parity baseline, sonra yalnizca encoder Mamba swap.`
