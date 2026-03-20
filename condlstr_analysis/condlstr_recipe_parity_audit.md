# CondLSTR Recipe Parity Audit

Tarih: 2026-03-20

Amac:
- `CondLSTR` kod tabaninda lane `head/target/matcher/loss/postprocess` zincirini kod seviyesinde haritalamak.
- Bu zinciri mevcut `LSTR + Mamba + dense head` portumuzla birebir karsilastirmak.
- "CondLSTR-benzeri" ile "CondLSTR parity" arasindaki farki netlestirmek.

Incelenen ana kaynak:
- `/home/alki/projects/CondLSTR/modeling/models/models/lane/cond_lstr_2d_res18.py`
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/cond_lstr_2d.py`
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/loss.py`
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/matcher.py`
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/postprocess.py`
- `/home/alki/projects/CondLSTR/modeling/models/backbones/transformer/transformer.py`
- `/home/alki/projects/CondLSTR/modeling/models/backbones/stdcnet/__init__.py`
- `/home/alki/projects/CondLSTR/modeling/models/backbones/stdcnet/model_stages.py`
- `/home/alki/projects/CondLSTR/data/transforms/lane/transforms.py`
- `/home/alki/projects/CondLSTR/data/transforms/lane/utils.py`
- `/home/alki/projects/CondLSTR/modeling/inferences/lane/lane_det_2d.py`
- `/home/alki/projects/CondLSTR/modeling/metrics/lane/lane_det_2d.py`

Karsilastirma icin bakilan yerel dosyalar:
- `/home/alki/projects/LSTR/models/LSTR_CULANE_2k_mamba_dense.py`
- `/home/alki/projects/LSTR/models/dynamic_lane_head.py`
- `/home/alki/projects/LSTR/models/condlstr_dense_matcher.py`
- `/home/alki/projects/LSTR/models/condlstr_dense_criterion.py`
- `/home/alki/projects/LSTR/models/condlstr_dense_postprocess.py`
- `/home/alki/projects/LSTR/utils/condlstr_dense_targets.py`
- `/home/alki/projects/LSTR/config/LSTR_CULANE_2k_mamba_dense.json`
- `/home/alki/projects/LSTR/test/culane.py`

## 1. Kisacik Sonuc

Tek cumlelik yargi:

`Bizim branch CondLSTR'nin fikir ailesinde, ama recipe parity'de degil.`

Bu yarginin nedeni:
- Dense dynamic head fikri benziyor.
- Row-wise target/matcher mantigi benziyor.
- Fakat query olusumu, spatial feature kaynagi, supervision cozunurlugu, loss formu, decode semantigi ve calibration defaultlari ayni degil.

Bu nedenle bugunku branch icin en dogru tanim:

`CondLSTR-inspired dense adapter on top of LSTR + Mamba`

Bu branch icin yanlis tanim:

`CondLSTR parity port`

## 2. CondLSTR End-to-End Veri Akisi

CondLSTR zinciri su sekilde calisiyor:

1. Dataset lane point listesi veriyor.
2. Transform pipeline augment sonrasi her lane icin raster `img_mask` olusturuyor.
3. `STDCNet(BiSeNet)` tabanli backbone fused feature uretiyor.
4. Transformer bu fused feature u encoder memory olarak kullaniyor.
5. Decoder `20 x 1` target grid uzerinde query tensor uretiyor.
6. Detector query branch ile `object/class/range/params` cikariyor.
7. Dynamic conv branch ayni query'lerin uretdigi agirliklarla spatial map uzerinde `mask/reg` cikariyor.
8. Loss native lane maskelerden row-wise target cikariyor.
9. Hungarian matcher `obj + cls + loc + reg + range` cost ile esleme yapiyor.
10. Postprocess dense map'ten lane point listesi decode ediyor.
11. Inference katmani pointleri orijinal koordinata geri tasiyor.
12. Metric chamfer-distance tabanli lane matching yapiyor.

Bu zincirde en kritik nokta su:

CondLSTR sadece bir `dynamic head` degil. Basarisi, `query formulation + fused feature source + native target generation + matcher/loss + decode` birlikteliginden geliyor.

## 3. Data ve Target Uretimi

### 3.1 Transform Seviyesi

Dosya:
- `/home/alki/projects/CondLSTR/data/transforms/lane/transforms.py`

Kritik satirlar:
- `GenerateLaneLine2D.__call__`: 57-121
- Lane raster mask uretimi: 86-97
- Single-class dataset icin `lane_attris = 0`: 99-106

Ne yapiyor:
- Orijinal `lane_points` augment ediliyor.
- Her lane icin ayri bir binary mask ciziliyor.
- Sonuc `img_mask` olarak `H x W x M` seklinde saklaniyor.
- CULane gibi attribute olmayan veri setinde class etiketi varsayilan `0`.

Sonuc:
- CondLSTR native supervision olarak lane mask ile egitiliyor.
- Row-wise target sonradan bu maskeden turetiliyor.

### 3.2 Collate ve Padding

Dosya:
- `/home/alki/projects/CondLSTR/data/transforms/lane/utils.py`

Kritik satirlar:
- `collate_fn_padded`: 62-74

Ne yapiyor:
- Hem `img`, hem `img_mask` padded tensor halinde batchleniyor.
- Lane sayisi degisken olsa da batch icinde tek tensorde tasinabiliyor.

### 3.3 Loss Preprocess ile GT Contract

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/loss.py`

Kritik satirlar:
- `CondLSTR2DLoss.preprocess`: 235-319

Ne yapiyor:
- `gt_masks` ve `gt_labels` alip current dense grid boyutuna getiriyor.
- `gt_row_rng`, `gt_row_loc`, `gt_row_reg`, `gt_row_loc_mask`, `gt_row_reg_mask`, `gt_label_obj`, `gt_label_cls` uretiyor.
- `min_points` filtresi uyguluyor.
- `row_rng` normalize ediliyor.
- `row_reg_mask` yalnizca `line_width * 4` civarinda aciliyor.

Ana sonuc:
- CondLSTR target contract'i native mask supervision'dan doguyor.
- Bizdeki gibi legacy label tensor -> lane point -> rowwise adapter degil.

## 4. Backbone, Neck ve Spatial Feature Kaynagi

### 4.1 Wrapper Seviyesi

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/models/lane/cond_lstr_2d_res18.py`

Kritik satirlar:
- Model config: 14-58
- Forward akisi: 80-109

Operasyonel ayarlar:
- `img_backbone = STDCNet(backbone='ResNet18')`
- `det_backbone = Transformer(...)`
- `src_shape=(24, 42)`
- `tgt_shape=(20, 1)`
- `d_model=256`
- `num_encoder_layers=2`
- `num_decoder_layers=4`
- `return_intermediate_dec=True`

Detector ayarlari:
- `disable_coords=True`
- `branch_channels=256`
- `min_points=2`
- `line_width=16`
- `score_thresh=0.7`
- `eos_coef=0.4`
- `mask_downscale=1`

### 4.2 Ayrik Neck Var mi?

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/backbones/stdcnet/__init__.py`
- `/home/alki/projects/CondLSTR/modeling/models/backbones/stdcnet/model_stages.py`

Kritik satirlar:
- `STDCNet` aslinda `BiSeNet` kullaniyor: `__init__.py` 13-19
- `ContextPath` ve ARM modulleri: `model_stages.py` 117-264
- `FeatureFusionModule`: `model_stages.py` 284-314
- `BiSeNet.forward`: `model_stages.py` 426-462

Yargi:
- Wrapper seviyesinde ayrik bir `neck.create(...)` cagrisi yok.
- Ama backbone "sadece raw ResNet feature" degil.
- `BiSeNet` icinde context-path + attention-refinement + feature-fusion var.
- `feat_res8` ile `feat_cp8` birlestirilip `feat_out` uretiliyor.

Bu nedenle:
- CondLSTR kodunda disaridan baglanmis bir FPN yok.
- Ama backbone icinde net bir multi-scale fusion var.
- Yani "neck yok" demek eksik olur.
- Dogrusu su: `ayri neck modulu yok, ama backbone icinde BiSeNet tarzi fusion var`.

### 4.3 Spatial Dense Feature Tam Olarak Ne?

Kritik satirlar:
- `img_feat = self.img_backbone(img)`: `cond_lstr_2d_res18.py` 96
- `enc_outs = self.det_backbone.forward_encoder(img_feat, src_mask)`: 99
- `enc_feat = enc_outs[0].view(...).permute(...)`: 102
- `f_mask = [enc_feat] * len(det_feat)`: 105

Yani dense head'in spatial girdisi:
- Raw `layer2` veya `layer4` degil.
- `BiSeNet` ile fuse edilmis backbone feature'in transformer encoder memory hali.
- Sonra tekrar `B x C x H x W` formatina donusturulen `enc_feat`.

Bu nokta parity icin kritik.

## 5. Query Formulation ve Query Semantics

### 5.1 CondLSTR Query Sayisi

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/models/lane/cond_lstr_2d_res18.py`

Kritik satirlar:
- `tgt_shape=(20, 1)`: 25

Yorum:
- CondLSTR query sayisi fiilen `20`.
- Query'ler tek boyutlu target grid uzerinde diziliyor.

### 5.2 Query'lerin Icerigi Nasil Olusuyor?

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/backbones/transformer/transformer.py`

Kritik satirlar:
- `forward_decoder`: 181-256
- `tgt_mask` yaratimi: 224-225
- `tgt_pos_embed = self.tgt_pos_embed(tgt_mask)`: 226-227
- `if tgt is None: tgt = tgt_pos_embed`: 232-233

Bu cok kritik bir parity noktasi:
- CondLSTR burada klasik DETR/LSTR tarzi ayri bir `query_embed.weight` vermiyor.
- Decoder input'u dogrudan `learned target positional embedding`.
- Yani query slot'lari "20 x 1 learned target grid" semantigine sahip.

Bu nedenle:
- CondLSTR query semantics'i, bizim LSTR query embedding semantigimizle birebir ayni degil.
- Sadece query sayisinin farkli olmasi degil, query'nin ne oldugu da farkli.

### 5.3 Decoder Ciktisi Neye Donusuyor?

Kritik satirlar:
- `ys = [x.view(h_tgt, w_tgt, bs, -1).permute(2, 3, 0, 1)...]`: `transformer.py` 255
- `det_feat = [x.squeeze(-1).transpose(1, 2) for x in det_feat]`: `cond_lstr_2d_res18.py` 103

Yani:
- Her decoder layer output'u `B x C x 20 x 1`
- Sonra `B x 20 x C` formatina cevriliyor.
- Head query branch bu tensoru kullaniyor.

## 6. Head Yapisi

### 6.1 Detector Seviyesi

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/cond_lstr_2d.py`

Kritik satirlar:
- `DynamicMaskHead`: 14-92
- `CtnetHead`: 112-153
- `CondLSTR2D.__init__`: 156-236
- `_forward`: 238-260

### 6.2 Query Branch

`CtnetHead` su cikislari uretiyor:
- `logits`: objectness, 2 kanal
- `attris`: class logits
- `ranges`: 2 kanal
- `params`: mask + reg dynamic param'larinin birlestirilmis hali

Bu da su anlama geliyor:
- Objectness/background ayrimi explicit 2-logit.
- Class branch ayri.
- Range branch ayri.
- Dynamic branch parametreleri tek tensor olarak uretiliyor, sonra split ediliyor.

### 6.3 Dynamic Spatial Branch

`DynamicMaskHead`:
- Tek katmanli dynamic 1x1 conv kullaniyor.
- `disable_coords=False` ise normalize koordinatlari feature'a concatenate ediyor.
- Mask branch son bias'ina `prior_prob=0.01` tabanli bias ekliyor.

Wrapper'da actual setting:
- `disable_coords=True`

Yani actual CondLSTR recipe:
- Koordinat eklenmeden dynamic conv.

### 6.4 Range Contract

Kritik satir:
- `lane_ranges = lane_ranges.sigmoid()`: `cond_lstr_2d.py` 253

Bu da su anlama gelir:
- Range prediction training ve inference oncesi [0, 1] araligina sikistirilir.
- GT row range de normalized oldugu icin output contract ayni domain'de tutulur.

### 6.5 Dense Output Resolution

Kritik satirlar:
- `mask_shape = img_shape // mask_downscale`: `cond_lstr_2d.py` 257
- `regs = F.interpolate(...)`: 258
- `masks = F.interpolate(...)`: 259

Res18 wrapper'da:
- `mask_downscale=1`

Sonuc:
- Dense supervision grid'i input shape seviyesine kadar geri buyutuluyor.
- CondLSTR loss/postprocess low-res feature grid uzerinde dogrudan kalmiyor.

## 7. Matcher

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/matcher.py`

Kritik satirlar:
- `forward`: 31-139

Cost terimleri:
- `cost_obj`
- `cost_cls`
- `cost_row_loc`
- `cost_row_iou`
- `cost_row_reg`
- `cost_row_rng`

Loc cost formu:
- `cost_loc = cost_row_loc + cost_row_iou * 2.0`

Objectness semantics:
- `softmax(logits_obj)` uzerinden foreground score `index 0`
- background `index 1`

Class semantics:
- `ignore_index = 255` destekli

Reg cost:
- Dense reg map uzerinde maskeli L1

Range cost:
- Normalized row range L1

Yargi:
- Matcher CondLSTR'de ciddi bicimde row-wise geometry odakli.
- Sadece objectness ile query secmiyor.
- `row_iou` cost matcher icin temel bir parca.

## 8. Loss

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/loss.py`

Kritik satirlar:
- `SetCriterion.__init__`: 16-36
- `loss_obj`: 38-48
- `loss_cls`: 50-69
- `loss_loc`: 82-127
- `loss_reg`: 129-148
- `loss_rng`: 150-163
- Weighted sum: 179-201
- `CondLSTR2DLoss.forward`: 321-351

Kayip terimleri:
- `loss_obj`
- `loss_cls`
- `loss_loc`
- `loss_reg`
- `loss_rng`

Asil dikkat edilmesi gerekenler:
- `loss_loc` icinde hem row L1 hem `2 * row_iou` var.
- Ayrik bir aktif `dense_mask CE` kaybi yok.
- Kodda row classification CE denemesi yorum satirinda birakilmis.

Default detector loss agirliklari:
- `obj_weight=10`
- `cls_weight=10`
- `loc_weight=1`
- `reg_weight=1`
- `rng_weight=20`

Object EOS:
- Wrapper recipe'de `eos_coef=0.4`

Training derinligi:
- Eger decoder output list ise tum layer'lar uzerinden loss toplanabiliyor.
- Bu da intermediate decoder supervision verdigi anlamina geliyor.

## 9. Postprocess

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/postprocess.py`

Kritik satirlar:
- `__call__`: 21-109

Akis:
- `scores_obj = softmax(logits_obj)[..., 0]`
- `scores_cls = softmax(logits_cls)`
- Mask logits width ekseninde softmax aliniyor.
- Row center expectation ile bulunuyor.
- Row center round edilip o kolondan regression offset cekiliyor.
- `row_loc = rounded_center + reg_offset`
- `score_thresh` ile query seciliyor.
- `row_rng` normalized range'den image row'a donusturuluyor.
- `min_points` filtresi uygulanıyor.
- `mask_downscale` ile point koordinati geri carpiliyor.
- Opsiyonel smoothing var.

Bu decode mantigi teknik olarak onemli:
- Train-time row center semantigi ile uyumlu.
- Salt `argmax` decode degil.
- Expectation + round + gather yapiliyor.

## 10. Inference ve Metric

### 10.1 Inference

Dosya:
- `/home/alki/projects/CondLSTR/modeling/inferences/lane/lane_det_2d.py`

Kritik satirlar:
- 18-85

Ne yapiyor:
- `lane_points`, `lane_scores`, `lane_attris` aliyor.
- `scale_factor` ve `image_offset` ile original image frame'e geri tasiyor.

### 10.2 Metric

Dosya:
- `/home/alki/projects/CondLSTR/modeling/metrics/lane/lane_det_2d.py`

Kritik satirlar:
- `LaneDet2DMetric`: 9-73
- `LaneDistance`: 76-286

Ne yapiyor:
- Chamfer-distance tabanli lane matching kullaniyor.
- Attribute-based alt metrikleri de tutuyor.

Yargi:
- CondLSTR'nin resmi eval semantigi, bizim CULane official evaluator hattimizla birebir ayni degil.

## 11. Bizim LSTR Dense Port ile Birebir Karsilastirma

### 11.1 Target Uretimi

Bizde:
- Legacy LSTR label tensor -> lane point listesi
- Lane point listesi -> row-wise dense target

Dosyalar:
- `/home/alki/projects/LSTR/models/LSTR_CULANE_2k_mamba_dense.py`: 33-135
- `/home/alki/projects/LSTR/utils/condlstr_dense_targets.py`: 86-195

Yargi:
- Fikir dogru.
- Ama native mask supervision parity yok.
- Bu bir adapter katmani.

### 11.2 Query Sayisi ve Query Semantics

Bizde:
- `num_queries=7`
- `dec_layers=2`
- `attn_dim=32`

Dosya:
- `/home/alki/projects/LSTR/config/LSTR_CULANE_2k_mamba_dense.json`: 43-66

Ayrica:
- Dense head query girdisi `hs[-1]`
- Yani sadece final decoder layer kullaniliyor

Dosya:
- `/home/alki/projects/LSTR/models/LSTR_CULANE_2k_mamba_dense.py`: 176-180

Yargi:
- Query sayisi farkli.
- Decoder derinligi farkli.
- Hidden dim farkli.
- Query semantigi de farkli, cunku bizim taraf LSTR query embedding tabanli; CondLSTR taraf learned target-grid positional embedding tabanli.

Bu fark "sadece kapasite farki" degil, dogrudan recipe farki.

### 11.3 Spatial Feature Kaynagi

Bizde:
- `dense_feature = layer2(p)`
- Dense head spatial girdisi bu `layer2`

Dosya:
- `/home/alki/projects/LSTR/models/LSTR_CULANE_2k_mamba_dense.py`: 169-178

Yargi:
- CondLSTR'nin fuse edilmis encoder memory spatial map'i ile ayni degil.
- Biz daha erken ve daha hafif feature kullaniyoruz.

### 11.4 Dynamic Head Parametrization

Bizde:
- Query branch ayri `object_logits`, `class_logits`, `ranges`, `mask_params`, `reg_params` uretiyor.
- Spatial branch ayri `mask_branch` ve `reg_branch`.

Dosya:
- `/home/alki/projects/LSTR/models/dynamic_lane_head.py`: 30-88
- `/home/alki/projects/LSTR/models/dynamic_lane_head.py`: 91-143
- `/home/alki/projects/LSTR/models/dynamic_lane_head.py`: 146-266

Yargi:
- Dinamik conv fikri ayni ailede.
- Tek katmanli dynamic conv fikri de benzer.
- Ama birebir ayni sınıf/senaryo degil.
- CondLSTR actual recipe `disable_coords=True`; bizde baseline `dense_use_coords=true`.

### 11.5 Range Contract

Bizde:
- `pred_ranges` ham lineer branch cikisi
- Detector forward icinde `sigmoid` uygulanmiyor

Dosya:
- `/home/alki/projects/LSTR/models/dynamic_lane_head.py`: 81-88
- `/home/alki/projects/LSTR/models/LSTR_CULANE_2k_mamba_dense.py`: 178-180

Yargi:
- Bu parity acigidir.
- Ham lineer output yanlis olmak zorunda degil.
- Ama CondLSTR recipe ile ayni degil.

### 11.6 Loss ve Matcher

Matcher:
- Bizim matcher CondLSTR'ye oldukca yakin
- `object + class + row_location + row_iou + row_reg + row_range`

Dosya:
- `/home/alki/projects/LSTR/models/condlstr_dense_matcher.py`: 23-235

Criterion:
- Bizde ayrik `loss_dense_mask` CE var
- `row_iou` loss config ile kapatilabiliyor

Dosya:
- `/home/alki/projects/LSTR/models/condlstr_dense_criterion.py`: 12-210

Config:
- `dense_enable_row_iou_loss=false`
- `dense_object_eos_coef=0.1`

Dosya:
- `/home/alki/projects/LSTR/config/LSTR_CULANE_2k_mamba_dense.json`: 60-66

Ana farklar:
- CondLSTR loc loss = `row_l1 + 2 * row_iou`
- Bizde row_iou default kapali
- CondLSTR'de aktif `dense_mask CE` yok
- Bizde aktif `loss_dense_mask` var
- CondLSTR obj/cls/rng agirliklari cok daha sert
- Bizde baseline agirliklar 1.0

Bu parity farki bugunku calibration tartismasi icin birinci derece onemdedir.

### 11.7 Objectness / Background Semantics

CondLSTR:
- `target_classes` default `1` yani background
- Matched query'ler `gt_label_obj = 0` ile foreground

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/loss.py`: 42-48

Bizde:
- Ayni semantik
- `gt_label_obj` target conversion'da `0`
- Criterion default object target `1`

Dosyalar:
- `/home/alki/projects/LSTR/utils/condlstr_dense_targets.py`: 167-174
- `/home/alki/projects/LSTR/models/condlstr_dense_criterion.py`: 194-209

Yargi:
- Objectness class semantics temel olarak ayni.
- Sorun "foreground/background tanimi ters" degil.
- Sorun agirlik, calibration ve query dagilimi tarafinda.

### 11.8 Decode Parity

CondLSTR:
- Expectation -> round -> gather reg -> add offset

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/detectors/lane/cond_lstr_2d/postprocess.py`: 28-35

Bizde:
- Argmax -> reg_map[row, col]

Dosya:
- `/home/alki/projects/LSTR/models/condlstr_dense_postprocess.py`: 59-70

Ayrica:
- Bizim matcher/loss row center hesaplamasi expectation tabanli

Dosya:
- `/home/alki/projects/LSTR/models/condlstr_dense_matcher.py`: 248-252
- `/home/alki/projects/LSTR/models/condlstr_dense_criterion.py`: 69

Yargi:
- Train-time semantik ile eval decode semantigi bizde ayni degil.
- CondLSTR bu konuda daha tutarli.

### 11.9 Inference Threshold

CondLSTR recipe:
- `score_thresh=0.7`

Dosya:
- `/home/alki/projects/CondLSTR/modeling/models/models/lane/cond_lstr_2d_res18.py`: 49-54

Bizde evaluator:
- Dense decode sabit `score_thresh=0.5`

Dosya:
- `/home/alki/projects/LSTR/test/culane.py`: 75-80

Yargi:
- Threshold tuning kok neden degil.
- Ama reference inference parity de yok.

## 12. En Kritik Parity Aciklari

Etki sirasi acisindan en kritik parity aciklari:

1. Query formulation parity yok.
2. Spatial feature source parity yok.
3. Native GT mask supervision parity yok.
4. Range output domain parity yok.
5. Loss parity yok.
6. Decode parity yok.
7. Calibration defaults parity yok.
8. Intermediate decoder supervision parity yok.

Burada ozellikle ilk alti madde recipe seviyesinde fark yaratiyor.

## 13. Su Ana Kadarki Deneylerin Bu Audit Acisindan Anlami

Simdiki deneyler tamamen anlamsiz degil.

Ne ogrettiler:
- `eos` calibration acisindan gercek bir sinyal.
- Expectation decode parity icin kritik.
- Sadece `sigmoid` eklemek yeterli degil.
- Sadece agirliklari CondLSTR gibi yapmak da yetmeyebilir.

Ama bu deneylerin siniri su:
- Bunlar parity olmayan bir branch uzerinde yapildi.
- O yuzden sonuc "CondLSTR bunu boyle yapiyor" anlamina gelmez.
- Ancak "bizim minimal port hangi eksenlerde kiriliyor" sorusuna cevap verir.

## 14. Bu Audit'ten Cikan Dogrudan Yargi

Bu audit sonrasi teknik olarak su cumle kurulabilir:

`Bizim mevcut branch CondLSTR recipe'sinin sadece bir alt kumesini tasiyor.`

Bu nedenle:
- Daha fazla rastgele knob taramasi getirisi azalan bir faza girdi.
- Bundan sonraki dogru asama `CondLSTR parity audit -> targeted parity implementation -> controlled ablation`.

## 15. Sonraki Asama icin Oncelikli Sorular

Kod bazli kapatilmasi gereken sorular:

1. LSTR query embedding tabanli form ile CondLSTR target-grid query form arasindaki fark nasil ele alinacak?
2. Spatial dense branch `layer2` uzerinde mi kalacak, yoksa encoder-memory-benzeri fused feature mi kullanilacak?
3. Native GT mask supervision'a daha yakin bir hedef uretim yolu gerekli mi?
4. Range branch output'u CondLSTR gibi `sigmoid` domain'e alinmali mi?
5. Loss parity icin `row_iou on + dense_mask CE ablation` zorunlu mu?
6. Decode parity icin expectation tabanli resmi yol mu esas alinacak?
7. Intermediate decoder layer supervision gerekli mi?
8. BiSeNet-benzeri fused source eksikligi bizim tarafta structural bottleneck mi?

## 16. Operasyonel Oneri

Bu rapordan cikan pratik tavsiye:

1. Yeni ana faz "CondLSTR parity implementation audit" olmali.
2. Bundan sonra knob denemeleri yalnizca parity checklist maddelerine bagli ve kontrollu yapilmali.
3. "Basit tuning" ile "recipe parity" kesin olarak birbirinden ayrilmali.

Bu rapora gore bugunku en dogru stratejik cümle:

`Artik local tuning degil, component-level CondLSTR parity calismasi yapilacak.`
