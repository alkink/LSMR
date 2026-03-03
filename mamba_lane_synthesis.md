# Mamba-Lane: SOTA Mimariler Sentezi ve Hibrit Model Tasarımı

Bu doküman, günümüze kadarki 9 farklı üst düzey şerit tespit ve otonom sürüş yörünge mimarisinin **(LSTR, CLRNet, CondLaneNet, CondLSTR, GANet, LAformer, LaneAF, PriorLane, RCLane)** derinlemesine analizlerinin ortak bir paydada eritildiği sentez dosyasıdır. Amacımız, **"Vision Mamba" (VSS)** altyapısını kullanarak hıza ve keskinliğe dayalı, rakiplerini geride bırakacak yepyeni bir melez (hybrid) Mamba-Lane mimarisi tasarlamaktır.

---

## 1. Mimarilerin Kıyaslamalı Özeti

| Model Modülü | LSTR | CLRNet | CondLaneNet | CondLSTR | GANet | LAformer | LaneAF | PriorLane | RCLane | Mamba-Lane (Hedef Tasarım) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ana Paradigma** | Polinom Regresyonu | Nokta Regresyonu (Line IoU) | Dinamik Evrişim + Maske | Dinamik Evrişim + Maske | Keypoint Offset (Global Association) | Yörünge Tahmini (Vektörel) | Affinity Fields (VAA, HAA) | Segmentasyon (SegFormer) | Relay Chain (Oklar + Mesafe) | **Koşullu Mamba Regresyonu / Maskesi** |
| **Backbone (Omurga)** | Mini-ResNet | ResNet / DLA34 | ResNet | ResNet / STDC | ResNet / HRNet | VectorNet (PointSubGraph) | DLA34 / ResNet / ERFNet | Mix Transformer (MiT) | Mix Transformer (MiT) | **Vision Mamba (VSS) / Hafif CNN + VSS** |
| **Neck (Özellik Birleştirme)** | Sadece 1x1 Conv | FPN | TransConvFPN (Transformer) | Transformer Encoder | FPN + Deformable Conv (LFA) | LaneGCN (Agent-Lane Cross Attn) | DLA Up-sampling | MLP Linear Fuse | MLP Linear Fuse | **Mamba Feature Aggregator (Local Mamba)** |
| **Head (Tahmin Kafası)** | Object Queries (Transformer Dec.) | ROIGather + Öğrenilebilir Priors | CondLaneHead (Dinamik Ağırlık) | CtnetHead (Sorgularla Dinamik Ağırlık) | Heatmap + Kök Offsetleri | Laplace Decoder (GRU) | VAF / HAF Heatmap | SegFormer Mask Head | RCLaneHead (5 Yön/Mesafe Kafası) | **1B Mamba Decoder + Dinamik Evrişim (veya LineIoU)** |
| **Loss (Kayıp)** | Bipartite L1 (Polinom) | Focal + LineIoU + SmoothL1 | Focal + L1 (Offset) | Dice + L1 (Offset) | Focal + L1 (Offset) | Laplace NLL (Varyans/Dağılım) | Binary C.E + L1 VAF/HAF | CrossEntropy (OHEM) | CrossEntropy (OHEM) + SmoothL1 | **Focal (Sınıf) + Dice (Maske) / LineIoU** |
| **Ekstra Özellik** | - | İteratif İyileştirme | Adım Adım Evrişim | Transformer + CNN Karışımı | Deformable CNN (LFA) | Belirsizlik (Scale) Skoru | Vektör Tarlaları | OpenStreetMap Harita Füzyonu | Zincirleme Oklar | **Uzun Dizi Hafızası, Maske Bağımsızlık** |

---

## 2. Her Modelden Öğrendiğimiz Altın Dersler (Ne Almalıyız?)

Yeni modelimizi sıfırdan kurarken, tekerleği yeniden icat etmek yerine her SOTA makalenin şahikasını (en iyi kısmını) alıp Mamba zincirine oturtmalıyız:

1. **LSTR'den:** "Görüntüyü boydan boya dizmek (sequence) ve Object Query'ler (öğrenilebilir tohumlar) kullanıp Bipartite Matching yapmak". Bu, Post-processing NMS belasını çözer.
2. **CLRNet'ten:** "Polinom katsayıları esnek değildir, bunun yerine 72 adet somut X-noktası tahmini yap ve **LineIoU Loss** ile bunları şerit gibi eğit." (Çok kritik bir başarı sırrı)
3. **CondLaneNet & CondLSTR'den:** "Şeridi direk tahmin etmek yerine, her şerit için özel bir **Dinamik CNN Filtresi (Ağırlığı)** tahmin et ve bu filtreyle bütün haritayı çarpıp sadece o şeridi parlat."
4. **GANet'ten:** "Standart kare evrişimler şeritler için yetersizdir. Çevreyi taramak için Context lazım." (Onlarda Deformable Conv vardı, bizde Vision Mamba'nın uzun taramaları olacak).
5. **LaneAF & RCLane'den:** Maske tabanlı modellerin matematiksel güzelliği (oklar ve vektör tarlaları). Vektörel mantık Mamba'nın ardışık adım (step-by-step) öngörü yeteneğiyle harmanlanabilir.
6. **PriorLane'den:** Kameranın yetmediği yerde HD-Map (Harita) verisini sisteme dahil edip Cross-Attention benzeri bir melezlemeyle (Mamba ile daha ucuz şekilde) başarım katlamak.
7. **LAformer'dan:** "Tahmin ettiğin rotanın yanına bir de **Scale (Eminlik / Varyans)** parametresi ver". Sisli veya silik yollarda otonom sürüş sistemine "Burada çizgi var diyorum ama sadece %40 eminim, dikkatli ol" dedirtebilmek.

---

## 3. Mamba-Lane: Nihai Tasarım Önerisi (Architecture Blueprint)

Tüm bulgular ışığında, **"CondMamba-Lane"** (Koşullu Mamba Şerit Tespiti) kod adlı yeni melez modelin iş akışı şöyle tasarlanmalıdır:

### Aşama 1: Görüntü Özellik Çıkarımı (Backbone)
**Tasarım:** Sadece ResNet kullanmak yerine, görüntü `(3 x 320 x 800)` **Vision Mamba (VSS) Bloklarına** veya çok hafif bir CNN'in ardından VSS bloklarına verilir.
**Açıklama:** Mamba, Transformer gibi çalışıp tüm pikseller arasındaki global ilişkiyi doğrusal sürede \(O(N)\) kurar. H/32, W/32 ebatında `Ana Özellik Haritası` çıkar.

### Aşama 2: Şerit Tohumları ve Mamba Decoder
**Tasarım:** LSTR'daki `N=10` adet şerit sorgusunu al. Bu 10 sorguyu (token), Ana Özellik Haritası'ndan çekilen Mamba dizisiyle etkileşime sok (Cross-Mamba veya Mamba Mixer).
**Çıktı:** Elimizde 10 adet ultra-zengin vektör (her biri bir potansiyel şeridi temsil eder) kalır.

### Aşama 3: CondLSTR'vari Dinamik Üretim (Prediction Head)
**Tasarım:** 10 vektörü basit MLP katmanlarına ver. MLP'ler şunları üretsin:
- **Sınıflandırma:** Bu şerit aktif mi? (0-1)
- **Belirsizlik (LAformer):** Bu şerit hakkında ne kadar eminim? (Scale skoru)
- **Koşullu Ağırlıklar (CondLaneNet):** Bu şeridi haritadan söküp alacak "Özel Sihirli Gözlük" (Dinamik Convolution Kernel Parametreleri).

### Aşama 4: Haritayı Yansıtma (Generation)
**Tasarım:** Aşama 1'de Mamba'dan çıkan `Ana Özellik Haritası` ile Aşama 3'ten gelen `Özel Sihirli Gözlük` (Filtreler) `torch.bmm` ile çarpılır.
**Çıktı:** 10 Adet pırıl pırıl, maske veya (CLRNet gibi) doğrudan nokta koordinatı. Şablonlara oturtulmuş yüksek hassasiyetli şeritler. Segmentasyon problemi olmadığı için NMS veya kümelenmeye ihtiyaç duyulmaz.

### Aşama 5: Optimizasyon (Loss)
- Şeritlerin pikselleri veya noktaları için **CLRNet'in LineIoU** Kaybı (mükemmel örtüşme takibi).
- Maskelerin keskinliği için Dice Loss.
- Seçilebilen maskeler için Hungarian Matching.

## Sonuç
Bu yapı; LSTR'ın eşleştirme matematiğini, Mamba'nın lineer hızda global bakış yeteneğini, CondLSTR'nin dinamik kernel başlığını ve CLRNet'in LineIoU başarısını tek bir çatı altında birleştiren, literatürde daha önce (bildiğimiz kadarıyla) hiç denenmemiş **SOTA kırıcı** bir formül olacaktır.
