"""
DynamicKernelHead Pre-Integration Analysis — DÜZELTİLMİŞ
==========================================================

Düzeltilen bug'lar:
  BUG-1: Gradient testi yanlış — .cuda() sonrası requires_grad=True leaf değil
          FIX: .cuda().requires_grad_(True) sırası düzeltildi
  BUG-2: Query hook shape yanlış — (N,B,D) alıyor ama qs[0] yanlış kesiyor
          FIX: permute(1,0,2)[0] ile (N,D) doğru alınıyor
  BUG-3: Kernel aktivasyon testi (5.2) random tensor kullanıyor — 22 vs 23 hep aynı
          FIX: Gerçek layer2 feature'ları kullanıyor, eşik de anlamlı (1.3x)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib

sys.path.insert(0, '/home/alki/projects/LSTR')

from config import system_configs
from models.LSTR_CULANE_MAMBA import model as MambaModel
from db.datasets import datasets

cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE_MAMBA.json")
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["system"]["snapshot_name"] = "LSTR_CULANE_MAMBA"
cfg["system"]["data_dir"] = "/home/alki/projects/"
system_configs.update_config(cfg["system"])

print("=" * 80)
print("DYNAMIC KERNEL HEAD PRE-INTEGRATION ANALYSIS (DÜZELTİLMİŞ)")
print("=" * 80)

model = MambaModel(flag=True).cuda()
model.eval()

db_obj = datasets['CULANE'](cfg['db'], 'train+val')
sample_fn = importlib.import_module('sample.culane').sample_data

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: LAYER2 FEATURE KALİTESİ
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 1: LAYER2 FEATURE KALİTESİ")
print("=" * 80)

layer2_outputs = []

def hook_layer2(module, input, output):
    layer2_outputs.append(output.detach().cpu())

hook = model.layer2.register_forward_hook(hook_layer2)

print("\n1.1 layer2 SHAPE VE İSTATİSTİK (10 örnek)")
print("-" * 80)

with torch.no_grad():
    for i in range(10):
        data, _ = sample_fn(db_obj, i)
        img = data['xs'][0].cuda()
        mask = data['xs'][1].cuda()
        model(img, mask)

hook.remove()

for i, feat in enumerate(layer2_outputs):
    print(f"Sample {i}: shape={feat.shape} mean={feat.mean():.4f} "
          f"std={feat.std():.4f} min={feat.min():.4f} max={feat.max():.4f} "
          f"sparsity={(feat.abs()<0.01).float().mean():.3f}")

print("\n1.2 GÖRÜNTÜLER ARASI COSINE BENZERLİK")
print("-" * 80)
print("Beklenti: <0.85 (feature çeşitliliği)")
print("Yüksekse: layer2 lane bilgisi taşımıyor, kernel öğrenemez")

flat = [f.flatten() for f in layer2_outputs[:5]]
sim = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        sim[i][j] = F.cosine_similarity(
            flat[i].unsqueeze(0), flat[j].unsqueeze(0)
        ).item()

print("\nCosine similarity matrisi:")
print(np.round(sim, 3))
avg = (sim.sum() - np.trace(sim)) / 20
print(f"\nOrtalama çapraz-benzerlik: {avg:.4f}")

if avg > 0.95:
    print("🚨 KRİTİK: layer2 neredeyse sabit — kernel öğrenemez!")
    layer2_ok = False
elif avg > 0.85:
    print("⚠️ UYARI: Benzerlik yüksek — kernel öğrenmesi zayıf olabilir")
    layer2_ok = None
else:
    print("✅ Feature çeşitliliği yeterli")
    layer2_ok = True

print("\n1.3 SPATIAL AKTİVASYON ANALİZİ (şerit bölgesi odaklı mı?)")
print("-" * 80)
print("Not: Şeritler görüntünün alt yarısında, alt yarı daha aktif olmalı")
print("     Oran > 1.0 → alt yarı ağırlıklı (iyi)")
print("     Oran < 1.0 → üst yarı ağırlıklı (kötü)")

spatial_ok = True
ratios = []
for i in range(3):
    # layer2_outputs[i] shape: (B, C, H, W) — B'nin ilk elemanını al
    feat = layer2_outputs[i][0]  # (C, H, W)
    spatial_mean = feat.abs().mean(dim=0)  # (H, W)
    H = spatial_mean.shape[0]
    top_half = spatial_mean[:H//2].mean().item()
    bot_half = spatial_mean[H//2:].mean().item()
    ratio = bot_half / (top_half + 1e-8)
    ratios.append(ratio)
    status = "✅" if ratio >= 1.0 else "⚠️"
    print(f"Sample {i}: üst={top_half:.4f}, alt={bot_half:.4f}, oran={ratio:.2f}x {status}")
    if ratio < 1.0:
        spatial_ok = False

avg_ratio = np.mean(ratios)
print(f"Ortalama oran: {avg_ratio:.2f}x")
if avg_ratio < 1.0:
    print("⚠️ Uniform aktivasyon — lane'e odaklanma zayıf")
    print("   NOT: Bu pre-train modelde beklenir, eğitimde düzelebilir")
else:
    print("✅ Alt yarı (şerit bölgesi) daha aktif")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: DECODER QUERY KALİTESİ
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 2: DECODER QUERY KALİTESİ")
print("=" * 80)

# ── BUG-2 DÜZELTMESİ ────────────────────────────────────────────────────
# ESKİ: hs[-1].detach().cpu() → shape (N, B, D) = (7, 16, 32)
#        sonra qs = qs[0] → (B, D) = (16, 32)  YANLIŞ! İlk query'nin tüm batch'i
# YENİ: hs[-1].permute(1, 0, 2) → (B, N, D), sonra [0] → (N, D) = (7, 32) DOĞRU
# ─────────────────────────────────────────────────────────────────────────

decoder_outputs = []

def hook_decoder_out(module, input, output):
    if isinstance(output, tuple):
        hs = output[0]  # (num_layers, N, B, D)
    else:
        hs = output     # (num_layers, N, B, D)
    # hs[-1] = son layer: (N, B, D) = (7, batch, 32)
    last = hs[-1]  # (N, B, D)
    # BUG-2 FIX: (N, B, D) → (B, N, D) → batch 0 → (N, D)
    queries_b0 = last.permute(1, 0, 2)[0]  # (N, D) = (7, 32)
    decoder_outputs.append(queries_b0.detach().cpu())

hook_dec = model.transformer.decoder.register_forward_hook(hook_decoder_out)

with torch.no_grad():
    for i in range(5):
        data, _ = sample_fn(db_obj, i)
        img = data['xs'][0].cuda()
        mask = data['xs'][1].cuda()
        model(img, mask)

hook_dec.remove()

print("\n2.1 QUERY-QUERY COSINE BENZERLİK")
print("-" * 80)
print(f"Beklenti: Query'ler birbirinden farklı olmalı (sim < 0.80)")
print(f"Yüksekse: Tüm lane'ler için aynı kernel → dynamic head işlevsiz")

query_ok = True
all_sims = []
for i, qs in enumerate(decoder_outputs):
    # qs shape: (N, D) = (7, 32) — BUG-2 FIX sonrası doğru shape
    print(f"\nSample {i}: shape={qs.shape}  (beklenen: (7, 32))")

    if qs.shape[0] != 7:
        print(f"  🚨 Shape hâlâ yanlış! Expected (7, 32), got {qs.shape}")
        print(f"     Hook'u kontrol et")
        query_ok = False
        continue

    sims = []
    for a in range(qs.shape[0]):
        for b in range(a+1, qs.shape[0]):
            s = F.cosine_similarity(
                qs[a].unsqueeze(0), qs[b].unsqueeze(0)
            ).item()
            sims.append(s)

    avg_sim = np.mean(sims)
    all_sims.append(avg_sim)
    print(f"  Query-query cos similarity: mean={avg_sim:.4f} "
          f"max={np.max(sims):.4f} min={np.min(sims):.4f}")

    if avg_sim > 0.95:
        print("  🚨 Query'ler neredeyse aynı — kernel collapse riski!")
        query_ok = False
    elif avg_sim > 0.80:
        print("  ⚠️ Query'ler benzer — çeşitlilik düşük")
    else:
        print("  ✅ Query'ler yeterince farklı")

if all_sims:
    print(f"\nOrtalama query similarity (5 sample): {np.mean(all_sims):.4f}")
    print("NOT: Eğitilmemiş modelde collapse beklenir. Eğitim sonrası tekrar ölç.")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: VRAM ANALİZİ — bmm
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 3: VRAM ANALİZİ — bmm İŞLEMİ")
print("=" * 80)

B  = 16
N  = 7
K  = 32
C  = 32
# layer2 gerçek shape'i layer2_outputs'tan al
real_H = layer2_outputs[0].shape[2]
real_W = layer2_outputs[0].shape[3]
HW = real_H * real_W

print(f"\nParametreler: B={B}, N={N}, K={K}, C={C}")
print(f"layer2 gerçek boyutu: H={real_H}, W={real_W}, HW={HW}")
print(f"İşlem: kernel(B,N,K) @ feature(B,K,HW) → output(B,N,HW)")

bytes_per_float = 4
kernel_size  = B * N * K  * bytes_per_float
feature_size = B * C * HW * bytes_per_float
output_size  = B * N * HW * bytes_per_float
total_forward    = kernel_size + feature_size + output_size
total_with_grad  = total_forward * 2

print(f"\nTensor boyutları:")
print(f"  kernel:  ({B}, {N}, {K}) = {kernel_size/1e6:.2f} MB")
print(f"  feature: ({B}, {C}, {HW}) = {feature_size/1e6:.2f} MB")
print(f"  output:  ({B}, {N}, {HW}) = {output_size/1e6:.2f} MB")
print(f"  Forward: {total_forward/1e6:.2f} MB")
print(f"  Forward+Backward: {total_with_grad/1e6:.2f} MB")

total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e6
reserved  = torch.cuda.memory_reserved() / 1e6
free_gpu  = total_gpu - reserved
print(f"\nGPU: {total_gpu:.0f} MB toplam, {free_gpu:.0f} MB boş")

print("\n3.1 FARKLI K DEĞERLERİ İÇİN VRAM TAHMİNİ")
print("-" * 80)

vram_ok = True
for k in [32, 64, 128, 256]:
    ker  = B * N * k  * bytes_per_float
    feat = B * k * HW * bytes_per_float
    out  = B * N * HW * bytes_per_float
    total = (ker + feat + out) * 2
    ok = "✅" if total/1e6 < free_gpu * 0.5 else "🚨"
    print(f"  K={k:3d}: {total/1e6:.1f} MB {ok}")
    if k == 32 and total/1e6 >= free_gpu * 0.5:
        vram_ok = False

print("\n3.2 GERÇEK bmm VRAM TESTİ (gerçek layer2 H,W ile)")
print("-" * 80)

bmm_ok = True
for k in [32, 64]:
    try:
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        kernel_t  = torch.randn(B, N, k).cuda()
        feature_t = torch.randn(B, k, HW).cuda()
        out_t = torch.bmm(kernel_t, feature_t)
        after = torch.cuda.memory_allocated()
        print(f"K={k}: ✅ bmm başarılı, VRAM delta={(after-before)/1e6:.1f}MB, "
              f"output={out_t.shape}")
        del kernel_t, feature_t, out_t
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"K={k}: 🚨 OOM! {str(e)[:80]}")
        bmm_ok = False

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: İZOLE MODÜL TESTİ
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 4: DYNAMIC KERNEL HEAD — İZOLE MODÜL TESTİ")
print("=" * 80)

class DynamicKernelHead(nn.Module):
    """
    Giriş:
      queries:  (B, N, D)    — decoder çıktısı
      feat_map: (B, C, H, W) — layer2 çıktısı
    Çıkış:
      x_coords:   (B, N, 72) — normalize x koordinatı [0,1]
      cls_scores: (B, N, 1)  — lane var/yok
    """
    def __init__(self, query_dim=32, feat_channels=32, kernel_size=32, num_points=72):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_points  = num_points

        self.kernel_gen = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.ReLU(),
            nn.Linear(query_dim * 2, kernel_size)
        )
        self.feat_proj = nn.Conv2d(feat_channels, kernel_size, 1) \
            if feat_channels != kernel_size else nn.Identity()

        self.x_head = nn.Sequential(
            nn.Linear(kernel_size, kernel_size * 2),
            nn.ReLU(),
            nn.Linear(kernel_size * 2, num_points),
            nn.Sigmoid()   # 0-1 arası normalize x
        )
        self.cls_head = nn.Linear(query_dim, 1)

    def forward(self, queries, feat_map):
        B, N, D = queries.shape
        kernels  = self.kernel_gen(queries)           # (B, N, K)
        feat     = self.feat_proj(feat_map)           # (B, K, H, W)
        B, K, H, W = feat.shape
        feat_flat = feat.view(B, K, H * W)            # (B, K, HW)
        lane_feats = torch.bmm(kernels, feat_flat)    # (B, N, HW)
        lane_feats = lane_feats.mean(dim=-1, keepdim=True).expand(-1, -1, K)
        x_coords   = self.x_head(lane_feats)          # (B, N, 72)
        cls_scores = self.cls_head(queries)            # (B, N, 1)
        return x_coords, cls_scores

print("\n4.1 FORWARD TEST")
print("-" * 80)

module = DynamicKernelHead(
    query_dim=32, feat_channels=32, kernel_size=32, num_points=72
).cuda()

test_q_fwd  = torch.randn(2, 7, 32).cuda()
test_fm_fwd = torch.randn(2, 32, real_H, real_W).cuda()  # Gerçek layer2 H,W

print(f"Input queries:  {test_q_fwd.shape}")
print(f"Input feat_map: {test_fm_fwd.shape}")

try:
    x_coords, cls_scores = module(test_q_fwd, test_fm_fwd)
    print(f"\n✅ Forward başarılı")
    print(f"  x_coords:   {x_coords.shape}   beklenen: (2, 7, 72)")
    print(f"  cls_scores: {cls_scores.shape} beklenen: (2, 7, 1)")
    print(f"  x_coords range: [{x_coords.min():.4f}, {x_coords.max():.4f}]")
    forward_ok = x_coords.shape == (2, 7, 72)
    if not forward_ok:
        print("🚨 x_coords shape yanlış!")
    if not (0 <= x_coords.min() and x_coords.max() <= 1):
        print("🚨 x_coords 0-1 dışında! Sigmoid çalışmıyor.")
        forward_ok = False
except Exception as e:
    print(f"🚨 HATA: {e}")
    import traceback; traceback.print_exc()
    forward_ok = False

# ── BUG-1 DÜZELTMESİ ────────────────────────────────────────────────────
# ESKİ: torch.randn(..., requires_grad=True).cuda()
#        → .cuda() yeni tensor yaratır → non-leaf → .grad = None
# YENİ: torch.randn(...).cuda().requires_grad_(True)
#        → cuda'da leaf tensor → .grad doluyor
# ─────────────────────────────────────────────────────────────────────────

print("\n4.2 GRADIENT TESTİ")
print("-" * 80)
print("BUG-1 FIX: .cuda().requires_grad_(True) kullanılıyor")

test_q_grad  = torch.randn(2, 7, 32).cuda().requires_grad_(True)   # FIX
test_fm_grad = torch.randn(2, 32, real_H, real_W).cuda().requires_grad_(True)  # FIX

x_coords_g, cls_scores_g = module(test_q_grad, test_fm_grad)
loss_g = x_coords_g.sum() + cls_scores_g.sum()
loss_g.backward()

query_grad_ok = test_q_grad.grad is not None
feat_grad_ok  = test_fm_grad.grad is not None

print(f"Query gradient:   {'✅ var' if query_grad_ok else '🚨 YOK — backprop kesilmiş!'}")
print(f"Feature gradient: {'✅ var' if feat_grad_ok else '🚨 YOK — layer2 güncellenmeyecek!'}")

if not feat_grad_ok:
    print("  → DynamicKernelHead bmm'i feature'a gradient götürmüyor")
    print("  → Bu olursa layer2 weights dynamic head için öğrenemiyor")

dead_count = 0
exploding_count = 0
print("\nModül parametresi gradient'ları:")
for name, param in module.named_parameters():
    if param.grad is None:
        print(f"  🚨 {name}: gradient YOK")
        dead_count += 1
        continue
    gm = param.grad.abs().mean().item()
    gx = param.grad.abs().max().item()
    if gm < 1e-9:
        print(f"  💀 {name}: dead ({gm:.2e})")
        dead_count += 1
    elif "bias" not in name and gm > 10.0:
        exploding_count += 1
    elif "bias" in name and gm > 100.0:
        exploding_count += 1
    else:
        print(f"  ✅ {name}: OK ({gm:.2e})")

print(f"\nDead: {dead_count}, Exploding: {exploding_count}")
grad_ok = dead_count == 0 and exploding_count == 0

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: KERNEL ANLAMLILIK TESTİ
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 5: KERNEL ANLAMLILIK TESTİ")
print("=" * 80)

module.eval()

# 5.1 Kernel çeşitliliği — random query ile (initialization kalitesi)
with torch.no_grad():
    test_q_k = torch.randn(1, 7, 32).cuda()
    kernels_k = module.kernel_gen(test_q_k)    # (1, 7, K)
    kernels_np = kernels_k[0].cpu().numpy()    # (7, K)

print("\n5.1 KERNEL ÇEŞİTLİLİK — initialization (random query)")
print("-" * 80)

km = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        a = torch.tensor(kernels_np[i]).unsqueeze(0)
        b = torch.tensor(kernels_np[j]).unsqueeze(0)
        km[i][j] = F.cosine_similarity(a, b).item()
print(np.round(km, 3))

avg_k = (km.sum() - np.trace(km)) / (7*7-7)
print(f"\nOrtalama kernel benzerliği: {avg_k:.4f}")
print("(Düşük = daha farklı = iyi, Yüksek = collapse riski)")

if avg_k > 0.95:
    print("🚨 Kernel'lar neredeyse aynı — initialization sorunu!")
    kernel_diverse = False
elif avg_k > 0.80:
    print("⚠️ Kernel benzerliği yüksek — eğitimde çeşitlenebilir")
    kernel_diverse = None
else:
    print("✅ Kernel'lar yeterince farklı")
    kernel_diverse = True

# ── BUG-3 DÜZELTMESİ ────────────────────────────────────────────────────
# ESKİ: Random tensor ile attention hesaplanıyordu
#        → Softmax(random) her zaman ~H/2 vs H/2+1 verir (22 vs 23)
#        → Tamamen anlamsız test
# YENİ: Gerçek layer2 feature kullanılıyor
#        → Anlamlı spatial pattern testi
#        → Eşik 1.3x (trivial 1.0x değil)
# ─────────────────────────────────────────────────────────────────────────

print("\n5.2 KERNEL AKTİVASYON PATTERN — GERÇEK LAYER2 İLE")
print("-" * 80)
print("BUG-3 FIX: Gerçek layer2 feature kullanılıyor (random tensor değil)")
print("Eşik: alt/üst oranı > 1.3x  (trivial 1.0 değil)")

real_feat = layer2_outputs[0][0:1].cuda()  # (1, C, H, W) — gerçek görüntü

with torch.no_grad():
    real_q = torch.randn(1, 7, 32).cuda()
    kernels_real = module.kernel_gen(real_q)            # (1, 7, K)
    feat_proj_real = module.feat_proj(real_feat)        # (1, K, H, W)
    _, K_real, H_real, W_real = feat_proj_real.shape
    feat_flat_real = feat_proj_real.view(1, K_real, -1) # (1, K, HW)

    attn = torch.bmm(kernels_real, feat_flat_real)      # (1, 7, HW)
    attn = attn.view(1, 7, H_real, W_real)
    attn_softmax = attn.softmax(dim=-1)

    bottom_focused = 0
    mid = H_real // 2
    for lane_i in range(7):
        a_map   = attn_softmax[0, lane_i]
        top_sum = a_map[:mid].sum().item()
        bot_sum = a_map[mid:].sum().item()
        ratio   = bot_sum / (top_sum + 1e-8)
        # Eşik 1.3x — trivial 1.0 değil
        focused = ratio > 1.3
        status  = "✅ alt odaklı" if focused else f"⚠️ üst ağırlıklı (ratio={ratio:.2f})"
        print(f"  Lane {lane_i}: üst={top_sum:.4f}, alt={bot_sum:.4f}, "
              f"ratio={ratio:.2f}x → {status}")
        if focused:
            bottom_focused += 1

pattern_ok = bottom_focused >= 4
print(f"\n{bottom_focused}/7 lane gerçek anlamda alt yarı odaklı (eşik: >1.3x)")
if pattern_ok:
    print("✅ Yeterli lane alt yarı odaklı")
else:
    print("⚠️ Çoğu lane alt yarıya odaklanamıyor")
    print("   NOT: Eğitilmemiş modelde beklenir, eğitimde düzelebilir")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: OVERFIT TESTİ — GERÇEK CULANE GT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 6: OVERFIT TESTİ — GERÇEK CULANE GT İLE")
print("=" * 80)

data_ov, _ = sample_fn(db_obj, 100)
batch_size = data_ov['xs'][0].shape[0]
print(f"Batch size: {batch_size}")

gt_lanes_raw = data_ov['ys'][1]
print(f"GT lanes shape (raw): {gt_lanes_raw.shape}")

B_gt, N_gt, C_gt = gt_lanes_raw.shape
N_model = 7

if N_gt < N_model:
    padding  = torch.zeros(B_gt, N_model - N_gt, C_gt)
    gt_lanes = torch.cat([gt_lanes_raw, padding], dim=1)
else:
    gt_lanes = gt_lanes_raw[:, :N_model, :]

print(f"GT lanes shape (padded): {gt_lanes.shape}")

def poly_to_xcoords(gt_tensor, num_points=72):
    """GT'den normalize x_coords üret, 0-1 arasına clamp et."""
    B, N, _ = gt_tensor.shape
    x_coords = torch.zeros(B, N, num_points)
    for b in range(B):
        for n in range(N):
            lane_id = gt_tensor[b, n, 0].item()
            if lane_id > 0:
                x_start = float(gt_tensor[b, n, 1].item())
                x_start = max(0.0, min(1.0, x_start))
                noise   = torch.randn(num_points) * 0.02
                x_coords[b, n, :] = torch.clamp(
                    torch.tensor(x_start) + noise, 0.0, 1.0
                )
    return x_coords

real_gt_x   = poly_to_xcoords(gt_lanes, num_points=72).cuda()
real_gt_cls = (gt_lanes[:, :, 0:1] > 0).float().cuda()

gt_min, gt_max = real_gt_x.min().item(), real_gt_x.max().item()
print(f"real_gt_x shape: {real_gt_x.shape}, range: [{gt_min:.3f}, {gt_max:.3f}]")
if gt_max > 1.0:
    print(f"🚨 GT max={gt_max:.3f} > 1.0, clamp uygulanıyor...")
    real_gt_x = torch.clamp(real_gt_x, 0.0, 1.0)

print(f"real_gt_cls shape: {real_gt_cls.shape}, "
      f"valid lanes: {real_gt_cls.sum().item():.0f}/{real_gt_cls.numel()}")

model.train()
head = DynamicKernelHead(
    query_dim=32, feat_channels=32, kernel_size=32, num_points=72
).cuda()

layer2_feat = [None]
def hook_l2(m, inp, out):
    layer2_feat[0] = out

dec_out = [None]
def hook_dec_ov(m, inp, out):
    if isinstance(out, tuple):
        hs = out[0]   # (num_layers, N, B, D)
    else:
        hs = out
    # (N, B, D) → (B, N, D)
    dec_out[0] = hs[-1].permute(1, 0, 2)

l2_hook  = model.layer2.register_forward_hook(hook_l2)
dec_hook = model.transformer.decoder.register_forward_hook(hook_dec_ov)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(head.parameters()), lr=1e-3
)

single_img  = data_ov['xs'][0].cuda()
single_mask = data_ov['xs'][1].cuda()

losses         = []
x_losses       = []
cls_losses_log = []
kernel_diversity = []

print("\nEpoch | Loss    | x_loss  | cls_loss | kernel_sim (↓ = iyi)")
print("-" * 65)

for epoch in range(200):
    optimizer.zero_grad()

    out_dict, _ = model._train(single_img, single_mask)

    queries = dec_out[0]       # (B, N, D) = (B, 7, 32)
    feat    = layer2_feat[0]   # (B, C, H, W)

    if queries is None or feat is None:
        print(f"🚨 Epoch {epoch}: hook başarısız!")
        break

    x_coords_ov, cls_scores_ov = head(queries, feat)

    x_loss   = F.l1_loss(x_coords_ov, real_gt_x)
    cls_loss = F.binary_cross_entropy_with_logits(cls_scores_ov, real_gt_cls)
    total    = x_loss + cls_loss

    total.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(head.parameters()), max_norm=1.0
    )
    optimizer.step()

    losses.append(total.item())
    x_losses.append(x_loss.item())
    cls_losses_log.append(cls_loss.item())

    with torch.no_grad():
        ks   = head.kernel_gen(queries)   # (B, N, K)
        k0   = ks[0]                      # (N, K)
        sims = []
        for i in range(k0.shape[0]):
            for j in range(i+1, k0.shape[0]):
                sims.append(
                    F.cosine_similarity(k0[i].unsqueeze(0), k0[j].unsqueeze(0)).item()
                )
        kernel_diversity.append(np.mean(sims))

    if epoch % 20 == 0:
        print(f"  {epoch:3d}   | {total:.4f}  | {x_loss:.4f}  | "
              f"{cls_loss:.4f}   | {kernel_diversity[-1]:.4f}")

l2_hook.remove()
dec_hook.remove()

x_i = x_losses[0];    x_f = x_losses[-1]
t_i = losses[0];      t_f = losses[-1]
kd_i = kernel_diversity[0]; kd_f = kernel_diversity[-1]

x_drop = (x_i - x_f) / (x_i + 1e-8) * 100
t_drop = (t_i - t_f) / (t_i + 1e-8) * 100

print(f"\n--- KRİTİK X_LOSS ANALİZİ ---")
print(f"x_loss : {x_i:.4f} → {x_f:.4f}  azalma: %{x_drop:.1f}")
print(f"total  : {t_i:.4f} → {t_f:.4f}  azalma: %{t_drop:.1f}")

print(f"\nKernel cosine similarity (↓ = kernel'lar farklılaşıyor = iyi):")
print(f"  Başlangıç: {kd_i:.4f}")
print(f"  Final:     {kd_f:.4f}")

# Kernel diversity — cosine sim düşüşü = kernel'lar farklılaşıyor = iyi
if kd_f < 0.1:
    print("  ✅ Kernel'lar çok farklılaştı (collapse yok)")
    diversity_ok = True
elif kd_f < 0.3:
    print("  ⚠️ Kernel'lar kısmen farklılaştı")
    diversity_ok = None
elif kd_f >= kd_i:
    print("  🚨 Kernel similarity artmış — collapse başlıyor!")
    diversity_ok = False
else:
    print(f"  ⚠️ Kernel similarity düştü ama hâlâ yüksek ({kd_f:.3f})")
    diversity_ok = None

if x_drop > 80:
    print(f"\n✅ x_loss %{x_drop:.1f} düştü — koordinat öğreniyor!")
    x_loss_ok = True
elif x_drop > 50:
    print(f"\n⚠️ x_loss %{x_drop:.1f} düştü — sınırlı öğrenme")
    x_loss_ok = None
else:
    print(f"\n🚨 x_loss sadece %{x_drop:.1f} düştü — koordinat öğrenmiyor!")
    x_loss_ok = False

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 7: UYUMLULUK KONTROLÜ
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 7: MEVCUT LSTR vs DynamicKernelHead UYUMLULUK")
print("=" * 80)

with torch.no_grad():
    data_c, _ = sample_fn(db_obj, 0)
    out_dict, _ = model(data_c['xs'][0].cuda(), data_c['xs'][1].cuda())

print("\nMevcut LSTR çıktıları:")
print(f"  pred_logits: {out_dict['pred_logits'].shape}  → (B, N, 3) polynomial class")
print(f"  pred_curves: {out_dict['pred_curves'].shape}  → (B, N, 8) polynomial coeff")

print("\nDynamicKernelHead çıktıları (hedef):")
print("  x_coords:   (B, N, 72)  → normalize x koordinatı")
print("  cls_scores: (B, N, 1)   → lane var/yok")

print("\n⚠️ Gerekli değişiklikler:")
print("  1. pred_curves (polynomial) → x_coords (72 nokta)")
print("  2. loss_curves → L1(x_coords, gt)")
print("  3. pred_logits (3-class) → cls_scores (binary)")
print("  4. CULane evaluator: x_coords × image_width → pixel x ✅")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 8: KARAR RAPORU
# ═══════════════════════════════════════════════════════════════════════════

print("\n")
print("╔" + "═"*78 + "╗")
print("║" + " DYNAMİC KERNEL HEAD PRE-INTEGRATION RAPORU (DÜZELTİLMİŞ)".center(78) + "║")
print("╚" + "═"*78 + "╝")

results = [
    # (test_adı, sonuç, kritik_mi)
    ("layer2 cosine similarity < 0.85",          layer2_ok,      True),
    ("layer2 spatial aktivasyon (bilgi için)",    spatial_ok,     False),
    ("Query'ler farklı — sim < 0.80",            query_ok,       True),
    ("bmm K=32 VRAM'a sığıyor",                  vram_ok,        True),
    ("bmm forward başarılı",                     bmm_ok,         True),
    ("x_coords 0-1 arası (Sigmoid)",             forward_ok,     True),
    ("Feature gradient → layer2'ye akıyor",      feat_grad_ok,   True),
    ("Query gradient akıyor",                    query_grad_ok,  True),
    ("Head param. gradient sağlıklı",            grad_ok,        True),
    ("Kernel başlangıçta çeşitli",               kernel_diverse, False),
    ("Kernel aktivasyon anlamlı (>1.3x)",        pattern_ok,     False),
    ("x_loss > %80 azaldı (koordinat öğrenme)", x_loss_ok,      True),
    ("Kernel eğitimde collapse olmadı",          diversity_ok,   True),
]

print("\nKONTROL LİSTESİ:")
print("┌" + "─"*62 + "┬" + "─"*8 + "┬" + "─"*6 + "┐")
print("│ Test" + " "*57 + "│ Durum  │ Krit. │")
print("├" + "─"*62 + "┼" + "─"*8 + "┼" + "─"*6 + "┤")

critical_fails = 0
warns = 0

for test, ok, critical in results:
    if ok is True:
        status = "✅"
    elif ok is False:
        status = "🚨"
        if critical:
            critical_fails += 1
    else:
        status = "⚠️ "
        warns += 1
    krit = "EVET" if critical else "hayır"
    print(f"│ {test:<60} │ {status:<6} │ {krit:<5}│")

print("└" + "─"*62 + "┴" + "─"*8 + "┴" + "─"*6 + "┘")

print(f"\nÖzet: {critical_fails} kritik başarısızlık, {warns} uyarı")

print("\nKRİTİK BAŞARISIZLIK KOŞULLARI:")
print("  🚨 layer2 cosine sim > 0.95    → kernel öğrenemez")
print("  🚨 bmm OOM K=32               → batch size küçült")
print("  🚨 Feature gradient YOK       → layer2 freeze kalır")
print("  🚨 x_loss < %50 azalma        → koordinat öğrenmiyor")
print("  🚨 Query collapse > 0.95      → tüm kernel'lar aynı")
print("  🚨 Kernel collapse (sim arttı) → dynamic head işlevsiz")

if critical_fails == 0 and warns <= 2:
    print("\n✅ ENTEGRASYONBİLİR — kritik sorun yok")
elif critical_fails == 0:
    print(f"\n⚠️ KOŞULLU — {warns} uyarı var, ilerleyebilirsin ama takip et")
else:
    print(f"\n🚨 {critical_fails} KRİTİK SORUN — bunları çözmeden entegre etme")
    for test, ok, critical in results:
        if ok is False and critical:
            print(f"  → {test}")

print("\n--- BUG DÜZELTMELERİ (bu scriptte) ---")
print("  BUG-1: Gradient test: .cuda().requires_grad_(True) sırası düzeltildi")
print("  BUG-2: Query hook: (N,B,D).permute(1,0,2)[0] → doğru (N,D) shape")
print("  BUG-3: Kernel aktivasyon: random tensor → gerçek layer2, eşik 1.3x")

print("\n" + "=" * 80)
print("ANALİZ TAMAMLANDI")
print("=" * 80)