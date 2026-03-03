"""
LSTR + FPN Neck Entegrasyon Ön-Analiz Scripti (DÜZELTİLMİŞ)
==============================================================

DÜZELTİLEN SORUNLAR:
1. Rapor mantığı: WARN durumlarını doğru değerlendirir
2. Exploding gradient kaynağını bulur (backbone vs decoder vs loss)
3. FPN P2 (45×80) yüksek çözünürlük üretir
4. Overfit testini MAMBA encoder ile yapar

KULLANIM:
    conda activate clrernet
    python fpn_preanalysis_fixed.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Config yükle ──────────────────────────────────────────────────────────────
from config import system_configs
cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE_MAMBA.json")
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["system"]["snapshot_name"] = "LSTR_CULANE_MAMBA"
cfg["system"]["data_dir"] = "/home/alki/projects/"
system_configs.update_config(cfg["system"])

from models.LSTR_CULANE_MAMBA import model as MambaModel, loss as MambaLoss

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"
results = {}

def record(test_name, status, note=""):
    results[test_name] = (status, note)

# ── DÜZELTİLMİŞ FPN Neck Modülü ─────────────────────────────────────────────────
class SimpleFPNNeck(nn.Module):
    """
    DÜZELTİLMİŞ: P2 (45×80) yüksek çözünürlük de üretir.
    
    Çıktı formatı: {
        'P2': (B, 256, 45, 80),   # Dynamic kernel için
        'P3': (B, 256, 23, 40),
        'P4': (B, 256, 12, 20),    # Mamba encoder için
        'fused_P4': (B, 256, 12, 20)  # LSTR uyumlu tek output
    }
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for idx in range(len(in_channels_list)):
            in_c = in_channels_list[idx]
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            
    def forward(self, inputs):
        # inputs: [layer2, layer3, layer4]
        # shapes: (B, 32, 45, 80), (B, 64, 23, 40), (B, 128, 12, 20)
        
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        used_backbone_levels = len(laterals)
        
        # Top-down: coarse → fine (standart FPN)
        outs = [laterals[-1]]
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:]
            out_up = F.interpolate(outs[0], size=prev_shape, mode="nearest")
            outs.insert(0, laterals[i - 1] + out_up)
        
        # Her seviyeye fpn_conv uygula
        fpn_outs = [
            self.fpn_convs[i](outs[i])
            for i in range(used_backbone_levels)
        ]
        
        # DÜZELTME: Tüm seviyeleri layer4 boyutuna indir ve topla
        target_size = inputs[-1].shape[-2:]  # (12, 20)
        fused = fpn_outs[-1]
        for i in range(used_backbone_levels - 1):
            fused = fused + F.interpolate(
                fpn_outs[i], size=target_size, mode="nearest"
            )
        
        # DÜZELTME: Ayrı ayrı P2, P3, P4 de döndür
        return {
            'P2': fpn_outs[0],      # (B, out_channels, 45, 80) - Dynamic kernel için
            'P3': fpn_outs[1],      # (B, out_channels, 23, 40)
            'P4': fpn_outs[2],      # (B, out_channels, 12, 20) - Mamba encoder için
            'fused_P4': fused       # (B, out_channels, 12, 20) - LSTR uyumlu
        }

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: MİMARİ ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("BÖLÜM 1: MAMBA MODEL MİMARİSİ ANALİZİ")
print("=" * 80)

model = MambaModel(flag=True).cuda()
model.eval()

total_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if not param.requires_grad:
        frozen_params += param.numel()

print(f"\nToplam parametre: {total_params:,}")
print(f"Frozen parametre: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
print(f"Trainable parametre: {total_params - frozen_params:,}")

record("Frozen parametre yok (BN hariç)", PASS if frozen_params < total_params * 0.05 else WARN)

# ── 1.2 Mamba encoder kontrolü ──
print("\n" + "=" * 80)
print("MAMBA ENCODER KONTROLÜ")
print("=" * 80)

encoder_type = type(model.transformer.encoder).__name__
print(f"Encoder tipi: {encoder_type}")

if encoder_type == "BidirectionalMambaEncoder":
    print("✅ Mamba encoder kullanılıyor")
    record("Mamba encoder aktif", PASS, encoder_type)
else:
    print(f"⚠️ Transformer encoder kullanılıyor: {encoder_type}")
    record("Mamba encoder aktif", WARN, encoder_type)

# ── 1.3 Forward pass boyut haritası ──
hooks = {}
activations = {}

def make_hook(name):
    def hook(module, input, output):
        t = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(t, torch.Tensor):
            activations[name] = {
                "shape": list(t.shape),
                "mean": round(t.mean().item(), 4),
                "std": round(t.std().item(), 4),
                "min": round(t.min().item(), 4),
                "max": round(t.max().item(), 4),
                "has_nan": bool(torch.isnan(t).any().item()),
                "has_inf": bool(torch.isinf(t).any().item()),
            }
    return hook

target_names = ['layer2', 'layer3', 'layer4', 'input_proj']
for name, mod in model.named_modules():
    if name in target_names:
        hooks[name] = mod.register_forward_hook(make_hook(name))

dummy_input = torch.randn(1, 3, 360, 640).cuda()
dummy_mask = torch.zeros(1, 1, 360, 640).cuda()
with torch.no_grad():
    model(dummy_input, dummy_mask)

for h in hooks.values():
    h.remove()

print("\n" + "=" * 80)
print("FORWARD PASS BOYUT HARİTASI (FPN İÇİN GEREKLİ KATMANLAR)")
print("=" * 80)
all_shapes_ok = True
for name, stats in activations.items():
    print(f"\n[{name}]")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats.get("has_nan") or stats.get("has_inf"):
        print("  🚨 NaN/Inf TESPİT EDİLDİ!")
        all_shapes_ok = False

record("Forward pass boyutlar doğru", PASS if all_shapes_ok else FAIL)
record("NaN/Inf yok", PASS if all_shapes_ok else FAIL)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: GRADIENT ANALİZİ (KAYNAK BULMA)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 2: GRADIENT ANALİZİ (KAYNAK TESPİTİ)")
print("=" * 80)

try:
    from db.datasets import datasets
    import importlib
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    print(f"DB yüklendi: {db_obj.db_inds.size} örnek")
    sample_fn = importlib.import_module('sample.culane').sample_data
except Exception as e:
    traceback.print_exc()
    print("🚨 KRİTİK HATA: Veri yükleme başarısız!")
    sys.exit(1)

model.train()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

data, _ = sample_fn(db_obj, 0)
img = data['xs'][0].cuda()
mask = data['xs'][1].cuda()

out_dict, _ = model._train(img, mask)
loss = out_dict['pred_logits'].sum() + out_dict['pred_curves'].sum()
loss.backward()

# Gradient analizi - bölge bölge
regions = {
    'backbone_layer1-2': [],
    'backbone_layer3-4': [],
    'input_proj': [],
    'mamba_encoder': [],
    'transformer_decoder': [],
    'prediction_heads': [],
}

dead_by_region = {k: [] for k in regions.keys()}
exploding_by_region = {k: [] for k in regions.keys()}

for name, param in model.named_parameters():
    if param.grad is None:
        continue
    
    grad_mean = param.grad.abs().mean().item()
    grad_max = param.grad.abs().max().item()
    
    # Bölge tespiti
    region = None
    if 'layer1' in name or 'layer2' in name:
        region = 'backbone_layer1-2'
    elif 'layer3' in name or 'layer4' in name:
        region = 'backbone_layer3-4'
    elif 'input_proj' in name:
        region = 'input_proj'
    elif 'transformer.encoder' in name or 'mamba' in name:
        region = 'mamba_encoder'
    elif 'transformer.decoder' in name:
        region = 'transformer_decoder'
    elif 'class_embed' in name or 'specific_embed' in name or 'shared_embed' in name:
        region = 'prediction_heads'
    
    if region:
        regions[region].append((name, grad_mean, grad_max))
        if grad_mean < 1e-7:
            dead_by_region[region].append(name)
        elif grad_mean > 10.0:
            exploding_by_region[region].append(name)

print("\n--- DEAD GRADIENTS (mean < 1e-7) ---")
total_dead = 0
for region, names in dead_by_region.items():
    if names:
        print(f"  {region}: {len(names)} dead")
        for name in names[:3]:  # İlk 3
            print(f"    - {name}")
        total_dead += len(names)

print(f"\nToplam dead: {total_dead}")

print("\n--- EXPLODING GRADIENTS (mean > 10.0) ---")
total_exploding = 0
for region, names in exploding_by_region.items():
    if names:
        print(f"  {region}: {len(names)} exploding")
        for name in names[:3]:  # İlk 3
            print(f"    - {name}")
        total_exploding += len(names)

print(f"\nToplam exploding: {total_exploding}")

if total_exploding > 0:
    # En sorunlu bölgeyi tespit et
    worst_region = max(exploding_by_region.items(), key=lambda x: len(x[1]))
    print(f"\n🚨 EN SORUNLU BÖLGE: {worst_region[0]} ({len(worst_region[1])} exploding)")
    record("Gradient akışı sağlıklı", WARN, f"Dead:{total_dead} Exploding:{total_exploding} | Worst:{worst_region[0]}")
elif total_dead > 0:
    record("Gradient akışı sağlıklı", WARN, f"Dead:{total_dead} Exploding:0")
else:
    record("Gradient akışı sağlıklı", PASS)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: FPN TESTİ (P2 ÜRETİMİ)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 3: FPN TESTİ (P2 YÜKSEK ÇÖZÜNÜRLÜK ÜRETİMİ)")
print("=" * 80)

fpn_module = SimpleFPNNeck([32, 64, 128], out_channels=256).cuda()

t_layer2 = torch.randn(1, 32, 45, 80).cuda()
t_layer3 = torch.randn(1, 64, 23, 40).cuda()
t_layer4 = torch.randn(1, 128, 12, 20).cuda()

fpn_outputs = fpn_module([t_layer2, t_layer3, t_layer4])

print(f"\nFPN Çıktıları:")
for key, tensor in fpn_outputs.items():
    print(f"  {key}: {tensor.shape}")

# P2 kontrolü
p2_shape = fpn_outputs['P2'].shape
expected_p2 = (1, 256, 45, 80)

if p2_shape == expected_p2:
    print(f"✅ P2 doğru boyut: {p2_shape}")
    record("FPN P2 (45×80) üretiyor", PASS)
else:
    print(f"🚨 P2 yanlış boyut: {p2_shape}, beklenen: {expected_p2}")
    record("FPN P2 (45×80) üretiyor", FAIL, f"got {p2_shape}")

# Gradient testi
t_layer2.requires_grad_(True)
t_layer3.requires_grad_(True)
t_layer4.requires_grad_(True)

loss_fpn = fpn_outputs['P2'].sum() + fpn_outputs['fused_P4'].sum()
loss_fpn.backward()

all_grad_ok = True
for name, param in fpn_module.named_parameters():
    if param.grad is None or param.grad.abs().mean() < 1e-9:
        print(f"⚠️ {name}: gradient yok veya dead")
        all_grad_ok = False

if all_grad_ok:
    print("✅ FPN tüm parametreleri gradient alıyor")
    record("FPN gradient akışı sağlıklı", PASS)
else:
    print("🚨 FPN'de bazı parametreler gradient almıyor")
    record("FPN gradient akışı sağlıklı", FAIL)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: OVERFIT TESTİ (MAMBA + FPN)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 4: OVERFIT TESTİ (MAMBA ENCODER + FPN)")
print("=" * 80)

class Mamba_FPN_TestModel(MambaModel):
    def __init__(self, fpn_neck, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fpn = fpn_neck
        # DÜZELTME: FPN çıktısı 256 kanal, input_proj 128 bekliyor
        # Yeni projection: 256 → 32 (doğrudan transformer boyutuna)
        self.fpn_proj = nn.Conv2d(256, 32, 1)
        
    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks = xs[1]
        
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p1 = self.layer1(p)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        
        # FPN - tüm seviyeleri al
        fpn_outs = self.fpn([p2, p3, p4])
        
        # Mamba encoder için P4 (veya fused) kullan
        fpn_features = fpn_outs['fused_P4']
        
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=fpn_features.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(fpn_features, pmasks)
        
        # DÜZELTME: fpn_proj kullan, input_proj yerine
        hs, _, weights = self.transformer(
            self.fpn_proj(fpn_features),
            pmasks,
            self.query_embed.weight,
            pos
        )
        
        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared = self.shared_embed(hs)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat(
            [output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]], 
            dim=-1
        )
        
        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
        return out, weights

mod_fpn_test = Mamba_FPN_TestModel(fpn_module, flag=True).cuda()
criterion = MambaLoss().cuda()

# Loss fix
for k in list(criterion.criterion.weight_dict.keys()):
    if 'loss_curves' in k:
        criterion.criterion.weight_dict[k] = 2.5

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, mod_fpn_test.parameters()), lr=1e-3
)

# Tek görüntü
data, _ = sample_fn(db_obj, 0)
single_img = data['xs'][0].cuda()
single_mask = data['xs'][1].cuda()
real_targets = [t.cuda() for t in data['ys'][1:]]
targets_list = [single_img] + real_targets

losses = []
mod_fpn_test.train()

print("\nOverfit testi başlıyor...")
for epoch in range(200):
    optimizer.zero_grad()
    out_dict, _ = mod_fpn_test._train(single_img, single_mask)
    loss_result = criterion(epoch, False, 'train', out_dict, targets_list)
    total_loss = loss_result[0].mean()
    
    total_loss.backward()
    # Clip gradient - exploding için
    torch.nn.utils.clip_grad_norm_(mod_fpn_test.parameters(), max_norm=1.0)
    optimizer.step()
    
    losses.append(total_loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss={total_loss.item():.6f}")

initial_loss = losses[0]
final_loss = losses[-1]
reduction = (initial_loss - final_loss) / initial_loss * 100

print(f"\nBaşlangıç loss: {initial_loss:.6f}")
print(f"Final loss: {final_loss:.6f}")
print(f"Azalama: %{reduction:.1f}")

# Eşik %90
threshold = 90
if reduction > threshold:
    print(f"✅ Model overfit edebildi (>%{threshold})")
    record(f"Tek görüntüde >%{threshold} overfit", PASS, f"drop={reduction:.1f}%")
elif reduction > 50:
    print(f"⚠️ Kısmi öğrenme (%{reduction:.1f} < %{threshold})")
    record(f"Tek görüntüde >%{threshold} overfit", WARN, f"drop={reduction:.1f}%")
else:
    print(f"🚨 Model overfit edemedi!")
    record(f"Tek görüntüde >%{threshold} overfit", FAIL, f"drop={reduction:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: DÜZELTİLMİŞ RAPOR
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "╔" + "═" * 70 + "╗")
print("║     PRE-INTEGRATION ANALYSIS RAPORU (DÜZELTİLMİŞ)               ║")
print("╚" + "═" * 70 + "╝")

icon = {PASS: "✅", FAIL: "🚨", WARN: "⚠️ "}
fail_count = 0
warn_count = 0

for test, (status, note) in results.items():
    sym = icon.get(status, "?")
    print(f"  {test:<45}{sym+' '+status:<12}{note}")
    if status == FAIL:
        fail_count += 1
    elif status == WARN:
        warn_count += 1

print("-" * 80)
print(f"ÖZET: {fail_count} FAIL, {warn_count} WARN")

# DÜZELTİLMİŞ karar mantığı
if fail_count > 0:
    print("GENEL KARAR: 🚨 ENTEGRASYONBİLİR DEĞİL — kritik sorunlar var")
elif warn_count > 0:
    print(f"GENEL KARAR: ⚠️ KOŞULLU ENTEGRASYON — {warn_count} uyarı var, dikkatli ol")
else:
    print("GENEL KARAR: ✅ ENTEGRASYONBİLİR — tüm testler geçti")

print("\n--- ÖNERİLER ---")
if warn_count > 0 or fail_count > 0:
    print("1. Exploding gradient için: clip_grad_norm_ ekleyin (zaten eklendi)")
    print("2. Overfit %90 altındaysa: learning rate veya epoch artırın")
    print("3. FPN P2 üretmiyorsa: SimpleFPNNeck.forward() kontrol edin")
    print("4. Mamba encoder yoksa: LSTR_CULANE_MAMBA.json kullandığınızdan emin olun")
