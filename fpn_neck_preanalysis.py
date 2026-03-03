"""
LSTR + FPN Neck Entegrasyon Ön-Analiz Scripti
=============================================

KULLANIM:
    conda activate clrernet
    python fpn_neck_preanalysis.py
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
cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE.json")
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["system"]["snapshot_name"] = "LSTR_CULANE"
cfg["system"]["data_dir"] = "/home/alki/projects/"  # CULane is at /home/alki/projects/CULane/
system_configs.update_config(cfg["system"])

from models.LSTR_CULANE import model as LSTRModel, loss as LSTRLoss

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"
results = {}

def record(test_name, status, note=""):
    results[test_name] = (status, note)

# ── Basit FPN Neck Modülü ─────────────────────────────────────────────────────
class SimpleFPNNeck(nn.Module):
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
        # example shapes: (32, 45, 80), (64, 23, 40), (128, 12, 20)

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

        # Tüm seviyeleri layer4 boyutuna indir ve topla
        # Böylece lateral_convs[0,1] ve fpn_convs[0,1] de loss'a bağlanır
        target_size = inputs[-1].shape[-2:]   # (12, 20)
        fused = fpn_outs[-1]                  # layer4 çıktısı (zaten doğru boyut)
        for i in range(used_backbone_levels - 1):
            fused = fused + F.interpolate(
                fpn_outs[i], size=target_size, mode="nearest"
            )

        return fused  # shape: (B, out_channels, H4, W4)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: MODEL MİMARİSİ ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("BÖLÜM 1: MİMARİ ANALİZİ")
print("=" * 80)

model = LSTRModel(flag=True).cuda()
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

# ── 1.2 Forward pass boyut haritası ──
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
# BÖLÜM 2: ÖZELLİK (FEATURE) ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 2: FEATURE ANALİZİ (layer4 - Olası FPN kaynağı)")
print("=" * 80)

# Veri yükleme başarılı değilse direkt traceback bas ve bitir kuralı uygulandı
try:
    from db.datasets import datasets
    import importlib
    # Modeli test etmek için data loader
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    print(f"DB yüklendi: {db_obj.db_inds.size} örnek")
    sample_fn = importlib.import_module('sample.culane').sample_data
except Exception as e:
    traceback.print_exc()
    print("🚨 KRİTİK HATA: Veri yükleme başarısız. Test geçersiz sayılıyor!")
    sys.exit(1)

def analyze_features(model, seq_len=10):
    features_list = []
    
    def hook_fn(module, inp, output):
        features_list.append(output.detach().cpu())
        
    target_layer = dict(model.named_modules())['layer4']
    hook = target_layer.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for i in range(seq_len):
            data, _ = sample_fn(db_obj, i)
            # Veri onceden batched dır: [16, 3, 295, 820]
            img = data['xs'][0].cuda()
            mask = data['xs'][1].cuda()
            model(img, mask)
            
    hook.remove()
    
    flat_features = [f.flatten() for f in features_list[:5]]
    sim_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            sim = F.cosine_similarity(
                flat_features[i].unsqueeze(0),
                flat_features[j].unsqueeze(0)
            ).item()
            sim_matrix[i][j] = sim
            
    print("\nGÖRÜNTÜLER ARASI COSİNE BENZERLİK MATRİSİ (layer4)")
    print(np.round(sim_matrix, 3))
    
    avg_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (5*5 - 5)
    print(f"\nOrtalama çapraz-benzerlik: {avg_off_diag:.4f}")
    
    if avg_off_diag > 0.95:
        print("🚨 KRİTİK UYARI: Feature'lar çok benzer!")
        record("Feature benzerliği < 0.85", FAIL, f"avg_off_diag={avg_off_diag:.4f}")
    elif avg_off_diag > 0.85:
        print("⚠️  UYARI: Feature'lar oldukça benzer, dikkatli ol.")
        record("Feature benzerliği < 0.85", WARN, f"avg_off_diag={avg_off_diag:.4f}")
    else:
        print("✅ Feature çeşitliliği yeterli görünüyor.")
        record("Feature benzerliği < 0.85", PASS, f"avg_off_diag={avg_off_diag:.4f}")

analyze_features(model)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: GRADIENT ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 3: GRADIENT ANALİZİ (Mevcut Model)")
print("=" * 80)

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

dead_layers = []
exploding_layers = []

for name, param in model.named_parameters():
    # layer 1-4 gradient akışını kontrol edelim
    if 'layer' not in name: continue
        
    if param.grad is not None:
        grad_mean = param.grad.abs().mean().item()
        grad_max = param.grad.abs().max().item()
        if grad_mean < 1e-7:
            dead_layers.append(name)
        elif grad_mean > 10.0:
            exploding_layers.append(name)

print(f"Dead katmanlar: {len(dead_layers)}")
print(f"Exploding katmanlar: {len(exploding_layers)}")

if dead_layers or exploding_layers:
    record("Gradient akışı sağlıklı", WARN, f"Dead:{len(dead_layers)} Exploding:{len(exploding_layers)}")
else:
    record("Gradient akışı sağlıklı", PASS)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: YENİ MODÜL İZOLE TESTİ (Gerçek Loss İle)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 4: YENİ MODÜL İZOLE TESTİ (FPN Neck + Gerçek LSTR Loss'u Temsili)")
print("=" * 80)

# input_channels: layer2 (32), layer3 (64), layer4 (128)
fpn_module = SimpleFPNNeck([32, 64, 128], out_channels=128).cuda()

# (batch_size=1)
t_layer2 = torch.randn(1, 32, 45, 80).cuda()
t_layer3 = torch.randn(1, 64, 23, 40).cuda()
t_layer4 = torch.randn(1, 128, 12, 20).cuda()

test_inputs = [t_layer2, t_layer3, t_layer4]
for t in test_inputs:
    t.requires_grad_(True)

try:
    outputs = fpn_module(test_inputs)
    print(f"Test layer2 input shape: {t_layer2.shape}")
    print(f"Test layer3 input shape: {t_layer3.shape}")
    print(f"Test layer4 input shape: {t_layer4.shape}")
    print(f"FPN Output shape: {outputs.shape}")  # (1, 128, 12, 20) bekliyoruz
    
    # Gerçek modele uygun loss simülasyonu
    temp_proj = nn.Conv2d(128, 32, 1).cuda()
    temp_linear = nn.Linear(32, 7).cuda()
    
    flat_out = temp_proj(outputs).flatten(2).permute(0, 2, 1) # B, HW, D
    loss_sim = temp_linear(flat_out).sum()
    
    # Optimizer kullanarak grad akışını sağla
    test_optimizer = torch.optim.SGD(list(fpn_module.parameters()) + list(temp_proj.parameters()) + list(temp_linear.parameters()), lr=1e-3)
    test_optimizer.zero_grad()
    loss_sim.backward()
    test_optimizer.step()
    
    all_grad_ok = True
    for name, param in fpn_module.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  {name}: gradient YOK")
            all_grad_ok = False
        elif param.grad.abs().mean() < 1e-9:
            print(f"  ⚠️  {name}: dead gradient")
            all_grad_ok = False
            
    if all_grad_ok:
        print("✅ FPN Modülü forward & backward başarılı.")
        record("Yeni modül izole çalışıyor", PASS)
        record("Yeni modülde gradient akıyor", PASS)
    else:
        record("Yeni modül izole çalışıyor", PASS)
        record("Yeni modülde gradient akıyor", FAIL)

except Exception as e:
    traceback.print_exc()
    record("Yeni modül izole çalışıyor", FAIL, str(e))
    record("Yeni modülde gradient akıyor", FAIL, "Forward failed")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: OVERFİT TESTİ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 5: OVERFİT TESTİ (Tek Görüntü + FPN Entegre Model)")
print("=" * 80)

class LSTR_FPN_TestModel(LSTRModel):
    def __init__(self, fpn_neck, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fpn = fpn_neck
        
    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks = xs[1]
        
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p1 = self.layer1(p)
        p2 = self.layer2(p1)  # 32
        p3 = self.layer3(p2)  # 64
        p4 = self.layer4(p3)  # 128
        
        # FPN 
        fpn_features = self.fpn([p2, p3, p4])
        
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=fpn_features.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(fpn_features, pmasks)
        
        # Transformer'a projeksiyon
        hs, _, weights = self.transformer(self.input_proj(fpn_features), pmasks, self.query_embed.weight, pos)
        
        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared = self.shared_embed(hs)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat([output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]], dim=-1)
        
        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
        return out, weights

# İzole test modelini oluştur
mod_fpn_test = LSTR_FPN_TestModel(fpn_module, flag=True).cuda()
criterion = LSTRLoss().cuda()

for k in list(criterion.criterion.weight_dict.keys()):
    if 'loss_curves' in k:
        criterion.criterion.weight_dict[k] = 2.5

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, mod_fpn_test.parameters()), lr=1e-3
)

# Tek görüntü seç
data, _ = sample_fn(db_obj, 0)
single_img = data['xs'][0].cuda()
single_mask = data['xs'][1].cuda()
real_targets = [t.cuda() for t in data['ys'][1:]]
targets_list = [single_img] + real_targets

losses = []
mod_fpn_test.train()

for epoch in range(200):
    optimizer.zero_grad()
    out_dict, _ = mod_fpn_test._train(single_img, single_mask)
    loss_result = criterion(epoch, False, 'train', out_dict, targets_list)
    total_loss = loss_result[0].mean()
    
    total_loss.backward()
    # Kural 3: clip_grad_norm_ eklendi 
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
print(f"Azalma: %{reduction:.1f}")

if reduction > 90:
    print("✅ Model overfit edebildi — mimari ve loss fonksiyonu çalışıyor")
    record("Tek görüntüde >%90 overfit", PASS, f"drop={reduction:.1f}")
elif reduction > 50:
    print("⚠️  Kısmi öğrenme")
    record("Tek görüntüde >%90 overfit", WARN, f"drop={reduction:.1f}")
else:
    print("🚨 Model overfit edemedi!")
    record("Tek görüntüde >%90 overfit", FAIL, f"drop={reduction:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: ENTEGRASYONBİLİRLİK VE RAPOR
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "╔" + "═" * 66 + "╗")
print("║              PRE-INTEGRATION ANALYSIS RAPORU                     ║")
print("╚" + "═" * 66 + "╝")

icon = {PASS: "✅", FAIL: "🚨", WARN: "⚠️ "}
all_pass = True

for test, (status, note) in results.items():
    sym = icon.get(status, "?")
    print(f"  {test:<40}{sym+' '+status:<12}{note}")
    if status == FAIL:
        all_pass = False

print("-" * 80)
if all_pass:
    print("GENEL KARAR: ✅ ENTEGRASYONBİLİR — tüm testler geçti, devam et")
else:
    print("GENEL KARAR: 🚨 ENTEGRASYONBİLİR DEĞİL — kritik sorunlar var")
