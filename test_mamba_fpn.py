"""
Mamba + FPN Model Test Scripti
==============================

Amaç: LSTR_CULANE_MAMBA_FPN modelinin doğruluğunu doğrula
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import system_configs

# Config yükle
cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE_MAMBA_FPN.json")
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["system"]["snapshot_name"] = "LSTR_CULANE_MAMBA_FPN"
cfg["system"]["data_dir"] = "/home/alki/projects/"
system_configs.update_config(cfg["system"])

from models.LSTR_CULANE_MAMBA_FPN import model as MambaFPNModel, loss as MambaFPNLoss

print("=" * 80)
print("MAMBA + FPN MODEL TEST")
print("=" * 80)

# Model oluştur
model = MambaFPNModel(flag=True).cuda()
model.eval()

print("\n--- 1. Model Parametreleri ---")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Toplam parametre: {total_params:,}")
print(f"Trainable: {trainable_params:,}")

# FPN parametreleri
fpn_params = sum(p.numel() for p in model.fpn.parameters())
fpn_proj_params = sum(p.numel() for p in model.fpn_proj.parameters())
print(f"FPN parametre: {fpn_params:,} ({fpn_params//1000}K)")
print(f"fpn_proj parametre: {fpn_proj_params:,}")
print(f"Toplam FPN + proj: {fpn_params + fpn_proj_params:,}")

# Mamba encoder parametreleri
mamba_enc_params = sum(p.numel() for p in model.transformer.encoder.parameters())
print(f"Mamba encoder parametre: {mamba_enc_params:,}")

print("\n--- 2. Forward Pass Test ---")
dummy_input = torch.randn(1, 3, 360, 640).cuda()
dummy_mask = torch.zeros(1, 1, 360, 640).cuda()

# FPN outputlarını kontrol et için hook
def fpn_hook(module, input, output):
    print(f"FPN outputs:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")

fpn_hook_handle = model.fpn.register_forward_hook(fpn_hook)

with torch.no_grad():
    out_dict, weights = model(dummy_input, dummy_mask)

fpn_hook_handle.remove()

print(f"pred_logits shape: {out_dict['pred_logits'].shape}")
print(f"pred_curves shape: {out_dict['pred_curves'].shape}")

print("\n--- 3. Gradient Flow Test ---")
model.train()

# Yeni forward pass grad için
out_dict, weights = model(dummy_input, dummy_mask)

# Küçük loss oluştur
loss = out_dict['pred_logits'].sum() + out_dict['pred_curves'].sum()
loss.backward()

# Gradient kontrolü
dead_count = 0
exploding_count = 0

print("\nKritik parametreler:")
for name, param in model.named_parameters():
    if param.grad is None:
        continue
    
    grad_mean = param.grad.abs().mean().item()
    
    if 'fpn' in name or 'mamba' in name:
        status = "OK"
        if grad_mean < 1e-7:
            status = "DEAD"
            dead_count += 1
        elif grad_mean > 10.0:
            status = "EXPLODING"
            exploding_count += 1
        print(f"  {name:50s} {status:12s} grad={grad_mean:.2e}")

print(f"\nGradient özeti: {dead_count} dead, {exploding_count} exploding")

print("\n--- 4. Overfit Test (Tek Görüntü) ---")
from db.datasets import datasets
import importlib

try:
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    sample_fn = importlib.import_module('sample.culane').sample_data
    
    criterion = MambaFPNLoss().cuda()
    # Lean FPN (64ch) için optimize edilmiş LR
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3  # Lean FPN için 1e-3 yeterli
    )
    
    data, _ = sample_fn(db_obj, 0)
    single_img = data['xs'][0].cuda()
    single_mask = data['xs'][1].cuda()
    real_targets = [t.cuda() for t in data['ys'][1:]]
    targets_list = [single_img] + real_targets
    
    losses = []
    max_epochs = 200  # Lean FPN ile 200 epoch yeterli
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        out_dict, _ = model._train(single_img, single_mask)
        loss_result = criterion(epoch, False, 'train', out_dict, targets_list)
        total_loss = loss_result[0].mean()
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={total_loss.item():.6f}")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nBaşlangıç: {initial_loss:.6f}")
    print(f"Final: {final_loss:.6f}")
    print(f"Azalama: %{reduction:.1f}")
    
    print(f"\n{'='*60}")
    if reduction > 90:
        print("✅ Overfit BAŞARILI (>90%) - Lean FPN model çalışıyor")
    elif reduction > 80:
        print(f"⚠️ Overfit kısmi başarılı (%{reduction:.1f}) - Kabul edilebilir")
    elif reduction > 60:
        print(f"⚠️ Orta seviye overfit (%{reduction:.1f}) - Geliştirme gerekli")
        print(f"   Öneri: lr=2e-3 dene veya 300 epoch çalıştır")
    else:
        print(f"❌ Overfit başarısız (%{reduction:.1f}) - Ciddi sorun var")
        
except Exception as e:
    print(f"Overfit testi hatası: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST TAMAMLANDI")
print("=" * 80)
