"""
Position Encoding + tgt Init Düzeltme Test
==========================================

İki kritik düzeltmeyi doğrula:
1. Position encoding scale: std≈0.1 (eskiden 22x memory)
2. tgt = query_embed * 0.1 (eskiden zeros)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import system_configs

# Mamba FPN config yükle
cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE_MAMBA_FPN.json")
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["system"]["snapshot_name"] = "LSTR_CULANE_MAMBA_FPN"
cfg["system"]["data_dir"] = "/home/alki/projects/"
system_configs.update_config(cfg["system"])

from models.LSTR_CULANE_MAMBA_FPN import model as MambaFPNModel, loss as MambaFPNLoss

print("=" * 80)
print("DÜZELTME DOĞRULAMA TESTİ")
print("=" * 80)

model = MambaFPNModel(flag=True).cuda()
model.train()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Position Encoding Scale
# ═══════════════════════════════════════════════════════════════════════════

print("\n--- TEST 1: Position Encoding Scale ---")

dummy_input = torch.randn(1, 3, 360, 640).cuda()
dummy_mask = torch.zeros(1, 1, 360, 640).cuda()

# Hook ile position encoding'i ve memory'i yakala
pos_captured = None
memory_captured = None

def hook_encoder(module, args, output):
    """Mamba encoder output (memory)"""
    global memory_captured
    if isinstance(output, tuple):
        memory_captured = output[0].clone()
    else:
        memory_captured = output.clone()

enc_hook = model.transformer.encoder.register_forward_hook(hook_encoder)

with torch.no_grad():
    out_dict, _ = model(dummy_input, dummy_mask)

enc_hook.remove()

# Position encoding direkt hesapla
pmasks = F.interpolate(dummy_mask[:, 0, :, :][None], 
                       size=(12, 20)).to(torch.bool)[0]
fpn_feat = torch.randn(1, 32, 12, 20).cuda()
pos = model.position_embedding(fpn_feat, pmasks)

pos_std = pos.std().item()
pos_l2 = torch.norm(pos).item()

print(f"Position Encoding:")
print(f"  Std: {pos_std:.4f}")
print(f"  L2 norm: {pos_l2:.2f}")

if memory_captured is not None:
    mem_std = memory_captured.std().item()
    mem_l2 = torch.norm(memory_captured).item()
    print(f"Memory (encoder output):")
    print(f"  Std: {mem_std:.4f}")
    print(f"  L2 norm: {mem_l2:.2f}")
    
    ratio = pos_l2 / (mem_l2 + 1e-8)
    print(f"\nPos/Memory L2 ratio: {ratio:.2f}")
    
    if ratio < 5.0:
        print("✅ Position encoding scale uyumlu!")
    else:
        print(f"⚠️ Ratio hâlâ yüksek ({ratio:.1f}x)")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: tgt Init (Zeros vs Query_Embed)
# ═══════════════════════════════════════════════════════════════════════════

print("\n--- TEST 2: tgt Init ---")

decoder_layer_input = None

def hook_decoder_layer(module, args):
    global decoder_layer_input
    decoder_layer_input = args[0].clone()

dec_hook = model.transformer.decoder.layers[0].register_forward_pre_hook(hook_decoder_layer)

with torch.no_grad():
    out_dict, _ = model(dummy_input, dummy_mask)

dec_hook.remove()

if decoder_layer_input is not None:
    tgt_mean = decoder_layer_input.mean().item()
    tgt_std = decoder_layer_input.std().item()
    tgt_l2 = torch.norm(decoder_layer_input).item()
    
    print(f"Decoder Layer 0 Input (tgt):")
    print(f"  Shape: {decoder_layer_input.shape}")
    print(f"  Mean: {tgt_mean:.6f}")
    print(f"  Std: {tgt_std:.6f}")
    print(f"  L2 norm: {tgt_l2:.4f}")
    
    if tgt_std > 1e-6:
        print("✅ tgt artık zeros değil!")
    else:
        print("❌ tgt hâlâ zeros!")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Dead Gradient Check
# ═══════════════════════════════════════════════════════════════════════════

print("\n--- TEST 3: Dead Gradient Check ---")

out_dict, _ = model(dummy_input, dummy_mask)
loss = out_dict['pred_logits'].sum() + out_dict['pred_curves'].sum()
loss.backward()

self_attn = model.transformer.decoder.layers[0].self_attn
embed_dim = self_attn.embed_dim

dead_count = 0
exploding_count = 0

critical_params = [
    ("self_attn.in_proj_weight", self_attn.in_proj_weight),
    ("self_attn.in_proj_bias", self_attn.in_proj_bias),
    ("self_attn.out_proj.weight", self_attn.out_proj.weight),
    ("self_attn.out_proj.bias", self_attn.out_proj.bias),
]

for name, param in critical_params:
    if param.grad is not None:
        grad_mean = param.grad.abs().mean().item()
        grad_max = param.grad.abs().max().item()
        
        status = "OK"
        if grad_mean < 1e-7:
            status = "💀 DEAD"
            dead_count += 1
        elif grad_mean > 10.0:
            status = "🔥 EXPLODING"
            exploding_count += 1
        
        print(f"  {name:35s} {status:15s} mean={grad_mean:.2e} max={grad_max:.2e}")
    else:
        print(f"  {name:35s} ❌ NO GRADIENT")

print(f"\nSonuç: {dead_count} dead, {exploding_count} exploding")

if dead_count == 0 and exploding_count == 0:
    print("✅ Tüm gradientler sağlıklı!")
elif dead_count == 0:
    print("⚠️ Dead gradient yok ama exploding var (clip_grad ile kontrol edilebilir)")
else:
    print("❌ Hâlâ dead gradient var!")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Overfit Test (Gerçek Loss ile)
# ═══════════════════════════════════════════════════════════════════════════

print("\n--- TEST 4: Overfit Test (Gerçek Loss) ---")

from db.datasets import datasets
import importlib

try:
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    sample_fn = importlib.import_module('sample.culane').sample_data
    
    criterion = MambaFPNLoss().cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )
    
    data, _ = sample_fn(db_obj, 0)
    single_img = data['xs'][0].cuda()
    single_mask = data['xs'][1].cuda()
    real_targets = [t.cuda() for t in data['ys'][1:]]
    targets_list = [single_img] + real_targets
    
    losses = []
    max_epochs = 300  # 200'den artır ��� eğri hâlâ düşüyor
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        out_dict, _ = model._train(single_img, single_mask)
        loss_result = criterion(epoch, False, 'train', out_dict, targets_list)
        total_loss = loss_result[0].mean()
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 30 == 0:  # 300 epoch için 30'er basamak
            print(f"Epoch {epoch:3d}: loss={total_loss.item():.4f}")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nBaşlangıç: {initial_loss:.4f}")
    print(f"Final: {final_loss:.4f}")
    print(f"Azalma: %{reduction:.1f}")
    
    if reduction > 90:
        print("✅ Overfit BAŞARILI")
    elif reduction > 80:
        print("⚠️ Overfit kısmi başarılı")
    else:
        print("❌ Overfit yetersiz")

except Exception as e:
    print(f"Overfit testi hatası: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TAMAMLANDI")
print("=" * 80)
