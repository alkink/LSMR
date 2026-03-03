"""
Attention Saturation Analizi (Basitleştirilmiş)
===============================================

Sorun: decoder.layers[0].self_attn'da
- in_proj_weight → DEAD (mean=0.0)
- in_proj_bias → EXPLODING (mean=1.64e+03)
- out_proj.weight → DEAD (mean=0.0)
- out_proj.bias → EXPLODING (mean=5.73e+03)

Hipotez: Softmax saturation — bazı attention weight'ler ~1.0, diğerleri ~0.0

Bu script:
1. Decoder layer input'unu (tgt) yakala
2. Manuel olarak Q,K,V hesapla
3. Attention weights dağılımını kontrol et
4. clip_grad_norm OLMADAN backward yap
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import system_configs

# Mamba config yükle
cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE_MAMBA.json")
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["system"]["snapshot_name"] = "LSTR_CULANE_MAMBA"
cfg["system"]["data_dir"] = "/home/alki/projects/"
system_configs.update_config(cfg["system"])

from models.LSTR_CULANE_MAMBA import model as MambaModel

print("=" * 80)
print("ATTENTION SATURATION ANALİZİ")
print("=" * 80)

model = MambaModel(flag=True).cuda()
model.train()

# Veri yükle
try:
    from db.datasets import datasets
    import importlib
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    sample_fn = importlib.import_module('sample.culane').sample_data
except Exception as e:
    print(f"Veri yükleme hatası: {e}")
    sys.exit(1)

data, _ = sample_fn(db_obj, 0)
img = data['xs'][0].cuda()
mask = data['xs'][1].cuda()

print(f"\nInput: {img.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: Decoder Layer Input'u Yakala
# ═══════════════════════════════════════════════════════════════════════════

decoder_layer_0_input = None

def hook_decoder_layer_pre(module, args):
    """Decoder layer 0'a girerken tgt'yi yakala"""
    global decoder_layer_0_input
    # args = (tgt, memory, ...)
    decoder_layer_0_input = args[0].clone() if len(args) > 0 else None

decoder_hook = model.transformer.decoder.layers[0].register_forward_pre_hook(hook_decoder_layer_pre)

# Forward pass
with torch.no_grad():
    out_dict, _ = model._train(img, mask)

decoder_hook.remove()

print("\n" + "=" * 80)
print("BÖLÜM 1: Decoder Layer 0 Input (tgt)")
print("=" * 80)

if decoder_layer_0_input is not None:
    print(f"\ntgt shape: {decoder_layer_0_input.shape}")
    print(f"  Mean: {decoder_layer_0_input.mean().item():.6f}")
    print(f"  Std: {decoder_layer_0_input.std().item():.6f}")
    print(f"  Min: {decoder_layer_0_input.min().item():.6f}")
    print(f"  Max: {decoder_layer_0_input.max().item():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: Manuel Attention Computation
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 2: Self-Attention Manuel Hesaplama")
print("=" * 80)

self_attn = model.transformer.decoder.layers[0].self_attn
embed_dim = self_attn.embed_dim
num_heads = self_attn.num_heads
head_dim = embed_dim // num_heads

# Split in_proj_weight: [W_q; W_k; W_v]
W_q = self_attn.in_proj_weight[:embed_dim, :]
W_k = self_attn.in_proj_weight[embed_dim:2*embed_dim, :]
W_v = self_attn.in_proj_weight[2*embed_dim:, :]

b_q = self_attn.in_proj_bias[:embed_dim]
b_k = self_attn.in_proj_bias[embed_dim:2*embed_dim]
b_v = self_attn.in_proj_bias[2*embed_dim:]

# Self-attention: query = key = value = tgt
tgt = decoder_layer_0_input  # (L, B, D)
L, B, D = tgt.shape

# Q, K, V computation
q = F.linear(tgt, W_q, b_q)  # (L, B, D)
k = F.linear(tgt, W_k, b_k)
v = F.linear(tgt, W_v, b_v)

print(f"\nQ, K, V shapes: {q.shape}")
print(f"Q mean: {q.mean().item():.6f}, std: {q.std().item():.6f}")
print(f"K mean: {k.mean().item():.6f}, std: {k.std().item():.6f}")
print(f"V mean: {v.mean().item():.6f}, std: {v.std().item():.6f}")

# Reshape for multi-head
q = q.view(L, B, num_heads, head_dim).permute(1, 2, 0, 3)  # (B, H, L, D_h)
k = k.view(L, B, num_heads, head_dim).permute(1, 2, 0, 3)
v = v.view(L, B, num_heads, head_dim).permute(1, 2, 0, 3)

# Attention scores
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, L, L)
attn_weights = F.softmax(attn_scores, dim=-1)

print(f"\nAttention weights shape: {attn_weights.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: Attention Weights Analizi
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 3: Attention Saturation Analizi")
print("=" * 80)

for h in range(num_heads):
    weights_h = attn_weights[0, h, :, :]  # (L, L) for batch 0
    
    # Her query için max attention weight
    max_weights = weights_h.max(dim=-1)[0]
    min_weights = weights_h.min(dim=-1)[0]
    entropy = -(weights_h * torch.log(weights_h + 1e-10)).sum(dim=-1)
    
    uniform_entropy = torch.log(torch.tensor(L, dtype=torch.float32, device=weights_h.device))
    
    print(f"\nHead {h}:")
    print(f"  Max attention per query: mean={max_weights.mean().item():.4f}, std={max_weights.std().item():.4f}")
    print(f"  Min attention per query: mean={min_weights.mean().item():.6f}")
    print(f"  Entropy per query: mean={entropy.mean().item():.4f} / {uniform_entropy.item():.4f} (uniform)")
    
    saturated = (max_weights > 0.95).sum().item()
    print(f"  Saturated queries (>0.95): {saturated}/{L} ({saturated/L*100:.1f}%)")
    
    if saturated > L // 2:
        print(f"  🔴 HIGH SATURATION in head {h}!")

# ═════════════��═════════════════════════════════════════════════════════════
# BÖLÜM 4: Gradient Analizi (NO CLIPPING)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 4: Gradient Analizi (NO CLIPPING)")
print("=" * 80)

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

out_dict, _ = model._train(img, mask)
loss = out_dict['pred_logits'].sum() + out_dict['pred_curves'].sum()

print(f"\nLoss: {loss.item():.6f}")
print("Backward pass...")

loss.backward()

print("\n" + "-" * 80)
print("Decoder Layer 0 Self-Attn Gradients:")
print("-" * 80)

# in_proj_weight gradients (split Q, K, V)
if self_attn.in_proj_weight.grad is not None:
    grad_W_q = self_attn.in_proj_weight.grad[:embed_dim, :]
    grad_W_k = self_attn.in_proj_weight.grad[embed_dim:2*embed_dim, :]
    grad_W_v = self_attn.in_proj_weight.grad[2*embed_dim:, :]
    
    print(f"\nin_proj_weight gradients:")
    print(f"  W_q: mean={grad_W_q.abs().mean().item():.2e}, max={grad_W_q.abs().max().item():.2e}")
    print(f"  W_k: mean={grad_W_k.abs().mean().item():.2e}, max={grad_W_k.abs().max().item():.2e}")
    print(f"  W_v: mean={grad_W_v.abs().mean().item():.2e}, max={grad_W_v.abs().max().item():.2e}")
    
    if grad_W_q.abs().mean().item() < 1e-7:
        print(f"  💀 W_q gradient DEAD!")
    if grad_W_k.abs().mean().item() < 1e-7:
        print(f"  💀 W_k gradient DEAD!")
    if grad_W_v.abs().mean().item() < 1e-7:
        print(f"  💀 W_v gradient DEAD!")

# in_proj_bias gradients
if self_attn.in_proj_bias.grad is not None:
    grad_b_q = self_attn.in_proj_bias.grad[:embed_dim]
    grad_b_k = self_attn.in_proj_bias.grad[embed_dim:2*embed_dim]
    grad_b_v = self_attn.in_proj_bias.grad[2*embed_dim:]
    
    print(f"\nin_proj_bias gradients:")
    print(f"  b_q: mean={grad_b_q.abs().mean().item():.2e}, max={grad_b_q.abs().max().item():.2e}")
    print(f"  b_k: mean={grad_b_k.abs().mean().item():.2e}, max={grad_b_k.abs().max().item():.2e}")
    print(f"  b_v: mean={grad_b_v.abs().mean().item():.2e}, max={grad_b_v.abs().max().item():.2e}")
    
    if grad_b_q.abs().mean().item() > 10.0:
        print(f"  🔥 b_q gradient EXPLODING!")
    if grad_b_k.abs().mean().item() > 10.0:
        print(f"  🔥 b_k gradient EXPLODING!")
    if grad_b_v.abs().mean().item() > 10.0:
        print(f"  🔥 b_v gradient EXPLODING!")

# out_proj gradients
if self_attn.out_proj.weight.grad is not None:
    grad_out_w = self_attn.out_proj.weight.grad
    print(f"\nout_proj.weight gradient:")
    print(f"  mean={grad_out_w.abs().mean().item():.2e}, max={grad_out_w.abs().max().item():.2e}")
    if grad_out_w.abs().mean().item() < 1e-7:
        print(f"  💀 out_proj.weight gradient DEAD!")

if self_attn.out_proj.bias.grad is not None:
    grad_out_b = self_attn.out_proj.bias.grad
    print(f"\nout_proj.bias gradient:")
    print(f"  mean={grad_out_b.abs().mean().item():.2e}, max={grad_out_b.abs().max().item():.2e}")
    if grad_out_b.abs().mean().item() > 10.0:
        print(f"  🔥 out_proj.bias gradient EXPLODING!")

# ═══════════════════════════════════════════════════════════════════════════
# SONUÇ
# ═══════════════════════════════════════════════��═══════════════════════════

print("\n" + "=" * 80)
print("SONUÇ")
print("=" * 80)

print("""
Analiz tamamlandı:
1. Decoder layer 0 input (tgt) statistics
2. Q, K, V projection statistics
3. Attention saturation per head
4. Gradient flow (no clipping)

OLASI NEDENLER:

A) Softmax Saturation:
   Eğer attention weights çok extreme (0.99 vs 0.001),
   softmax backward'da gradient neredeyse sıfır olur.

B) tgt Init Sorunu:
   Decoder başlangıçta tgt=zeros kullanıyor.
   Tüm query'ler aynı başlıyorsa, self-attention saturation.

C) Scale Mismatch:
   Mamba encoder output vs decoder query embedding scale.
""")

print("=" * 80)
