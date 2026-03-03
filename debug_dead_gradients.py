"""
Decoder Dead Gradient Analizi
==============================

Amaç: transformer.decoder.layers.0.self_attn neden dead gradient üretiyor?
Bunun Mamba encoder çıktısı ile bir bağlantısı var mı?
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

from models.LSTR_CULANE_MAMBA import model as MambaModel, loss as MambaLoss

print("=" * 80)
print("DECODER DEAD GRADIENT ANALİZİ")
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

print("\n--- Forward Pass ---")
print(f"Input: {img.shape}")

# Hook ile decoder inputunu kaydet
decoder_input = None
decoder_self_attn_input = None

def hook_decoder_input(module, input):
    global decoder_input
    # input[0] = tgt (queries), input[1] = memory
    decoder_input = (input[0].clone(), input[1].clone())
    return input

def hook_self_attn(module, input):
    global decoder_self_attn_input
    # Self-attn input: query (tgt)
    decoder_self_attn_input = input[0].clone()
    return input

# Decoder'a hook
decoder_hook = model.transformer.decoder.register_forward_pre_hook(hook_decoder_input)
self_attn_hook = model.transformer.decoder.layers[0].self_attn.register_forward_pre_hook(hook_self_attn)

with torch.no_grad():
    out_dict, _ = model._train(img, mask)

decoder_hook.remove()
self_attn_hook.remove()

print(f"\nDecoder input shapes:")
print(f"  tgt (queries): {decoder_input[0].shape}")  # (num_queries, B, D)
print(f"  memory (encoder out): {decoder_input[1].shape}")  # (HW, B, D)

print(f"\nSelf-attention input (tgt):")
print(f"  Shape: {decoder_self_attn_input.shape}")
print(f"  Mean: {decoder_self_attn_input.mean().item():.6f}")
print(f"  Std: {decoder_self_attn_input.std().item():.6f}")
print(f"  Min: {decoder_self_attn_input.min().item():.6f}")
print(f"  Max: {decoder_self_attn_input.max().item():.6f}")

# --- Gradient Analizi ---
print("\n" + "=" * 80)
print("GRADIENT AKIŞI ANALİZİ")
print("=" * 80)

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

out_dict, _ = model._train(img, mask)
loss = out_dict['pred_logits'].sum() + out_dict['pred_curves'].sum()
loss.backward()

# Dead gradient detaylı analiz
dead_layers = []

for name, param in model.named_parameters():
    if param.grad is None:
        continue
    
    grad_mean = param.grad.abs().mean().item()
    grad_std = param.grad.abs().std().item()
    grad_max = param.grad.abs().max().item()
    
    # Dead threshold (çok sıkı)
    if grad_mean < 1e-7:
        status = "💀 DEAD"
        dead_layers.append(name)
    elif grad_mean < 1e-5:
        status = "⚠️ NEAR-DEAD"
    elif grad_mean > 10.0:
        status = "🔥 EXPLODING"
    else:
        status = "✅ OK"
    
    # Sadece decoder ve ilgili kısımları yazdır
    if 'decoder' in name or 'input_proj' in name or 'query_embed' in name:
        print(f"{name:60s} {status:15s} mean={grad_mean:.2e} std={grad_std:.2e} max={grad_max:.2e}")

print(f"\n💀 Toplam dead layer: {len(dead_layers)}")

# --- Sorun Tespiti ---
print("\n" + "=" * 80)
print("OLASI SORUNLAR")
print("=" * 80)

# 1. Query embedding kontrolü
query_embed = model.query_embed.weight
print(f"\n1. Query Embedding (learnable):")
print(f"   Shape: {query_embed.shape}")
print(f"   Mean: {query_embed.mean().item():.6f}")
print(f"   Std: {query_embed.std().item():.6f}")
print(f"   Gradient var mı: {query_embed.grad is not None}")
if query_embed.grad is not None:
    print(f"   Grad mean: {query_embed.grad.abs().mean().item():.2e}")

# 2. Mamba encoder output kontrolü
print(f"\n2. Mamba Encoder Output (memory):")
print(f"   Shape: {decoder_input[1].shape}")
print(f"   Mean: {decoder_input[1].mean().item():.6f}")
print(f"   Std: {decoder_input[1].std().item():.6f}")

# 3. Decoder self-attn ağırlıkları
self_attn = model.transformer.decoder.layers[0].self_attn
print(f"\n3. Decoder Self-Attention Weights:")
print(f"   in_proj_weight shape: {self_attn.in_proj_weight.shape}")
print(f"   in_proj_bias shape: {self_attn.in_proj_bias.shape}")
print(f"   out_proj.weight shape: {self_attn.out_proj.weight.shape}")
print(f"   out_proj.bias shape: {self_attn.out_proj.bias.shape}")

# in_proj_weight aslında 3 concat: W_q, W_k, W_v
embed_dim = self_attn.embed_dim
W_q = self_attn.in_proj_weight[:embed_dim, :]
W_k = self_attn.in_proj_weight[embed_dim:2*embed_dim, :]
W_v = self_attn.in_proj_weight[2*embed_dim:, :]

print(f"\n   W_q (Query proj): mean={W_q.mean().item():.6f}, std={W_q.std().item():.6f}")
print(f"   W_k (Key proj):   mean={W_k.mean().item():.6f}, std={W_k.std().item():.6f}")
print(f"   W_v (Value proj): mean={W_v.mean().item():.6f}, std={W_v.std().item():.6f}")

# Gradientlerini kontrol et
if self_attn.in_proj_weight.grad is not None:
    grad_W_q = self_attn.in_proj_weight.grad[:embed_dim, :]
    grad_W_k = self_attn.in_proj_weight.grad[embed_dim:2*embed_dim, :]
    grad_W_v = self_attn.in_proj_weight.grad[2*embed_dim:, :]
    
    print(f"\n   W_q gradient: mean={grad_W_q.abs().mean().item():.2e}")
    print(f"   W_k gradient: mean={grad_W_k.abs().mean().item():.2e}")
    print(f"   W_v gradient: mean={grad_W_v.abs().mean().item():.2e}")
else:
    print(f"\n   💀 in_proj_weight gradient YOK!")

# --- HIPOTEZLER ---
print("\n" + "=" * 80)
print("OLASI NEDENLER")
print("=" * 80)

print("""
Hipotez 1: Query embedding çok küçük veya sıfır yakın
  → Self-attention query'leri zayıf, gradient akışı azalır

Hipotez 2: Mamba encoder output scale uyuşmazlığı
  → Encoder çıktıları çok büyük/küçük
  → Attention weights extrem değerler alıyor
  → Gradient vanishing/exploding

Hipotez 3: Learning rate çok küçük
  → 1e-4 çok düşük olabilir, bazı katmanlar öğrenemiyor

Hipotez 4: Initialization sorunu
  → Self-attention ağırlıkları kötü initialize edildi
  → Xavier/Uniform yerine başka bir şey kullanılmış olabilir
""")

# --- ÖNERİLER ---
print("=" * 80)
print("ÖNERİLER")
print("=" * 80)

print("""
1. Learning rate artır: 1e-4 → 1e-3 (overfit testinde kullanıldı)
2. LayerNorm yerine RMSNorm dene (Mamba ile daha iyi çalışabilir)
3. Self-attention init kontrol et: Xavier normal olmalı
4. Gradient clipping: max_norm=1.0 zaten ekli
5. Pre-LN vs Post-LN kontrol et: Şu an Post-LN
""")
