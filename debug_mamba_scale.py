"""
Mamba Encoder Output Scale Analizi
===================================

Sorun 3: Mamba encoder output (memory) scale kontrolü

Gözlenen:
- Memory: mean=0.0, std=1.0 (çok şüpheli)
- Bu LayerNorm sonrası gibi

Hipotez: Mamba encoder output'u zaten normalize olmuş,
ama decoder query embedding (tgt) normalize edilmemiş.
Cross-attention'da scale mismatch oluşuyor.

Bu script:
1. Mamba encoder'dan ÖNCE ve SONRA feature scale'ini ölç
2. LayerNorm var mı kontrol et
3. tgt (decoder query) ile memory scale karşılaştır
4. Position encoding scale'ini kontrol et
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
print("MAMBA ENCODER OUTPUT SCALE ANALİZİ")
print("=" * 80)

model = MambaModel(flag=True).cuda()
model.eval()

# Hook değişkenleri
mamba_input = None
mamba_output = None
pos_encoding = None
query_embedding = None

def hook_mamba_encoder(module, args, output):
    """
    Mamba encoder forward_hook receives:
        args: (src, src_key_padding_mask, pos, ...)
        output: (encoder_out, attn_weights) - tuple
    
    Note: PyTorch forward_hook signature is hook(module, args, output)
    """
    global mamba_input, mamba_output
    
    # Extract src from args
    src = args[0] if args else None
    mamba_input = src.clone() if src is not None else None
    
    # Output is a tuple: (encoder_output, attention_weights)
    if isinstance(output, tuple):
        mamba_output = output[0].clone()  # encoder_output only
    else:
        mamba_output = output.clone()

def hook_query_embed(module, args, output):
    """Query embedding forward_hook"""
    global query_embedding
    # Query embedding is just the weight tensor, output is the tensor itself
    if isinstance(output, tuple):
        query_embedding = output[0].clone()
    else:
        query_embedding = output.clone()

# Hook'ları kaydet
mamba_hook = model.transformer.encoder.register_forward_hook(hook_mamba_encoder)
query_hook = model.query_embed.register_forward_hook(hook_query_embed)

# Dummy input
dummy_input = torch.randn(1, 3, 360, 640).cuda()
dummy_mask = torch.zeros(1, 1, 360, 640).cuda()

with torch.no_grad():
    out_dict, _ = model(dummy_input, dummy_mask)

mamba_hook.remove()
query_hook.remove()

print("\n" + "=" * 80)
print("BÖLÜM 1: Mamba Encoder Input/Output Scale")
print("=" * 80)

if mamba_input is not None:
    print(f"\nMamba Encoder INPUT (transformer_input):")
    print(f"  Shape: {mamba_input.shape}")
    print(f"  Mean: {mamba_input.mean().item():.6f}")
    print(f"  Std: {mamba_input.std().item():.6f}")
    print(f"  Min: {mamba_input.min().item():.6f}")
    print(f"  Max: {mamba_input.max().item():.6f}")
    print(f"  L2 norm: {torch.norm(mamba_input).item():.6f}")
    # Shape is (HW, B, D), not (B, C, H, W)
    if len(mamba_input.shape) == 3:
        print(f"  Per-channel mean (first 8): {mamba_input[:, 0, :8].mean(dim=0).cpu().numpy()}")
        print(f"  Per-channel std (first 8): {mamba_input[:, 0, :8].std(dim=0).cpu().numpy()}")

if mamba_output is not None:
    # Output is (HW, B, D)
    print(f"\nMamba Encoder OUTPUT (memory):")
    print(f"  Shape: {mamba_output.shape}")
    print(f"  Mean: {mamba_output.mean().item():.6f}")
    print(f"  Std: {mamba_output.std().item():.6f}")
    print(f"  Min: {mamba_output.min().item():.6f}")
    print(f"  Max: {mamba_output.max().item():.6f}")
    print(f"  L2 norm: {torch.norm(mamba_output).item():.6f}")
    
    # Per-channel stats (first 8 channels)
    print(f"  Per-channel mean (first 8): {mamba_output[:, 0, :8].mean(dim=0).cpu().numpy()}")
    print(f"  Per-channel std (first 8): {mamba_output[:, 0, :8].std(dim=0).cpu().numpy()}")
    
    # Check if it looks LayerNorm-ed
    if abs(mamba_output.mean().item()) < 0.1 and abs(mamba_output.std().item() - 1.0) < 0.2:
        print(f"  ⚠️ Output looks LayerNorm-ed! (mean≈0, std≈1)")

print("\n" + "=" * 80)
print("BÖLÜM 2: Query Embedding Scale")
print("=" * 80)

if query_embedding is not None:
    print(f"\nQuery Embedding (learnable):")
    print(f"  Shape: {query_embedding.shape}")
    print(f"  Mean: {query_embedding.mean().item():.6f}")
    print(f"  Std: {query_embedding.std().item():.6f}")
    print(f"  Min: {query_embedding.min().item():.6f}")
    print(f"  Max: {query_embedding.max().item():.6f}")
    print(f"  L2 norm: {torch.norm(query_embedding).item():.6f}")

print("\n" + "=" * 80)
print("BÖLÜM 3: Position Encoding Scale")
print("=" * 80)

# Get position encoding
dummy_mask_bool = dummy_mask[:, 0, :, :] == 0
with torch.no_grad():
    pos = model.position_embedding(dummy_input.permute(0, 2, 3, 1), dummy_mask_bool)

print(f"\nPosition Encoding:")
print(f"  Shape: {pos.shape}")
print(f"  Mean: {pos.mean().item():.6f}")
print(f"  Std: {pos.std().item():.6f}")
print(f"  Min: {pos.min().item():.6f}")
print(f"  Max: {pos.max().item():.6f}")
print(f"  L2 norm: {torch.norm(pos).item():.6f}")

print("\n" + "=" * 80)
print("BÖLÜM 4: Scale Karşılaştırma")
print("=" * 80)

if mamba_input is not None and mamba_output is not None and query_embedding is not None:
    print(f"\n{'Component':<30} {'Mean':>12} {'Std':>12} {'L2':>12}")
    print("-" * 70)
    
    mamba_input_flat = mamba_input.flatten()
    mamba_output_flat = mamba_output.flatten()
    query_flat = query_embedding.flatten()
    pos_flat = pos.flatten()
    
    print(f"{'Mamba Input':<30} {mamba_input.mean():>12.6f} {mamba_input.std():>12.6f} {torch.norm(mamba_input):>12.2f}")
    print(f"{'Mamba Output (memory)':<30} {mamba_output.mean():>12.6f} {mamba_output.std():>12.6f} {torch.norm(mamba_output):>12.2f}")
    print(f"{'Query Embedding':<30} {query_embedding.mean():>12.6f} {query_embedding.std():>12.6f} {torch.norm(query_embedding):>12.2f}")
    print(f"{'Position Encoding':<30} {pos.mean():>12.6f} {pos.std():>12.6f} {torch.norm(pos):>12.2f}")

print("\n" + "=" * 80)
print("BÖLÜM 5: Mamba LayerNorm Kontrolü")
print("=" * 80)

# Check if Mamba encoder has LayerNorm
mamba_encoder = model.transformer.encoder
print(f"\nBidirectionalMambaEncoder yapısı:")
print(f"  Type: {type(mamba_encoder)}")

if hasattr(mamba_encoder, 'layers'):
    for i, layer in enumerate(mamba_encoder.layers):
        print(f"\n  Layer {i}:")
        print(f"    Has forward_mamba: {hasattr(layer, 'forward_mamba')}")
        print(f"    Has backward_mamba: {hasattr(layer, 'backward_mamba')}")
        print(f"    Has norm: {hasattr(layer, 'norm')}")
        if hasattr(layer, 'norm'):
            print(f"    Norm type: {type(layer.norm)}")

print("\n" + "=" * 80)
print("ANALİZ SONUCU")
print("=" * 80)

print("""
Bu analiz Mamba encoder output'unun scale'ini kontrol etti:

1. Mamba Input → Output scale değişimi:
   - Eğer std≈1 ise LayerNorm var demektir
   - Transformer decoder query embedding ile uyumlu olmalı

2. Query Embedding vs Memory:
   - İkisi de benzer scale'de olmalı
   - Çok fark varsa cross-attention'da sorun çıkar

3. Position Encoding:
   - Position encoding memory'e ekleniyor (pos + memory)
   - Position encoding çok büyükse memory'yi domine eder

OLASI SORUNLAR:

A) Mamba Output LayerNorm'lu:
   Mamba encoder içinde LayerNorm varsa output ≈ N(0,1)
   Decoder query embedding de N(0,1) olmalı.
   
B) Position Encoding Scale:
   Position encoding çok büyükse (örn: mean=10), memory'yi bozar.
   Position encoding küçükse (örn: mean=0.1), etkisiz olur.

C) tgt Init Sorunu:
   Decoder başlangıçta tgt=zeros kullanıyor. Tüm query'ler aynı →
   self-attention saturation → dead gradient.
""")

print("=" * 80)
print("TAMAMLANDI")
print("=" * 80)
