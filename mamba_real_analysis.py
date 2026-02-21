"""
LSTR + Gerçek mamba-ssm Entegrasyon Ön-Analiz Scripti
======================================================

KURULUM GEREKSİNİMLERİ (elle kur):
    conda activate clrernet
    pip install causal-conv1d --no-build-isolation
    pip install mamba-ssm --no-build-isolation

Kurulum sonrası çalıştır:
    conda activate clrernet && python mamba_real_analysis.py
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
results = {}  # test adı → durum

def record(test_name, status, note=""):
    results[test_name] = (status, note)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: MODEL MİMARİSİ ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("BÖLÜM 1.1: PARAMETRE HARİTASI")
print("=" * 80)

model = LSTRModel(flag=True).cuda()
model.eval()

total_params    = 0
frozen_params   = 0
non_bn_frozen   = []

for name, param in model.named_parameters():
    total_params += param.numel()
    if not param.requires_grad:
        frozen_params += param.numel()
        if "bn" not in name.lower() and "norm" not in name.lower():
            non_bn_frozen.append(name)
    status = "TRAINABLE" if param.requires_grad else "FROZEN"
    print(f"  {name:<60} | {str(list(param.shape)):<25} | {status}")

print(f"\nToplam parametre : {total_params:,}")
print(f"Frozen           : {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
print(f"Trainable        : {total_params - frozen_params:,}")

# LSTR FrozenBatchNorm2d kasıtlı — kontrol: BN dışı frozen var mı?
if non_bn_frozen:
    print(f"[WARN] BN/Norm dışı frozen parametre bulundu: {non_bn_frozen}")
    record("Frozen param sadece BN (kasıtlı)", WARN, str(non_bn_frozen))
else:
    print("✅ Frozen parametreler yalnızca BatchNorm — beklenen davranış")
    record("Frozen param sadece BN (kasıtlı)", PASS)

# ── 1.2 Forward pass boyut haritası ──────────────────────────────────────────
print("\n" + "=" * 80)
print("BÖLÜM 1.2: FORWARD PASS BOYUT HARİTASI")
print("=" * 80)

hooks_store = {}
activations = {}

def make_hook(name):
    def hook(module, inp, output):
        t = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(t, torch.Tensor):
            activations[name] = {
                "shape"   : list(t.shape),
                "mean"    : round(t.mean().item(), 4),
                "std"     : round(t.std().item(),  4),
                "has_nan" : bool(torch.isnan(t).any().item()),
                "has_inf" : bool(torch.isinf(t).any().item()),
            }
    return hook

target_names = ['layer1', 'layer4', 'input_proj',
                'transformer.encoder', 'transformer.decoder']
for name, mod in model.named_modules():
    if name in target_names:
        hooks_store[name] = mod.register_forward_hook(make_hook(name))

dummy_input = torch.randn(1, 3, 360, 640).cuda()
dummy_mask  = torch.zeros(1, 1, 360, 640).cuda()

with torch.no_grad():
    _ = model(dummy_input, dummy_mask)

for h in hooks_store.values():
    h.remove()

EXPECTED = {
    "layer1"              : [1, 16, 90, 160],
    "layer4"              : [1, 128, 12, 20],
    "input_proj"          : [1, 32, 12, 20],
}

all_shapes_ok = True
for name, stats in activations.items():
    print(f"\n[{name}]")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats.get("has_nan"):
        print("  🚨 NaN TESPİT EDİLDİ!")
        all_shapes_ok = False
    if stats.get("has_inf"):
        print("  🚨 Inf TESPİT EDİLDİ!")
        all_shapes_ok = False
    if name in EXPECTED:
        if stats["shape"] != EXPECTED[name]:
            print(f"  🚨 Boyut UYUMSUZ! Beklenen {EXPECTED[name]}, Gerçek {stats['shape']}")
            all_shapes_ok = False
        else:
            print(f"  ✅ Boyut doğru")

seq_len = activations.get("input_proj", {}).get("shape", [0,0,0,0])
if len(seq_len) == 4:
    hw = seq_len[2] * seq_len[3]
    print(f"\n[input_proj → Transformer girdi adaptasyonu]")
    print(f"  (B={seq_len[0]}, C={seq_len[1]}, H={seq_len[2]}, W={seq_len[3]})")
    print(f"  Flatten → (HW={hw}, B={seq_len[0]}, D={seq_len[1]}) — LSTR seq-first formatı")
    print(f"  Mamba   → permute(1,0,2) → (B={seq_len[0]}, HW={hw}, D={seq_len[1]})")

record("Forward pass boyutlar doğru", PASS if all_shapes_ok else FAIL)
record("NaN/Inf yok", PASS if all_shapes_ok else FAIL)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: FEATURE ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 2: FEATURE BENZERLİĞİ (input_proj)")
print("=" * 80)

# Gercek veri yukle
real_images = None
real_masks = None
real_targets = None
try:
    from db.datasets import datasets
    import importlib
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    print(f"DB yuklendi: {db_obj.db_inds.size} ornek")
    sample_fn = importlib.import_module('sample.culane').sample_data
    data, _ = sample_fn(db_obj, 0)
    real_images  = data['xs'][0].cuda()
    real_masks   = data['xs'][1].cuda()
    real_targets = [t.cuda() for t in data['ys'][1:]]
    print(f'PASS: Gercek CULANE verisi yuklendi (B={real_images.shape[0]})')
except Exception:
    print('[WARN] Gercek veri yuklenemedi, torch.randn kullanilacak:')
    import traceback; traceback.print_exc()

features_list = []

def feat_hook(module, inp, output):
    features_list.append(output.detach().cpu())

target_layer = dict(model.named_modules())['input_proj']
h = target_layer.register_forward_hook(feat_hook)

model.eval()
with torch.no_grad():
    if real_images is not None:
        # 5 farkli gercek goruntu ile test
        for i in range(min(5, real_images.shape[0])):
            model(real_images[i:i+1], real_masks[i:i+1])
        print("Feature benzerligi testi: gercek CULane goruntuleri kullanildi.")
    else:
        for _ in range(5):
            img = torch.randn(1, 3, 360, 640).cuda()
            model(img, dummy_mask)
        print("Feature benzerligi testi: torch.randn kullanildi (gercek veri yok).")

h.remove()

flat_features = [f.flatten() for f in features_list[:5]]
sim_matrix = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        sim = F.cosine_similarity(
            flat_features[i].unsqueeze(0),
            flat_features[j].unsqueeze(0)
        ).item()
        sim_matrix[i][j] = sim

print("Cosine Benzerlik Matrisi (input_proj, 5 rastgele görüntü):")
print(np.round(sim_matrix, 3))

avg_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (5*5 - 5)
print(f"\nOrtalama çapraz-benzerlik (N=5, istatistiksel gösterge): {avg_off_diag:.4f}")
print("Not: 10 sample istatistiksel olarak kesin değildir — gösterge niteliğindedir.")

if avg_off_diag > 0.95:
    print("🚨 KRİTİK: Feature'lar çok benzer — Mamba bu noktadan öğrenemez!")
    record("Feature benzerliği < 0.85", FAIL, f"avg_off_diag={avg_off_diag:.4f}")
elif avg_off_diag > 0.85:
    print("⚠️  Yüksek benzerlik — dikkat et")
    record("Feature benzerliği < 0.85", WARN, f"avg_off_diag={avg_off_diag:.4f}")
else:
    print("✅ Feature çeşitliliği yeterli")
    record("Feature benzerliği < 0.85", PASS, f"avg_off_diag={avg_off_diag:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: GRADIENT ANALİZİ — mevcut encoder
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 3: GRADIENT ANALİZİ — mevcut transformer.encoder")
print("=" * 80)

model.train()
dim_input = torch.randn(1, 3, 360, 640).cuda()
out_dict, _ = model._train(dim_input, dummy_mask)

# Sadece model çıkışı üzerinden prox backward (gerçek AELoss gerektirmez burada)
dummy_loss = out_dict['pred_logits'].sum() + out_dict['pred_curves'].sum()
model.zero_grad()
dummy_loss.backward()

dead_layers      = []
exploding_layers = []

for name, param in model.named_parameters():
    if 'transformer.encoder' not in name:
        continue
    if param.grad is not None:
        gm = param.grad.abs().mean().item()
        mx = param.grad.abs().max().item()
        s  = "OK"
        if gm < 1e-7:
            s = "⚠️ DEAD"
            dead_layers.append(name)
        elif mx > 10.0:
            s = "🚨 EXPLODING"
            exploding_layers.append(name)
        print(f"  {name:<55} | mean={gm:.2e} | max={mx:.2e} | {s}")
    else:
        print(f"  {name:<55} | grad=None")

print(f"\nDead layers     : {len(dead_layers)}")
print(f"Exploding layers: {len(exploding_layers)}")

if dead_layers or exploding_layers:
    record("Mevcut encoder gradient sağlıklı", WARN,
           f"dead={dead_layers}, exploding={exploding_layers}")
else:
    print("✅ Gradient akışı sağlıklı")
    record("Mevcut encoder gradient sağlıklı", PASS)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: GERÇEK MAMBA-SSM İZOLE TEST
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 4: GERÇEK MAMBA-SSM KURULUM + İZOLE TEST")
print("=" * 80)

try:
    from mamba_ssm import Mamba
    print("✅ mamba-ssm import başarılı")
    record("Gerçek mamba-ssm kurulumu başarılı", PASS)
except ImportError as e:
    print(f"🚨 mamba-ssm import HATASI: {e}")
    print("   ➜ Kur: pip install mamba-ssm causal-conv1d --no-build-isolation")
    record("Gerçek mamba-ssm kurulumu başarılı", FAIL, str(e))
    print("\n[DURDURULDU] mamba-ssm kurulmadan devam edilemez.")
    import sys; sys.exit(1)

d_model = system_configs.attn_dim  # 32

mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).cuda()
mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).cuda()

test_input = torch.randn(1, 240, d_model, device='cuda', requires_grad=True)

# Forward-only
out_fwd = mamba_fwd(test_input)
print(f"Forward Mamba çıktı boyutu: {out_fwd.shape}")  # (1, 240, 32)

# Bidirectional (flip + reverse)
out_bwd     = mamba_bwd(torch.flip(test_input, dims=[1]))
out_bwd     = torch.flip(out_bwd, dims=[1])
out_bidir   = out_fwd + out_bwd
print(f"Bidirectional (add) çıktı : {out_bidir.shape}")  # (1, 240, 32)

loss_b = out_bidir.sum()
loss_b.backward()

inp_grad_mean = test_input.grad.abs().mean().item()
print(f"\nGiriş gradient mean: {inp_grad_mean:.2e}")

all_grad_ok = True
for name, param in list(mamba_fwd.named_parameters()) + list(mamba_bwd.named_parameters()):
    if param.grad is None:
        print(f"  ⚠️  {name}: gradient YOK")
        all_grad_ok = False
    elif param.grad.abs().mean() < 1e-9:
        gv = param.grad.abs().mean().item()
        print(f"  ⚠️  {name}: dead gradient ({gv:.2e})")
        all_grad_ok = False
    else:
        gv = param.grad.abs().mean().item()
        print(f"  ✅ {name}: {gv:.2e}")

bidir_shape_ok = (out_bidir.shape == (1, 240, d_model))
if not bidir_shape_ok:
    print(f"🚨 Bidirectional çıktı shapi yanlış: {out_bidir.shape}")

record("Mamba forward/bidirectional çalışıyor", PASS if bidir_shape_ok else FAIL)
if all_grad_ok and inp_grad_mean > 1e-7:
    print("\n✅ Gerçek Mamba kernel gradient akışı tamam")
    record("Mamba gradient akışı sağlıklı", PASS, f"inp_grad={inp_grad_mean:.2e}")
else:
    print(f"\n🚨 Gradient sorunu! inp_grad={inp_grad_mean:.2e}")
    record("Mamba gradient akışı sağlıklı", FAIL, f"inp_grad={inp_grad_mean:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: WRAPPER SHAPE DOĞRULAMASI
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 5: BidirectionalMambaEncoder SHAPE DOĞRULAMASI")
print("=" * 80)

# enc_attn_weights shape analizi (test/culane.py:106-169)
# Hook: lambda self, input, output: enc_attn_weights.append(output[1]))
# self_attn → (q, k, v) → returns (attn_output, attn_weights)
# attn_weights shape: (B*num_heads, HW, HW)
# enc_attn_weights[0] → (num_heads*B, HW, HW)
# reshape(shape + shape) burada shape=(12,20) → (HW, HW)=(240,240)
# Mevcut hook: output[1] from self_attn → (B*num_heads, HW, HW)
# [0] indisi → ilk batch → (num_heads, HW, HW) shape'i kalmaz
# Gerçek hook: layerde output[1] shape = (B*num_heads, HW, HW)
# [0] → (HW, HW) = (240, 240) → reshape(12,20,12,20) ✓
# Bu yüzden dummy_weights = (num_heads, HW, HW) → output[1] → mevcut hook
# [0] → (HW, HW) → reshape(12,20,12,20) ✓

print("enc_attn_weights shape analizi:")
print("  test/culane.py:106 hook: output[1] = self_attn weight → (B*num_heads, HW, HW)")
print("  enc_attn_weights[0] → (HW, HW)=(240,240), reshape(12,20,12,20) ✓")
print("  Wrapper dummy_weights: (num_heads, HW, HW) → [0] → (HW, HW) ✓")
num_heads = system_configs.num_heads
print(f"  num_heads = system_configs.num_heads = {num_heads}")

from models.py_utils.mamba_encoder import BidirectionalMambaEncoder

enc = BidirectionalMambaEncoder().cuda()

src = torch.randn(240, 1, 32).cuda()  # LSTR seq-first
out, wts = enc(src)

assert out.shape == (240, 1, 32),             f"out shape yanlis: {out.shape}"
assert wts.shape == (num_heads * 1, 240, 240), f"wts B=1 shape yanlis: {wts.shape}"

# Kritik: B>1 testi — gercek egitimde batch_size=16
src_b16 = torch.randn(240, 16, 32).cuda()
out_b16, wts_b16 = enc(src_b16)
ok_b16 = wts_b16.shape == (num_heads * 16, 240, 240)
print(f"B=16 Test: out={out_b16.shape}, wts={wts_b16.shape} -> {'PASS' if ok_b16 else 'FAIL (gercek egitimde patlar)'}")

# Simüle reshape test (test/culane.py:169)
target_shape = (12, 20)
try:
    reshaped = wts[0].reshape(target_shape + target_shape)
    assert reshaped.shape == (12, 20, 12, 20)
    print(f"✅ out : {out.shape}")
    print(f"✅ wts : {wts.shape}")
    print(f"✅ wts[0].reshape(12,20,12,20): {reshaped.shape} — test/culane.py:169 uyumlu")
    record("enc_attn_weights shape doğrulandı", PASS,
           f"wts[0].reshape(12,20,12,20)={reshaped.shape}")
    record("Wrapper shape testi geçti (240,1,32)", PASS)
except AssertionError as e:
    print(f"🚨 Shape assertion hatası: {e}")
    record("enc_attn_weights shape doğrulandı", FAIL, str(e))
    record("Wrapper shape testi geçti (240,1,32)", FAIL, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: OVERFİT TESTİ — GERÇEK MAMBA İLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 6: OVERFİT TESTİ — GERÇEK BidirectionalMambaEncoder")
print("=" * 80)

# ─── NOT: Gerçek CULane annotation yerine shapesi doğru dummy targets kullanıyoruz.
# pts = 18 (culane.py:111), max_lanes = 7 (LSTR config: num_queries)
# Target format: [class=1, lower, upper, 18*xs, 18*ys] = 1+2+18+18 = 39 float
# Her batch öğesi = (2_lanes, 39_floats)
# AELoss.forward → targets[0]=visual_tensor, targets[i+1]=label per batch item

model_mamba = LSTRModel(flag=True).cuda()
model_mamba.eval()

original_enc = model_mamba.transformer.encoder
net = model_mamba.transformer.encoder = BidirectionalMambaEncoder().cuda()
criterion = LSTRLoss().cuda()
# loss_curves: 5 -> 2.5 hem primary hem aux (spike fix)
for k in list(criterion.criterion.weight_dict.keys()):
    if 'loss_curves' in k:
        criterion.criterion.weight_dict[k] = 2.5
print(f"Patched weight_dict: {criterion.criterion.weight_dict}")
optimizer  = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_mamba.parameters()), lr=1e-3
)

# Tam batch kullan: kp.py:258 [tgt[0] for tgt in targets[1:]] batch_size kadar eleman bekliyor
if real_images is not None:
    single_img  = real_images   # [16, 3, 295, 820]
    single_mask = real_masks    # [16, 1, 295, 820]
    # real_targets each is [batch_size, n_lanes, 73] — exact training format
    targets_list = [real_images] + real_targets
    print(f"Gercek CULANE tam batch ile overfit: B={single_img.shape[0]}")
else:
    n_pts = 18
    tgt = torch.zeros(2, 1 + 2 + n_pts*2).cuda()
    tgt[:, 0] = 1; tgt[:, 1] = 0.8; tgt[:, 2] = 0.3
    tgt[:, 3:3+n_pts]  = torch.linspace(0.3, 0.7, n_pts).cuda()
    tgt[:, 3+n_pts:]   = torch.linspace(0.3, 0.8, n_pts).cuda()
    single_img  = torch.randn(1, 3, 360, 640).cuda()
    single_mask = torch.zeros(1, 1, 360, 640).cuda()
    targets_list = [single_img, tgt.unsqueeze(0)]
    print("Dummy goruntu kullaniliyor.")

losses_history = []
model_mamba.train()

print("200 epoch eğitim (ADD stratejisi bidirectional Mamba)...")
for epoch in range(200):
    optimizer.zero_grad()
    out_pred, _ = model_mamba._train(single_img, single_mask)
    loss_result  = criterion(epoch, False, 'train', out_pred, targets_list)
    total_loss   = loss_result[0].mean()

    if not math.isfinite(total_loss.item()):
        print(f"[Epoch {epoch}] Loss NaN/Inf — erken durdurma!")
        record("Overfit %90+ (gerçek Mamba)", FAIL, "Loss diverged")
        break

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_mamba.parameters(), 1.0)
    optimizer.step()
    lv = total_loss.item()
    losses_history.append(lv)
    
    # Spike detection
    if epoch > 0 and lv > losses_history[-2] * 2:
        print(f"  [Epoch {epoch}] SPIKE: {losses_history[-2]:.4f} -> {lv:.4f}")

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: {total_loss.item():.4f}")
else:
    initial = losses_history[0]
    final   = losses_history[-1]
    drop    = (initial - final) / initial * 100
    print(f"\nBaşlangıç : {initial:.4f}")
    print(f"Final     : {final:.4f}")
    print(f"Düşüş     : %{drop:.1f}")

    if drop > 90:
        print("✅ Overfit başarılı — mimari uyumlu")
        record("Overfit %90+ (gerçek Mamba)", PASS, f"drop={drop:.1f}%")
    elif drop > 50:
        print("⚠️  Kısmi öğrenme — d_state=32 veya lr=5e-4 ile tekrar dene")
        record("Overfit %90+ (gerçek Mamba)", WARN, f"drop={drop:.1f}%")
    else:
        print("🚨 Overfit başarısız!")
        print("   1. positional encoding Mamba'da ağırlık taşımıyor olabilir (yavaş convergence)")
        print("   2. LR çok yüksek → clip_grad_norm dene (zaten 1.0 uygulandı)")
        print("   3. d_state=32 ile tekrar dene")
        record("Overfit %90+ (gerçek Mamba)", FAIL, f"drop={drop:.1f}%")

# Encoder'ı geri yükle
model_mamba.transformer.encoder = original_enc

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7: KARAR RAPORU
# ══════════════════════════════════════════════════════════════════════════════

print("\n")
print("╔" + "═"*68 + "╗")
print("║   PRE-INTEGRATION ANALYSIS RAPORU — LSTR + Mamba                  ║")
print("╚" + "═"*68 + "╝")
print(f"\nModel              : LSTR (Mini-ResNet, CULane config)")
print(f"Entegre edilecek   : BidirectionalMambaEncoder (gerçek mamba-ssm)")
print(f"Entegrasyon noktası: model.transformer.encoder")
print(f"GPU                : {torch.cuda.get_device_name(0)}")
print(f"CUDA               : {torch.version.cuda}")
print(f"PyTorch            : {torch.__version__}")

header = f"{'Test':<50}{'Durum':<10}{'Notlar'}"
print(f"\n{header}")
print("-" * 100)

icon = {PASS: "✅", FAIL: "🚨", WARN: "⚠️ "}
all_pass = True
for test, (status, note) in results.items():
    sym = icon.get(status, "?")
    print(f"  {test:<50}{sym+' '+status:<12}{note}")
    if status == FAIL:
        all_pass = False

print("-" * 100)
print("\nGENEL KARAR:")
if all_pass:
    fails  = [k for k, (s, _) in results.items() if s == FAIL]
    warns  = [k for k, (s, _) in results.items() if s == WARN]
    if not fails and not warns:
        print("✅ ENTEGRASYONBİLİR — tüm testler geçti")
    elif not fails:
        print(f"⚠️  KOŞULLU — uyarılar mevcut: {warns}")
    else:
        print(f"🚨 ENTEGRASYONBİLİR DEĞİL — kritik: {fails}")
else:
    fails = [k for k, (s, _) in results.items() if s == FAIL]
    print(f"🚨 ENTEGRASYONBİLİR DEĞİL — kritik sorunlar: {fails}")

print("""
ÖNERİLEN SONRAKI ADIMLAR:
  Tüm testler geçtiyse:
    1. models/py_utils/mamba_encoder.py projeye eklendi (hazır)
    2. kp.py veya LSTR_CULANE.py'de:
         from models.py_utils.mamba_encoder import BidirectionalMambaEncoder
         model.transformer.encoder = BidirectionalMambaEncoder()
    3. Baseline LSTR F1 ölç → Mamba versiyonu eğit → karşılaştır
    4. Backbone (VMamba) geçişinden önce encoder ablation'ı tamamla

  Overfit başarısız olduysa:
    → mamba_encoder.py'de d_state=32 yap ve tekrar dene
    → lr=5e-4 ve 400 epoch dene
    → AELoss Hungarian matching instabilite için loss weight'leri azalt (loss_curves * 2 yerine * 1)
""")
