"""
LSTR + Gerçek mamba-ssm — Konsolide Ön-Analiz Scripti
=======================================================

Önceki 3 scriptin (mamba_real_analysis.py, mamba_3tests.py, mamba_isolation.py)
birleştirilmiş ve düzeltilmiş hali.

Düzeltilen sorunlar:
  - Veri yükleme hatası tam traceback ile raporlanıyor
  - B=1 ve B=16 wrapper shape testi dinamik
  - enc_attn_weights reshape hardcoded (12,20) yerine dinamik
  - Baseline transformer overfit testi eklendi (ablation için)
  - B=1 isolation testi eklendi (batch etkisi izole)
  - d_state=16 vs d_state=32 karşılaştırması
  - loss_curves=2.5 spike fix dahil
  - all_pass mantığı düzeltildi (önceki sürümde ters çalışıyordu)
  - net = model_mamba.transformer.encoder ataması kaldırıldı (bug)

KURULUM:
    conda activate clrernet
    pip install causal-conv1d mamba-ssm --no-build-isolation

ÇALIŞTIR:
    conda activate clrernet && python mamba_real_analysis.py
"""

import os
import sys
import json
import math
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
from config import system_configs
cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE.json")
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["system"]["snapshot_name"] = "LSTR_CULANE"
cfg["system"]["data_dir"] = "/home/alki/projects/"
system_configs.update_config(cfg["system"])

from models.LSTR_CULANE import model as LSTRModel, loss as LSTRLoss

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
results = {}

def record(test_name, status, note=""):
    results[test_name] = (status, note)
    icon = {"PASS": "✅", "FAIL": "🚨", "WARN": "⚠️ "}
    print(f"  → KAYIT [{icon.get(status,'?')} {status}] {test_name}: {note}")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: MODEL MİMARİSİ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 1.1: PARAMETRE HARİTASI")
print("=" * 80)

model = LSTRModel(flag=True).cuda()
model.eval()

total_params  = 0
frozen_params = 0
non_bn_frozen = []

for name, param in model.named_parameters():
    total_params += param.numel()
    if not param.requires_grad:
        frozen_params += param.numel()
        if "bn" not in name.lower() and "norm" not in name.lower():
            non_bn_frozen.append(name)
    status = "TRAINABLE" if param.requires_grad else "FROZEN"
    print(f"  {name:<60} | {str(list(param.shape)):<25} | {status}")

print(f"\nToplam  : {total_params:,}")
print(f"Frozen  : {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
print(f"Trainable: {total_params - frozen_params:,}")

if non_bn_frozen:
    print(f"[WARN] BN dışı frozen: {non_bn_frozen}")
    record("Frozen param sadece BN (kasıtlı)", WARN, str(non_bn_frozen))
else:
    print("✅ Frozen = yalnızca BatchNorm — beklenen")
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
                "has_nan" : bool(torch.isnan(t).any()),
                "has_inf" : bool(torch.isinf(t).any()),
            }
    return hook

for name, mod in model.named_modules():
    if name in ['layer1', 'layer4', 'input_proj', 'transformer.encoder', 'transformer.decoder']:
        hooks_store[name] = mod.register_forward_hook(make_hook(name))

dummy_input = torch.randn(1, 3, 360, 640).cuda()
dummy_mask  = torch.zeros(1, 1, 360, 640).cuda()

with torch.no_grad():
    _ = model(dummy_input, dummy_mask)

for h in hooks_store.values():
    h.remove()

# Gerçek çözünürlük 295×820 olabilir — EXPECTED dummy input içindir
EXPECTED_DUMMY = {
    "layer1"    : [1, 16, 90, 160],
    "layer4"    : [1, 128, 12, 20],
    "input_proj": [1, 32, 12, 20],
}

all_shapes_ok = True
for name, stats in activations.items():
    print(f"\n[{name}]")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats.get("has_nan"):
        print("  🚨 NaN!")
        all_shapes_ok = False
    if stats.get("has_inf"):
        print("  🚨 Inf!")
        all_shapes_ok = False
    if name in EXPECTED_DUMMY:
        if stats["shape"] != EXPECTED_DUMMY[name]:
            print(f"  🚨 Boyut uyumsuz: beklenen {EXPECTED_DUMMY[name]}")
            all_shapes_ok = False
        else:
            print(f"  ✅ Boyut doğru")

# Dinamik HW bilgisi — sonraki testlerde kullanılacak
proj_shape = activations.get("input_proj", {}).get("shape", [1, 32, 12, 20])
DUMMY_H, DUMMY_W = proj_shape[2], proj_shape[3]
DUMMY_HW = DUMMY_H * DUMMY_W
print(f"\n[input_proj] Dummy HW = {DUMMY_H}×{DUMMY_W} = {DUMMY_HW} token")

record("Forward pass boyutlar doğru", PASS if all_shapes_ok else FAIL)
record("NaN/Inf yok", PASS if all_shapes_ok else FAIL)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: VERİ YÜKLEME — TAM TRACEBACK
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 2: GERÇEK CULane VERİSİ YÜKLEME")
print("=" * 80)

real_images  = None
real_masks   = None
real_targets = None

try:
    from db.datasets import datasets
    import importlib
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    print(f"DB yüklendi: {db_obj.db_inds.size} örnek")
    sample_fn = importlib.import_module('sample.culane').sample_data
    data, _   = sample_fn(db_obj, 0)
    real_images  = data['xs'][0].cuda()
    real_masks   = data['xs'][1].cuda()
    real_targets = [t.cuda() for t in data['ys'][1:]]
    print(f"✅ Gerçek veri yüklendi: B={real_images.shape[0]}, img={real_images.shape}")
    print(f"   masks: {real_masks.shape}")
    print(f"   targets: {len(real_targets)} tensor, ilk shape: {real_targets[0].shape}")
    record("Gerçek CULane veri yükleme", PASS, f"B={real_images.shape[0]}")
except Exception:
    print("🚨 Gerçek veri yüklenemedi — tam traceback:")
    traceback.print_exc()
    print("\n[!] torch.randn ile devam edilecek — feature similarity sonuçları GEÇERSİZ sayılacak")
    record("Gerçek CULane veri yükleme", FAIL, "traceback yukarıda")

# Gerçek veri boyutlarını al (varsa), yoksa dummy boyutlar
if real_images is not None:
    REAL_H = real_images.shape[2] // 32  # backbone stride=32
    REAL_W = real_images.shape[3] // 32
    REAL_HW = REAL_H * REAL_W
    print(f"   Gerçek backbone çıkış: {REAL_H}×{REAL_W} = {REAL_HW} token")
else:
    REAL_H, REAL_W, REAL_HW = DUMMY_H, DUMMY_W, DUMMY_HW

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: FEATURE BENZERLİĞİ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 3: FEATURE BENZERLİĞİ (input_proj) — BASELINE vs MAMBA KARŞILAŞTIRMASI")
print("=" * 80)

features_list = []

def feat_hook(module, inp, output):
    features_list.append(output.detach().cpu())

target_layer = dict(model.named_modules())['input_proj']
h_feat = target_layer.register_forward_hook(feat_hook)

model.eval()
with torch.no_grad():
    if real_images is not None:
        for i in range(min(5, real_images.shape[0])):
            model(real_images[i:i+1], real_masks[i:i+1])
        print("Gerçek CULane görüntüleri kullanıldı.")
    else:
        for _ in range(5):
            model(torch.randn(1, 3, 360, 640).cuda(), dummy_mask)
        print("⚠️  torch.randn kullanıldı — sonuç geçersiz!")

h_feat.remove()

if len(features_list) >= 5:
    flat_features = [f.flatten() for f in features_list[:5]]
    sim_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            sim_matrix[i][j] = F.cosine_similarity(
                flat_features[i].unsqueeze(0),
                flat_features[j].unsqueeze(0)
            ).item()

    print("\nCosine Benzerlik Matrisi (input_proj, 5 görüntü):")
    print(np.round(sim_matrix, 3))
    avg_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / 20
    print(f"\nOrtalama çapraz-benzerlik: {avg_off_diag:.4f}")
    print("NOT: Bu değer baseline transformer için de aynı çıkacak (aynı input_proj).")
    print("Kritik olan bu değerin Mamba öğrenmesini engellememesi — overfit testi bunu kanıtlar.")

    if avg_off_diag > 0.95:
        note = f"avg={avg_off_diag:.4f} — gerçek veriyle yüksek ama overfit testi belirleyici"
        record("Feature benzerliği (bilgi amaçlı)", WARN, note)
    elif avg_off_diag > 0.85:
        record("Feature benzerliği (bilgi amaçlı)", WARN, f"avg={avg_off_diag:.4f}")
    else:
        record("Feature benzerliği (bilgi amaçlı)", PASS, f"avg={avg_off_diag:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: GRADIENT ANALİZİ — MEVCUT ENCODER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 4: GRADIENT ANALİZİ — mevcut transformer.encoder")
print("=" * 80)

model.train()
out_dict, _ = model._train(dummy_input, dummy_mask)
dummy_loss = out_dict['pred_logits'].sum() + out_dict['pred_curves'].sum()
model.zero_grad()
dummy_loss.backward()

dead_layers = []
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

print(f"\nDead: {len(dead_layers)} | Exploding: {len(exploding_layers)}")
if dead_layers or exploding_layers:
    record("Mevcut encoder gradient sağlıklı", WARN,
           f"dead={dead_layers}, exploding={exploding_layers}")
else:
    print("✅ Gradient akışı sağlıklı")
    record("Mevcut encoder gradient sağlıklı", PASS)

model.eval()

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: GERÇEK MAMBA-SSM KURULUM + İZOLE TEST
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 5: GERÇEK MAMBA-SSM KURULUM + İZOLE TEST")
print("=" * 80)

try:
    from mamba_ssm import Mamba
    print("✅ mamba-ssm import başarılı")
    record("Gerçek mamba-ssm kurulumu başarılı", PASS)
except ImportError as e:
    print(f"🚨 mamba-ssm import HATASI: {e}")
    print("   pip install mamba-ssm causal-conv1d --no-build-isolation")
    record("Gerçek mamba-ssm kurulumu başarılı", FAIL, str(e))
    print("\n[DURDURULDU] mamba-ssm olmadan devam edilemez.")
    sys.exit(1)

d_model = system_configs.attn_dim  # 32

mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).cuda()
mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).cuda()

test_input = torch.randn(1, DUMMY_HW, d_model, device='cuda', requires_grad=True)

out_fwd  = mamba_fwd(test_input)
out_bwd  = torch.flip(mamba_bwd(torch.flip(test_input, [1])), [1])
out_bidir = out_fwd + out_bwd
print(f"Forward çıktı     : {out_fwd.shape}")
print(f"Bidirectional çıktı: {out_bidir.shape}")

out_bidir.sum().backward()
inp_grad_mean = test_input.grad.abs().mean().item()
print(f"\nGiriş gradient mean: {inp_grad_mean:.2e}")

all_grad_ok = True
for name, param in list(mamba_fwd.named_parameters()) + list(mamba_bwd.named_parameters()):
    if param.grad is None:
        print(f"  ⚠️  {name}: gradient YOK")
        all_grad_ok = False
    elif param.grad.abs().mean() < 1e-9:
        print(f"  ⚠️  {name}: dead gradient ({param.grad.abs().mean():.2e})")
        all_grad_ok = False
    else:
        print(f"  ✅ {name}: {param.grad.abs().mean():.2e}")

bidir_shape_ok = (out_bidir.shape == (1, DUMMY_HW, d_model))
if all_grad_ok and bidir_shape_ok:
    print("\n✅ Gerçek Mamba kernel gradient akışı tamam")
    record("Mamba forward/bidirectional çalışıyor", PASS)
    record("Mamba gradient akışı sağlıklı", PASS, f"inp_grad={inp_grad_mean:.2e}")
else:
    record("Mamba forward/bidirectional çalışıyor", FAIL if not bidir_shape_ok else PASS)
    record("Mamba gradient akışı sağlıklı", FAIL if not all_grad_ok else PASS)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: WRAPPER SHAPE DOĞRULAMASI — B=1 ve B=16
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 6: BidirectionalMambaEncoder SHAPE DOĞRULAMASI")
print("=" * 80)

from models.py_utils.mamba_encoder import BidirectionalMambaEncoder

enc       = BidirectionalMambaEncoder().cuda()
num_heads = system_configs.num_heads
print(f"num_heads = {num_heads}")

# B=1 testi
src_b1   = torch.randn(DUMMY_HW, 1, d_model).cuda()
out_b1, wts_b1 = enc(src_b1)
ok_b1    = (out_b1.shape == (DUMMY_HW, 1, d_model)) and \
           (wts_b1.shape == (num_heads * 1, DUMMY_HW, DUMMY_HW))
print(f"B=1: out={out_b1.shape}, wts={wts_b1.shape} → {'✅ PASS' if ok_b1 else '🚨 FAIL'}")

# B=16 testi — gerçek eğitim batch size
src_b16  = torch.randn(DUMMY_HW, 16, d_model).cuda()
out_b16, wts_b16 = enc(src_b16)
ok_b16   = (out_b16.shape == (DUMMY_HW, 16, d_model)) and \
           (wts_b16.shape == (num_heads * 16, DUMMY_HW, DUMMY_HW))
print(f"B=16: out={out_b16.shape}, wts={wts_b16.shape} → {'✅ PASS' if ok_b16 else '🚨 FAIL (egitimde patlayacak)'}")

# Dinamik reshape testi — test/culane.py için
# shape = f_map.shape[-2:] ile dinamik alınıyor, hardcoded değil
# Gerçek veri boyutuyla test et
try:
    reshaped = wts_b1[0].reshape(DUMMY_H, DUMMY_W, DUMMY_H, DUMMY_W)
    print(f"✅ wts_b1[0].reshape({DUMMY_H},{DUMMY_W},{DUMMY_H},{DUMMY_W}): {reshaped.shape}")
    record("enc_attn_weights shape doğrulandı", PASS,
           f"wts[0].reshape({DUMMY_H},{DUMMY_W},{DUMMY_H},{DUMMY_W})={reshaped.shape}")
except Exception as e:
    print(f"🚨 reshape FAIL: {e}")
    record("enc_attn_weights shape doğrulandı", FAIL, str(e))

if ok_b1 and ok_b16:
    record("Wrapper shape testi geçti (B=1 ve B=16)", PASS)
else:
    record("Wrapper shape testi geçti (B=1 ve B=16)", FAIL,
           f"B1={'OK' if ok_b1 else 'FAIL'} B16={'OK' if ok_b16 else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7: OVERFİT TESTLERİ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("BÖLÜM 7: OVERFİT TESTLERİ")
print("=" * 80)

def make_targets(img, tgt_tensor_list):
    """targets_list formatı: [image_tensor] + [per_sample_labels]"""
    return [img] + tgt_tensor_list

def run_overfit(label, img, mask, targets_list, use_mamba=True, d_state=16, epochs=200):
    """
    use_mamba=True  → BidirectionalMambaEncoder
    use_mamba=False → orijinal LSTR transformer encoder (baseline)
    """
    print(f"\n{'─'*60}")
    print(f"TEST: {label}")
    print(f"  use_mamba={use_mamba}, d_state={d_state}, B={img.shape[0]}, epochs={epochs}")
    print(f"{'─'*60}")

    m = LSTRModel(flag=True).cuda()

    if use_mamba:
        class Bidir(nn.Module):
            def __init__(self):
                super().__init__()
                nh = system_configs.num_heads
                self.nh = nh
                self.fwd = nn.ModuleList([
                    Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                    for _ in range(system_configs.enc_layers)])
                self.bwd = nn.ModuleList([
                    Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                    for _ in range(system_configs.enc_layers)])
                self.norms = nn.ModuleList([
                    nn.LayerNorm(d_model)
                    for _ in range(system_configs.enc_layers)])

            def forward(self, src, src_key_padding_mask=None, pos=None):
                if pos is not None:
                    src = src + pos
                x = src.permute(1, 0, 2)   # (HW,B,D) → (B,HW,D)
                for f, b, n in zip(self.fwd, self.bwd, self.norms):
                    x = n(x + f(x) + torch.flip(b(torch.flip(x, [1])), [1]))
                out = x.permute(1, 0, 2)   # → (HW,B,D)
                B = out.shape[1]
                wts = torch.zeros(B * self.nh, out.shape[0], out.shape[0], device=out.device)
                return out, wts

        m.transformer.encoder = Bidir().cuda()

    crit = LSTRLoss().cuda()
    for k in list(crit.criterion.weight_dict.keys()):
        if 'loss_curves' in k:
            crit.criterion.weight_dict[k] = 2.5  # spike fix

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, m.parameters()), lr=1e-3)

    losses = []
    spikes = []
    m.train()

    for epoch in range(epochs):
        opt.zero_grad()
        out_pred, _ = m._train(img, mask)
        loss_val    = crit(epoch, False, 'train', out_pred, targets_list)[0].mean()

        if not math.isfinite(loss_val.item()):
            print(f"  [Epoch {epoch}] NaN/Inf — durduruluyor.")
            return None

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        lv = loss_val.item()
        if epoch > 0 and lv > losses[-1] * 2:
            spikes.append((epoch, losses[-1], lv))
            print(f"  [Epoch {epoch}] SPIKE: {losses[-1]:.4f} → {lv:.4f}")
        losses.append(lv)

        if epoch % 40 == 0:
            print(f"  Epoch {epoch:3d}: {lv:.4f}")

    init  = losses[0]
    final = losses[-1]
    drop  = (init - final) / init * 100
    print(f"\n  Başlangıç: {init:.4f} | Final: {final:.4f} | Düşüş: %{drop:.1f}")
    print(f"  Spike sayısı: {len(spikes)}")

    if drop > 90:
        print(f"  → PASS (>90%)")
    elif drop > 75:
        print(f"  → WARN (75-90%)")
    else:
        print(f"  → FAIL (<75%)")

    return drop

# ── Veri hazırla ──────────────────────────────────────────────────────────────
if real_images is not None:
    img_b1   = real_images[0:1]
    mask_b1  = real_masks[0:1]
    tgt_b1   = real_targets[0][0:1]
    tgts_b1  = make_targets(img_b1, [tgt_b1])

    img_b16  = real_images
    mask_b16 = real_masks
    tgts_b16 = make_targets(img_b16, real_targets)
else:
    n_pts   = 18
    tgt_dummy = torch.zeros(2, 1 + 2 + n_pts*2).cuda()
    tgt_dummy[:, 0] = 1
    tgt_dummy[:, 1] = 0.8
    tgt_dummy[:, 2] = 0.3
    tgt_dummy[:, 3:3+n_pts] = torch.linspace(0.3, 0.7, n_pts).cuda()
    tgt_dummy[:, 3+n_pts:]  = torch.linspace(0.3, 0.8, n_pts).cuda()

    img_b1   = torch.randn(1, 3, 360, 640).cuda()
    mask_b1  = torch.zeros(1, 1, 360, 640).cuda()
    tgts_b1  = make_targets(img_b1, [tgt_dummy.unsqueeze(0)])

    img_b16  = img_b1
    mask_b16 = mask_b1
    tgts_b16 = tgts_b1

# ── Test A: Baseline transformer, B=1 ────────────────────────────────────────
drop_baseline = run_overfit(
    "BASELINE transformer, B=1 (ablation referansı)",
    img_b1, mask_b1, tgts_b1,
    use_mamba=False, epochs=200
)

# ── Test B: Mamba d_state=16, B=1 ────────────────────────────────────────────
drop_mamba_b1_d16 = run_overfit(
    "Mamba d_state=16, B=1",
    img_b1, mask_b1, tgts_b1,
    use_mamba=True, d_state=16, epochs=200
)

# ── Test C: Mamba d_state=32, B=1 — kapasite karşılaştırması ─────────────────
drop_mamba_b1_d32 = run_overfit(
    "Mamba d_state=32, B=1 (kapasite artırımı)",
    img_b1, mask_b1, tgts_b1,
    use_mamba=True, d_state=32, epochs=200
)

# ── Test D: Mamba d_state=16, B=16 — tam batch ───────────────────────────────
drop_mamba_b16 = run_overfit(
    "Mamba d_state=16, B=16 (tam batch)",
    img_b16, mask_b16, tgts_b16,
    use_mamba=True, d_state=16, epochs=200
)

# ── Overfit kayıtları ──────────────────────────────────────────────────────────
def overfit_status(drop, label):
    if drop is None:
        record(label, FAIL, "NaN/Inf")
    elif drop > 90:
        record(label, PASS, f"drop={drop:.1f}%")
    elif drop > 75:
        record(label, WARN, f"drop={drop:.1f}%")
    else:
        record(label, FAIL, f"drop={drop:.1f}%")

overfit_status(drop_baseline,      "Overfit: Baseline transformer B=1")
overfit_status(drop_mamba_b1_d16,  "Overfit: Mamba d16 B=1")
overfit_status(drop_mamba_b1_d32,  "Overfit: Mamba d32 B=1")
overfit_status(drop_mamba_b16,     "Overfit: Mamba d16 B=16 (tam batch)")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8: KARŞILAŞTIRMA TABLOSU + KARAR RAPORU
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n")
print("╔" + "═"*68 + "╗")
print("║   PRE-INTEGRATION ANALYSIS RAPORU — LSTR + Mamba                  ║")
print("╚" + "═"*68 + "╝")
print(f"\nModel              : LSTR (Mini-ResNet, CULane config)")
print(f"Entegre edilecek   : BidirectionalMambaEncoder (gerçek mamba-ssm kernel)")
print(f"Entegrasyon noktası: model.transformer.encoder")
print(f"GPU                : {torch.cuda.get_device_name(0)}")
print(f"CUDA               : {torch.version.cuda}")
print(f"PyTorch            : {torch.__version__}")

# Overfit karşılaştırma tablosu
print(f"\n{'─'*60}")
print("OVERFİT KARŞILAŞTIRMASI (ablation referansı)")
print(f"{'─'*60}")
rows = [
    ("Baseline transformer B=1",    drop_baseline),
    ("Mamba d_state=16   B=1",      drop_mamba_b1_d16),
    ("Mamba d_state=32   B=1",      drop_mamba_b1_d32),
    ("Mamba d_state=16   B=16",     drop_mamba_b16),
]
for label, drop in rows:
    if drop is None:
        verdict = "🚨 NaN"
    elif drop > 90:
        verdict = f"✅ %{drop:.1f}"
    elif drop > 75:
        verdict = f"⚠️  %{drop:.1f}"
    else:
        verdict = f"🚨 %{drop:.1f}"
    print(f"  {label:<35} {verdict}")

# Tüm test sonuçları
print(f"\n{'Test':<52}{'Durum':<10}{'Notlar'}")
print("─" * 100)
icon = {PASS: "✅", FAIL: "🚨", WARN: "⚠️ "}
fails = [k for k, (s, _) in results.items() if s == FAIL]
warns = [k for k, (s, _) in results.items() if s == WARN]
for test, (status, note) in results.items():
    sym = icon.get(status, "?")
    print(f"  {test:<52}{sym+' '+status:<12}{note}")
print("─" * 100)

print("\nGENEL KARAR:")
if not fails:
    if not warns:
        print("✅ ENTEGRASYONBİLİR — tüm testler geçti")
    else:
        print(f"⚠️  KOŞULLU — uyarılar mevcut: {warns}")
else:
    print(f"🚨 ENTEGRASYONBİLİR DEĞİL — kritik sorunlar: {fails}")

# Ablation yorumu
print("\nABLATION YORUMU:")
if drop_baseline and drop_mamba_b1_d16:
    diff = drop_mamba_b1_d16 - drop_baseline
    if abs(diff) < 5:
        print(f"  ✅ Mamba (%{drop_mamba_b1_d16:.1f}) ≈ Transformer (%{drop_baseline:.1f}) — overfit kapasitesi eşdeğer")
    elif diff > 0:
        print(f"  ✅ Mamba (%{drop_mamba_b1_d16:.1f}) > Transformer (%{drop_baseline:.1f}) — Mamba daha iyi ezberliyor")
    else:
        print(f"  ⚠️  Mamba (%{drop_mamba_b1_d16:.1f}) < Transformer (%{drop_baseline:.1f}) — kapasite farkı var, d_state artır")

if drop_mamba_b16 and drop_mamba_b1_d16:
    diff_batch = drop_mamba_b1_d16 - drop_mamba_b16
    if diff_batch > 10:
        print(f"  ℹ️  B=16 düşüşü (%{drop_mamba_b16:.1f}) batch etkisinden — normal, gerçek eğitimde optimizer'ın avantajı var")

print("""
ÖNERİLEN SONRAKI ADIMLAR:
  Tüm testler geçtiyse:
    1. models/py_utils/mamba_encoder.py projeye eklendi (hazır)
    2. kp.py veya LSTR_CULANE.py'de encoder'ı değiştir:
         from models.py_utils.mamba_encoder import BidirectionalMambaEncoder
         model.transformer.encoder = BidirectionalMambaEncoder()
    3. Baseline LSTR CULane F1 ölç (0.64 referans)
    4. Mamba versiyonu tam eğit → F1 karşılaştır
    5. Backbone (VMamba) geçişine encoder ablation'dan sonra geç

  Overfit başarısız veya Mamba < Baseline ise:
    → d_state=32 ile mamba_encoder.py'yi güncelle
    → lr=5e-4, 400 epoch dene
    → loss_curves weight'ini 2.0'a düşür
""")