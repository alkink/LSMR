"""
3 Hedefli Test:
1. Gerçek CULane veri yükleme — tam traceback
2. B=2 wrapper shape testi
3. loss_curves=2.5 ile 200 epoch overfit — spike kayboldu mu?
"""
import os, sys, json, math, traceback
import torch, torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import system_configs
with open('config/LSTR_CULANE.json') as f:
    cfg = json.load(f)
cfg['system']['snapshot_name'] = 'LSTR_CULANE'
cfg['system']['data_dir'] = '/home/alki/projects/'
system_configs.update_config(cfg['system'])

from models.LSTR_CULANE import model as LSTRModel, loss as LSTRLoss
from models.py_utils.mamba_encoder import BidirectionalMambaEncoder

# ─── TEST 1: Gerçek veri yükleme ─────────────────────────────────────────────
print("=" * 60)
print("TEST 1: GERCEK CULANE VERI YUKLEME")
print("=" * 60)
real_images = real_masks = real_targets = None
try:
    from db.datasets import datasets
    import importlib
    db_obj = datasets['CULANE'](cfg['db'], 'train+val')
    print(f"DB yuklendi. Index boyutu: {db_obj.db_inds.size}")
    sample_fn = importlib.import_module('sample.culane').sample_data
    data, _ = sample_fn(db_obj, 0)
    real_images  = data['xs'][0].cuda()
    real_masks   = data['xs'][1].cuda()
    real_targets = [t.cuda() for t in data['ys'][1:]]
    print(f"PASS: Gercek CULANE goruntu yuklendi")
    print(f"  images shape : {real_images.shape}")
    print(f"  masks  shape : {real_masks.shape}")
    print(f"  target tensors: {len(real_targets)}, shapes: {[t.shape for t in real_targets[:2]]}")
except Exception:
    print("FAIL: Veri yuklenemedi — tam traceback:")
    traceback.print_exc()

# ─── TEST 2: B=2 wrapper shape testi ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: B=2 WRAPPER SHAPE")
print("=" * 60)
num_heads = system_configs.num_heads  # 2
enc = BidirectionalMambaEncoder().cuda()

# B=1
src_b1 = torch.randn(240, 1, 32).cuda()
out_b1, wts_b1 = enc(src_b1)
ok_b1 = wts_b1.shape == (num_heads * 1, 240, 240)
print(f"B=1: out={out_b1.shape}, wts={wts_b1.shape} -> {'PASS' if ok_b1 else 'FAIL'}")

# B=2
src_b2 = torch.randn(240, 2, 32).cuda()
out_b2, wts_b2 = enc(src_b2)
ok_b2 = wts_b2.shape == (num_heads * 2, 240, 240)
print(f"B=2: out={out_b2.shape}, wts={wts_b2.shape} -> {'PASS' if ok_b2 else 'FAIL (BUG)'}")

# test/culane.py:169 reshape simulasyonu (B=1 ile olduğu gibi hook cagrilir)
try:
    reshaped = wts_b1[0].reshape(12, 20, 12, 20)
    print(f"wts_b1[0].reshape(12,20,12,20): {reshaped.shape} -> PASS")
except Exception as e:
    print(f"reshape FAIL: {e}")

# ─── TEST 3: loss_curves=2.5 overfit testi ────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: loss_curves=2.5 ile 200 EPOCH OVERFIT (SPIKE KONTROLU)")
print("=" * 60)

# loss_curves weight'ini override et
class PatchedLSTRLoss(LSTRLoss):
    def __init__(self):
        super().__init__()
        # loss_curves: 5 -> 2.5 (hem ana hem aux)
        for k in list(self.criterion.weight_dict.keys()):
            if 'loss_curves' in k:
                self.criterion.weight_dict[k] = 2.5
        print(f"loss weight_dict: {self.criterion.weight_dict}")

model_m = LSTRModel(flag=True).cuda()
model_m.transformer.encoder = BidirectionalMambaEncoder().cuda()
criterion = PatchedLSTRLoss().cuda()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_m.parameters()), lr=1e-3
)

# Gercek veri varsa tam batch kullan — kp.py:258 targets[1:] iterates over batch
if real_images is not None:
    single_img  = real_images   # [16, 3, 295, 820]
    single_mask = real_masks    # [16, 1, 295, 820]
    # data['ys'] = [images, gt_lanes_b0, ..., gt_lanes_b15]
    # her gt_lanes_bi: [batch_size, n_lanes, 73]
    targets_list = [real_images] + real_targets
    print(f"Gercek CULANE tam batch: B={single_img.shape[0]}")
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

losses_hist = []
spikes = []
model_m.train()

for epoch in range(200):
    optimizer.zero_grad()
    out_pred, _ = model_m._train(single_img, single_mask)
    loss_result  = criterion(epoch, False, 'train', out_pred, targets_list)
    total_loss   = loss_result[0].mean()

    if not math.isfinite(total_loss.item()):
        print(f"[Epoch {epoch}] NaN/Inf — durduruluyor.")
        break

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_m.parameters(), 1.0)
    optimizer.step()

    lv = total_loss.item()
    if epoch > 0 and lv > losses_hist[-1] * 2:
        spikes.append((epoch, losses_hist[-1], lv))
        print(f"  [Epoch {epoch}] SPIKE: {losses_hist[-1]:.4f} -> {lv:.4f}")
    losses_hist.append(lv)

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: {lv:.4f}")

if losses_hist:
    init, final = losses_hist[0], losses_hist[-1]
    drop = (init - final) / init * 100
    print(f"\nBaslangic: {init:.4f} | Final: {final:.4f} | Dusus: %{drop:.1f}")
    print(f"Toplam spike sayisi: {len(spikes)}")
    if spikes:
        print("Spike detayi:", spikes)
    if len(spikes) == 0:
        print("PASS: loss_curves=2.5 ile spike kayboldu.")
    elif drop > 90:
        print("PARTIAL: %90+ overfit saglandı ama spike devam ediyor — daha derin sorun.")
    else:
        print("FAIL: Overfit saglanamadi ve spike var.")
