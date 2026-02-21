"""
Isolation Test: B=1 overfit ile batch boyutu etkisini izole et.
+ d_state=32 ile kapasite testi
"""
import os, sys, json, math
import torch
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import system_configs
with open('config/LSTR_CULANE.json') as f:
    cfg = json.load(f)
cfg['system']['snapshot_name'] = 'LSTR_CULANE'
cfg['system']['data_dir'] = '/home/alki/projects/'
system_configs.update_config(cfg['system'])

from models.LSTR_CULANE import model as LSTRModel, loss as LSTRLoss
from models.py_utils.mamba_encoder import BidirectionalMambaEncoder
from mamba_ssm import Mamba

# ── Gerçek veri yükle ────────────────────────────────────────────────────────
from db.datasets import datasets
import importlib
db_obj = datasets['CULANE'](cfg['db'], 'train+val')
sample_fn = importlib.import_module('sample.culane').sample_data
data, _ = sample_fn(db_obj, 0)
real_images  = data['xs'][0].cuda()
real_masks   = data['xs'][1].cuda()
real_targets = [t.cuda() for t in data['ys'][1:]]
print(f"Veri yuklendi: B={real_images.shape[0]}, img={real_images.shape}")

def run_overfit(label, img, mask, targets_list, d_state=16, epochs=200):
    print(f"\n{'='*55}")
    print(f"TEST: {label}")
    print(f"  d_state={d_state}, B={img.shape[0]}, epochs={epochs}")
    print(f"{'='*55}")

    model_m = LSTRModel(flag=True).cuda()

    # BidirectionalMambaEncoder'i d_state parametresiyle oluştur
    import torch.nn as nn
    class Bidir(nn.Module):
        def __init__(self):
            super().__init__()
            attn_dim   = system_configs.attn_dim
            num_heads  = system_configs.num_heads
            enc_layers = system_configs.enc_layers
            self.num_heads = num_heads
            self.mamba_fwd = nn.ModuleList([
                Mamba(d_model=attn_dim, d_state=d_state, d_conv=4, expand=2)
                for _ in range(enc_layers)])
            self.mamba_bwd = nn.ModuleList([
                Mamba(d_model=attn_dim, d_state=d_state, d_conv=4, expand=2)
                for _ in range(enc_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(attn_dim) for _ in range(enc_layers)])

        def forward(self, src, src_key_padding_mask=None, pos=None):
            if pos is not None: src = src + pos
            x = src.permute(1, 0, 2)  # (HW,B,D) → (B,HW,D)
            for fwd, bwd, norm in zip(self.mamba_fwd, self.mamba_bwd, self.norms):
                x = norm(x + fwd(x) + torch.flip(bwd(torch.flip(x, [1])), [1]))
            out = x.permute(1, 0, 2)
            B = out.shape[1]
            dummy_wts = torch.zeros(B * self.num_heads, out.shape[0], out.shape[0], device=out.device)
            return out, dummy_wts

    model_m.transformer.encoder = Bidir().cuda()

    criterion = LSTRLoss().cuda()
    for k in list(criterion.criterion.weight_dict.keys()):
        if 'loss_curves' in k:
            criterion.criterion.weight_dict[k] = 2.5

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_m.parameters()), lr=1e-3)

    losses = []
    spikes = []
    model_m.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out_pred, _ = model_m._train(img, mask)
        loss_result  = criterion(epoch, False, 'train', out_pred, targets_list)
        total_loss   = loss_result[0].mean()

        if not math.isfinite(total_loss.item()):
            print(f"  [Epoch {epoch}] NaN/Inf — durduruluyor.")
            return None

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_m.parameters(), 1.0)
        optimizer.step()

        lv = total_loss.item()
        if epoch > 0 and lv > losses[-1] * 2:
            spikes.append((epoch, losses[-1], lv))
            print(f"  [Epoch {epoch}] SPIKE: {losses[-1]:.4f} -> {lv:.4f}")
        losses.append(lv)

        if epoch % 40 == 0:
            print(f"  Epoch {epoch:3d}: {lv:.4f}")

    init, final = losses[0], losses[-1]
    drop = (init - final) / init * 100
    print(f"\n  Baslangic: {init:.4f} | Final: {final:.4f} | Dusus: %{drop:.1f}")
    print(f"  Spike sayisi: {len(spikes)}")

    if drop > 90:
        verdict = "PASS (>90%)"
    elif drop > 75:
        verdict = "WARN (75-90%)"
    else:
        verdict = "FAIL (<75%)"
    print(f"  Sonuc: {verdict}")
    return drop

# ── Test A: B=1, d_state=16 ─────────────────────────────────────────────────
# B=1: tek goruntu, targets[1:] icin sadece 1 eleman gerekiyor
img_b1   = real_images[0:1]
mask_b1  = real_masks[0:1]
# kp.py:258 gt_cluxy = [tgt[0] for tgt in targets[1:]]
# targets[1:] listesinin uzunlugu == batch_size == 1 olmali.
# real_targets[0] shape: [16, n_lanes, 73] → [0] = tek ornek
tgt_b1 = real_targets[0][0:1]   # [1, n_lanes, 73]
targets_b1 = [img_b1, tgt_b1]

drop_a = run_overfit("B=1, d_state=16", img_b1, mask_b1, targets_b1, d_state=16)

# ── Test B: B=1, d_state=32 — kapasite artırımı ─────────────────────────────
drop_b = run_overfit("B=1, d_state=32", img_b1, mask_b1, targets_b1, d_state=32)

# ── Özet ────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("OZET")
print("="*55)
print(f"  B=16 d16 (onceki): %75.4")
print(f"  B=1  d16         : %{drop_a:.1f}" if drop_a else "  B=1 d16: FAIL (NaN)")
print(f"  B=1  d32         : %{drop_b:.1f}" if drop_b else "  B=1 d32: FAIL (NaN)")
if drop_a and drop_a > 90:
    print("\n  KARAR: %75.4 sorunu batch boyutundan kaynaklaniyordu.")
    print("  Mamba kapasitesi yeterli — entegrasyon devam edebilir.")
elif drop_b and drop_b > 90:
    print("\n  KARAR: d_state=16 yetersiz, d_state=32 gerekli.")
    print("  mamba_encoder.py'de d_state=32 yap ve tekrar egit.")
else:
    print("\n  KARAR: Mimari kapasite sorunu var — daha derin inceleme gerekli.")
