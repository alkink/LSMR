"""
Progressive Scaling Training Script — LSTR Baseline vs Mamba Ablation
=======================================================================

Plan:
  Stage 1: train_100  (100 images)  — 50 epoch  → Baseline F1 vs Mamba F1
  Stage 2: train_2k   (2000 images) — 100 epoch → Baseline F1 vs Mamba F1
  Stage 3: Full train (87k images)  — 500k iter → Kazanan mimari ile

Kullanım:
  python progressive_train.py --stage 100 --mode baseline
  python progressive_train.py --stage 100 --mode mamba
  python progressive_train.py --stage 2k  --mode baseline
  python progressive_train.py --stage 2k  --mode mamba

Değerlendirme (her aşamadan sonra):
  python test.py LSTR_CULANE --testiter <iter> --modality eval --split testing
  cd lane_evaluation && bash run.sh
"""

import os
import sys
import json
import queue
import random
import argparse
import importlib
import threading
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Progressive Scaling Train")
parser.add_argument("--stage",  choices=["100", "2k"], required=True,
                    help="Eğitim seti: '100' (100 görüntü) veya '2k' (2000 görüntü)")
parser.add_argument("--mode",   choices=["baseline", "mamba"], required=True,
                    help="Mimari: 'baseline' (orijinal transformer) veya 'mamba'")
parser.add_argument("--iter",   type=int, default=0,
                    help="Snapshot'tan devam (opsiyonel)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
from config import system_configs

cfg_file = os.path.join(system_configs.config_dir, "LSTR_CULANE.json")
with open(cfg_file) as f:
    cfg = json.load(f)

# Stage'e göre split ve iteration ayarla
STAGE_SETTINGS = {
    "100": {
        "train_split": "train_100",
        "max_iter":    800,       # 100 img ÷ batch16 × 50 epoch = ~312 iter → 800 yeterli
        "snapshot":    200,
        "val_iter":    0,         # küçük sette validation devre dışı
        "display":     100,
    },
    "2k": {
        "train_split": "train_2k",
        "max_iter":    12500,     # 2000 img ÷ batch16 × 100 epoch = 12500 iter
        "snapshot":    2500,
        "val_iter":    0,
        "display":     500,
    },
}

stage_cfg = STAGE_SETTINGS[args.stage]
snapshot_name = f"LSTR_CULANE_{args.stage}_{args.mode}"

cfg["system"]["snapshot_name"] = snapshot_name
cfg["system"]["data_dir"]      = "/home/alki/projects/"
cfg["system"]["train_split"]   = stage_cfg["train_split"]
cfg["system"]["max_iter"]      = stage_cfg["max_iter"]
cfg["system"]["snapshot"]      = stage_cfg["snapshot"]
cfg["system"]["val_iter"]      = stage_cfg["val_iter"]
cfg["system"]["display"]       = stage_cfg["display"]

system_configs.update_config(cfg["system"])

print(f"\n{'='*60}")
print(f"Progressive Training — Stage={args.stage}, Mode={args.mode}")
print(f"  split    : {stage_cfg['train_split']}")
print(f"  max_iter : {stage_cfg['max_iter']}")
print(f"  snapshot : {snapshot_name}")
print(f"{'='*60}\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
from db.datasets import datasets

db = datasets["CULANE"](cfg["db"], stage_cfg["train_split"])
print(f"Dataset yüklendi: {db.db_inds.size} eğitim örneği")

# Validation için val seti (küçük)
val_db = datasets["CULANE"](cfg["db"], "val")

# ── Model ─────────────────────────────────────────────────────────────────────
from nnet.py_factory import NetworkFactory

if args.mode == "mamba":
    # Mamba encoder'ı modele ekle
    import torch.nn as nn
    from mamba_ssm import Mamba as MambaSSM

    d_model    = system_configs.attn_dim   # 32
    d_state    = 16
    enc_layers = system_configs.enc_layers  # 2
    num_heads  = system_configs.num_heads   # 2

    class BidirMamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.nh = num_heads
            self.fwd = nn.ModuleList([
                MambaSSM(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                for _ in range(enc_layers)])
            self.bwd = nn.ModuleList([
                MambaSSM(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                for _ in range(enc_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(enc_layers)])

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

    nnet = NetworkFactory(flag=True)
    nnet.model.module.transformer.encoder = BidirMamba().cuda()

    # loss_curves=2.5 (spike fix)
    for k in list(nnet.loss.criterion.weight_dict.keys()):
        if 'loss_curves' in k:
            nnet.loss.criterion.weight_dict[k] = 2.5

    mamba_params = sum(p.numel() for p in nnet.model.module.transformer.encoder.parameters())
    print(f"Mamba encoder parametresi: {mamba_params:,}")

else:
    # Baseline — orijinal transformer, değişiklik yok
    nnet = NetworkFactory(flag=True)

total_params = sum(p.numel() for p in nnet.model.parameters())
print(f"Toplam model parametresi : {total_params:,}")

# ── Veri yükleme (train.py'dan adapt) ────────────────────────────────────────
from torch.multiprocessing import Process, Queue, Pool
from utils import stdout_to_tqdm
import models.py_utils.misc as utils
from tqdm import tqdm

def prefetch_data(db, q, sample_fn):
    ind = 0
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_fn(db, ind)
            q.put(data)
        except Exception:
            traceback.print_exc()

def pin_memory(data_q, pinned_q, sema):
    while True:
        data = data_q.get()
        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]
        pinned_q.put(data)
        if sema.acquire(blocking=False):
            return

sample_fn = importlib.import_module(f"sample.{db.data}").sample_data

training_queue       = Queue(system_configs.prefetch_size)
pinned_training_q    = queue.Queue(system_configs.prefetch_size)
training_sema        = threading.Semaphore()
training_sema.acquire()

task = Process(target=prefetch_data, args=(db, training_queue, sample_fn))
task.daemon = True
task.start()

pin_thread = threading.Thread(
    target=pin_memory, args=(training_queue, pinned_training_q, training_sema))
pin_thread.daemon = True
pin_thread.start()

# ── Eğitim döngüsü ────────────────────────────────────────────────────────────
if args.iter:
    lr = system_configs.learning_rate / (system_configs.decay_rate ** (args.iter // system_configs.stepsize))
    nnet.load_params(args.iter)
    nnet.set_lr(lr)
    print(f"Snapshot'tan devam: iter={args.iter}, lr={lr}")
else:
    nnet.set_lr(system_configs.learning_rate)

nnet.cuda()
nnet.train_mode()

metric_logger = utils.MetricLogger(delimiter="  ")
metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

start_iter   = args.iter
max_iter     = system_configs.max_iter
snapshot_dir = system_configs.snapshot_dir
display      = system_configs.display

print(f"\nEğitim başlıyor: iter {start_iter+1} → {max_iter}")
print(f"Snapshot her {system_configs.snapshot} iter'da kaydediliyor: {snapshot_dir}\n")

with stdout_to_tqdm() as save_stdout:
    for iteration in metric_logger.log_every(
        tqdm(range(start_iter + 1, max_iter + 1), file=save_stdout, ncols=67),
        print_freq=10
    ):
        training = pinned_training_q.get(block=True)
        save     = bool(display and iteration % display == 0)

        (set_loss, loss_dict) = nnet.train(iteration, save, "train", **training)
        (ld_red, ld_red_unscaled, ld_red_scaled, lv) = loss_dict
        metric_logger.update(loss=lv, **ld_red_scaled, **ld_red_unscaled)
        metric_logger.update(class_error=ld_red["class_error"])
        metric_logger.update(lr=system_configs.learning_rate)
        del set_loss

        if iteration % system_configs.snapshot == 0:
            nnet.save_params(iteration)
            print(f"\n[Snapshot] iter={iteration} kaydedildi → {snapshot_dir}")

print(f"\nEğitim tamamlandı: {max_iter} iter")
print(f"\nDeğerlendirme için:")
print(f"  python test.py {snapshot_name} --testiter {max_iter} --modality eval --split testing")
print(f"  cd lane_evaluation && bash run.sh")
