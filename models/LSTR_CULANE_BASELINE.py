"""
LSTR_CULANE_BASELINE — Full CULane (500k iter) Baseline Transformer
=====================================================================

LSTR_CULANE_MAMBA ile birebir aynı şartlar:
  - Aynı config: LSTR_CULANE_BASELINE.json (500k iter, stepsize=450k, batch=16)
  - Aynı data_dir: /home/alki/projects/
  - Aynı loss_curves=2.5 (Mamba ile fair comparison)
  - Mimari: Orijinal Transformer encoder (Mamba yok)

Kullanım:
  python train.py LSTR_CULANE_BASELINE --threads 16
  python test.py LSTR_CULANE_BASELINE --testiter 500000 --modality eval --split testing
"""
from models.LSTR_CULANE_BASELINE_LC25 import model, loss  # noqa: F401
