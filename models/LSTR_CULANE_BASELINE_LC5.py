"""
LSTR_CULANE_BASELINE_LC5 — Full CULane (500k iter) Baseline Transformer
========================================================================

LSTR_CULANE ile birebir aynı mimari ve loss weight.
Fark: data_dir düzeltildi (/home/alki/projects/)

Karşılaştırma amacı:
  - LSTR_CULANE_BASELINE_LC5  → Transformer + loss_curves=5.0 (default)
  - LSTR_CULANE_MAMBA_LC5     → Mamba       + loss_curves=5.0 (default)

Kullanım:
  python train.py LSTR_CULANE_BASELINE_LC5 --threads 16
  python test.py LSTR_CULANE_BASELINE_LC5 --testiter 500000 --modality eval --split testing
"""
from models.LSTR_CULANE import model, loss  # noqa: F401
# loss_curves override YOK — AELoss default değeri kullanılır (5.0)
