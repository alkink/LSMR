"""
LSTR_CULANE_BASELINE_LC25 — Orijinal transformer, loss_curves=2.5

Ablation amacı: Mamba ile aynı loss weight kullanarak
"F1 farkı mimari mi yoksa loss weight mi?" sorusunu yanıtlamak.

Tek fark: loss_curves ve loss_curves_0 = 2.5 (baseline'da 5.0)
Mimari: LSTR_CULANE.py ile birebir aynı
"""
from models.LSTR_CULANE import model  # noqa: F401 — mimari tamamen aynı
from models.py_utils import AELoss
from config import system_configs


class loss(AELoss):
    def __init__(self):
        super(loss, self).__init__(
            debug_path=system_configs.result_dir,
            aux_loss=system_configs.aux_loss,
            num_classes=system_configs.lane_categories,
            dec_layers=system_configs.dec_layers
        )
        # Mamba ile aynı loss weight — fairness ablation
        for k in list(self.criterion.weight_dict.keys()):
            if 'loss_curves' in k:
                self.criterion.weight_dict[k] = 2.5
        print(f"[BASELINE_LC25] weight_dict: {self.criterion.weight_dict}")
