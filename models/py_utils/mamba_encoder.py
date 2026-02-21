"""
LSTR + Gerçek mamba-ssm Bidirectional Encoder Modülü

Kullanım:
    from models.py_utils.mamba_encoder import BidirectionalMambaEncoder
    model.transformer.encoder = BidirectionalMambaEncoder().cuda()

Gereksinim: pip install mamba-ssm causal-conv1d
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from config import system_configs


class BidirectionalMambaEncoder(nn.Module):
    """
    LSTR transformer.encoder'ı yerine geçen gerçek Mamba tabanlı encoder.

    Causality Çözümü:
        Mamba doğası gereği causal (soldan-sağa). Feature map'te sağdaki
        lane token'ları soldakileri göremez. Çözüm: iki yönlü Mamba bloğu.
        - Forward  pass: x olduğu gibi geçer
        - Backward pass: x flip edilir, sonuç geri flip edilir
        - İkisi toplanır → Bidirectional

    Sequence boyutu notu:
        240 token (12×20), standart attention'a karşı Mamba'nın belirgin
        hız avantajının ortaya çıktığı eşiğin (>512 token) altında.
        Avantaj burada FPS değil, O(N) bellek kompleksitesinde.

    LSTR forward signature uyumu:
        Mevcut encoder: (src, src_key_padding_mask=None, pos=None) → (out, weights)
        src boyutu : (HW=240, B=1, D=32)  ← PyTorch seq-first
        Mamba girdisi: (B=1, HW=240, D=32) ← batch-first → permute gerekli

    enc_attn_weights shape:
        test/culane.py:169 → enc_attn_weights[0].reshape(shape + shape)
        shape = conv_features.shape[-2:] = (12, 20)
        reshape(shape + shape) = reshape(12, 20, 12, 20) ← (HW, HW) olarak görülüyor
        Aslında hook output[1] alıyor → self_attn weight shape: (B*num_heads, HW, HW)
        Bizim wrapper'da: (num_heads, HW, HW) döndürülüyor.
        enc_attn_weights[0] → (HW, HW) = (240, 240), reshape(12,20,12,20) → OK
    """

    def __init__(self):
        super().__init__()
        d_model        = system_configs.attn_dim        # 32 (CULane config)
        d_state        = 16                              # SSM state boyutu
        d_conv         = 4                               # Causal conv kernel
        expand         = 2                               # inner = d_model * expand = 64
        n_layers       = system_configs.enc_layers       # 2 (CULane config)
        self.n_layers  = n_layers
        self.num_heads = system_configs.num_heads        # 2 (CULane config)

        self.mamba_fwd_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.mamba_bwd_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, src, src_key_padding_mask=None, pos=None):
        """
        Args:
            src: (HW, B, D) — LSTR seq-first formatı
            src_key_padding_mask: (B, HW) — padding mask (opsiyonel)
            pos: (HW, B, D) — positional encoding (opsiyonel)
        Returns:
            out: (HW, B, D)
            dummy_weights: (num_heads, HW, HW) — test/culane.py:169 uyumlu
        """
        if pos is not None:
            src = src + pos

        # (HW, B, D) → (B, HW, D)  Mamba batch-first bekliyor
        x = src.permute(1, 0, 2)

        for fwd, bwd, norm in zip(self.mamba_fwd_layers, self.mamba_bwd_layers, self.norms):
            x_fwd = fwd(x)
            x_bwd = torch.flip(bwd(torch.flip(x, dims=[1])), dims=[1])
            x = norm(x + x_fwd + x_bwd)   # residual + add bidirectional

        # (B, HW, D) → (HW, B, D)  LSTR seq-first'e geri
        out = x.permute(1, 0, 2)

        # enc_attn_weights: test/culane.py:169'da
        #   enc_attn_weights[0].reshape(shape + shape)
        #   shape = (12, 20) → reshape(12, 20, 12, 20)
        # self_attn output[1] → (B*num_heads, HW, HW) 
        B = out.shape[1]
        src_hw = out.shape[0]  # 240
        dummy_weights = torch.zeros(
            B * self.num_heads, src_hw, src_hw, device=out.device
        )
        return out, dummy_weights
