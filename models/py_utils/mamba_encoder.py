"""Mamba tabanlı bidirectional encoder + lane-aware scan order."""

import math
import os

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from config import system_configs


class BidirectionalMambaEncoder(nn.Module):
    """
    LSTR encoder yerine kullanılan bidirectional Mamba encoder.

    Ek olarak lane-aware scan order uygular:
    - Varsayılan: none (geriye dönük uyumluluk)
    - Config/env ile: col-bu veya col-td seçilebilir
    - Mamba işleminden sonra token sırası tekrar raster düzene geri çevrilir.

    Bu sayede downstream mask/decoder kontratı bozulmadan, encoder içinde
    lane yönüne hizalı sıra bilgisi kullanılmış olur.
    """

    def __init__(self):
        super().__init__()
        d_model = system_configs.attn_dim
        d_state = 16
        d_conv = 4
        expand = 2
        n_layers = system_configs.enc_layers

        self.num_heads = system_configs.num_heads
        cfg_scan_order = getattr(system_configs, "scan_order", "none")
        cfg_scan_hw = getattr(system_configs, "scan_hw", "")
        env_scan_order = os.environ.get("LSTR_MAMBA_SCAN_ORDER")
        env_scan_hw = os.environ.get("LSTR_MAMBA_SCAN_HW")

        self.scan_order = (env_scan_order if env_scan_order is not None else cfg_scan_order).lower()
        self.scan_hw_env = (env_scan_hw if env_scan_hw is not None else cfg_scan_hw)

        self.mamba_fwd_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.mamba_bwd_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        print(
            f"[MambaEncoder] scan_order={self.scan_order}, "
            f"scan_hw={'auto' if not self.scan_hw_env else self.scan_hw_env}"
        )

    def _resolve_hw(self, seq_len):
        # 1) Explicit override (recommended when trying new input sizes)
        if self.scan_hw_env:
            try:
                h_str, w_str = self.scan_hw_env.split(",")
                h, w = int(h_str), int(w_str)
                if h > 0 and w > 0 and h * w == seq_len:
                    return h, w
            except Exception:
                pass

        # 2) Common feature-map sizes in this project
        known = {
            260: (10, 26),
            240: (12, 20),
        }
        if seq_len in known:
            return known[seq_len]

        # 3) Fallback: pick factor pair closest to lane-like aspect ratio
        #    (wide feature map is typical for road images)
        target_ratio = 2.6
        best = None
        best_score = float("inf")
        upper = int(math.sqrt(seq_len))
        for h in range(1, upper + 1):
            if seq_len % h != 0:
                continue
            w = seq_len // h
            hh, ww = (h, w) if w >= h else (w, h)
            ratio = float(ww) / float(hh)
            score = abs(ratio - target_ratio)
            if score < best_score:
                best_score = score
                best = (hh, ww)
        return best

    def _build_permutation(self, seq_len, device):
        if self.scan_order in ("none", "raster"):
            return None, None

        hw = self._resolve_hw(seq_len)
        if hw is None:
            return None, None

        h, w = hw
        if h * w != seq_len:
            return None, None

        base = torch.arange(seq_len, device=device, dtype=torch.long).view(h, w)

        # Raster (top->bottom, left->right) -> Column-wise (left->right), bottom->top
        if self.scan_order == "col-bu":
            perm = base.flip(0).transpose(0, 1).contiguous().view(-1)
        elif self.scan_order == "col-td":
            perm = base.transpose(0, 1).contiguous().view(-1)
        else:
            # Unknown mode => safe fallback
            return None, None

        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(seq_len, device=device, dtype=torch.long)
        return perm, inv

    def forward(self, src, src_key_padding_mask=None, pos=None):
        del src_key_padding_mask  # kept for signature compatibility

        if pos is not None:
            src = src + pos

        # (HW, B, D) -> (B, HW, D)
        x = src.permute(1, 0, 2).contiguous()
        seq_len = x.shape[1]

        if not hasattr(self, "_scan_debug_printed"):
            h_w = self._resolve_hw(seq_len)
            print(f"[MambaEncoder][scan-debug] seq_len={seq_len}, resolved_hw={h_w}")
            self._scan_debug_printed = True

        perm, inv = self._build_permutation(seq_len=seq_len, device=x.device)
        if perm is not None:
            x = x.index_select(1, perm)

        for fwd, bwd, norm in zip(self.mamba_fwd_layers, self.mamba_bwd_layers, self.norms):
            x_fwd = fwd(x)
            x_bwd = torch.flip(bwd(torch.flip(x, dims=[1]).contiguous()), dims=[1]).contiguous()
            x = norm(x + x_fwd + x_bwd)

        # restore raster order so decoder/mask alignment remains unchanged
        if inv is not None:
            x = x.index_select(1, inv)

        out = x.permute(1, 0, 2).contiguous()

        bsz = out.shape[1]
        dummy_weights = torch.zeros(
            bsz * self.num_heads, seq_len, seq_len, device=out.device, dtype=out.dtype
        )
        return out, dummy_weights
