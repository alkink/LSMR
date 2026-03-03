"""
2k Mamba + VSSD-style non-causal encoder deneyi.

Taban model:
  [models.LSTR_CULANE_MAMBA](models/LSTR_CULANE_MAMBA.py:1)

Bu dosya sadece encoder'ı non-causal Mamba2 bloğu ile değiştirir.
Decoder/head/loss hattı korunur.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import system_configs
from models.LSTR_CULANE_MAMBA import model as _BaseMambaModel
from models.LSTR_CULANE_MAMBA import loss  # noqa: F401

try:
    from mamba_ssm import Mamba2  # type: ignore
except Exception:
    Mamba2 = None


_HEADDIM_MISMATCH_WARNED = set()


def _ensure_stride8_channel_last(x_bsc: torch.Tensor) -> torch.Tensor:
    if (x_bsc.stride(0) % 8 == 0) and (x_bsc.stride(2) % 8 == 0):
        return x_bsc

    bsz, seq, ch = x_bsc.shape
    padded = torch.empty((bsz, seq, ch * 8), device=x_bsc.device, dtype=x_bsc.dtype)
    padded[..., ::8] = x_bsc
    return padded[..., ::8]


def _mamba2_stride_safe_for_cuda(layer: nn.Module) -> bool:
    """
    Mamba2 non-mem-eff yolunda çağrılan causal_conv1d kernel için
    in_proj çıkış kanal sayısının 8'e bölünebilir olmasını zorunlu tutar.
    """
    try:
        in_proj_out = int(layer.in_proj.weight.shape[0])  # type: ignore[attr-defined]
    except Exception:
        return True
    return (in_proj_out % 8) == 0


def _build_mamba2(d_model: int, d_state: int, d_conv: int, headdim: int) -> Tuple[nn.Module, Dict[str, int]]:
    if Mamba2 is None:
        raise ImportError("Mamba2 import edilemedi. `mamba-ssm` sürümünü güncelleyin.")

    def _with_mem_eff_priority(base_kw: Dict[str, int]) -> list[dict]:
        return [
            {**base_kw, "use_mem_eff_path": False, "sequence_parallel": False},
            {**base_kw, "sequence_parallel": False},
            {**base_kw, "use_mem_eff_path": False},
            dict(base_kw),
            {**base_kw, "use_mem_eff_path": True},
        ]

    hd_candidates = [int(headdim), 32, 16, 8, 4, 2, 1]
    unique_hd = []
    for h in hd_candidates:
        if h > 0 and h not in unique_hd:
            unique_hd.append(h)

    candidate_kwargs = []
    for h in unique_hd:
        if d_model % h != 0:
            continue
        for base_kw in [
            {"d_model": d_model, "d_state": d_state, "d_conv": d_conv, "headdim": h},
            {"d_model": d_model, "d_state": d_state, "headdim": h},
            {"d_model": d_model, "d_state": d_state, "d_conv": d_conv},
            {"d_model": d_model, "d_state": d_state},
        ]:
            candidate_kwargs.extend(_with_mem_eff_priority(base_kw))

    for base_kw in [
        {"d_ssm": d_model, "d_state": d_state, "headdim": int(headdim)},
        {"d_ssm": d_model, "d_state": d_state},
    ]:
        candidate_kwargs.extend(_with_mem_eff_priority(base_kw))

    last_err: Optional[Exception] = None
    for kw in candidate_kwargs:
        try:
            layer = Mamba2(**kw)
            if torch.cuda.is_available() and not _mamba2_stride_safe_for_cuda(layer):
                last_err = RuntimeError(
                    "Mamba2 candidate CUDA stride koşulunu sağlamıyor "
                    f"(in_proj_out={int(layer.in_proj.weight.shape[0])}, kw={kw})."  # type: ignore[attr-defined]
                )
                continue

            selected_headdim = kw.get("headdim", None)
            if selected_headdim is not None and int(selected_headdim) != int(headdim):
                warn_key = (int(d_model), int(d_state), int(d_conv), int(headdim), int(selected_headdim))
                if warn_key not in _HEADDIM_MISMATCH_WARNED:
                    warnings.warn(
                        "VSSD encoder Mamba2 headdim isteği runtime uyumu için değiştirildi "
                        f"(requested={int(headdim)} -> selected={int(selected_headdim)}).",
                        RuntimeWarning,
                    )
                    _HEADDIM_MISMATCH_WARNED.add(warn_key)
            return layer, kw
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Mamba2 oluşturulamadı (d_model={d_model}, d_state={d_state}, d_conv={d_conv}, headdim={headdim})"
    ) from last_err


class NonCausalMambaEncoder(nn.Module):
    """
    LSTR encoder imzasına uyumlu non-causal Mamba2 encoder.

    Girdi/çıktı imzası:
      [models.py_utils.transformer.TransformerEncoder.forward()](models/py_utils/transformer.py:78)
    """

    def __init__(
        self,
        d_model: int = 32,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
    ):
        super().__init__()
        self.num_heads = int(num_heads)

        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        self.merge = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(int(num_layers))])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(int(num_layers))])

        chosen_kwargs: Optional[Dict[str, int]] = None
        for _ in range(int(num_layers)):
            layer_fwd, kw = _build_mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, headdim=headdim)
            layer_bwd, _ = _build_mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, headdim=headdim)
            self.fwd.append(layer_fwd)
            self.bwd.append(layer_bwd)
            if chosen_kwargs is None:
                chosen_kwargs = {k: int(v) for k, v in kw.items() if isinstance(v, int)}

        self.mamba2_kwargs = chosen_kwargs or {}
        self.fallback_fwd = nn.ModuleList()
        self.fallback_bwd = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.fallback_fwd.append(nn.Identity())
            self.fallback_bwd.append(nn.Identity())
        self.runtime_fallback_used = False
        self._warned_runtime_fallback = False

    def forward(self, src, src_key_padding_mask=None, pos=None):
        del src_key_padding_mask
        if pos is not None:
            src = src + pos

        # (S, B, C) -> (B, S, C)
        x = src.permute(1, 0, 2).contiguous()
        x = _ensure_stride8_channel_last(x)

        for i, (f, b, m, n) in enumerate(zip(self.fwd, self.bwd, self.merge, self.norm)):
            try:
                xf = f(x)
                xb = torch.flip(b(torch.flip(x, dims=[1]).contiguous()), dims=[1])
            except RuntimeError as e:
                if "causal_conv1d with channel last layout requires strides" not in str(e):
                    raise
                if isinstance(self.fallback_fwd[i], nn.Identity):
                    self.fallback_fwd[i] = __import__("mamba_ssm").Mamba(  # type: ignore[attr-defined]
                        d_model=x.shape[-1], d_state=16, d_conv=4
                    ).to(x.device)
                    self.fallback_bwd[i] = __import__("mamba_ssm").Mamba(  # type: ignore[attr-defined]
                        d_model=x.shape[-1], d_state=16, d_conv=4
                    ).to(x.device)
                if not self._warned_runtime_fallback:
                    warnings.warn(
                        "VSSD encoder Mamba2 runtime stride kısıtına takıldı; Mamba(v1) fallback ile devam ediyor.",
                        RuntimeWarning,
                    )
                    self._warned_runtime_fallback = True
                self.runtime_fallback_used = True
                xf = self.fallback_fwd[i](x)
                xb = torch.flip(self.fallback_bwd[i](torch.flip(x, dims=[1]).contiguous()), dims=[1])
            x = n(x + m(torch.cat([xf, xb], dim=-1)))

        # (B, S, C) -> (S, B, C)
        out = x.permute(1, 0, 2)

        # test/culane debug hook uyumu için attention-like placeholder
        bsz = out.shape[1]
        seq = out.shape[0]
        weights = torch.zeros(bsz * self.num_heads, seq, seq, device=out.device, dtype=out.dtype)
        return out, weights


class model(_BaseMambaModel):
    def __init__(self, flag=False):
        super().__init__(flag=flag)

        d_model = int(getattr(self.transformer, "d_model", system_configs.attn_dim))
        enc_layers = int(system_configs.enc_layers)
        num_heads = int(system_configs.num_heads)

        self.transformer.encoder = NonCausalMambaEncoder(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            headdim=32,
            num_layers=enc_layers,
            num_heads=num_heads,
        )

        print(
            "[LSTR_CULANE_2k_mamba_vssd] encoder -> NonCausalMambaEncoder "
            f"(layers={enc_layers}, d_model={d_model}, kwargs={self.transformer.encoder.mamba2_kwargs})"
        )

