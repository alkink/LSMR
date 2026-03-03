"""Mamba decoder varyant katmanları ve stack wrapper'ları."""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn


_HEADDIM_MISMATCH_WARNED = set()

try:
    from mamba_ssm import Mamba
except Exception as e:  # pragma: no cover
    raise ImportError(
        "mamba_ssm import edilemedi. Önce `pip install mamba-ssm causal-conv1d` çalıştırın."
    ) from e

try:
    from mamba_ssm import Mamba2  # type: ignore
except Exception:
    Mamba2 = None


def _prep_with_pos(
    memory: torch.Tensor,
    query: torch.Tensor,
    pos: Optional[torch.Tensor] = None,
    query_pos: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mem = memory if pos is None else memory + pos
    qry = query if query_pos is None else query + query_pos
    return mem, qry


def _ensure_stride8_channel_last(x_bsc: torch.Tensor) -> torch.Tensor:
    """
    causal_conv1d CUDA kernel için gerekli stride hizalamasını sağlar.
    Bazı ortamlarda stride(2) % 8 != 0 olduğunda Mamba2 runtime hatası verir.
    """
    if (x_bsc.stride(0) % 8 == 0) and (x_bsc.stride(2) % 8 == 0):
        return x_bsc

    bsz, seq, ch = x_bsc.shape
    padded = torch.empty(
        (bsz, seq, ch * 8),
        device=x_bsc.device,
        dtype=x_bsc.dtype,
    )
    padded[..., ::8] = x_bsc
    return padded[..., ::8]


def _is_attn_like_layer(layer: nn.Module) -> bool:
    return hasattr(layer, "multihead_attn") and hasattr(layer, "self_attn")


def _mamba2_stride_safe_for_cuda(layer: nn.Module) -> bool:
    """
    Mamba2'nin non-mem-eff yolunda çağırdığı causal_conv1d CUDA kernel'i için
    in_proj çıkış kanal sayısının 8'e bölünebilir olmasını zorunlu kılıyoruz.

    Neden: ileri geçişte split/transposed ara tensörde stride(2) pratikte
    `in_proj_out_dim` değerine eşitleniyor; bu değer 8'in katı değilse
    runtime'da stride hatası alınabiliyor.
    """
    try:
        in_proj_out = int(layer.in_proj.weight.shape[0])  # type: ignore[attr-defined]
    except Exception:
        # Güvenli tarafta kal: katman yapısı beklenenden farklıysa filtreleme yapma.
        return True
    return (in_proj_out % 8) == 0


def _build_mamba2(d_model: int, d_state: int, d_conv: int, headdim: int):
    if Mamba2 is None:
        raise ImportError("Mamba2 import edilemedi. mamba-ssm sürümünü güncelleyin.")

    def _with_mem_eff_priority(base_kw: dict[str, int]) -> list[dict[str, Any]]:
        # Öncelik: Triton stride kısıtına takılmamak için önce memory-efficient path kapalı dene.
        # Bazı mamba-ssm sürümlerinde bu argüman olmayabilir; candidate loop zaten güvenli şekilde fallback yapar.
        return [
            {**base_kw, "use_mem_eff_path": False, "sequence_parallel": False},
            {**base_kw, "sequence_parallel": False},
            {**base_kw, "use_mem_eff_path": False},
            dict(base_kw),
            {**base_kw, "use_mem_eff_path": True},
        ]

    hd_candidates = [headdim, 32, 16, 8, 4, 2, 1]
    hd_unique = []
    for h in hd_candidates:
        h = int(h)
        if h <= 0:
            continue
        if h not in hd_unique:
            hd_unique.append(h)

    candidate_kwargs = []
    for h in hd_unique:
        if d_model % h != 0:
            continue
        base_candidates = [
            {"d_model": d_model, "d_state": d_state, "d_conv": d_conv, "headdim": h},
            {"d_model": d_model, "d_state": d_state, "headdim": h},
            {"d_model": d_model, "d_state": d_state, "d_conv": d_conv},
            {"d_model": d_model, "d_state": d_state},
        ]
        for base_kw in base_candidates:
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
                        "Mamba2 headdim isteği çalışma zamanı uyumu için değiştirildi "
                        f"(requested={int(headdim)} -> selected={int(selected_headdim)}).",
                        RuntimeWarning,
                    )
                    _HEADDIM_MISMATCH_WARNED.add(warn_key)
            return layer, kw
        except Exception as e:  # pragma: no cover
            last_err = e
            continue

    raise RuntimeError(f"Mamba2 oluşturulamadı (d_model={d_model}, d_state={d_state}, headdim={headdim}).") from last_err


class MambaDecoderLayerA(nn.Module):
    """Varyant A — Prefix concat."""

    def __init__(self, d_model: int = 32, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # memory: (S,B,C), query: (L,B,C)
        lq = query.shape[0]
        combined = torch.cat([memory, query], dim=0)     # (S+L,B,C)
        out_bsc = self.mamba(combined.permute(1, 0, 2))  # (B,S+L,C)
        out = out_bsc.permute(1, 0, 2)                   # (S+L,B,C)
        new_query = out[-lq:]                            # (L,B,C)
        return self.norm(new_query + query)


class MambaDecoderLayerB_Naive(nn.Module):
    """Varyant B — Naive loop referansı."""

    def __init__(self, d_model: int = 32, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # memory: (S,B,C), query: (L,B,C)
        lq = query.shape[0]
        new_qs = []
        for i in range(lq):
            qi = query[i : i + 1]                           # (1,B,C)
            combined = torch.cat([memory, qi], dim=0)       # (S+1,B,C)
            out_bsc = self.mamba(combined.permute(1, 0, 2)) # (B,S+1,C)
            out = out_bsc.permute(1, 0, 2)                  # (S+1,B,C)
            new_qs.append(out[-1:] + qi)
        new_query = torch.cat(new_qs, dim=0)
        return self.norm(new_query)


class MambaDecoderLayerB_Batch(nn.Module):
    """Varyant B — Batch optimize (eğitimde kullanılacak)."""

    def __init__(self, d_model: int = 32, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # memory: (S,B,C), query: (L,B,C)
        s, bsz, ch = memory.shape
        lq = query.shape[0]

        mem_exp = memory.unsqueeze(2).expand(s, bsz, lq, ch).reshape(s, bsz * lq, ch)  # (S,B*L,C)
        q_flat = query.permute(1, 0, 2).reshape(bsz * lq, ch).unsqueeze(0)              # (1,B*L,C)
        combined = torch.cat([mem_exp, q_flat], dim=0)                                  # (S+1,B*L,C)

        out_bsc = self.mamba(combined.permute(1, 0, 2))                                 # (B*L,S+1,C)
        new_q_flat = out_bsc[:, -1, :]                                                  # (B*L,C)
        new_query = new_q_flat.view(bsz, lq, ch).permute(1, 0, 2).contiguous()         # (L,B,C)
        return self.norm(new_query + query)


class MambaDecoderLayerC(nn.Module):
    """Varyant C — Bidirectional query-aware."""

    def __init__(self, d_model: int = 32, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.linear_merge = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward_branches(self, memory: torch.Tensor, query: torch.Tensor):
        # memory: (S,B,C), query: (L,B,C)
        s, bsz, ch = memory.shape
        lq = query.shape[0]

        mem_exp = memory.unsqueeze(2).expand(s, bsz, lq, ch).reshape(s, bsz * lq, ch)  # (S,B*L,C)
        q_flat = query.permute(1, 0, 2).reshape(bsz * lq, ch).unsqueeze(0)              # (1,B*L,C)
        combined = torch.cat([mem_exp, q_flat], dim=0)                                  # (S+1,B*L,C)
        combined_bsc = combined.permute(1, 0, 2).contiguous()                           # (B*L,S+1,C)

        fwd_all = self.mamba_fwd(combined_bsc)
        fwd_last = fwd_all[:, -1, :]                                                     # (B*L,C)

        bwd_in = torch.flip(combined_bsc, dims=[1])
        bwd_all = self.mamba_bwd(bwd_in)
        bwd_last = bwd_all[:, 0, :]                                                      # (B*L,C)

        merged = self.linear_merge(torch.cat([fwd_last, bwd_last], dim=-1))             # (B*L,C)
        new_q = merged.view(bsz, lq, ch).permute(1, 0, 2).contiguous()                  # (L,B,C)
        return self.norm(new_q + query), fwd_last, bwd_last

    def forward(self, memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        out, _fwd, _bwd = self.forward_branches(memory, query)
        return out


class MambaDecoderLayerD_Mamba2(nn.Module):
    """Varyant D — Query-aware Mamba2 (B varyantının Mamba2 karşılığı)."""

    def __init__(self, d_model: int = 32, d_state: int = 64, d_conv: int = 4, headdim: int = 32):
        super().__init__()
        self.mamba, self.mamba2_kwargs = _build_mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=headdim,
        )
        self.mamba_fallback = Mamba(d_model=d_model, d_state=max(16, int(d_state // 4)), d_conv=d_conv)
        self.runtime_fallback_used = False
        self._warned_runtime_fallback = False
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        s, bsz, ch = memory.shape
        lq = query.shape[0]

        mem_exp = memory.unsqueeze(2).expand(s, bsz, lq, ch).reshape(s, bsz * lq, ch)  # (S,B*L,C)
        q_flat = query.permute(1, 0, 2).reshape(bsz * lq, ch).unsqueeze(0)              # (1,B*L,C)
        combined = torch.cat([mem_exp, q_flat], dim=0)                                  # (S+1,B*L,C)

        combined_bsc = combined.permute(1, 0, 2).contiguous()                            # (B*L,S+1,C)
        combined_bsc = _ensure_stride8_channel_last(combined_bsc)
        try:
            out_bsc = self.mamba(combined_bsc)                                            # (B*L,S+1,C)
        except RuntimeError as e:
            if "causal_conv1d with channel last layout requires strides" not in str(e):
                raise
            if not self._warned_runtime_fallback:
                warnings.warn(
                    "Mamba2 runtime stride kısıtına takıldı; Varyant D bu oturumda Mamba(v1) fallback ile devam ediyor.",
                    RuntimeWarning,
                )
                self._warned_runtime_fallback = True
            self.runtime_fallback_used = True
            out_bsc = self.mamba_fallback(combined_bsc.contiguous())
        new_q_flat = out_bsc[:, -1, :]                                                  # (B*L,C)
        new_query = new_q_flat.view(bsz, lq, ch).permute(1, 0, 2).contiguous()         # (L,B,C)
        return self.norm(new_query + query)


class _BaseMambaDecoder(nn.Module):
    def __init__(
        self,
        layer_cls,
        num_layers: int = 2,
        d_model: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        return_intermediate: bool = True,
        layer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        kwargs = {} if layer_kwargs is None else dict(layer_kwargs)
        self.layers = nn.ModuleList(
            [layer_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, **kwargs) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    def _forward_layer(
        self,
        layer: nn.Module,
        output: torch.Tensor,
        memory: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if _is_attn_like_layer(layer):
            return layer(output, memory, pos=pos, query_pos=query_pos)
        mem, qry = _prep_with_pos(memory, output, pos=pos, query_pos=query_pos)
        return layer(mem, qry)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        del tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = self._forward_layer(layer, output, memory, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        output = self.norm(output)
        if self.return_intermediate:
            intermediate[-1] = output
            return torch.stack(intermediate)
        return output


class MambaDecoderA(_BaseMambaDecoder):
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        return_intermediate: bool = True,
    ):
        super().__init__(
            layer_cls=MambaDecoderLayerA,
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
        )


class MambaDecoderB(_BaseMambaDecoder):
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        return_intermediate: bool = True,
    ):
        super().__init__(
            layer_cls=MambaDecoderLayerB_Batch,
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
        )


class MambaDecoderC(_BaseMambaDecoder):
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        return_intermediate: bool = True,
    ):
        super().__init__(
            layer_cls=MambaDecoderLayerC,
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
        )


class MambaDecoderD(_BaseMambaDecoder):
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 32,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 32,
        return_intermediate: bool = True,
    ):
        super().__init__(
            layer_cls=MambaDecoderLayerD_Mamba2,
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
            layer_kwargs={"headdim": int(headdim)},
        )


class MambaDecoderE_Hybrid(_BaseMambaDecoder):
    """Varyant E — katman sırası olarak Hybrid (MambaB -> CrossAttn)."""

    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 2,
        dim_feedforward: int = 128,
        return_intermediate: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(
            layer_cls=MambaDecoderLayerB_Batch,
            num_layers=1,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
        )

        from models.py_utils.transformer import TransformerDecoderLayer

        layers: list[nn.Module] = [
            MambaDecoderLayerB_Batch(d_model=d_model, d_state=d_state, d_conv=d_conv)
        ]
        for _ in range(max(int(num_layers) - 1, 0)):
            layers.append(
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=int(num_heads),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    activation="relu",
                    normalize_before=False,
                )
            )
        self.layers = nn.ModuleList(layers)


def build_variant_decoder(
    variant: str,
    num_layers: int,
    d_model: int,
    d_state: int,
    d_conv: int,
    return_intermediate: bool,
    headdim: int = 32,
    num_heads: int = 2,
    dim_feedforward: int = 128,
) -> nn.Module:
    v = variant.upper()
    if v == "A":
        return MambaDecoderA(
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
        )
    if v == "B":
        return MambaDecoderB(
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
        )
    if v == "C":
        return MambaDecoderC(
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            return_intermediate=return_intermediate,
        )
    if v == "D":
        return MambaDecoderD(
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=headdim,
            return_intermediate=return_intermediate,
        )
    if v == "E":
        return MambaDecoderE_Hybrid(
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            return_intermediate=return_intermediate,
            dropout=0.1,
        )
    raise ValueError(f"Unknown variant: {variant}")

