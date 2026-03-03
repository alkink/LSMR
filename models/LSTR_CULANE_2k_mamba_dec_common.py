"""
Common model wrapper for 2k Mamba + decoder-variant experiments (A/B/C).

Backbone + encoder + loss are inherited from
[models.LSTR_CULANE_MAMBA](models/LSTR_CULANE_MAMBA.py:1).
Only decoder is replaced with the selected Mamba decoder variant.
"""

from __future__ import annotations

from config import system_configs
from models.LSTR_CULANE_MAMBA import model as _MambaEncoderBaseModel
from mamba_decoder_analysis.decoder_variants import build_variant_decoder


class BaseMambaDecoderVariantModel(_MambaEncoderBaseModel):
    def __init__(
        self,
        flag: bool = False,
        decoder_variant: str = "A",
        d_state: int = 16,
        d_conv: int = 4,
        headdim: int = 32,
    ):
        variant = decoder_variant.upper()
        state = int(d_state)
        conv = int(d_conv)
        head_dim = int(headdim)

        super().__init__(flag=flag)

        self.decoder_variant = variant
        self.decoder_d_state = state
        self.decoder_d_conv = conv
        self.decoder_headdim = head_dim

        old_decoder = self.transformer.decoder
        num_layers = len(old_decoder.layers)
        d_model = int(getattr(self.transformer, "d_model", system_configs.attn_dim))
        return_intermediate = bool(getattr(old_decoder, "return_intermediate", True))

        new_decoder = build_variant_decoder(
            variant=self.decoder_variant,
            num_layers=num_layers,
            d_model=d_model,
            d_state=self.decoder_d_state,
            d_conv=self.decoder_d_conv,
            return_intermediate=return_intermediate,
            headdim=self.decoder_headdim,
            num_heads=int(system_configs.num_heads),
            dim_feedforward=int(system_configs.dim_feedforward),
        )

        # Eski decoder'ın cihazını koru; model daha sonra .cuda()/.to(...) alırsa birlikte taşınır.
        try:
            old_device = next(old_decoder.parameters()).device
            new_decoder = new_decoder.to(old_device)
        except StopIteration:
            pass

        self.transformer.decoder = new_decoder

        print(
            "[LSTR_CULANE_2k_mamba_dec_common] "
            f"decoder -> variant={self.decoder_variant}, "
            f"layers={num_layers}, d_model={d_model}, "
            f"d_state={self.decoder_d_state}, d_conv={self.decoder_d_conv}, "
            f"headdim={self.decoder_headdim}"
        )

