"""Thin alias for 2k Mamba + decoder variant A experiment."""

from models.LSTR_CULANE_MAMBA import loss  # noqa: F401
from models.LSTR_CULANE_2k_mamba_dec_common import BaseMambaDecoderVariantModel


class model(BaseMambaDecoderVariantModel):
    def __init__(self, flag=False):
        super().__init__(flag=flag, decoder_variant="A", d_state=16, d_conv=4)

