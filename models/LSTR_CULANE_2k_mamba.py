"""Thin redirect for 2k Mamba configuration."""

# NetworkFactory imports this module via snapshot_name.
# Keep this file as a stable alias to the actual Mamba implementation.
from models.LSTR_CULANE_MAMBA import model, loss  # noqa: F401
