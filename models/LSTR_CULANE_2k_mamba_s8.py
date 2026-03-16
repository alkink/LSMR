"""2k Mamba encoder ablation using stride-8 backbone features directly."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LSTR_CULANE_MAMBA import model as _BaseMambaModel
from models.LSTR_CULANE_MAMBA import loss  # noqa: F401


class model(_BaseMambaModel):
    def __init__(self, flag=False):
        super().__init__(flag=flag)

        in_ch = int(self.layer2[-1].bn2.num_features)
        d_model = int(self.transformer.d_model)
        self.s8_proj = nn.Identity() if in_ch == d_model else nn.Conv2d(in_ch, d_model, kernel_size=1)

        print(
            f"[LSTR_CULANE_2k_mamba_s8] stride-8 ablation active: "
            f"layer2_channels={in_ch}, d_model={d_model}"
        )

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks = xs[1]

        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)

        transformer_input = self.s8_proj(p)
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=transformer_input.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(transformer_input, pmasks)

        hs, _, weights = self.transformer(transformer_input, pmasks, self.query_embed.weight, pos)
        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared = self.shared_embed(hs)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat(
            [output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]], dim=-1
        )

        out = {"pred_logits": output_class[-1], "pred_curves": output_specific[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(output_class, output_specific)
        return out, weights

