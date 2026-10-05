"""DINOv2-SALAD global descriptor (Izquierdo & Civera, CVPR 2024), inference only.

`backbone.py` and `aggregator.py` are taken from VGGT-Long's `LoopModels`, which adapted
them from https://github.com/serizba/salad; `dinov2/` is facebookresearch/dinov2, loaded
through torch.hub from this directory.
"""

import torch.nn as nn

from .aggregator import SALAD
from .backbone import DINOv2


class SaladModel(nn.Module):
    """Backbone + aggregator, with the state-dict layout of the released checkpoint."""

    def __init__(self):
        super().__init__()
        self.backbone = DINOv2('dinov2_vitb14', num_trainable_blocks=4, norm_layer=True,
                               return_token=True)
        self.aggregator = SALAD(num_channels=768, num_clusters=64, cluster_dim=128,
                                token_dim=256)

    def forward(self, x):
        return self.aggregator(self.backbone(x))
