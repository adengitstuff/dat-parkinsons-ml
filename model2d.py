"""
2D backbone for striatal slab images.

Deliberately boring: an ImageNet-pretrained ResNet-18 with a linear head.
The whole hypothesis being tested is "pretraining + preserved spatial
resolution", so the architecture should add nothing else that could confound it.

Resolution budget, for contrast with the 3D net:
    224x224 input, ResNet stride 32 -> 7x7 feature map over the striatum.
    The 3D net reached 2x2x2 over an entire head.
"""

import torch.nn as nn
from torchvision.models import resnet18, resnet34


def build_model(arch="resnet18", pretrained=True, dropout=0.3, in_ch=3):
    fn = {"resnet18": resnet18, "resnet34": resnet34}[arch]
    m = fn(weights="IMAGENET1K_V1" if pretrained else None)
    if in_ch != 3:
        old = m.conv1
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        with_ = old.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
        m.conv1.weight.data = with_
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
    return m


class Wrapped(nn.Module):
    """Squeezes the logit so BCEWithLogitsLoss gets a 1-D tensor."""

    def __init__(self, **kw):
        super().__init__()
        self.net = build_model(**kw)

    def forward(self, x):
        return self.net(x).squeeze(1)