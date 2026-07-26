"""
The reasoning is: 'try to get real numbers fast', so I'll try, instead of a MedicalNet or any fancy
multi-region thinking, a simple 3D CNN for now. This is to pass smoke tests, get a general inference time, get real numbers and 
the real 'baseline' of the problem!!


Simple 3D CNN baseline - from scratch, no pretrained weights.
GroupNorm throughout (not BatchNorm) - safe for small batch sizes,
and since there's no pretrained checkpoint here, there's nothing lost by it.
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, groups=8):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
        nn.GroupNorm(groups, out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleCNN3D(nn.Module):
    """
    Input:  (B, 1, 96, 96, 96)
    Output: (B,)  - one raw logit per volume (bcewithlogits forf the sigmoid!)
    """
    def __init__(self, base_channels=16):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            conv_block(1, c),         # 96 -> 48
            conv_block(c, c * 2),     # 48 -> 24
            conv_block(c * 2, c * 4),  # 24 -> 12
            conv_block(c * 4, c * 8),  # 12 -> 6
            conv_block(c * 8, c * 8),  # 6 -> 3
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(c * 8, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)              # (B, C, 1, 1, 1)
        x = x.flatten(1)              # (B, C)
        x = self.fc(x)                # (B, 1)
        return x.squeeze(1)           # (B,)  - one logit per sample


if __name__ == "__main__":
    model = SimpleCNN3D()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {n_params:,}")

    dummy = torch.randn(4, 1, 96, 96, 96)
    out = model(dummy)
    print(f"input shape:  {tuple(dummy.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    assert out.shape == (4,), f"expected (4,), got {out.shape}"
    print("model self-test passed")