import torch
from lpips import LPIPS


class OneChannelLPIPS(LPIPS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(**kwargs)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        # Convert to 3 channels
        in0 = in0.view(-1, 1, in0.shape[2], in0.shape[3])
        in0 = in0.repeat(1, 3, 1, 1)
        in1 = in1.view(-1, 1, in1.shape[2], in1.shape[3])
        in1 = in1.repeat(1, 3, 1, 1)
        return super().forward(in0, in1, retPerLayer=retPerLayer, normalize=normalize)
