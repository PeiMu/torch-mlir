import torch
import numpy as np

def forward(proposals, gt, means=(0., 0., 0., 0.),
                stds=(1., 1., 1., 1.)):
    # assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)

    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    # means = deltas.new_tensor(means).unsqueeze(0)

    # stds = deltas.new_tensor(stds).unsqueeze(0)
    means = torch.Tensor([0., 0., 0., 0.]).unsqueeze(0)
    stds = torch.Tensor([1., 1., 1., 1.]).unsqueeze(0)

    deltas = deltas.sub_(means).div_(stds)

    return deltas


proposals = torch.randn(10, 4)
gt = torch.randn(10, 4)
forward(proposals, gt)