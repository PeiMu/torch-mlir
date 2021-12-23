import torch
import numpy as np

def forward(rois,
                deltas,
                means,
                stds,
                max_shape=None):
	wh_ratio_clip=16 / 1000,
	clip_border=True,
	add_ctr_clamp=False,
	ctr_clamp=32
	# means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(-1) // 4)
	# stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 4)
	denorm_deltas = deltas * stds + means
	dx = denorm_deltas[:, 0::4]
	dy = denorm_deltas[:, 1::4]
	dw = denorm_deltas[:, 2::4]
	dh = denorm_deltas[:, 3::4]
	max_ratio = torch.abs(torch.log(torch.tensor(wh_ratio_clip)))
	dw = dw.clamp(min=-max_ratio, max=max_ratio)
	dh = dh.clamp(min=-max_ratio, max=max_ratio)
	print("dw: \n", dw)
	# Compute center of each roi
	px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
	py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
	# Compute width/height of each roi
	pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
	ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
	# Use exp(network energy) to enlarge/shrink each roi
	gw = pw * dw.exp()
	gh = ph * dh.exp()
	# Use network energy to shift the center of each roi
	gx = px + pw * dx
	gy = py + ph * dy
	# Convert center-xy/width/height to top-left, bottom-right
	x1 = gx - gw * 0.5
	y1 = gy - gh * 0.5
	x2 = gx + gw * 0.5
	y2 = gy + gh * 0.5
	if max_shape is not None:
		x1 = x1.clamp(min=0, max=max_shape[1])
		y1 = y1.clamp(min=0, max=max_shape[0])
		x2 = x2.clamp(min=0, max=max_shape[1])
		y2 = y2.clamp(min=0, max=max_shape[0])
	bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
	return bboxes


rois = torch.Tensor([[0., 0., 1., 1.],
                         [0., 0., 1., 1.],
                         [0., 0., 1., 1.],
                         [5., 5., 5., 5.]])
deltas = torch.Tensor([[0., 0., 0., 0.],
                           [1., 1., 1., 1.],
                           [0., 0., 2., -1.],
                           [0.7, -1.9, -0.5, 0.3]])
means = torch.Tensor([[0., 0., 0., 0.]])
stds = torch.Tensor([[1., 1., 1., 1.]])
forward(rois, deltas, means, stds, max_shape=(32, 32, 3))
