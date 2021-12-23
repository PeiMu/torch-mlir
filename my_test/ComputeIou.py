import torch
import numpy as np

def forward(bbox1, bbox2):
    rec1 = bbox1[:4]
    rec2 = bbox2[:4]
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1)
    S_rec2 = (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1)

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = torch.maximum(rec1[1], rec2[1])
    right_line = torch.minimum(rec1[3], rec2[3])
    top_line = torch.maximum(rec1[0], rec2[0])
    bottom_line = torch.minimum(rec1[2], rec2[2])
    # j = (left_line >= right_line).item()
    # judge if there is an intersect

    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line + 1) * (
            bottom_line - top_line + 1)
        return intersect / (sum_area - intersect)


bbox1 = torch.rand(4)
bbox2 = torch.rand(4)
forward(bbox1, bbox2)
