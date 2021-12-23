# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torchvision.models as models
import numpy as np
from typing import Tuple

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export


# ==============================================================================

class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.resnet = models.resnet18()
        self.train(False)

    @export
    @annotate_args([
        None,
        ([-1, 3, -1, -1], torch.float32, True),
    ])
    def forward(self, img):
        return self.resnet.forward(img)


@register_test_case(module_factory=lambda: ResNet18Module())
def ResNet18Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 224, 224))


class IouOfModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, bbox1, bbox2):
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
        lt = torch.maximum(bbox1[:, :2], bbox2[:, :2])
        rb = torch.minimum(bbox1[:, 2:], bbox2[:, 2:])

        overlap_coord = (rb - lt).clip(0)
        overlap = overlap_coord[:, 0] * overlap_coord[:, 1]
        union = area1 + area2 - overlap

        return overlap / union


@register_test_case(module_factory=lambda: IouOfModule())
def IouOfModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 4), tu.rand(1024, 4))


class Bbox2DeltaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, proposals, gt, means,
                stds):
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
        means = means.unsqueeze(0)
        stds = stds.unsqueeze(0)

        deltas = deltas.sub_(means).div_(stds)

        return deltas


@register_test_case(module_factory=lambda: Bbox2DeltaModule())
def Bbox2DeltaModule_basic(module, tu: TestUtils):
    means = torch.Tensor([0., 0., 0., 0.])
    stds = torch.Tensor([1., 1., 1., 1.])
    module.forward(tu.rand(10, 4), tu.rand(10, 4), means, stds)


class Delta2BboxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True)
    ])
    def forward(self, rois,
                deltas,
                means,
                stds):
        max_shape=(32, 32, 3)
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


@register_test_case(module_factory=lambda: Delta2BboxModule())
def Delta2BboxModule_basic(module, tu: TestUtils):
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
    module.forward(rois, deltas, means, stds)


class GetMatchingScoresModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x_a, y_a, x_b, y_b, offset_a, offset_b, one):
        # x_a, y_a = point_a
        # x_b, y_b = point_b

        # x_coarse_b = x_a - offset[:, 0:1]
        # y_coarse_b = y_a - offset[:, 1:2]
        x_coarse_b = x_a - offset_a
        y_coarse_b = y_a - offset_b
        dis_x = abs(x_coarse_b - x_b)
        dis_y = abs(y_coarse_b - y_b)

        # new fashion
        # return torch.exp((dis_x * dis_x + dis_y * dis_y) / (-2.0 * 5))
        # old fashion
        return (one / (dis_x + one)) * (one / (dis_y + one))


@register_test_case(module_factory=lambda: GetMatchingScoresModule())
def GetMatchingScoresModule_basic(module, tu: TestUtils):
    M = 1
    x_a = torch.randn(M, 1)
    y_a = torch.randn(M, 1)
    x_b = torch.randn(M, 1)
    y_b = torch.randn(M, 1)
    offset = torch.randn(M, 2)
    one = torch.Tensor([[1.]])
    module.forward(x_a, y_a, x_b, y_b, offset[:, 0:1], offset[:, 1:2], one)
    # module.forward((tu.rand(1, 1), tu.rand(1, 1),),
    #                (tu.rand(1, 1), tu.rand(1, 1),), tu.rand(1, 2))


class GnerateHoiPredictionsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    # def intersect1d(tensor1, tensor2):
    #     aux = torch.cat((tensor1, tensor2),dim=0)
    # aux = aux.sort()[0]
    # return aux[:-1][(aux[1:] == aux[:-1]).data]
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, max_rel, interacted_sub_ids,
                interacted_obj_ids, clses_rel, hoi_score,
                detections_sub, detections_obj,
                hoi_threshold):
        hoi_triplet = torch.cat((interacted_sub_ids.view(max_rel, 1).float(),
                                 interacted_obj_ids.view(max_rel, 1).float(),
                                 clses_rel.view(max_rel, 1),
                                 hoi_score.view(max_rel, 1)), -1)

        hoi_predictions = torch.cat(
            (detections_sub[interacted_sub_ids.view(max_rel).long(), :],
             detections_obj[interacted_obj_ids.view(max_rel).long(), :],
             hoi_triplet[:, -2:]), dim=1)

        # sort by hoi_score
        sort_ind = hoi_triplet[:, -1].sort(descending=True)[1]
        hoi_triplet = hoi_triplet[sort_ind].cpu().numpy()
        hoi_predictions = hoi_predictions[sort_ind].cpu().numpy()

        # filtering out same hoi triplets and a certain threshold
        # hoi_triplet[:, [0, 1, 2]] means [subject id, object id, relation]
        uniq_id = torch.unique(torch.FloatTensor([hoi_triplet[:, [0, 1, 2]]]),
                               dim=0, sorted=True)[1]
        # uniq_id = np.unique(
        #     hoi_triplet[:, [0, 1, 2]].astype(np.int),
        #     axis=0,
        #     return_index=True)[1]
        thresh_id = np.where(hoi_triplet[:, -1] > hoi_threshold)[0]
        selected_hoi_id = np.intersect1d(uniq_id, thresh_id)
        hoi_predictions = hoi_predictions[selected_hoi_id]

        if len(hoi_predictions) == 0:
            subject_predictions = np.array([]).reshape(-1, 6)
            object_predictions = np.array([]).reshape(-1, 6)
            rel_predictions = np.array([0, 0.0]).reshape(-1, 2)

        else:
            subject_predictions = hoi_predictions[..., :6]
            object_predictions = hoi_predictions[..., 6:12]
            rel_predictions = hoi_predictions[..., 12:14]

            # map objects classes from [0, 1] to [1, 2]
            object_predictions[..., -1] += 1

        return subject_predictions, object_predictions, rel_predictions


@register_test_case(module_factory=lambda: GnerateHoiPredictionsModule())
def GnerateHoiPredictionsModule_basic(module, tu: TestUtils):
    module.forward(1, torch.tensor([0]),
                   torch.tensor([2]), tu.rand(1, 1),
                   torch.tensor([0.2179]),
                   tu.rand(1024, 6), tu.rand(2048, 6), 0.0)


class ComputeIouModule(torch.nn.Module):
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, bbox1, bbox2):
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

        if left_line <= right_line and top_line <= bottom_line:
            intersect = (right_line - left_line + 1) * (
                bottom_line - top_line + 1)
            return intersect / (sum_area - intersect)


@register_test_case(module_factory=lambda: ComputeIouModule())
def ComputeIouModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4))


class ComputeIouMatModule(torch.nn.Module):
    def compute_IOU(self, bbox1, bbox2):
        if bbox1[1, -1] == bbox2[1, -1]:
            rec1 = bbox1[1, :4]
            rec2 = bbox2[1, :4]
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1)
            S_rec2 = (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line + 1) * (
                        bottom_line - top_line + 1)
                return intersect / (sum_area - intersect)
        else:
            return 0
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, bbox_list1, bbox_list2):
        iou_mat = torch.zeros((len(bbox_list1), len(bbox_list2)))
        # if len(bbox_list1) == 0 or len(bbox_list2) == 0:
        #     return {}, {}
        # for i, bbox1 in enumerate(bbox_list1):
        #     for j, bbox2 in enumerate(bbox_list2):
        #         # iou_i =
        #         iou_mat[i, j] = self.compute_IOU(bbox1, bbox2)
        #         # print(iou_i)

        iou_mat_ov = iou_mat.clone()
        iou_mat[iou_mat >= 0.5] = 1
        iou_mat[iou_mat < 0.5] = 0

        match_pairs = torch.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pairs_ov = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pairs_ov[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pairs_ov[pred_id].append(iou_mat_ov[match_pairs[0][i],
                                                          pred_id])
        return match_pairs_dict, match_pairs_ov

@register_test_case(module_factory=lambda: ComputeIouMatModule())
def ComputeIouMatModule_basic(module, tu: TestUtils):
    module.forward([tu.rand(1024, 4)], [tu.rand(1024, 4)])


class CombinaClsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.mp = torch.nn.MaxPool2d(kernel_size=(1,2), stride=1)
    @export
    @annotate_args([
        None,
        ([4, 5, 6, 7], torch.float32, True),
    ])
    def forward(self, hm_rel):
        hm_rel = hm_rel.permute(1,0,2,3)
        tmp_ca = hm_rel[1:2]
        tmp_fa = hm_rel[3:]
        tmp_No = hm_rel[0:1]
        tmp_dr = hm_rel[2:3]
        tmp_res = torch.cat([tmp_ca, tmp_fa], dim=0).permute(2,3,1,0)
        if tmp_res.shape[2]!=1 or tmp_res.shape[3]!=1:
            tmp_res = self.mp(tmp_res)
        tmp_res = tmp_res.permute(3,2,0,1)
        # hm_rel = torch.cat([tmp_No, tmp_res, tmp_dr], dim=0)
        hm_rel = hm_rel.permute(1,0,2,3)
        return hm_rel 

@register_test_case(module_factory=lambda: CombinaClsModule())
def CombinaClsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7))


class PermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4, 2], torch.float32, True)
    ])
    def forward(self, x):
        return x.permute(0, 2, 1)

@register_test_case(module_factory=lambda: PermuteModule())
def PermuteModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))