import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class NWDAssigner(BaseAssigner):
    """
    Assign a corresponding gt bbox or background to each bbox.
    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Only remove the "ignore_iof_thr" from the arguments, cause there is no need for iof.

    """

    def __init__(self,
                 pos_nwd_thr,
                 neg_nwd_thr,
                 min_pos_nwd=.0,
                 gt_max_assign_all=True,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 ):
        self.pos_iou_thr = pos_nwd_thr
        self.neg_iou_thr = neg_nwd_thr
        self.min_pos_iou = min_pos_nwd
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality 

    def nwd_calcualter(gt_bboxes, bboxes):
        """
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
        """
        batch_shape = gt_bboxes.shape[:-2]
        rows = gt_bboxes.size(-2)
        cols = bboxes.size(-2)

        if rows*cols == 0:
            return gt_bboxes.new(batch_shape + (rows, cols))

        # don't mind about whether its neg/pos, NWD is calculating the distance.
        width_1, width_2 = gt_bboxes[..., 2] - gt_bboxes[..., 0], bboxes[..., 2] - bboxes[..., 0]
        height_1, height_2 = gt_bboxes[..., 3] - gt_bboxes[..., 1], bboxes[..., 3] - bboxes[..., 1]

        center_x1, center_y1 =  (gt_bboxes[..., 2] + gt_bboxes[..., 0])/ 2.0, (bboxes[..., 2] + bboxes[..., 0]) / 2.0
        center_x2, center_y2 = (gt_bboxes[..., 3] + gt_bboxes[..., 1]) /2.0, (bboxes[..., 3] + bboxes[..., 1]) / 2.0

        
        WD_square = torch.pow(center_x1 - center_x2, 2) + torch.pow(center_y1-center_y2, 2) + torch.pow((width_1 - width_2)/2.0, 2) + torch.pow((height_1 - height_2)/2.0, 2)
        
        #TODO: write NWD
        NWDs = WD_square 

        return NWDs


    #TODO: replace iou_calculater with nwd_calculater
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass
