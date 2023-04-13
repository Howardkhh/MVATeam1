from ensemble_boxes import *
import json
import numpy as np
import torch
from torch import nn
# need normalization for bbox coordination
# read in json file and turn it into bboxes lists, scores lists and label lists
# bbox_per_file = {1:[], 2:[], ... 9699:[]}
# bbox_lists = [bbox_per_file1, ...  ]

def normalization(bbox, width=3840, height=2160):
    # Size: 3840x2160 for pub_test dataset
    bbox = [bbox[0]/(width*1.000000000000), bbox[1]/height, bbox[2]/(width*1.000000000), bbox[3]/height]
    return bbox

def denorm(bbox, width=3840, height=2160):
    bbox = [bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height]
    return bbox

def xywh2xyxy(xywh):
    bbox = [xywh[0], xywh[1], (xywh[0] + xywh[2]), (xywh[1] + xywh[3])]
    return bbox

def xyxy2xywh(xyxy):
    bbox = [xyxy[0], xyxy[1], (xyxy[2] - xyxy[0]), (xyxy[3]-xyxy[1])]
    return bbox

def bbox_formatting(bboxes, scores , image_id, output):
    for i in range(0, len(bboxes)):
        cur_box = dict()
        cur_box['image_id'] = image_id
        cur_box['bbox'] = xyxy2xywh(denorm(bboxes[i]))
        cur_box['score'] = scores[i]
        cur_box['category_id'] = 0
        output.append(cur_box)
    return output

class wbf_torch(nn.Module):
    def __init__(self, boxes_list, scores_list, labels_list,
                 iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
        super().__init__()
        self.boxes_list = boxes_list
        self.scores_list = scores_list
        self.labels_list = labels_list
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.conf_type = conf_type
        self.allows_overflow = allows_overflow
        if conf_type not in ['avg']:
            print('Unknown conf_type: {}. Must be "avg"'.format(conf_type))
            exit()
    def prefilter_boxes(self, weights):
        re = []
        for t in range(len(self.boxes_list)):
            for j in range(len(self.boxes_list[t]))
                score = scores[t][j]
                if score < self.iou_thr:
                    continue
                label = int(self.labels_list[t][j])
                box_part = self.boxes_list[t][j]
                x1 = float(box_part[0])
                y1 = float(box_part[1])
                x2 = float(box_part[2])
                y2 = float(box_part[3])
                b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]
                re.append(b)
        current_boxes = torch.Tensor(re)
        current_boxes[current_boxes[:, 1].argsort()[::-1]]
        return current_boxes
    
    def find_matching_box_fast(self, boxes_list, new_box):
        """
            Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
            (~100x). This was previously the bottleneck since the function is called for every entry in the array.
        """
        def bb_iou_array(boxes, new_box):
            # bb interesection over union
            xA = torch.maximum(boxes[:, 0], new_box[0])
            yA = torch.maximum(boxes[:, 1], new_box[1])
            xB = torch.minimum(boxes[:, 2], new_box[2])
            yB = torch.minimum(boxes[:, 3], new_box[3])

            interArea = torch.maximum(xB - xA, 0) * torch.maximum(yB - yA, 0)

            # compute the area of both the prediction and ground-truth rectangles
            boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

            iou = interArea / (boxAArea + boxBArea - interArea)

            return iou

        if boxes_list.shape[0] == 0:
            return -1, match_iou

        # boxes = np.array(boxes_list)
        boxes = boxes_list

        ious = bb_iou_array(boxes[:, 4:], new_box[4:])

        ious[boxes[:, 0] != new_box[0]] = -1

        best_idx = torch.argmax(ious)
        best_iou = ious[best_idx]

        if best_iou <= match_iou:
            best_iou = match_iou
            best_idx = -1

        return best_idx, best_iou
    def forward(self, weights):
        if len(filtered_boxes) == 0:
            return torch.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
        weights = torch.Tensor(weights)

        filtered_boxed = self.prefilter_boxes(weights)

        boxes = filtered_boxes
        new_boxes = []
        weighted_boxes = torch.empty((0, 8))
        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = self.find_matching_box_fast(weighted_boxes, boxes[j])

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if not self.allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
            overall_boxes.append(weighted_boxes)
        overall_boxes = np.concatenate(overall_boxes, axis=0)
        overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
        boxes = overall_boxes[:, 4:]
        scores = overall_boxes[:, 1]
        labels = overall_boxes[:, 0]
        return boxes, scores, labels


def weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.0,
        conf_type='avg',
        allows_overflow=False
):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] *  clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


#4567+0.5 #4567+0.55 #3567+0.55 #cascade+0.6
#2457
def ensemble(config_file, output_file, weights=[2,4,5,6,8], iou_thr=0.5, skip_box_thr= 0.001): #0.01
    test_img_num = 9699
    output = [] # final result
    files = []
    json_file = []
    bbox_lists = []
    score_lists = []
    label_lists = []
    with open(config_file) as f:
        files = f.read().splitlines()
    f.close()
    # print(files)
    for cur_file in files:
        json_data = []
        print(cur_file)
        with open(cur_file) as k:
            json_data = json.load(k)
        json_file.append(json_data)

    for jf in json_file:
        bbox_per_file = {k: [] for k in np.arange(start=1,stop=test_img_num+1) }
        score_per_file = {k: [] for k in np.arange(start=1,stop=test_img_num+1) }
        label_per_file = {k: [] for k in np.arange(start=1,stop=test_img_num+1) }

        for data in jf:
            bbox_norm = normalization(xywh2xyxy(data['bbox']))
            bbox_per_file[int(data['image_id'])].append(bbox_norm)
            score_per_file[data['image_id']].append(data['score'])
            label_per_file[data['image_id']].append(data['category_id'])
            # print(bbox_per_file)
        bbox_lists.append(bbox_per_file)
        score_lists.append(score_per_file)
        label_lists.append(label_per_file)
    # ensemble
    # bbox_cmb = [bboxes_file1, bboxes_file2] for image_id = 1 to 9969+1
    for image_id in range(1, test_img_num+1):
        bbox_cmb, score_cmb, label_cmb = [], [], []
        for idx in range(len(files)):
            bbox_cmb.append(bbox_lists[idx][image_id])
            score_cmb.append(score_lists[idx][image_id])
            label_cmb.append(label_lists[idx][image_id])
        pred_bboxes_per_image, pred_score_per_image, pred_label_per_image = \
            weighted_boxes_fusion( bbox_cmb, score_cmb, label_cmb, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # print(pred_bboxes_per_image)
        print('Current Progress: {} / {}'.format(image_id, test_img_num), end='\r')
        output = bbox_formatting(pred_bboxes_per_image, pred_score_per_image, image_id, output)
    
    with open(output_file, "w") as f:
        json.dump(output, f)
        
ensemble('config_wbf.txt', 'results.json', weights=[2,4,5,6,8])