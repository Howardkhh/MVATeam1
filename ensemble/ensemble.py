from ensemble_boxes import weighted_boxes_fusion
import json
import numpy as np
import argparse
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



#4567+0.5 #4567+0.55 #3567+0.55 #cascade+0.6
#2457
def ensemble(config_file, output_file, method, weights=[2,4,5,6,8], iou_thr=0.5, skip_box_thr= 0.001, sigma = 0.1): #0.01
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
        
        tot_box = 0    
        for i in range(0, len(bbox_cmb)):
            tot_box += len(bbox_cmb[i])

        if tot_box > 0:
            if method == 'wbf':
                pred_bboxes_per_image, pred_score_per_image, pred_label_per_image = \
                    weighted_boxes_fusion( bbox_cmb, score_cmb, label_cmb, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            elif method == 'snms':
                pred_bboxes_per_image, pred_score_per_image, pred_label_per_image = \
                    soft_nms( bbox_cmb, score_cmb, label_cmb, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        # print(pred_bboxes_per_image)
        print('Current Progress: {} / {}'.format(image_id, test_img_num), end='\r')
        output = bbox_formatting(pred_bboxes_per_image, pred_score_per_image, image_id, output)
    
    with open(output_file, "w") as f:
        json.dump(output, f)
    
    return output

parser = argparse.ArgumentParser(description='Ensemble Choices')
parser.add_argument("--method", help="Please select wbf or snms", choices=['wbf', 'snms'], default='wbf')
args = parser.parse_args()

ensemble('config.txt', 'results.json', weights=[2,4,5,6,8], method=args.method)