#!/usr/bin/env python
# coding: utf-8

# In[14]:


from ensemble_boxes import *
import json


# In[21]:


class json_ensembler:
    def __init__(self, prediction_txt, output_file):
        self.file_names = []
        self.weights = []
        self.json_file = []
        self.test_img_num = 9699
        self.output = []
        self.output_file = output_file
        # self.bboxesList = [] # shape of (json_cnt, test_img_cnt, bboxes_num)
        
        lines = []
        
        with open(prediction_txt) as f:
            lines = f.read().splitlines()

        for i in range(0, len(lines)):
            self.file_names.append(lines[i].split()[0])
            self.weights.append(float(lines[i].split()[1]))
            
        # print('Files: ', len(self.file_names))
        
        self.index_list = []
        self.prefix_sum = []
        
        for i in range(0, len(self.file_names)):
            self.index_list.append([0]*self.test_img_num)
            self.prefix_sum.append([0]*self.test_img_num)
        
        for cur_file in self.file_names:
            json_data = []
            with open(cur_file) as f:
                json_data = json.load(f)
            self.json_file.append(json_data)
        
        for i in range(0, len(self.file_names)):
            for index in range(0, len(self.json_file[i])):
                self.index_list[i][self.json_file[i][index]['image_id']-1] += 1
        
        for i in range(0, len(self.file_names)):
            self.prefix_sum[i][0] = self.index_list[i][0]
            for j in range(1, len(self.index_list[i])):
                self.prefix_sum[i][j] = self.prefix_sum[i][j-1] + self.index_list[i][j]
    
    def ensamble(self):
        
        for image in range(0, self.test_img_num ):#
            pred_bboxes_list = []
            pred_scores_list = []
            pred_label_list = []
            for i in range(0, len(self.json_file)):
                pred_bboxes = []
                pred_scores = []
                pred_labels = []
                start = 0 if image == 0 else self.prefix_sum[i][image-1]
                for j in range(start, self.prefix_sum[i][image]):
                    cur_box = self.json_file[i][j]['bbox']
                    cur_box[2] += cur_box[0]
                    cur_box[3] += cur_box[1]
                    cur_box[0] = cur_box[0] / 3840
                    cur_box[1] = cur_box[1] / 2160
                    cur_box[2] = cur_box[2] / 3840
                    cur_box[3] = cur_box[3] / 2160
                    
                    pred_scores.append(self.json_file[i][j]['score'])
                    pred_labels.append(0)
                    pred_bboxes.append(cur_box)
                    
                
                pred_bboxes_list.append(pred_bboxes)
                pred_scores_list.append(pred_scores)
                pred_label_list.append(pred_labels)
            
            iou_thr = 0.5
            skip_box_thr = 0.0001
            sigma = 0.1
            
            tot_box = 0
            
            for i in range(0, len(pred_bboxes_list)):
                # print(pred_bboxes_list[i])
                tot_box += len(pred_bboxes_list[i])
            
            if tot_box > 0:
                boxes, scores, labels = soft_nms(pred_bboxes_list, pred_scores_list,
                        pred_label_list, weights=self.weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
                # boxes, scores, labels = non_maximum_weighted(pred_bboxes_list, pred_scores_list,
                #         pred_label_list, weights=self.weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
                # boxes, scores, labels = weighted_boxes_fusion(pred_bboxes_list, pred_scores_list, pred_label_list,
                #                weights=self.weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            
                self.bbox_formatting(boxes, scores, image+1)
            print('Current Progress: {} / {}'.format(image+1, self.test_img_num), end='\r')
        
        with open(self.output_file, "w") as f:
            json.dump(self.output, f)
    
    def bbox_formatting(self, bboxes, scores, image_id):
        for i in range(0, len(bboxes)):
            cur_box = dict();
            cur_box['image_id'] = image_id
            cur_box['bbox'] = [bboxes[i][0]*3840, bboxes[i][1]*2160, 
                               (bboxes[i][2]-bboxes[i][0])*3840, (bboxes[i][3]-bboxes[i][1])*2160]
            cur_box['score'] = scores[i]
            cur_box['category_id'] = 0
            self.output.append(cur_box)
    
    def print(self):
        for file in range(0, len(self.index_list)):
            for i in range(0, len(self.index_list[file])):
                print('Result {}, Image {}'.format(file, i), self.prefix_sum[file][i])


# In[22]:


je = json_ensembler('config_weighted.txt', 'results.json')


# In[23]:


je.ensamble()

