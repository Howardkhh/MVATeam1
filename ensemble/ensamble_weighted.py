#!/usr/bin/env python
# coding: utf-8

# In[19]:


import json
import numpy as np


# In[29]:


def calculate_NMS(boxes, overlap_threshold=0.8):
    # input type of boxes - np array of [x_min, y_min, width, height, score] 
    if len(boxes) == 0:
        return []
    
    chosen_idx = []
    x_min = boxes[:, 0].astype(float)
    y_min = boxes[:, 1].astype(float)
    x_max = x_min + boxes[:, 2].astype(float)
    y_max = y_min + boxes[:, 3].astype(float)
    score = boxes[:, 4].astype(float)
    
    box_area = (x_max - x_min + 1) * (y_max - y_min + 1)
    indices = np.argsort(score)
    
    while(len(indices) > 0):
        last_idx = len(indices) - 1
        
        i = indices[last_idx]
        chosen_idx.append(i)
        suppress = [last_idx]
    
        for pos in range(0, last_idx):
            j = indices[pos]

            xx1 = max(x_min[i], x_min[j])
            yy1 = max(y_min[i], y_min[j])
            xx2 = min(x_max[i], x_max[j])
            yy2 = min(y_max[i], y_max[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            
            overlap = float(w * h) / box_area[j]
            
            if overlap > overlap_threshold:
                suppress.append(pos)

        indices = np.delete(indices, suppress)
        
    return boxes[chosen_idx]


# In[25]:


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
        
        for image in range(0, self.test_img_num):
            pred_bboxes_list = []
            
            for i in range(0, len(self.json_file)):
                pred_bboxes = []
                start = 0 if image == 0 else self.prefix_sum[i][image-1]
                for j in range(start, self.prefix_sum[i][image]):
                    cur_box = self.json_file[i][j]['bbox']
                    cur_box.append(self.json_file[i][j]['score']*self.weights[i])
                    pred_bboxes.append(np.array(cur_box))
                
                if len(pred_bboxes) == 0:
                    pred_bboxes = np.empty((0, 5), float)
                    pred_bboxes_list.append(pred_bboxes)
                else:
                    pred_bboxes_list.append(np.asarray(pred_bboxes))
            
            all_pred_bboxes = np.append(pred_bboxes_list[0], pred_bboxes_list[1], axis = 0)
            for i in range(2, len(self.json_file)):
                all_pred_bboxes = np.append(all_pred_bboxes, pred_bboxes_list[i], axis = 0)
            
            # print(len(all_pred_bboxes))
            
            NMS_boxes = calculate_NMS(all_pred_bboxes)
            
            # print(len(NMS_boxes))
            
            
            self.bbox_formatting(NMS_boxes, image+1)
            print('Current Progress: {} / {}'.format(image+1, self.test_img_num), end='\r')
        
        with open(self.output_file, "w") as f:
            json.dump(self.output, f)
            # print('Current Image {}'.format(image+1))
            # print('bbox_num: {}, {}'.format(len(pred_bboxes_list[0]), len(pred_bboxes_list[1])))
    
    
    def bbox_formatting(self, bboxes, image_id):
        for i in range(0, len(bboxes)):
            cur_box = dict();
            cur_box['image_id'] = image_id
            cur_box['bbox'] = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
            cur_box['score'] = bboxes[i][4]
            cur_box['category_id'] = 0
            self.output.append(cur_box)
    
    def print(self):
        for file in range(0, len(self.index_list)):
            for i in range(0, len(self.index_list[file])):
                print('Result {}, Image {}'.format(file, i), self.prefix_sum[file][i])


# In[26]:


je = json_ensembler('config_weighted.txt', 'results.json')


# In[27]:


je.ensamble()

