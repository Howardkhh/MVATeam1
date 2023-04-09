import os
import copy
import time
import argparse
import cv2
import numpy as np
import json

from pycocotools.coco import COCO

bbox_path = "../data/birds/"
img_file_path = os.listdir(bbox_path)

def paste(new_img, num):
    h, w = new_img.shape[:2]
    new_ann = []
    for _ in range(num):
        bbox = bbox_path + img_file_path[np.random.choice(len(img_file_path))]
        img = cv2.imread(bbox, cv2.IMREAD_UNCHANGED)
        obj_img = img[:, :, :3]
        orig_box_h, orig_box_w = obj_img.shape[0:2]
        
        if np.random.randint(2):
            box_w = int(np.random.choice(np.arange(400), p=num_w))
            box_h = max(6, int(orig_box_h*box_w/orig_box_w * np.random.uniform(0.8, 1.2)))

        else:
            box_h = int(np.random.choice(np.arange(400), p=num_h))
            box_w = max(6, int(orig_box_w*box_h/orig_box_h * np.random.uniform(0.8, 1.2)))

        obj_img = cv2.resize(obj_img, dsize=(box_w, box_h))

        x1, y1 = np.random.randint(w-box_w), np.random.randint(h-box_h)
        x2, y2 = x1 + box_w, y1 + box_h
        region_img = new_img[y1:y2, x1:x2]
        mask = obj_img != 0

        # fix contrast
        background_bright = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY).mean()
        contrast = -30
        brightness = 0
        if(background_bright <= 125): brightness = -background_bright-20
        obj_img = obj_img * (contrast/127 + 1) - contrast + brightness # 轉換公式
        obj_img = np.clip(obj_img, 0, 255)
        obj_img = np.uint8(obj_img)

        region_img[mask] = obj_img[mask]

        # applying the kernel to the input image
        size = 5
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        region_img = cv2.filter2D(region_img, -1, kernel_motion_blur)
        
        new_img[y1:y2, x1:x2] = region_img
        new_ann.append({"id": -1, 
                        "image_id": -1, 
                        "category_id": 0, 
                        "iscrowd": 0, 
                        "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]], 
                        "area": int(box_w * box_h), 
                        "bbox": [x1, y1, box_w, box_h]})
    return new_img, new_ann
        

coco_train = COCO("../data/mva2023_sod4bird_train/annotations/split_train_coco.json")
coco_val = COCO("../data/mva2023_sod4bird_train/annotations/split_val_coco.json")
coco_merged = COCO("../data/mva2023_sod4bird_train/annotations/merged_train.json")
img_dir = "../data/mva2023_sod4bird_train/images/"
dest_dir = "../data/mva2023_sod4bird_train_extended/"

if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)
if not os.path.isdir(dest_dir+"annotations"):
    os.mkdir(dest_dir+"annotations")
if not os.path.isdir(dest_dir+"images"):
    os.mkdir(dest_dir+"images")

num_objects = [0] * 1000
max_num = 0
total_objects = 0

imgIds = coco_merged.getImgIds()
imgIds_train = coco_train.getImgIds()
imgIds_val = coco_val.getImgIds()

merged_json = {"info": {"description": "MVA2023 Challenge Dataset", "year": 2023, "date_created": "2023/01/09"}, "licenses": ["Copyright (c) 2023 MVA organization and Toyota Technological Institute. https://docs.google.com/document/d/1zozNSsahDWNELrW3N6mzL33iOEBK_IdVFkJaKzS6ZoY/edit?usp=sharing"], "categories": [{"name": "bird", "supercategory": "bird", "id": 0}], "images": [], "annotations": []}
train_json = {"info": {"description": "MVA2023 Challenge Dataset", "year": 2023, "date_created": "2023/01/09"}, "licenses": ["Copyright (c) 2023 MVA organization and Toyota Technological Institute. https://docs.google.com/document/d/1zozNSsahDWNELrW3N6mzL33iOEBK_IdVFkJaKzS6ZoY/edit?usp=sharing"], "categories": [{"name": "bird", "supercategory": "bird", "id": 0}], "images": [], "annotations": []}
val_json = {"info": {"description": "MVA2023 Challenge Dataset", "year": 2023, "date_created": "2023/01/09"}, "licenses": ["Copyright (c) 2023 MVA organization and Toyota Technological Institute. https://docs.google.com/document/d/1zozNSsahDWNELrW3N6mzL33iOEBK_IdVFkJaKzS6ZoY/edit?usp=sharing"], "categories": [{"name": "bird", "supercategory": "bird", "id": 0}], "images": [], "annotations": []}

for id in imgIds:
    ann = coco_train.getAnnIds(id)
    num_objects[len(ann)] += 1
    max_num = max(max_num, len(ann))
    total_objects += len(ann)

num_objects = np.array(num_objects[:max_num+10]) + 1
print(num_objects)

num_w = [0] * 400
num_h = [0] * 400

annIds = coco_merged.getAnnIds()
for id in annIds:
    ann = coco_merged.loadAnns(id)[0]
    w = ann["bbox"][2]
    h = ann["bbox"][3]
    num_w[w] += 1
    num_h[h] += 1

print(num_w)
print(num_h)

num_w = np.array(num_w) / np.sum(num_w)
num_h = np.array(num_h) / np.sum(num_h)

cur_imgId = 1
cur_annId = 0

for count, id in enumerate(imgIds):
    print(count, end='\r')
    img_description = coco_merged.loadImgs(id)[0]
    img_annIds = coco_merged.getAnnIds(id)
    img_anns = coco_merged.loadAnns(img_annIds)
    img = cv2.imread(img_dir + img_description["file_name"], cv2.IMREAD_UNCHANGED)

    cv2.imwrite(dest_dir+f"images/{cur_imgId:06d}.jpg", img)

    merged_json["images"].append({"id": cur_imgId, "width": 3840, "height": 2160, "file_name": f"{cur_imgId:06d}.jpg", "date_captured": ""})
    if id in imgIds_train:
        train_json["images"].append({"id": cur_imgId, "width": 3840, "height": 2160, "file_name": f"{cur_imgId:06d}.jpg", "date_captured": ""})
    else:
        val_json["images"].append({"id": cur_imgId, "width": 3840, "height": 2160, "file_name": f"{cur_imgId:06d}.jpg", "date_captured": ""})
    for ann in img_anns.copy():
        ann["id"] = cur_annId
        ann["image_id"] = cur_imgId
        cur_annId += 1
        merged_json["annotations"].append(ann.copy())
        if id in imgIds_train:
            train_json["annotations"].append(ann.copy())
        else:
            val_json["annotations"].append(ann.copy())
    cur_imgId += 1

    for i in range(9):
        target_num = np.random.choice(np.arange(len(img_annIds)+1, len(img_annIds)+10), p=num_objects[len(img_annIds)+1:len(img_annIds)+10] / np.sum(num_objects[len(img_annIds)+1:len(img_annIds)+10]))
        new_img, new_ann = paste(img.copy(), target_num-len(img_annIds))
        cv2.imwrite(dest_dir+f"images/{cur_imgId:06d}.jpg", new_img)
        
        merged_json["images"].append({"id": cur_imgId, "width": 3840, "height": 2160, "file_name": f"{cur_imgId:06d}.jpg", "date_captured": ""})
        if id in imgIds_train:
            train_json["images"].append({"id": cur_imgId, "width": 3840, "height": 2160, "file_name": f"{cur_imgId:06d}.jpg", "date_captured": ""})
        else:
            val_json["images"].append({"id": cur_imgId, "width": 3840, "height": 2160, "file_name": f"{cur_imgId:06d}.jpg", "date_captured": ""})
        
        for ann in img_anns.copy() + new_ann:
            ann["id"] = cur_annId
            ann["image_id"] = cur_imgId
            cur_annId += 1
            merged_json["annotations"].append(ann.copy())
            if id in imgIds_train:
                train_json["annotations"].append(ann.copy())
            else:
                val_json["annotations"].append(ann.copy())
        cur_imgId += 1
    

merged_file = open(dest_dir + "annotations/merged_train.json", "w")
train_file = open(dest_dir + "annotations/split_train_coco.json", "w")
val_file = open(dest_dir + "annotations/split_val_coco.json", "w")
json.dump(merged_json, merged_file, indent=4)
json.dump(train_json, train_file, indent=4)
json.dump(val_json, val_file, indent=4)
merged_file.close()
train_file.close()
val_file.close()