from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime
# check: settubg dpi, score_thr as you wish.
def draw_box(img, imgId, anns_list, folder_name, model_name, score_thr = 0.3, dpi = 400):
    plt.figure(dpi=dpi)
    plt.imshow(img)
    
    for anns in anns_list:
        score = anns['score']
        if score >= score_thr:
            bbox = anns['bbox']
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

            text = f'{score:.2f}'
            x, y = bbox[0] + bbox[2], bbox[1] + bbox[3]
            plt.text(x, y, text, color='r', fontsize=5, ha='left', va='top')
    # layer: if you want all images to be put in same foler, change the tail
    # tail = '_' + model_name + '.png'
    tail = '.png'

    plt.axis('off')
    plt.savefig(folder_name+str(imgId)+tail, bbox_inches='tight', pad_inches=0)

def draw_model(current_time, model_name):
     # layer: if you want all images to be put in same foler, change the folder name
     # folder_name = current_time+"/"
    folder_name = current_time+"/"+model_name+"/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print("Result folder name is repeated.")
        return

    # check: list_to_draw and the model_name.json must match. I.e. having same png list.
    # check: set the correct path for list_to_draw and each model's json.
    list_to_draw = COCO("data/mva2023_sod4bird_pub_test/annotations/private_test_coco_empty_ann.json")
    imgIds = list_to_draw.getImgIds()
    result = list_to_draw.loadRes("final_submit_0776/private_test1/"+model_name+".json")

    for imgId in tqdm(imgIds):
        img = list_to_draw.loadImgs(imgId)[0]
        img = Image.open(str(pathlib.Path("data/mva2023_sod4bird_pub_test/images", img['file_name'])))

        anns_list = result.getAnnIds(imgId)
        anns_list = result.loadAnns(anns_list)
        draw_box(img, imgId, anns_list, folder_name, model_name)
        break # test_env: just for checking path and compile error

if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%m%d_%H_%M")
    model_name_list =["cascade_mask_internimage_h_fpn_40e_nwd_finetune","intern_h_public_nosahi",
                        "cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train","intern_h_public_nosahi_randflip",
                        "cascade_nwd_paste_howard_0604","intern_xl_public_nosahi_randflip","cascade_original","results_interImage",
                        "cascade_rcnn_sticker_61_2","results","centernet_slicing_01"]
    for model_name in model_name_list:
        draw_model(current_time, model_name)

# check
''' 
1. copy private_test_coco_empty_ann.json into data/annotations/
2. Choose a json file that exist in both private_test1
3. cd MVATeam1
4. python3 final_submit_0776/cmp_result.py
'''

