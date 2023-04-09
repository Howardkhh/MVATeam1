from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random
from PIL import Image
import pathlib
import matplotlib.pyplot as plt


# coco = COCO("data/mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json")
coco = COCO("data/mva2023_sod4bird_train/annotations/split_val_coco.json")

imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds)
resultsCOCO = coco.loadRes("work_dirs/eval_results.json")
N = len(imgs)
test_idx = random.sample(imgIds, 10)


for count, imgId in enumerate(test_idx):
    img = coco.loadImgs(imgId)[0]
    # print(resultsCOCO.loadImgs(imgId)[0])
    # print(str(pathlib.Path("data/mva2023_sod4bird_train/images", img['file_name'])))
    # print(coco.loadImgs(imgId)[0]["file_name"], resultsCOCO.loadImgs(imgId)[0]["file_name"])
    # img = Image.open(str(pathlib.Path("data/mva2023_sod4bird_pub_test/images", img['file_name'])))
    img = Image.open(str(pathlib.Path("data/mva2023_sod4bird_train/images", img['file_name'])))
    plt.figure(dpi=800)
    plt.imshow(img)
    anns = resultsCOCO.getAnnIds(imgId)
    anns = resultsCOCO.loadAnns(anns)
    # gt = coco.getAnnIds(imgId)
    # gt = coco.loadAnns(gt)
    print(anns)
    resultsCOCO.showAnns(anns, draw_bbox=True)
    # coco.showAnns(gt, draw_bbox=True)
    plt.savefig("vis_gt/000.jpg")
    break