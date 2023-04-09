from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt

dest_dir = "../data/mva2023_sod4bird_train_extended/"
coco = COCO(dest_dir + "annotations/merged_train.json")

imgIds = coco.getImgIds()
for id in imgIds:
    img_description = coco.loadImgs(id)[0]
    img = cv2.imread(dest_dir + "images/" + img_description["file_name"], cv2.IMREAD_UNCHANGED)
    plt.imshow(img)
    annIds = coco.getAnnIds(id)
    print(len(annIds))
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()