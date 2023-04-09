# credits to Bing
import argparse
from pycocotools.coco import COCO
from PIL import Image
import os
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--annotations', type=str, required=True, help='Path to annotations file')
args = parser.parse_args()

# Create cropped_bbox directory if it doesn't exist
if not os.path.exists('cropped_bbox'):
    os.makedirs('cropped_bbox')

# Initialize COCO API
coco = COCO(args.annotations)
img_file_path = Path(args.annotations).parent.parent

# Get image ids
img_ids = coco.getImgIds()

# Initialize global index
index = 0

# Loop through images and annotations
for img_id in img_ids:
    # Load image
    img = coco.loadImgs(img_id)[0]
    I = Image.open(Path(img_file_path, "images", img['file_name']).absolute())

    # Get annotation ids for image
    ann_ids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(ann_ids)

    # Crop and save bounding boxes
    for ann in anns:
        bbox = ann['bbox']
        cropped_img = I.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        cropped_img.save(os.path.join('cropped_bbox', str(index) + '.jpg'))
        index += 1