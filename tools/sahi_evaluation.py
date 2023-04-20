from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import argparse
import pathlib
import json
from PIL import Image
import matplotlib.pyplot as plt
import random
import time
import os
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403

def parse_args():
    parser = argparse.ArgumentParser(
        description='test (and eval) a model using sahi')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('datadir')
    parser.add_argument('annotation')
    parser.add_argument('--score-threshold', default=0.3, type=float)
    parser.add_argument('--num-test', default=-1, type=int)
    parser.add_argument('--crop-size', default=800, type=int)
    parser.add_argument('--overlap-ratio', default=0.1, type=float)
    parser.add_argument('--out-file-name', default="eval_results.json", type=str)
    return parser.parse_args()

def sahi_validation(args):

    coco = COCO(args.annotation)

    imgIds = coco.getImgIds()

    if args.num_test > 0:
        test_N = args.num_test
        test_idx = random.sample(imgIds, test_N)
    else:
        test_N = len(imgIds)
        test_idx = imgIds

    if not pathlib.Path(args.checkpoint).is_file():
                time.sleep(5)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='mmdet',
        model_path=args.checkpoint,
        config_path=args.config,
        confidence_threshold=args.score_threshold,
        device="cuda", # or 'cuda:0'
    )
    results = []
    for count, imgId in enumerate(test_idx):
        img = coco.loadImgs(imgId)[0]
        print(f"{count}/{test_N}", end='\r')
        result = get_sliced_prediction(
            str(pathlib.Path(args.datadir, img['file_name'])),
            detection_model,
            slice_height = args.crop_size,
            slice_width = args.crop_size,
            overlap_height_ratio = args.overlap_ratio,
            overlap_width_ratio = args.overlap_ratio,
            # perform_standard_pred=False, # uncomment this line if the whole image size is different
            verbose=0).to_coco_predictions(image_id=imgId)
        results.extend(result)
    
    if len(results) == 0:
        print("No Dectected bbox, skipping evaluation!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    with open(args.out_file_name, "w") as f:
        json.dump(results, f)
    
    if args.num_test > 0:
        resultsCOCO = coco.loadRes(args.out_file_name)
        eval = COCOeval(coco, resultsCOCO, "bbox")
        eval.params.imgIds = test_idx      # set parameters as desired
        eval.evaluate();                # run per image evaluation
        eval.accumulate();              # accumulate per image results
        eval.summarize();               # display summary metrics of results
    else:
        print("Done generating test predictions!!!!!!")


if __name__ == '__main__':
    args = parse_args()
    sahi_validation(args)
