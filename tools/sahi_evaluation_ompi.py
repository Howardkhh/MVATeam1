import os

# proc_id = int(os.environ['SLURM_PROCID'])
# ntasks = int(os.environ['SLURM_NTASKS'])
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

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

import sys
import subprocess
import pickle

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

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

def get_dist_rank():
    return int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])

def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_rank()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        # ordered_results = []
        # for res in zip(*part_list):
        #     ordered_results.extend(list(res))
        # # the dataloader may pad some samples
        # ordered_results = ordered_results[:size]
        
        ret = []
        for i in range(len(part_list)):
            for j in range(len(part_list[i])):
                for k in range(len(part_list[i][j])):
                    ret.append(part_list[i][j][k])
        return ret


def init_dist():
    proc_id = int(os.environ['OMPI_COMM_WORLD_RANK'])
    ntasks = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # num_gpus = torch.cuda.device_count()
    # torch.cuda.set_device(proc_id % num_gpus)
    # specify master port
    if 'MASTER_PORT' in os.environ:
        port = os.environ['MASTER_PORT']
    else:
        sys.stderr.write("MASTER_PORT not defined")
        exit(1)
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    print(f"local rank={os.environ['OMPI_COMM_WORLD_LOCAL_RANK']}")
    os.environ['RANK'] = str(proc_id)

    dist.init_process_group(backend='nccl')

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

    # device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
    # print("device", device)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='mmdet',
        model_path=args.checkpoint,
        config_path=args.config,
        confidence_threshold=args.score_threshold,
        device='cuda', # or 'cuda:0'
    )
    # detection_model = nn.DataParallel(detection_model)
    # detection_model.to(torch.device(f"cuda:{os.environ['LOCAL_RANK']}"))
    # detection_model.to(device)

    # devices = list(range(torch.cuda.device_count()))

    results = []
   
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_idx))
    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=1,
        shuffle=False,
        num_workers=0,
        sampler=test_sampler
    )


    dataset = test_loader.dataset
    rank, size = get_dist_rank()

    for batch_idx, batch in enumerate(test_loader):
        batch = batch[0]
        print(f"Batch {batch_idx+1}/{len(test_loader)}", end='\r')
        results_batch = []
        with torch.no_grad():
            for imgId in batch:
                imgId_int = int(imgId)
                # sys.stderr.write(f"rank={rank} imgId={imgId_int}\n")
                # sys.stderr.write(f"coco.loadImgs(imgId)={coco.loadImgs(imgId_int)}\n")
                img = coco.loadImgs(imgId_int)[0]
                re = get_sliced_prediction(
                    str(pathlib.Path(args.datadir, img['file_name'])),
                    detection_model,
                    slice_height = args.crop_size,
                    slice_width = args.crop_size,
                    overlap_height_ratio = args.overlap_ratio,
                    overlap_width_ratio = args.overlap_ratio,
                    # perform_standard_pred=False, # uncomment this line if the whole image size is different
                    verbose=0).to_coco_predictions(image_id=imgId_int)
                results_batch.append(re)
        results.extend([r for r in results_batch if len(r) > 0])
    
    

    
    if len(results) == 0:
        print("No Dectected bbox, skipping evaluation!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    # gather
    print(results)
    results = collect_results_gpu(results, len(dataset))
    if rank == 0:
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
    init_dist()
    sahi_validation(args)
