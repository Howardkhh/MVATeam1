# Execution

## 0. Hardware Requirement
- Before running our model, please ensure that your GPUs have at least **26 GB** of VRAM.
- Our environment has been deployed into a Docker image that requires the NVIDIA Container Toolkit. The Docker image is based on CUDA 11.3, please make sure that your GPU driver version is at least **465.19.01**.

It is recommended to execute our code using V100 32GB or RTX A6000, as we have tested the following scripts and docker images on these devices.

## 1. Environment Preparation (MVATeam1)

Please clone our GitHub repository and switch to the `submit` branch:

```bash
git clone https://github.com/Howardkhh/MVATeam1.git
cd MVATeam1
git switch submit
```

The default branch is called `main`, whereas the code we had prepared before 23:59 PST on April 21, 2023, is located in the `submit` branch.

Before starting the docker container, please link the data folder to our repository:

```bash
ln -s <absolute path to the data folder> ./data
```

The docker container can be launched by:

```bash
make start
```

The docker image is built with the `Dockerfile` within the root of our repository, and has been uploaded to the Dockerhub. Invoking `make start` will also download the image.

As inferencing using MMDetection requires a larger shared memory space (`/dev/shm`), we set the Docker container to have 32GB of shared memory space. The size could be changed by editing the `SHMEM_SIZE` variable in the `Makefile`.

After launching the Docker container, the working directory should be the `MVATeam1` folder. Next, please install the MMDetection development package inside the container by:

```bash
make post_install
```

## 2. Model Weights

Our model weights are too large to be uploaded to GitHub. Please download the pre-trained weights from Google Drive:

```bash
gdown --folder https://drive.google.com/drive/folders/1zNZTGmlRVsPSpxrVwpik17I2w3XLbqDc?usp=share_link
```

A single folder named `final` should be downloaded to the root of our repository. 



**Important:** Please ensure that **all files** are fully downloaded (the progress bars should be at 100%). 

The size of the `final` folder should be 36 GB.
```bash
du -sh final
# output: 36G        final
```

## 3. Folder Contents 
Please make sure that the files are in the following format (only the most important files are listed):
```bash
MVATeam1
├── configs
├── data
│   ├── mva2023_sod4bird_private_test
│   │   ├── annotations
│   │   │   └── private_test_coco_empty_ann.json
│   │   └── images
│   │       └── ...
│   └── ...
├── ensemble
├── final
│   ├── baseline_centernet
│   │   ├── centernet_resnet18_140e_coco_inference.py
│   │   └── latest.pth
│   ├── cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train
│   │   └── latest.pth
│   ├── cascade_nwd_paste_howard
│   │   ├── cascade_nwd_paste_howard.zip
│   │   └── latest.pth
│   ├── cascade_rcnn_r50_fpn_40e_coco_finetune_sticker
│   │   └── latest.pth
│   ├── cascade_rcnn_r50_fpn_40e_coco_nwd_finetune
│   │   ├── cascade_rcnn_r50_fpn_40e_coco_finetune_original.py
│   │   └── latest.pth
│   ├── internimage_h_nwd
│   │   ├── intern_h_public.json
│   │   ├── intern_h_public_nosahi.json
│   │   ├── intern_h_public_nosahi_randflip.json
│   │   └── latest.pth
│   ├── internimage_xl_no_nwd
│   │   └── latest.pth
│   └── internimage_xl_nwd
│   	├── intern_xl_public_nosahi_randflip.json
│   	└── latest.pth   
└── ...
```

## 4. Inference
A bash script for inferencing has been prepared, please change the variable `<NUM_GPUS>` depending on your system:
```bash
bash inference_private_parallel.sh <NUM_GPUS>
# E.g.:
bash inference_private_parallel.sh 4
```

### **The final results to be evaluated is the file `results_team1.zip`!**

# Troubleshooting
## 1. Cannot Download Model Weights

In case of download quota exceeded, we provide another link. Please make sure the tarball is properly extracted.
```bash
gdown 'https://drive.google.com/u/3/uc?id=1fRW-3CVf3t5EQUjYfh6BQN27VvTRTn9a&export=download'
tar zxvf final.tar.gz
```
## 2. Internimage Custom Imports
If you encounter 
```KeyError: "CenterNet: 'InternImage is not in the models registry'"```\
or
```KeyError: "CenterNet: 'CustomLayerDecayOptimizerConstructor is not in the models registry'"```,\
please add
```python
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403
```
to the python script you are running.


# Auxiliary Training Data
We use an auxiliary dataset (https://www.kaggle.com/datasets/nelyg8002000/birds-flying) to augment our data during training. We have uploaded our dataset to the Google Drive. For detailed usage, please refer to the class: `MVAPasteBirds` in `MVATeam1/mmdet/datasets/pipelines/transforms.py`. It can be downloaded with the following commands:

```bash
cd data
gdown --fuzzy https://drive.google.com/file/d/1dcPNszz3h7Ntq0G_OfjEzGMwc9OAv7uQ/view?usp=share_link
unzip birds.zip
rm birds.zip
cd ..
```