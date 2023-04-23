# Execution

## 0. Hardware Requirements
- Before running our model, please ensure that your GPUs have at least **26 GB** of VRAM.
- Our environment has been deployed into a Docker image that requires the NVIDIA Container Toolkit. The Docker image is based on CUDA 11.3, please make sure that your GPU driver version is at least **465.19.01**.

It is recommended to execute our code using V100 32GB or RTX A6000, as we have tested the following scripts and docker images on these devices.

<br/>

## 1. Environment Preparation (MVATeam1)
### 1.1 Cloning Our Repository
Please clone our GitHub repository and switch to the `submit` branch:

```bash
git clone https://github.com/Howardkhh/MVATeam1.git
cd MVATeam1
git switch submit
```

The default branch is called `main`, whereas the code we had prepared before 23:59 PST on April 21, 2023, is located in the `submit` branch.

### **However, please follow this `main` branch's tutorial since it contains the most up-to-date information.**

<br/>

### 1.2 Preparing Private Test Data
Before launching the Docker container, ensure that the data folder is linked to our repository. The data folder should contain the private test folder named `mva2023_sod4bird_private_test`. Inside this folder, the annotation file should be named `private_test_coco_empty_ann.json` and placed under the `annotations` subfolder.

```bash
data
├── mva2023_sod4bird_private_test
│   ├── annotations
│   │   └── private_test_coco_empty_ann.json
│   └── images
│       ├── 00001.jpg
│       └── ...
└── ...
```

```bash
ln -s <absolute path to the data folder> ./data
```

<br/>

### 1.3 Launching Docker Container
The docker container can be launched by:

```bash
make start
```
After launching the Docker container, the working directory should be the `MVATeam1` folder. 

**All of the commands below should be executed in `MVATeam1` directory.**

Next, please install the MMDetection development package inside the container by:

```bash
make post_install
```
The docker image is built with the `Dockerfile` within the root of our repository, and has been uploaded to the Dockerhub. Invoking `make start` will also download the image.

As inferencing using MMDetection requires a larger shared memory space (`/dev/shm`), we set the Docker container to have 32GB of shared memory space. The size could be changed by editing the `SHMEM_SIZE` variable in the `Makefile`.

<br/>

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

<br/>

## 3. Folder Contents 
Please make sure that the files are in the following structure (only the most important files are listed):
```bash
MVATeam1
├── configs
├── data
│   ├── mva2023_sod4bird_private_test
│   │   ├── annotations
│   │   │   └── private_test_coco_empty_ann.json
│   │   └── images
│   │       ├── 00001.jpg
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
├── inference_private_parallel.sh
└── ...
```

<br/>

## 4. Inference
A bash script for inferencing has been prepared in the `MVATeam1` folder.

Please change the variable `<NUM_GPUS>` depending on your system, and then run the script under `MVATeam1` directory:
```bash
bash inference_private_parallel.sh <NUM_GPUS>
# E.g.:
bash inference_private_parallel.sh 4
```
**Note:** There is an alternative bash script called `inference_private.sh`, which is a sequential version that runs on a single GPU and produces the same output. However, we strongly recommend using the parallel version due to the extended inference time of our models.

This script automatically inferences with all of our models and runs the ensemble process. Upon completion, the final predictions are saved as `results.json` and zipped into `results_team1.zip`.

### **The final results to be evaluated is the file `results_team1.zip` under the `MVATeam1` folder!**

<br/>

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

<br/>

# Auxiliary Training Data
Open sourced datasets are allowed in this contest. We use an auxiliary dataset (https://www.kaggle.com/datasets/nelyg8002000/birds-flying) to augment our data during training. We have uploaded it to the Google Drive. For detailed usage, please refer to the class: `MVAPasteBirds` in `MVATeam1/mmdet/datasets/pipelines/transforms.py`. It can be downloaded with the following commands:

```bash
cd data
gdown --fuzzy https://drive.google.com/file/d/1dcPNszz3h7Ntq0G_OfjEzGMwc9OAv7uQ/view?usp=share_link
unzip birds.zip
rm birds.zip
cd ..
```
