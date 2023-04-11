# Installation
```shell
# requires cuda 11.5 and gcc10

conda create -n mva_team1 -c conda-forge -y python==3.10
conda activate mva_team1
pip3 install torch==1.11.0+cu115 torchvision==0.12.0+cu115  -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html
pip install timm opencv-python termcolor yacs pyyaml scipy

cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py

cd ..
pip install -v -e .
pip install -r requirements/sahi.txt

```
# Dataset
The dataset should be the same as baseline
```shell
data
 ├ drone2021
 ├ mva2023_sod4bird_train
 ├ mva2023_sod4bird_pub_test
 └ mva2023_sod4bird_private_test
```

# Training
## Centernet Baseline
```shell
# centernet r18
python3 tools/train.py  configs/mva2023_baseline/centernet_resnet18_140e_coco.py
python3 tools/train.py  configs/mva2023_baseline/centernet_resnet18_140e_coco_finetune.py
python3 hard_neg_example_tools/test_hard_neg_example.py \
    --config configs/mva2023_baseline/centernet_resnet18_140e_coco_sample_hard_negative.py \
    --checkpoint work_dirs/centernet_resnet18_140e_coco_finetune/latest.pth \
    --launcher none \
    --generate-hard-negative-samples True \
    --hard-negative-file work_dirs/centernet_resnet18_140e_coco_finetune/train_coco_hard_negative.json \
    --hard-negative-config num_max_det=10 pos_iou_thr=1e-5 score_thd=0.05 nms_thd=0.05
python3 tools/train.py configs/mva2023_baseline/centernet_resnet18_140e_coco_hard_negative_training.py

# centernet r50
python3 tools/train.py  configs/mva2023_baseline/centernet_resnet50_140e_coco.py
python3 tools/train.py  configs/mva2023_baseline/centernet_resnet50_140e_coco_finetune.py
python3 hard_neg_example_tools/test_hard_neg_example.py \
    --config configs/mva2023_baseline/centernet_resnet50_140e_coco_sample_hard_negative.py \
    --checkpoint work_dirs/centernet_resnet50_140e_coco_finetune/latest.pth \
    --launcher none \
    --generate-hard-negative-samples True \
    --hard-negative-file work_dirs/centernet_resnet50_140e_coco_finetune/train_coco_hard_negative.json \
    --hard-negative-config num_max_det=10 pos_iou_thr=1e-5 score_thd=0.05 nms_thd=0.05
python3 tools/train.py configs/mva2023_baseline/centernet_resnet50_140e_coco_hard_negative_training.py

# centernet intern image xl
python3 tools/train.py  configs/mva2023_baseline/centernet_internimage_xl_140e_coco.py
python3 tools/train.py  configs/mva2023_baseline/centernet_internimage_xl_140e_coco_finetune.py
python3 hard_neg_example_tools/test_hard_neg_example.py \
    --config configs/mva2023_baseline/centernet_internimage_xl_140e_coco_sample_hard_negative.py \
    --checkpoint work_dirs/centernet_internimage_xl_140e_coco_finetune/latest.pth \
    --launcher none \
    --generate-hard-negative-samples True \
    --hard-negative-file work_dirs/centernet_internimage_xl_140e_coco_finetune/train_coco_hard_negative.json \
    --hard-negative-config num_max_det=10 pos_iou_thr=1e-5 score_thd=0.05 nms_thd=0.05
python3 tools/train.py configs/mva2023_baseline/centernet_internimage_xl_140e_coco_hard_negative_training.py
```
## Cascade RCNN
```shell
# Cascade RCNN r18
python3 tools/train.py  configs/mva2023/cascade_rcnn_r18_fpn_140e_coco_nwd.py
python3 tools/train.py  configs/mva2023/cascade_rcnn_r18_fpn_20e_coco_nwd_finetune.py

# Cascade RCNN r50
python3 tools/train.py  configs/mva2023/cascade_rcnn_r50_fpn_140e_coco_nwd.py
python3 tools/train.py  configs/mva2023/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune.py

# Cascade intern image xl
python3 tools/train.py  configs/mva2023/cascade_rcnn_internimage_xl_fpn_140e_coco_nwd.py
python3 tools/train.py  configs/mva2023/cascade_rcnn_internimage_xl_fpn_20e_coco_nwd_finetune.py
```

# Inferencing
```shell
python3 tools/test.py configs/mva2023_baseline/centernet_resnet18_140e_coco_inference.py work_dirs/centernet_resnet18_140e_coco_hard_negative_training/latest.pth --format-only --eval-options jsonfile_prefix=centernet_resnet18_140e_coco_hard_negative_training
python3 tools/test.py configs/mva2023_baseline/centernet_resnet50_140e_coco_inference.py work_dirs/centernet_resnet50_140e_coco_hard_negative_training/latest.pth --format-only --eval-options jsonfile_prefix=centernet_resnet50_140e_coco_hard_negative_training
python3 tools/test.py configs/mva2023_baseline/centernet_internimage_xl_140e_coco_inference.py work_dirs/centernet_internimage_xl_140e_coco_hard_negative_training/latest.pth --format-only --eval-options jsonfile_prefix=centernet_internimage_xl_140e_coco_hard_negative_training

mv centernet_resnet18_140e_coco_hard_negative_training.bbox.json submit/centernet_resnet18_140e_coco_hard_negative_training.json 
mv centernet_resnet50_140e_coco_hard_negative_training.bbox.json submit/centernet_resnet50_140e_coco_hard_negative_training.json
mv centernet_internimage_xl_140e_coco_hard_negative_training.bbox.json submit/centernet_internimage_xl_140e_coco_hard_negative_training.json

python3 tools/sahi_evaluation.py configs/mva2023/cascade_rcnn_r18_fpn_20e_coco_nwd_finetune.py \
    work_dirs/cascade_rcnn_r18_fpn_20e_coco_nwd_finetune/latest.pth \
    data/mva2023_sod4bird_private_test/images/ \
    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
    --out-file-name cascade_rcnn_r18_fpn_20e_coco_nwd_finetune.json
python3 tools/sahi_evaluation.py configs/mva2023/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune.py \
    work_dirs/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune/latest.pth \
    data/mva2023_sod4bird_private_test/images/ \
    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
    --out-file-name cascade_rcnn_r50_fpn_40e_coco_nwd_finetune.json
python3 tools/sahi_evaluation.py configs/mva2023/cascade_rcnn_internimage_xl_fpn_20e_coco_nwd_finetune.py \
    work_dirs/cascade_rcnn_internimage_xl_fpn_20e_coco_nwd_finetune/latest.pth \
    data/mva2023_sod4bird_private_test/images/ \
    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
    --out-file-name cascade_rcnn_internimage_xl_fpn_20e_coco_nwd_finetune.json

mv work_dirs/cascade_rcnn_r18_fpn_20e_coco_nwd_finetune.json submit/cascade_rcnn_r18_fpn_20e_coco_nwd_finetune.json
mv work_dirs/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune.json submit/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune.json
mv work_dirs/cascade_rcnn_internimage_xl_fpn_20e_coco_nwd_finetune.json submit/cascade_rcnn_internimage_xl_fpn_20e_coco_nwd_finetune.json
```

# Ensemble
```shell
################################################################

# 1. List models we want to ensemble in config.txt.

# 2. Set weights for the corresponding models in ensemble.py
ensemble('config.txt', 'results.json', weights=[2,4,5,6,8])

# 3. Execute ensemble.py with argument --method to generate results.json
# The argument --method has choices ['wbf', 'snms] with 'wbf' being the default option
# To use the weighted boxes fusion ensembling method
python ensemble_wbf.py --method wbf

# To use the Soft NMS ensembling method
python ensemble_wbf.py --method snms

# 4. Compress the results.json to results.zip and we're done.
zip results.zip results.json

################################################################
```

# Trouble Shooting
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