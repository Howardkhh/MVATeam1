from set_lib_dir import LIB_ROOT_DIR
_base_ = './cascade_rcnn_r50_fpn_140e_coco_nwd.py',
data_root = LIB_ROOT_DIR + '/data/'

data = dict(
    train=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/merged_train.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    )
)        
runner = dict(max_epochs=20)

load_from = LIB_ROOT_DIR + '/work_dirs/cascade_rcnn_r50_fpn_140e_coco_nwd/latest.pth'

log_config = dict(
    interval=100,
)