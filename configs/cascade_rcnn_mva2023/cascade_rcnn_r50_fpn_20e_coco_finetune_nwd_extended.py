from set_lib_dir import LIB_ROOT_DIR
_base_ = ['./cascade_rcnn_r50_fpn_1x_coco_nwd.py',
    './sod4bird_dataset.py',
]
data_root = LIB_ROOT_DIR + '/data/'
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

load_from = LIB_ROOT_DIR + '/work_dirs/cascade_rcnn_r50_fpn_140e_coco/latest.pth'

log_config = dict(
    interval=100,
)

evaluation = dict(interval=999, metric='bbox')
data = dict(
    samples_per_gpu=8,
    train=dict(
        ann_file=data_root + 'mva2023_sod4bird_train_extended/annotations/split_train_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train_extended/images/'),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/'),
    test=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/'),
    )
