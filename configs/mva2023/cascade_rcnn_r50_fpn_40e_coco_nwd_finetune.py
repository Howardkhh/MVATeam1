from set_lib_dir import LIB_ROOT_DIR
_base_ = './cascade_rcnn_r50_fpn_140e_coco_nwd.py'
data_root = LIB_ROOT_DIR + '/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'),
                keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    train=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/merged_train.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
        ann_file=data_root + 'mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json',
        img_prefix=data_root + 'mva2023_sod4bird_pub_test/images/',
        test_pipeline=test_pipeline
    )
)        
runner = dict(max_epochs=40)

load_from = LIB_ROOT_DIR + '/work_dirs/cascade_rcnn_r50_fpn_140e_coco_nwd/latest.pth'

log_config = dict(
    interval=100,
)