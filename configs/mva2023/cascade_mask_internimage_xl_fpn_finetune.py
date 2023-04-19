from set_lib_dir import LIB_ROOT_DIR
data_root = LIB_ROOT_DIR + '/data/'
import os
# LIB_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    './drone_dataset_crop.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
# pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.6,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
]))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


data = dict(
    samples_per_gpu=2,
    # train=dict(pipeline=train_pipeline)
    )
# optimizer
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001 * 2, weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=39, layer_decay_rate=0.90,
                       depths=[5, 5, 24, 5], offset_lr_scale=0.01))
optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512))
resume_from = None
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

runner = dict(max_epochs=40)
evaluation = dict(interval=999, metric='bbox')
load_from = LIB_ROOT_DIR + '/work_dirs/cascade_mask_internimage_xl_fpn_3x_coco/latest.pth'
checkpoint_config = dict(
    interval=2,
    max_keep_ckpts=5,
    save_last=True,
)
log_config = dict(
    interval=100,
)

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 0],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    train=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/merged_train.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
        ann_file=data_root + 'mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json',
        img_prefix=data_root + 'mva2023_sod4bird_private_test/images/',
        pipeline=test_pipeline
    )
)