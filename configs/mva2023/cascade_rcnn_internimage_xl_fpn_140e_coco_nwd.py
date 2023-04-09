_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_nwd_mva2023.py',
    './drone_dataset_crop.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth'
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5),
)

optimizer = dict(
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=39,
        layer_decay_rate=0.9,
        depths=[5, 5, 24, 5],
        offset_lr_scale=0.01))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
runner = dict(max_epochs=140)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)