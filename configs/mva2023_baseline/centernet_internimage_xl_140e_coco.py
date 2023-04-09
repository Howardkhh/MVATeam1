_base_ = './centernet_resnet18_dcnv2_140e_coco.py'

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
        in_channel=1536,
        use_dcn=False)
    )

optimizer = dict(
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=39,
        layer_decay_rate=0.9,
        depths=[5, 5, 24, 5],
        offset_lr_scale=0.01))