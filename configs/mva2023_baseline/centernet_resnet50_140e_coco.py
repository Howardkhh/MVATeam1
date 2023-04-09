_base_ = './centernet_resnet18_dcnv2_140e_coco.py'

model = dict(
    backbone=dict(
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(use_dcn=False)
    )
