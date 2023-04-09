_base_ = ['./cascade_rcnn_r50_fpn_1x_coco.py',
    # '../mva2023_baseline/drone_dataset.py',
    '../mva2023_baseline/drone_dataset.py',
]
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=140)

log_config = dict(
    interval=100,
)

evaluation = dict(interval=999, metric='bbox')
