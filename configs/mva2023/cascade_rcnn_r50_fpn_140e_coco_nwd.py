_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_nwd_mva2023.py',
    './drone_dataset_crop.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
runner = dict(max_epochs=140)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)