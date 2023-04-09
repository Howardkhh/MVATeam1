from set_lib_dir import LIB_ROOT_DIR
_base_ = ['./cascade_rcnn_r50_fpn_1x_coco.py',
    './sod4bird_paste_dataset.py',
]
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=40)

load_from = LIB_ROOT_DIR + '/work_dirs/cascade_rcnn_r50_fpn_40e_coco_finetune_paste/latest.pth'

log_config = dict(
    interval=100,
)

evaluation = dict(interval=999, metric='bbox')
