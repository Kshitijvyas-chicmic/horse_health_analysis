# _base_ = [
#     'mmpose::_base_/default_runtime.py'
# ]

#default_scope = 'mmpose'

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook'),
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = []

# =========================
# Dataset
# =========================
keypoint_info = {
    0: dict(name='pastern_top', id=0, color=[255, 0, 0], type='upper', swap=''),
    1: dict(name='pastern_bottom', id=1, color=[255, 85, 0], type='lower', swap=''),
    2: dict(name='hoof_wall_top', id=2, color=[0, 255, 0], type='hoof', swap=''),
    3: dict(name='toe_tip', id=3, color=[0, 0, 255], type='hoof', swap='')
}

skeleton_info = {
    0: dict(link=('pastern_top', 'pastern_bottom'), color=[255, 128, 0]),
    1: dict(link=('hoof_wall_top', 'toe_tip'), color=[0, 128, 255])
}

dataset_info = dict(
    dataset_name='horse_hoof_side',
    paper_info=dict(
        author='',
        title='Horse Hoof Side View Keypoints',
        year=2025,
        homepage=''
    ),
    keypoint_info=keypoint_info,
    skeleton_info=skeleton_info,
    joint_weights=[1.0] * 4,
    sigmas=[0.03] * 4
)

num_keypoints = 4

# =========================
# Model
# =========================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='RTMPoseBackbone',
        in_channels=3,
        stem_channels=32,
        num_stages=4,
        stage_channels=[32, 64, 128, 256],
        out_channels=256
    ),
    neck=dict(
        type='RTMPoseNeck',
        in_channels=256,
        out_channels=256
    ),
    head=dict(
        type='RTMPoseHead',
        in_channels=256,
        out_channels=num_keypoints,
        input_size=(256, 256),
        heatmap_size=(64, 64),
        loss=dict(type='KeypointMSELoss', use_target_weight=True)
    ),
    test_cfg=dict(flip_test=False)
)

# =========================
# Pipelines
# =========================
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(
        type='GenerateTarget',
        encoder=dict(type='MSRAHeatmap', sigma=2)
    ),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs')
]

# =========================
# Dataloaders
# =========================
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        #data_root='data/',
        data_root='../data/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dataset_info,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        #data_root='data/',
        data_root='../data/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=dataset_info,
        pipeline=val_pipeline
    )
)

# =========================
# Optim & Schedule
# =========================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=30)
]

# train_cfg = dict(max_epochs=30, val_interval=1)
# val_cfg = dict()
# val_evaluator = dict(type='CocoMetric', score_mode='keypoint')



# =========================
# TRAINING CONTROL (TEST MODE)
# =========================

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Disable long COCO hooks
custom_hooks = []

# Make checkpoints simpler
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best=None,
        save_last=True
    )
)