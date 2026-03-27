# =========================
# Base RTMPose config
# =========================
#for potrait
#input_size=(192, 256),
#for landscape
#input_size=(256,192 ),

# for large data set
#_base_ = '../configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py'

#for medium data set
_base_ = '../configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'

#load_from = 'work_dirs/rtmpose_hoof_refined_rotation_experiment_v1/epoch_300.pth'

# =========================
# Work Directory
# =========================
work_dir = './work_dirs/rtmpose_hoof_manual_9_march'

default_scope = 'mmpose'

# =========================
# Keypoints
# =========================
num_keypoints = 4

#custom_hooks = []

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
    joint_weights=[1.0] * num_keypoints,
    sigmas=[0.03] * num_keypoints
)

# =========================
# Model override
# =========================
model = dict(
    head=dict(
        out_channels=num_keypoints,
        input_size=(192, 256),
        #in_featuremap_size=(8, 6),
        simcc_split_ratio=2.0,
    ),
    test_cfg=dict(flip_test=False),
)

# =========================
# Pipelines
# =========================
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.05),
    dict(type='RandomBBoxTransform', rotate_factor=10),

    # Very light flip (controlled risk)
    dict(type='RandomFlip', prob=0.2),

    # Appearance augmentation only
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(type='CLAHE', p=0.5),
            dict(
                type='CoarseDropout',
                max_holes=1,
                p=0.3
            ),
        ]
    ),

    # Controlled geometry (core decision)
    dict(
        type='TopdownAffine',
        input_size=(192, 256),
    ),

    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(192, 256),
            sigma=5.0,
            normalize=False
        )
    ),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale',padding=1.05),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]

# =========================
# Dataloaders
# =========================
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root='../data',
        ann_file='annotations/train_v7_fixed.json',
        data_prefix=dict(img='images/hq_consolidation_550/'),
        metainfo=dataset_info,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root='../data',
        ann_file='annotations/val_v7_fixed.json',
        data_prefix=dict(img='images/hq_consolidation_550/'),
        metainfo=dataset_info,
        pipeline=val_pipeline
    )
)

#val_evaluator = dict(type='CocoMetric', score_mode='keypoint')

val_evaluator = dict(
    type='CocoMetric',
    ann_file='../data/annotations/val_v7_fixed.json',
    score_mode='keypoint'
)

# =========================
# TRAINING CONTROL (TEST MODE)
# =========================

train_cfg = dict(
    #type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=10
)

# Fix LR for small batch size (8 vs 256)
base_lr = 5e-4
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# Fix Scheduler to fit 100 epochs
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500), # Warmup ~10 epochs (around 500-600 iters depending on batch)
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=150, # Start decay halfway
        end=300,  # End at 300
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True),
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Disable long COCO hooks
custom_hooks = []

# Make checkpoints simpler
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best=None,
        save_last=True
    )
)