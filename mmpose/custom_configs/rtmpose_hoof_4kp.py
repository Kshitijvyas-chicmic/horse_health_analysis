auto_scale_lr = dict(base_batch_size=1024)
backend_args = dict(backend='local')
base_lr = 0.001
codec = dict(
    input_size=(
        192,
        256,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = []
custom_imports = dict(imports=['tools.custom_vis_backend'], allow_failed_imports=False)
data_mode = 'topdown'
data_root = 'data/coco/'
dataset_info = dict(
    dataset_name='horse_hoof_side',
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    keypoint_info=dict({
        0:
        dict(
            color=[
                255,
                0,
                0,
            ],
            id=0,
            name='pastern_top',
            swap='',
            type='upper'),
        1:
        dict(
            color=[
                255,
                85,
                0,
            ],
            id=1,
            name='pastern_bottom',
            swap='',
            type='lower'),
        2:
        dict(
            color=[
                0,
                255,
                0,
            ],
            id=2,
            name='hoof_wall_top',
            swap='',
            type='hoof'),
        3:
        dict(color=[
            0,
            0,
            255,
        ], id=3, name='toe_tip', swap='', type='hoof')
    }),
    paper_info=dict(
        author='',
        homepage='',
        title='Horse Hoof Side View Keypoints',
        year=2025),
    sigmas=[
        0.03,
        0.03,
        0.03,
        0.03,
    ],
    skeleton_info=dict({
        0:
        dict(color=[
            255,
            128,
            0,
        ], link=(
            'pastern_top',
            'pastern_bottom',
        )),
        1:
        dict(color=[
            0,
            128,
            255,
        ], link=(
            'hoof_wall_top',
            'toe_tip',
        ))
    }))
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=5,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        max_keep_ckpts=1,
        rule='greater',
        save_best=None,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
exp_name = 'Horse_Health_Training'
keypoint_info = dict({
    0:
    dict(color=[
        255,
        0,
        0,
    ], id=0, name='pastern_top', swap='', type='upper'),
    1:
    dict(
        color=[
            255,
            85,
            0,
        ],
        id=1,
        name='pastern_bottom',
        swap='',
        type='lower'),
    2:
    dict(
        color=[
            0,
            255,
            0,
        ], id=2, name='hoof_wall_top', swap='', type='hoof'),
    3:
    dict(color=[
        0,
        0,
        255,
    ], id=3, name='toe_tip', swap='', type='hoof')
})
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
max_epochs = 300
mlflow_host = 'mlflow'
mlflow_ip = '172.19.0.2'
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.75),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=768,
        in_featuremap_size=(
            6,
            8,
        ),
        input_size=(
            192,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=4,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=False),
    type='TopdownPoseEstimator')
num_keypoints = 4
optim_wrapper = dict(
    optimizer=dict(lr=base_lr, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=150,
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=2.5e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = False
run_name = 'Hoof_4KP_Run_v6'
skeleton_info = dict({
    0:
    dict(color=[
        255,
        128,
        0,
    ], link=(
        'pastern_top',
        'pastern_bottom',
    )),
    1:
    dict(color=[
        0,
        128,
        255,
    ], link=(
        'hoof_wall_top',
        'toe_tip',
    ))
})
stage2_num_epochs = 30
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='/app/datasets/val_fixed.json',
        data_mode='topdown',
        data_prefix=dict(img='images/hq_consolidation_550/'),
        data_root='/app/data',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/app/datasets/val_fixed.json', type='CocoMetric')
tracking_uri = 'http://172.19.0.2:5000'
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/app/datasets/train_fixed.json',
        data_mode='topdown',
        data_prefix=dict(img='images/hq_consolidation_550/'),
        data_root='/app/data',
        metainfo=dict(
            dataset_name='horse_hoof_side',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=0,
                    name='pastern_top',
                    swap='',
                    type='upper'),
                1:
                dict(
                    color=[
                        255,
                        85,
                        0,
                    ],
                    id=1,
                    name='pastern_bottom',
                    swap='',
                    type='lower'),
                2:
                dict(
                    color=[
                        0,
                        255,
                        0,
                    ],
                    id=2,
                    name='hoof_wall_top',
                    swap='',
                    type='hoof'),
                3:
                dict(
                    color=[
                        0,
                        0,
                        255,
                    ],
                    id=3,
                    name='toe_tip',
                    swap='',
                    type='hoof')
            }),
            paper_info=dict(
                author='',
                homepage='',
                title='Horse Hoof Side View Keypoints',
                year=2025),
            sigmas=[
                0.03,
                0.03,
                0.03,
                0.03,
            ],
            skeleton_info=dict({
                0:
                dict(
                    color=[
                        255,
                        128,
                        0,
                    ],
                    link=(
                        'pastern_top',
                        'pastern_bottom',
                    )),
                1:
                dict(
                    color=[
                        0,
                        128,
                        255,
                    ], link=(
                        'hoof_wall_top',
                        'toe_tip',
                    ))
            })),
        pipeline=[
            dict(type='LoadImage'),
            dict(padding=1.05, type='GetBBoxCenterScale'),
            dict(rotate_factor=10, type='RandomBBoxTransform'),
            dict(prob=0.2, type='RandomFlip'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(p=0.5, type='CLAHE'),
                    dict(max_holes=1, p=0.3, type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=5.0,
                    type='SimCCLabel'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(padding=1.05, type='GetBBoxCenterScale'),
    dict(rotate_factor=10, type='RandomBBoxTransform'),
    dict(prob=0.2, type='RandomFlip'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(p=0.5, type='CLAHE'),
            dict(max_holes=1, p=0.3, type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=5.0,
            type='SimCCLabel'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/app/datasets/val_fixed.json',
        data_mode='topdown',
        data_prefix=dict(img='images/hq_consolidation_550/'),
        data_root='/app/data',
        metainfo=dict(
            dataset_name='horse_hoof_side',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=0,
                    name='pastern_top',
                    swap='',
                    type='upper'),
                1:
                dict(
                    color=[
                        255,
                        85,
                        0,
                    ],
                    id=1,
                    name='pastern_bottom',
                    swap='',
                    type='lower'),
                2:
                dict(
                    color=[
                        0,
                        255,
                        0,
                    ],
                    id=2,
                    name='hoof_wall_top',
                    swap='',
                    type='hoof'),
                3:
                dict(
                    color=[
                        0,
                        0,
                        255,
                    ],
                    id=3,
                    name='toe_tip',
                    swap='',
                    type='hoof')
            }),
            paper_info=dict(
                author='',
                homepage='',
                title='Horse Hoof Side View Keypoints',
                year=2025),
            sigmas=[
                0.03,
                0.03,
                0.03,
                0.03,
            ],
            skeleton_info=dict({
                0:
                dict(
                    color=[
                        255,
                        128,
                        0,
                    ],
                    link=(
                        'pastern_top',
                        'pastern_bottom',
                    )),
                1:
                dict(
                    color=[
                        0,
                        128,
                        255,
                    ], link=(
                        'hoof_wall_top',
                        'toe_tip',
                    ))
            })),
        pipeline=[
            dict(type='LoadImage'),
            dict(padding=1.05, type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/app/datasets/val_fixed.json',
    score_mode='keypoint',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(padding=1.05, type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
# vis_backends = [
#     dict(type='LocalVisBackend'),
#     dict(
#         save_dir='mlruns',
#         tracking_uri='http://172.19.0.2:5000',
#         type='SanitizedMLflowVisBackend'),
# ]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            exp_name='Horse_Health_Training',
            run_name='Hoof_4KP_Run_v6',
            save_dir='mlruns',
            tracking_uri='http://172.19.0.2:5000',
            type='SanitizedMLflowVisBackend'),
    ])
work_dir = 'mmpose/work_dirs/automation_run'
