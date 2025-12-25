# =========================
# Keypoint definition
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
