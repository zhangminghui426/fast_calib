"""默认配置。"""

# 体素化
voxel_size = (0.1, 0.1, 0.2)
point_cloud_range = (0.0, -5.0, -3.0, 10.0, 5.0, 2.0)
max_points_per_voxel = 32
max_num_voxels = 12000

# 网格尺寸
grid_size = (100, 100, 25)  # (Dx, Dy, Dz)

# VFE
vfe_filters = (32, 128)

# Backbone
sparse_filters = (64, 128, 256)
upsample_filters = (128, 128, 128)

# RPN
num_anchor_per_loc = 2
num_class = 1

# Anchor
anchor_sizes = [(0.55, 1.7, 1.18)]  # (w, l, h)
anchor_rotations = (0.0, 1.5707963)  # 0 和 90 度

# 训练
batch_size = 1
lr = 0.001
num_epochs = 14
cls_weight = 1.0
loc_weight = 2.0
score_thr = 0.3
