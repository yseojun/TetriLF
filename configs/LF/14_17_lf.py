_base_ = '../default.py'

expname = 'lf_14_17_feature_1222_test_120_60'
basedir = '/data/ysj/result/tetrirf/logs/'

data = dict(
    datadir='/data/ysj/dataset/lf_from_4dgs/14_17_60s_2_1800f_960_540',
    dataset_type='LF',  # Light Field dataset type
    white_bkgd=True,
    
    # LF-specific parameters
    uv_scale=1.0,       # Scale for UV (image) coordinates
    st_scale=0.1,      # Scale for ST (camera) coordinates
    grid_size_x=19,     # Number of views in x direction (auto-detect if not specified)
    grid_size_y=5,      # Number of views in y direction (auto-detect if not specified)
    
    # Test views (view indices to use for testing)
    # y=2 (middle row), x=3,6,9,12,15 (evenly spaced)
    # view_idx = y * grid_size_x + x = 2 * 19 + x
    test_frames=[41, 44, 47, 50, 53],  # Middle row, x=3,6,9,12,15
    world_size=[12, 3, 120, 60],
    
    # Not used for LF but needed for compatibility
    inverse_y=False,
    ndc=False,
    load2gpu_on_the_fly=True,
)

fine_model_and_render = dict(
    num_voxels=120**4,
    num_voxels_base=120**4,
    k0_type='PlaneGrid',
    k0_config=dict(factor=1),
    rgbnet_dim=96,
    RGB_model='MLP',
    rgbnet_depth=8,
    rgbnet_width=256,
    dynamic_rgbnet=True,
    viewbase_pe=4,
    stepsize=0.5,
)

fine_train = dict(
    N_iters=30000,
    N_rand=16384,          # Batch size
    ray_sampler='flatten',  # Use flatten for LF
    
    # Learning rates
    lrate_density=1e-1,
    lrate_k0=1e-3,
    lrate_rgbnet=1e-3,
    lrate_decay=20,
    
    # Total variation
    tv_every=1,
    tv_after=2000,
    tv_before=40000,
    tv_dense_before=40000,
    weight_tv_density=0,  # Not used for LF
    weight_tv_k0=1e-5,
    
    # Other losses
    weight_main=1.0,
    weight_l1_loss=0.001,  # L1 loss between adjacent frames
    weight_entropy_last=0,
    weight_rgbper=0,
    
    # Progressive growing
    pg_scale=[],
    pg_scale2=[],
    
    maskout_iter=20000,  # Not used for LF
)

coarse_train = dict(
    N_iters=0  # Skip coarse training for LF
)

