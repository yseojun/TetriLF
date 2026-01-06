_base_ = '../default.py'

# These will be overwritten by gen_train_all.py
expname = 'lf_shaman_1_0106_4_128_half_2'
basedir = '/data/ysj/result/tetrirf/logs/0106_4_128_half_2'

data = dict(
    datadir='/data/ysj/dataset/LF_video_half/shaman_1',
    dataset_type='LF',  # Light Field dataset type
    white_bkgd=True,
    
    # LF-specific parameters
    uv_scale=1.0,       # Scale for UV (image) coordinates
    st_scale=0.1,      # Scale for ST (camera) coordinates
    grid_size_x=9,      # Number of views in x direction (9x9 grid)
    grid_size_y=9,      # Number of views in y direction (9x9 grid)
    
    # Test views (view indices to use for testing)
    # x=2,4,6 and y=2,4,6 (3x3 grid of test views)
    # view_idx = y * grid_size_x + x = y * 9 + x
    test_frames=[20, 22, 24, 38, 40, 42, 56, 58, 60],  # (y,x): (2,2),(2,4),(2,6),(4,2),(4,4),(4,6),(6,2),(6,4),(6,6)
    world_size=[3, 3, 60, 30],
    
    # Not used for LF but needed for compatibility
    inverse_y=False,
    ndc=False,
    load2gpu_on_the_fly=True,
)

fine_model_and_render = dict(
    num_voxels=120**4,
    num_voxels_base=120**4,
    k0_type='PlaneGrid',
    k0_config=dict(factor=4),
    rgbnet_dim=96,
    RGB_model='MLP',
    rgbnet_depth=4,
    rgbnet_width=128,
    dynamic_rgbnet=True,
    viewbase_pe=4,
    stepsize=0.5,
)

fine_train = dict(
    N_iters=60000,
    N_rand=16384,          # Batch size (number of UVST points per iteration)
    ray_sampler='flatten',  # Use flatten for LF
    
    # Learning rates
    lrate_k0=1e-1,
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
    pg_scale=[500, 1000, 1500, 2000],
    pg_scale2=[3000, 5000, 7000, 9000],
    
    maskout_iter=20000,  # Not used for LF
)

coarse_train = dict(
    N_iters=0,
    N_rand=16384,
    ray_sampler='flatten',
    lrate_k0=1e-1,
    lrate_decay=20,
    weight_main=1.0,
    pg_scale=[],
    pg_scale2=[],
)

