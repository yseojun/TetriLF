"""
Test script for 4D Grid Sample and Interpolate CUDA extensions.
"""

import torch
import torch.nn as nn
import time

print("=" * 60)
print("Testing 4D Grid Sample and Interpolate CUDA Extensions")
print("=" * 60)

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    exit(1)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print()

# Import the modules
print("Loading CUDA extensions (JIT compile)...")
start = time.time()
from lib.grid_sample_4d import grid_sample_4d, interpolate_4d, GridSample4D, Interpolate4D
print(f"Loading time: {time.time() - start:.2f}s")
print()

def test_grid_sample_4d():
    """Test 4D grid sampling (quadrilinear interpolation)"""
    print("-" * 40)
    print("Test 1: grid_sample_4d")
    print("-" * 40)
    
    # Create test input: [N, C, X, Y, U, V]
    N, C = 1, 16
    X, Y, U, V = 8, 8, 64, 64
    input_tensor = torch.randn(N, C, X, Y, U, V, device='cuda', dtype=torch.float32)
    
    # Create grid: [N, N_pts, 4] with normalized coordinates in [-1, 1]
    N_pts = 1000
    grid = torch.rand(N, N_pts, 4, device='cuda', dtype=torch.float32) * 2 - 1  # [-1, 1]
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Grid shape: {grid.shape}")
    
    # Forward pass
    output = grid_sample_4d(input_tensor, grid, align_corners=True, padding_mode='zeros')
    print(f"  Output shape: {output.shape}")
    assert output.shape == (N, C, N_pts), f"Expected {(N, C, N_pts)}, got {output.shape}"
    
    # Test with requires_grad
    input_tensor.requires_grad_(True)
    output = grid_sample_4d(input_tensor, grid)
    loss = output.sum()
    loss.backward()
    
    assert input_tensor.grad is not None, "Gradient not computed!"
    print(f"  Gradient computed: {input_tensor.grad.shape}")
    print("  ✓ grid_sample_4d PASSED")
    print()
    return True


def test_interpolate_4d():
    """Test 4D interpolation (resize)"""
    print("-" * 40)
    print("Test 2: interpolate_4d")
    print("-" * 40)
    
    # Create test input: [N, C, X, Y, U, V]
    N, C = 1, 16
    X_in, Y_in, U_in, V_in = 4, 4, 32, 32
    input_tensor = torch.randn(N, C, X_in, Y_in, U_in, V_in, device='cuda', dtype=torch.float32)
    
    # Target size
    X_out, Y_out, U_out, V_out = 8, 8, 64, 64
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Target size: [{X_out}, {Y_out}, {U_out}, {V_out}]")
    
    # Forward pass
    output = interpolate_4d(input_tensor, size=[X_out, Y_out, U_out, V_out], align_corners=True)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (N, C, X_out, Y_out, U_out, V_out), f"Shape mismatch!"
    
    # Test with requires_grad
    input_tensor.requires_grad_(True)
    output = interpolate_4d(input_tensor, size=[X_out, Y_out, U_out, V_out])
    loss = output.sum()
    loss.backward()
    
    assert input_tensor.grad is not None, "Gradient not computed!"
    print(f"  Gradient computed: {input_tensor.grad.shape}")
    print("  ✓ interpolate_4d PASSED")
    print()
    return True


def test_interpolate_4d_downscale():
    """Test 4D interpolation downscaling"""
    print("-" * 40)
    print("Test 3: interpolate_4d (downscale)")
    print("-" * 40)
    
    N, C = 1, 8
    X_in, Y_in, U_in, V_in = 8, 8, 64, 64
    input_tensor = torch.randn(N, C, X_in, Y_in, U_in, V_in, device='cuda', dtype=torch.float32)
    
    X_out, Y_out, U_out, V_out = 4, 4, 32, 32
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Target size: [{X_out}, {Y_out}, {U_out}, {V_out}]")
    
    output = interpolate_4d(input_tensor, size=[X_out, Y_out, U_out, V_out])
    print(f"  Output shape: {output.shape}")
    assert output.shape == (N, C, X_out, Y_out, U_out, V_out)
    print("  ✓ interpolate_4d (downscale) PASSED")
    print()
    return True


def test_grid4d_class():
    """Test Grid4D class integration"""
    print("-" * 40)
    print("Test 4: Grid4D class")
    print("-" * 40)
    
    from lib.grid import create_grid
    
    # Create Grid4D
    channels = 16
    world_size = [8, 8, 64, 64]
    xyuv_min = [0.0, 0.0, 0.0, 0.0]
    xyuv_max = [1.0, 1.0, 1.0, 1.0]
    config = {}
    
    grid = create_grid('4D', 
                       channels=channels, 
                       world_size=world_size, 
                       xyuv_min=xyuv_min, 
                       xyuv_max=xyuv_max, 
                       config=config)
    grid = grid.cuda()
    
    print(f"  Grid4D created: {grid}")
    print(f"  xyuv_grid shape: {grid.xyuv_grid.shape}")
    
    # Test forward pass
    N_pts = 500
    xyuv_coords = torch.rand(N_pts, 4, device='cuda')  # [0, 1] coordinates
    xyuv_coords = xyuv_coords * (torch.tensor(xyuv_max, device='cuda') - torch.tensor(xyuv_min, device='cuda')) + torch.tensor(xyuv_min, device='cuda')
    
    output = grid(xyuv_coords)
    print(f"  Query coords shape: {xyuv_coords.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (N_pts, channels), f"Expected {(N_pts, channels)}, got {output.shape}"
    
    # Test scale_volume_grid
    new_world_size = [16, 16, 128, 128]
    grid.scale_volume_grid(new_world_size)
    print(f"  After scaling: xyuv_grid shape = {grid.xyuv_grid.shape}")
    assert grid.xyuv_grid.shape == (1, channels, 16, 16, 128, 128)
    
    print("  ✓ Grid4D class PASSED")
    print()
    return True


def test_performance():
    """Benchmark performance"""
    print("-" * 40)
    print("Test 5: Performance benchmark")
    print("-" * 40)
    
    N, C = 1, 32
    X, Y, U, V = 8, 8, 64, 64
    input_tensor = torch.randn(N, C, X, Y, U, V, device='cuda', dtype=torch.float32)
    
    N_pts = 100000
    grid = torch.rand(N, N_pts, 4, device='cuda', dtype=torch.float32) * 2 - 1
    
    # Warmup
    for _ in range(3):
        _ = grid_sample_4d(input_tensor, grid)
    torch.cuda.synchronize()
    
    # Benchmark grid_sample_4d
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        _ = grid_sample_4d(input_tensor, grid)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  grid_sample_4d: {elapsed/n_iters*1000:.3f} ms per call ({N_pts} points)")
    
    # Benchmark interpolate_4d
    for _ in range(3):
        _ = interpolate_4d(input_tensor, size=[16, 16, 128, 128])
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_iters):
        _ = interpolate_4d(input_tensor, size=[16, 16, 128, 128])
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  interpolate_4d: {elapsed/n_iters*1000:.3f} ms per call (8x8x64x64 -> 16x16x128x128)")
    
    print("  ✓ Performance benchmark PASSED")
    print()
    return True


def main():
    all_passed = True
    
    try:
        all_passed &= test_grid_sample_4d()
    except Exception as e:
        print(f"  ✗ grid_sample_4d FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_interpolate_4d()
    except Exception as e:
        print(f"  ✗ interpolate_4d FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_interpolate_4d_downscale()
    except Exception as e:
        print(f"  ✗ interpolate_4d (downscale) FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_grid4d_class()
    except Exception as e:
        print(f"  ✗ Grid4D class FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_performance()
    except Exception as e:
        print(f"  ✗ Performance benchmark FAILED: {e}")
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()



