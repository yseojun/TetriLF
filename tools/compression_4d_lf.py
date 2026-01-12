"""
Light Field 4D Grid Compression Tool

This script compresses Grid4D's xyuv_grid tensor [1, R, X, Y, U, V]
into video format for efficient storage and transmission.

For each (x, y) coordinate, R channels of U×V images are tiled and saved as frames.
Frame order: y is outer loop (frame_idx = y * X + x)
"""

import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_val = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_val

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : ((2**16-1)*np.clip(x,0,1)).astype(np.uint16)


def apply_colormap(img, colormap=cv2.COLORMAP_VIRIDIS):
    """Apply colormap to grayscale image for better visualization."""
    img_8b = to8b(img)
    return cv2.applyColorMap(img_8b, colormap)


def tile_maker_4d(channel_data, h=2160, w=3840):
    """
    Tile multiple feature channels into a single 2D image.
    
    Args:
        channel_data: tensor of shape [R, U, V] - R channels of U×V images
        h, w: output image size
    
    Returns:
        image: tensor of shape [h, w]
    """
    image = torch.zeros(h, w)
    
    R, U, V = channel_data.shape
    
    x, y = 0, 0
    for i in range(R):
        if y + V >= image.size(1):
            y = 0
            x = x + U
        assert x + U < image.size(0), f"tile_maker_4d: not enough space. Need more than {x + U} rows, but only have {image.size(0)}"
        
        image[x:x+U, y:y+V] = channel_data[i, :, :]
        y = y + V
    
    return image


def save_feature_visualization(channel_data, x_idx, y_idx, frameid, vis_dir, save_channels=False):
    """
    Save feature visualization images.
    
    Args:
        channel_data: tensor of shape [R, U, V]
        x_idx, y_idx: spatial indices
        frameid: frame index (checkpoint index)
        vis_dir: directory to save visualization images
        save_channels: if True, save individual channels as separate images
    """
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get numpy data
    data = channel_data.cpu().numpy()  # [R, U, V]
    R = data.shape[0]
    
    # Save tiled visualization (all channels in one image)
    tiled = tile_maker_4d(channel_data).cpu().numpy()
    
    # Save 8-bit grayscale version
    tiled_8b = to8b(tiled)
    cv2.imwrite(os.path.join(vis_dir, f'xyuv_x{x_idx}_y{y_idx}_frame_{frameid+1}.png'), tiled_8b)
    
    # Save colormap version for better visualization
    tiled_color = apply_colormap(tiled)
    cv2.imwrite(os.path.join(vis_dir, f'xyuv_x{x_idx}_y{y_idx}_color_frame_{frameid+1}.png'), tiled_color)
    
    # Save individual channels if requested
    if save_channels:
        channel_dir = os.path.join(vis_dir, f'channels_frame_{frameid+1}', f'x{x_idx}_y{y_idx}')
        os.makedirs(channel_dir, exist_ok=True)
        
        for ch_idx in range(R):
            ch_img = data[ch_idx]  # [U, V]
            
            # 8-bit grayscale
            ch_8b = to8b(ch_img)
            cv2.imwrite(os.path.join(channel_dir, f'ch_{ch_idx:02d}_gray.png'), ch_8b)
            
            # Colormap version
            ch_color = apply_colormap(ch_img)
            cv2.imwrite(os.path.join(channel_dir, f'ch_{ch_idx:02d}_color.png'), ch_color)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True,
                        help='checkpoint directory path')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames (checkpoints)')
    
    parser.add_argument("--qp", type=int, default=20,
                        help='qp value for video codec')

    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')

    parser.add_argument("--save_vis", action='store_true',
                        help='Save visualization images (8-bit) for viewing')

    parser.add_argument("--save_channels", action='store_true',
                        help='Save individual feature channels as separate images')

    args = parser.parse_args()

    # Quantization parameters
    bound_thres = 20
    low_bound = -bound_thres
    high_bound = bound_thres
    nbits = 2**16 - 1

    if args.logdir[-1] == '/':
        args.logdir = args.logdir[:-1]
    name = args.logdir.split('/')[-1]

    # 16비트 PNG 이미지 저장 폴더 (인코딩 전 원본)
    frames_dir = os.path.join(args.logdir, f'frames_4d_{args.qp}')
    # 압축된 비디오 저장 폴더
    compressed_dir = os.path.join(args.logdir, f'compressed_4d_{args.qp}')

    # Store grid sizes for each checkpoint
    all_grid_sizes = {}

    for frameid in tqdm(range(0, args.numframe)):
        
        tmp_file = os.path.join(args.logdir, f'fine_last_{frameid}.tar')

        if not os.path.isfile(tmp_file):
            tqdm.write(f"Checkpoint not found: {tmp_file}, skipping...")
            continue

        # Check if already processed (check first frame of this checkpoint)
        first_frame_file = os.path.join(frames_dir, f'xyuv_ckpt{frameid+1}_frame_1.png')
        if os.path.isfile(first_frame_file):
            tqdm.write(f"Already processed checkpoint {frameid}, skipping...")
            # Still need to load grid size for metadata
            ckpt = torch.load(tmp_file, map_location='cpu', weights_only=False)
            for key in ckpt['model_state_dict'].keys():
                if 'k0' in key and 'xyuv_grid' in key:
                    all_grid_sizes[frameid] = list(ckpt['model_state_dict'][key].size())
                    break
            continue

        tqdm.write(f"Loading Checkpoint {tmp_file}")
        ckpt = torch.load(tmp_file, map_location='cpu', weights_only=False)
        
        # Load xyuv_grid from checkpoint
        xyuv_grid = None
        for key in ckpt['model_state_dict'].keys():
            if 'k0' in key and 'xyuv_grid' in key:
                xyuv_grid = ckpt['model_state_dict'][key]
                tqdm.write(f"Found xyuv_grid: key={key}, shape={list(xyuv_grid.size())}")
                break

        if xyuv_grid is None:
            tqdm.write(f"Warning: xyuv_grid not found in checkpoint, skipping...")
            continue
        
        # xyuv_grid shape: [1, R, X, Y, U, V]
        grid_size = list(xyuv_grid.size())
        all_grid_sizes[frameid] = grid_size
        _, R, X, Y, U, V = grid_size
        
        tqdm.write(f"  Grid size: [1, R={R}, X={X}, Y={Y}, U={U}, V={V}]")
        tqdm.write(f"  Total frames for this checkpoint: {X * Y}")
        
        # Debug: feature grid value range
        tqdm.write(f"  Value range: min={xyuv_grid.min().item():.4f}, max={xyuv_grid.max().item():.4f}, mean={xyuv_grid.mean().item():.4f}")
        
        # Sanity check: check if values are within bound
        ra = xyuv_grid.abs().reshape(-1)
        ratio = (ra < bound_thres).sum() / ra.size(0)
        if ratio < 0.95:
            tqdm.write(f"Warning: xyuv_grid has {(1-ratio.item())*100:.2f}% values outside [-{bound_thres}, {bound_thres}]")
        
        os.makedirs(frames_dir, exist_ok=True)
        
        # Create visualization directory if needed
        vis_dir = None
        if args.save_vis or args.save_channels:
            vis_dir = os.path.join(args.logdir, f'visualization_4d_{args.qp}')
            os.makedirs(vis_dir, exist_ok=True)
        
        tpsnr = []
        
        # Process each (x, y) coordinate
        # Frame order: y is outer loop (frame_idx = y * X + x)
        frame_count = 0
        for y_idx in tqdm(range(Y), desc=f"Processing Y", leave=False):
            for x_idx in range(X):
                # Extract R channels of U×V for this (x, y)
                # xyuv_grid: [1, R, X, Y, U, V]
                raw_channel = xyuv_grid[0, :, x_idx, y_idx, :, :]  # [R, U, V]
                
                # Normalize to [0, 1]
                feat = (raw_channel - low_bound) / (high_bound - low_bound)
                
                # Quantize
                feat = torch.round(feat * nbits) / nbits
                
                # Clip to [0, 1]
                feat = torch.clamp(feat, 0, 1)
                
                # Calculate PSNR
                gt_feat = (raw_channel - low_bound) / (high_bound - low_bound)
                tpsnr.append(psnr(gt_feat, feat).item())
                
                # Tile the R channels into a single image
                img = tile_maker_4d(feat)
                
                # Save as 16-bit PNG
                # Frame numbering: 1-indexed for ffmpeg compatibility
                frame_idx = y_idx * X + x_idx + 1  # 1-indexed
                cv2.imwrite(os.path.join(frames_dir, f'xyuv_ckpt{frameid+1}_frame_{frame_idx}.png'), to16b(img.cpu().numpy()))
                
                # Save visualization images
                if args.save_vis or args.save_channels:
                    save_feature_visualization(feat, x_idx, y_idx, frameid, vis_dir, save_channels=args.save_channels)
                
                frame_count += 1
        
        tqdm.write(f"  Saved {frame_count} frames, avg PSNR: {np.mean(tpsnr):.2f} dB")
    
    # Save metadata
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get representative grid size (from first available checkpoint)
    if all_grid_sizes:
        first_frameid = min(all_grid_sizes.keys())
        representative_grid_size = all_grid_sizes[first_frameid]
    else:
        tqdm.write("Error: No checkpoints processed, cannot save metadata")
        sys.exit(1)
    
    torch.save({
        'grid_size': representative_grid_size,  # [1, R, X, Y, U, V]
        'all_grid_sizes': all_grid_sizes,  # Per-checkpoint grid sizes
        'bounds': (low_bound, high_bound),
        'nbits': nbits,
        'qp': args.qp,
        'grid_type': '4D',
        'frame_order': 'y_outer',  # y is outer loop: frame_idx = y * X + x
    }, os.path.join(frames_dir, f'grid_frame_meta.nf'))

    # Generate video compression script
    os.makedirs(compressed_dir, exist_ok=True)
    
    _, R, X, Y, U, V = representative_grid_size
    total_frames_per_ckpt = X * Y
    
    filename = '/dev/shm/planes_to_videos_4d_lf.sh'
    with open(filename, 'w') as f:
        # Compress each checkpoint's frames to a video
        for frameid in range(args.numframe):
            if frameid not in all_grid_sizes:
                continue
            
            if args.codec == 'h265':
                f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/xyuv_ckpt{frameid+1}_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {compressed_dir}/xyuv_grid_ckpt{frameid+1}.mp4\n")
            elif args.codec == 'mpg2':
                f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/xyuv_ckpt{frameid+1}_frame_%d.png -c:v mpeg2video -color_range pc -qscale:v {args.qp} {compressed_dir}/xyuv_grid_ckpt{frameid+1}.mpg\n")
        
        # Copy metadata and network files
        f.write(f'cp {frames_dir}/grid_frame_meta.nf {compressed_dir}/\n')
        f.write(f'cp {args.logdir}/rgbnet* {compressed_dir}/ 2>/dev/null || true\n')
        f.write(f'cp {args.logdir}/config.py {compressed_dir}/ 2>/dev/null || true\n')
        
        # Create compressed archive
        f.write(f'cd {args.logdir}\n')
        f.write(f'zip -r compressed_4d.zip compressed_4d_{args.qp}\n')

    os.system(f"bash {filename}")
    
    print(f"\n=== 완료 ===")
    print(f"프레임 이미지 (16-bit PNG): {frames_dir}")
    print(f"압축 비디오: {compressed_dir}")
    print(f"Grid 크기: [1, R={R}, X={X}, Y={Y}, U={U}, V={V}]")
    print(f"체크포인트당 프레임 수: {total_frames_per_ckpt}")

