"""
Light Field Feature Plane Compression Tool

This script compresses Light Field feature planes (6 planes: xy, uv, xu, xv, yu, yv)
into video format for efficient storage and transmission.

Unlike NeRF-based compression, Light Field doesn't have density grids.

This version combines all 6 planes into a single image for single-video compression.
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


def save_feature_visualization(plane_data, plane_name, frameid, vis_dir, save_channels=False):
    """
    Save feature plane visualization images.
    
    Args:
        plane_data: tensor of shape [1, C, H, W]
        plane_name: name of the plane (xy, uv, xu, xv, yu, yv)
        frameid: frame index
        vis_dir: directory to save visualization images
        save_channels: if True, save individual channels as separate images
    """
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get numpy data
    data = plane_data.squeeze(0).cpu().numpy()  # [C, H, W]
    num_channels = data.shape[0]
    
    # Save tiled visualization (all channels in one image)
    tiled = tile_maker(plane_data).cpu().numpy()
    
    # Save 8-bit grayscale version
    tiled_8b = to8b(tiled)
    cv2.imwrite(os.path.join(vis_dir, f'{plane_name}_tiled_frame_{frameid+1}.png'), tiled_8b)
    
    # Save colormap version for better visualization
    tiled_color = apply_colormap(tiled)
    cv2.imwrite(os.path.join(vis_dir, f'{plane_name}_tiled_color_frame_{frameid+1}.png'), tiled_color)
    
    # Save individual channels if requested
    if save_channels:
        channel_dir = os.path.join(vis_dir, f'channels_frame_{frameid+1}', plane_name)
        os.makedirs(channel_dir, exist_ok=True)
        
        for ch_idx in range(num_channels):
            ch_img = data[ch_idx]  # [H, W]
            
            # 8-bit grayscale
            ch_8b = to8b(ch_img)
            cv2.imwrite(os.path.join(channel_dir, f'ch_{ch_idx:02d}_gray.png'), ch_8b)
            
            # Colormap version
            ch_color = apply_colormap(ch_img)
            cv2.imwrite(os.path.join(channel_dir, f'ch_{ch_idx:02d}_color.png'), ch_color)


def tile_maker(feat_plane, h=2160, w=3840):
    """
    Tile multiple feature channels into a single 2D image.
    
    Args:
        feat_plane: tensor of shape [1, C, H, W]
        h, w: output image size
    
    Returns:
        image: tensor of shape [h, w]
    """
    image = torch.zeros(h, w)
    
    plane_h, plane_w = list(feat_plane.size())[-2:]
    
    x, y = 0, 0
    for i in range(feat_plane.size(1)):
        if y + plane_w >= image.size(1):
            y = 0
            x = x + plane_h
        assert x + plane_h < image.size(0), f"Tile_maker: not enough space. Need more than {x + plane_h} rows, but only have {image.size(0)}"
        
        image[x:x+plane_h, y:y+plane_w] = feat_plane[0, i, :, :]
        y = y + plane_w
    
    return image


def combine_plane_tiles(plane_tiles, layout=(3, 2), tile_h=2160, tile_w=3840):
    """
    Combine multiple plane tile images into a single large image.
    
    Args:
        plane_tiles: dict of plane_name -> tile tensor (shape [h, w])
        layout: (rows, cols) layout for combining tiles
        tile_h, tile_w: size of each tile
    
    Returns:
        combined: tensor of shape [rows * tile_h, cols * tile_w]
        tile_positions: dict mapping plane_name to (row, col) position
    """
    rows, cols = layout
    combined = torch.zeros(rows * tile_h, cols * tile_w)
    
    # Define tile positions (3x2 layout):
    # Row 0: xy, uv
    # Row 1: xu, xv
    # Row 2: yu, yv
    tile_order = ['xy', 'uv', 'xu', 'xv', 'yu', 'yv']
    tile_positions = {}
    
    for idx, plane_name in enumerate(tile_order):
        row = idx // cols
        col = idx % cols
        tile_positions[plane_name] = (row, col)
        
        if plane_name in plane_tiles:
            tile = plane_tiles[plane_name]
            r_start = row * tile_h
            r_end = r_start + tile_h
            c_start = col * tile_w
            c_end = c_start + tile_w
            combined[r_start:r_end, c_start:c_end] = tile
    
    return combined, tile_positions


def split_combined_image(combined, layout=(3, 2), tile_h=2160, tile_w=3840):
    """
    Split a combined image back into individual plane tiles.
    
    Args:
        combined: tensor of shape [rows * tile_h, cols * tile_w]
        layout: (rows, cols) layout of the combined image
        tile_h, tile_w: size of each tile
    
    Returns:
        plane_tiles: dict of plane_name -> tile tensor (shape [tile_h, tile_w])
    """
    rows, cols = layout
    tile_order = ['xy', 'uv', 'xu', 'xv', 'yu', 'yv']
    plane_tiles = {}
    
    for idx, plane_name in enumerate(tile_order):
        row = idx // cols
        col = idx % cols
        
        r_start = row * tile_h
        r_end = r_start + tile_h
        c_start = col * tile_w
        c_end = c_start + tile_w
        
        plane_tiles[plane_name] = combined[r_start:r_end, c_start:c_end].clone()
    
    return plane_tiles


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True,
                        help='checkpoint directory path')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames')
    
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

    # Tile layout: 3 rows x 2 cols
    TILE_LAYOUT = (3, 2)
    TILE_H = 2160
    TILE_W = 3840

    # 16비트 PNG 이미지 저장 폴더 (인코딩 전 원본) - 단일 이미지용
    frames_dir = os.path.join(args.logdir, f'frames_1plane_{args.qp}')
    # 압축된 비디오 저장 폴더
    compressed_dir = os.path.join(args.logdir, f'compressed_1plane_{args.qp}')

    # Light Field plane names (6 planes)
    LF_PLANE_NAMES = ['xy', 'uv', 'xu', 'xv', 'yu', 'yv']

    for frameid in tqdm(range(0, args.numframe)):
        
        # Check if already processed
        if os.path.isfile(os.path.join(frames_dir, f'combined_frame_{frameid+1}.png')):
            continue

        tmp_file = os.path.join(args.logdir, f'fine_last_{frameid}.tar')

        if not os.path.isfile(tmp_file):
            tqdm.write(f"Checkpoint not found: {tmp_file}, skipping...")
            continue

        tqdm.write(f"Loading Checkpoint {tmp_file}")
        ckpt = torch.load(tmp_file, map_location='cpu', weights_only=False)
        
        # Load planes from checkpoint
        planes = {}
        for key in ckpt['model_state_dict'].keys():
            if 'k0' in key and 'plane' in key:
                # Extract plane name (e.g., 'k0.xy_plane' -> 'xy_plane')
                plane_name = key.split('.')[-1]
                if plane_name.replace('_plane', '') in LF_PLANE_NAMES:
                    planes[plane_name] = ckpt['model_state_dict'][key]

        # Debug: print loaded planes
        tqdm.write(f"Loaded planes: {list(planes.keys())}")
        
        if len(planes) != 6:
            tqdm.write(f"Warning: Expected 6 planes, got {len(planes)}")
        
        plane_data = {}
        ratios = []
        tpsnr = []
        plane_sizes = {}
        
        for p in LF_PLANE_NAMES:
            plane_key = f"{p}_plane"
            
            if plane_key not in planes:
                tqdm.write(f"Warning: {plane_key} not found in checkpoint")
                continue
                
            raw_plane = planes[plane_key]
            plane_size = list(raw_plane.size())
            plane_sizes[plane_key] = plane_size
            
            # Debug: feature plane value range
            tqdm.write(f"  {plane_key}: shape={plane_size}, min={raw_plane.min().item():.4f}, max={raw_plane.max().item():.4f}, mean={raw_plane.mean().item():.4f}")
            
            # Sanity check: check if values are within bound
            ra = raw_plane.abs().reshape(-1)
            ratio = (ra < bound_thres).sum() / ra.size(0)
            if ratio < 0.95:
                tqdm.write(f"Warning: {plane_key} has {(1-ratio.item())*100:.2f}% values outside [-{bound_thres}, {bound_thres}]")
            ratios.append(ratio.item())
            
            # Normalize to [0, 1]
            feat = (raw_plane - low_bound) / (high_bound - low_bound)
            
            # Quantize
            feat = torch.round(feat * nbits) / nbits
            
            # Clip to [0, 1]
            feat = torch.clamp(feat, 0, 1)
            
            # Debug: normalized value range
            tqdm.write(f"  {plane_key} normalized: min={feat.min().item():.4f}, max={feat.max().item():.4f}, mean={feat.mean().item():.4f}")
            
            plane_data[p] = feat
            
            # Calculate PSNR
            gt_feat = (raw_plane - low_bound) / (high_bound - low_bound)
            tpsnr.append(psnr(gt_feat, feat).item())

        tqdm.write(f"ratio: {np.mean(ratios)*100:.2f}%   PSNR:{np.mean(tpsnr):.2f} qp:{args.qp}")
        os.makedirs(frames_dir, exist_ok=True)

        # Create visualization directory if needed
        vis_dir = None
        if args.save_vis or args.save_channels:
            vis_dir = os.path.join(args.logdir, f'visualization_1plane_{args.qp}')
            os.makedirs(vis_dir, exist_ok=True)

        # Create individual tile images for each plane
        plane_tiles = {}
        for plane_name, plane in plane_data.items():
            tile = tile_maker(plane, TILE_H, TILE_W)
            plane_tiles[plane_name] = tile
            
            # Save visualization images if requested
            if args.save_vis or args.save_channels:
                save_feature_visualization(plane, plane_name, frameid, vis_dir, save_channels=args.save_channels)

        # Combine all plane tiles into a single image
        combined_img, tile_positions = combine_plane_tiles(plane_tiles, TILE_LAYOUT, TILE_H, TILE_W)
        
        # Save combined image as 16-bit PNG
        cv2.imwrite(os.path.join(frames_dir, f'combined_frame_{frameid+1}.png'), to16b(combined_img.cpu().numpy()))
        
        # Save visualization of combined image if requested
        if args.save_vis:
            cv2.imwrite(os.path.join(vis_dir, f'combined_frame_{frameid+1}_8bit.png'), to8b(combined_img.cpu().numpy()))
            cv2.imwrite(os.path.join(vis_dir, f'combined_frame_{frameid+1}_color.png'), apply_colormap(combined_img.cpu().numpy()))

        # Save metadata (only once)
        if frameid == 0 or not os.path.isfile(os.path.join(frames_dir, f'planes_frame_meta.nf')):
            torch.save({
                'plane_size': plane_sizes,
                'bounds': (low_bound, high_bound),
                'nbits': nbits,
                'qp': args.qp,
                'plane_names': LF_PLANE_NAMES,
                'tile_layout': TILE_LAYOUT,
                'tile_h': TILE_H,
                'tile_w': TILE_W,
                'tile_positions': tile_positions,
                'combined_mode': True,  # Flag to indicate single-video mode
            }, os.path.join(frames_dir, f'planes_frame_meta.nf'))

    # Generate video compression script
    os.makedirs(compressed_dir, exist_ok=True)
    
    filename = '/dev/shm/planes_to_videos_lf.sh'
    with open(filename, 'w') as f:
        # Compress combined frames to single video
        if args.codec == 'h265':
            f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/combined_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {compressed_dir}/combined_planes.mp4\n")
        elif args.codec == 'mpg2':
            f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/combined_frame_%d.png -c:v mpeg2video -color_range pc -qscale:v {args.qp} {compressed_dir}/combined_planes.mpg\n")
        
        # Copy metadata and network files
        f.write(f'cp {frames_dir}/planes_frame_meta.nf {compressed_dir}/\n')
        f.write(f'cp {args.logdir}/rgbnet* {compressed_dir}/ 2>/dev/null || true\n')
        f.write(f'cp {args.logdir}/config.py {compressed_dir}/ 2>/dev/null || true\n')
        
        # Create compressed archive
        f.write(f'cd {args.logdir}\n')
        f.write(f'zip -r compressed_1plane.zip compressed_1plane_{args.qp}\n')

    os.system(f"bash {filename}")
    
    print(f"\n=== 완료 ===")
    print(f"프레임 이미지 (16-bit PNG): {frames_dir}")
    print(f"압축 비디오: {compressed_dir}")
    print(f"압축 방식: 단일 비디오 (6개 plane을 {TILE_LAYOUT[0]}x{TILE_LAYOUT[1]} 레이아웃으로 조합)")
    print(f"결합 이미지 크기: {TILE_LAYOUT[0] * TILE_H} x {TILE_LAYOUT[1] * TILE_W}")
    print(f"압축된 plane: {', '.join(LF_PLANE_NAMES)}")
