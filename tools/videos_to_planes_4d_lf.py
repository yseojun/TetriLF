"""
Light Field 4D Grid Decompression Tool

This script decompresses Grid4D's xyuv_grid tensor [1, R, X, Y, U, V]
from video format back to checkpoint format.

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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Timer:
    """Simple timer for measuring elapsed time."""
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, name):
        self.start_times[name] = time.perf_counter()
    
    def stop(self, name):
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        elapsed = time.perf_counter() - self.start_times[name]
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        return elapsed
    
    def get_total(self, name):
        if name in self.timings:
            return sum(self.timings[name])
        return 0
    
    def get_avg(self, name):
        if name in self.timings and len(self.timings[name]) > 0:
            return sum(self.timings[name]) / len(self.timings[name])
        return 0
    
    def get_count(self, name):
        if name in self.timings:
            return len(self.timings[name])
        return 0


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_val = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_val


def untile_image_4d(image, U, V, R):
    """
    Untile a 2D image back to multiple feature channels.
    
    Args:
        image: tensor of shape [H, W] (tiled image)
        U: height of each tile
        V: width of each tile
        R: number of channels (tiles)
    
    Returns:
        features: tensor of shape [R, U, V]
    """
    features = torch.zeros(R, U, V)
    
    x, y = 0, 0
    for i in range(R):
        if y + V >= image.size(1):
            y = 0
            x = x + U
        assert x + U < image.size(0), f"untile_image_4d: too many feature maps. x={x}, U={U}, image.size(0)={image.size(0)}"
        
        features[i, :, :] = image[x:x+U, y:y+V]
        y = y + V
    
    return features


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', required=True,
                        help='compressed files directory path')

    parser.add_argument('--model_template', type=str, default='fine_last_0.tar',
                        help='model template checkpoint for structure reference')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames (checkpoints)')

    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')

    parser.add_argument("--no_wandb", action='store_true',
                        help='disable wandb logging')

    args = parser.parse_args()

    # Output directory for restored checkpoints
    outdir = os.path.join(args.dir, '..', 'raw_out')
    os.makedirs(outdir, exist_ok=True)

    # Load template checkpoint for structure
    template_path = os.path.join(args.dir, '..', args.model_template)
    if not os.path.isfile(template_path):
        # Try to find any fine_last checkpoint
        parent_dir = os.path.join(args.dir, '..')
        fine_files = [f for f in os.listdir(parent_dir) if f.startswith('fine_last_') and f.endswith('.tar')]
        if fine_files:
            template_path = os.path.join(parent_dir, sorted(fine_files)[0])
        else:
            raise FileNotFoundError(f"No template checkpoint found in {parent_dir}")
    
    print(f"Loading template checkpoint: {template_path}")
    ckpt = torch.load(template_path, weights_only=False)

    # Load metadata
    meta_file = os.path.join(args.dir, 'grid_frame_meta.nf')
    assert os.path.isfile(meta_file), f"Metadata file not found: {meta_file}"
    meta = torch.load(meta_file, weights_only=False)
    
    low_bound, high_bound = meta['bounds']
    grid_size = meta['grid_size']  # [1, R, X, Y, U, V]
    all_grid_sizes = meta.get('all_grid_sizes', {})
    qp = meta.get('qp', 0)
    grid_type = meta.get('grid_type', '4D')
    frame_order = meta.get('frame_order', 'y_outer')
    
    _, R, X, Y, U, V = grid_size
    total_frames_per_ckpt = X * Y
    
    print(f"Metadata loaded:")
    print(f"  Bounds: [{low_bound}, {high_bound}]")
    print(f"  QP: {qp}")
    print(f"  Grid type: {grid_type}")
    print(f"  Grid size: [1, R={R}, X={X}, Y={Y}, U={U}, V={V}]")
    print(f"  Frame order: {frame_order}")
    print(f"  Total frames per checkpoint: {total_frames_per_ckpt}")

    name = args.dir.split('/')[-2]
    wandbrun = None
    if WANDB_AVAILABLE and not args.no_wandb:
        wandbrun = wandb.init(
            project="TeTriRF_LF",
            resume="allow",
            id='compression4DLF_' + name + '_' + args.codec,
        )

    # Initialize timer
    timer = Timer()

    # Decode videos to PNG frames using ffmpeg
    filename = '/dev/shm/videos_to_planes_4d_lf.sh'
    with open(filename, 'w') as f:
        f.write(f'cd {args.dir}\n')
        for frameid in range(args.numframe):
            if args.codec == 'mpg2':
                f.write(f"ffmpeg -y -i xyuv_grid_ckpt{frameid+1}.mpg -pix_fmt gray16be xyuv_ckpt{frameid+1}_frame_%d_out.png\n")
            else:
                f.write(f"ffmpeg -y -i xyuv_grid_ckpt{frameid+1}.mp4 -pix_fmt gray16be xyuv_ckpt{frameid+1}_frame_%d_out.png\n")
    
    print("Decoding videos to frames...")
    timer.start('video_decoding')
    os.system(f"bash {filename}")
    t_decode = timer.stop('video_decoding')
    print(f"  Video decoding time: {t_decode*1000:.2f} ms ({t_decode:.4f} s)")

    # Process each checkpoint
    print("\nRestoring checkpoints from decoded images...")
    timer.start('total_reconstruction')
    
    for frameid in tqdm(range(0, args.numframe)):
        
        timer.start('checkpoint_reconstruction')
        
        # Get grid size for this checkpoint (may differ due to progressive growing)
        if frameid in all_grid_sizes:
            ckpt_grid_size = all_grid_sizes[frameid]
        else:
            ckpt_grid_size = grid_size
        
        _, ckpt_R, ckpt_X, ckpt_Y, ckpt_U, ckpt_V = ckpt_grid_size
        ckpt_total_frames = ckpt_X * ckpt_Y
        
        # Check if first frame exists
        first_frame_path = os.path.join(args.dir, f"xyuv_ckpt{frameid+1}_frame_1_out.png")
        if not os.path.isfile(first_frame_path):
            tqdm.write(f"Warning: Frame file not found: {first_frame_path}, skipping checkpoint {frameid}")
            continue
        
        # Create a copy of the template checkpoint
        frame_ckpt = copy.deepcopy(ckpt)
        
        # Initialize empty xyuv_grid
        xyuv_grid = torch.zeros(1, ckpt_R, ckpt_X, ckpt_Y, ckpt_U, ckpt_V)
        
        success = True
        
        # Update model_kwargs world_size based on restored grid sizes
        if 'model_kwargs' in frame_ckpt and 'world_size' in frame_ckpt['model_kwargs']:
            new_world_size = [ckpt_X, ckpt_Y, ckpt_U, ckpt_V]
            old_world_size = frame_ckpt['model_kwargs']['world_size']
            
            if old_world_size != new_world_size:
                tqdm.write(f"  Updating model_kwargs world_size: {old_world_size} -> {new_world_size}")
                frame_ckpt['model_kwargs']['world_size'] = new_world_size
        
        # Restore grid from frames
        timer.start('grid_restore')
        
        for frame_idx_1based in tqdm(range(1, ckpt_total_frames + 1), desc=f"Restoring frames", leave=False):
            img_path = os.path.join(args.dir, f"xyuv_ckpt{frameid+1}_frame_{frame_idx_1based}_out.png")
            
            quant_img = cv2.imread(img_path, -1)
            
            if quant_img is None:
                tqdm.write(f"Warning: Could not read {img_path}, skipping frame")
                success = False
                break
            
            # Convert frame index to (x, y) coordinates
            # Frame order: y is outer loop (frame_idx = y * X + x)
            frame_idx_0based = frame_idx_1based - 1
            x_idx = frame_idx_0based % ckpt_X
            y_idx = frame_idx_0based // ckpt_X
            
            # Untile image to get R channels of U×V
            channel_data = untile_image_4d(
                torch.tensor(quant_img.astype(np.float32)) / int(2**16 - 1),
                ckpt_U,
                ckpt_V,
                ckpt_R
            )
            
            # Dequantize: convert from [0,1] back to original range
            channel_data = channel_data * (high_bound - low_bound) + low_bound
            
            # Assign to xyuv_grid
            xyuv_grid[0, :, x_idx, y_idx, :, :] = channel_data
        
        t_grid = timer.stop('grid_restore')
        
        if not success:
            timer.stop('checkpoint_reconstruction')
            continue
        
        # Update checkpoint with restored grid
        state_dict_key = 'k0.xyuv_grid'
        if state_dict_key not in frame_ckpt['model_state_dict']:
            tqdm.write(f"Warning: {state_dict_key} not in model_state_dict, trying to find correct key...")
            # Find the correct key
            found_key = None
            for key in frame_ckpt['model_state_dict'].keys():
                if 'xyuv_grid' in key:
                    found_key = key
                    break
            if found_key:
                state_dict_key = found_key
            else:
                tqdm.write(f"Error: Could not find xyuv_grid key in model_state_dict")
                timer.stop('checkpoint_reconstruction')
                continue
        
        frame_ckpt['model_state_dict'][state_dict_key] = xyuv_grid.clone()
        
        tqdm.write(f"  Restored xyuv_grid: shape={list(xyuv_grid.size())}, min={xyuv_grid.min().item():.4f}, max={xyuv_grid.max().item():.4f}")
        
        # Save restored checkpoint
        timer.start('checkpoint_save')
        output_path = os.path.join(outdir, f'fine_last_{frameid}.tar')
        torch.save(frame_ckpt, output_path)
        t_save = timer.stop('checkpoint_save')
        
        t_ckpt = timer.stop('checkpoint_reconstruction')
        tqdm.write(f"Saved restored model: {output_path} (grid: {t_grid*1000:.1f}ms, save: {t_save*1000:.1f}ms, total: {t_ckpt*1000:.1f}ms)")
    
    t_total_recon = timer.stop('total_reconstruction')

    # Copy rgbnet files to output directory
    rgbnet_files = glob.glob(os.path.join(args.dir, 'rgbnet*.tar'))
    for rgbnet_file in rgbnet_files:
        dest = os.path.join(outdir, os.path.basename(rgbnet_file))
        copyfile(rgbnet_file, dest)
        print(f"Copied {rgbnet_file} -> {dest}")

    # Copy config file if exists
    config_file = os.path.join(args.dir, 'config.py')
    if os.path.isfile(config_file):
        copyfile(config_file, os.path.join(outdir, 'config.py'))
        print(f"Copied config.py")

    # ========================================
    # Timing Summary
    # ========================================
    print("\n" + "=" * 70)
    print("  TIMING SUMMARY")
    print("=" * 70)
    print(f"  {'Stage':<35} {'Total (s)':>12} {'Avg/Ckpt (ms)':>18}")
    print("-" * 70)
    
    # Video decoding (all checkpoints at once)
    t_decode_total = timer.get_total('video_decoding')
    t_decode_per_ckpt = t_decode_total / args.numframe * 1000 if args.numframe > 0 else 0
    print(f"  {'Video Decoding (ffmpeg)':<35} {t_decode_total:>12.4f} {t_decode_per_ckpt:>18.2f}")
    
    # Grid restoration
    t_grid_total = timer.get_total('grid_restore')
    t_grid_avg = timer.get_avg('grid_restore') * 1000
    print(f"  {'Grid Restoration (PNG->Tensor)':<35} {t_grid_total:>12.4f} {t_grid_avg:>18.2f}")
    
    # Checkpoint saving
    t_save_total = timer.get_total('checkpoint_save')
    t_save_avg = timer.get_avg('checkpoint_save') * 1000
    print(f"  {'Checkpoint Saving':<35} {t_save_total:>12.4f} {t_save_avg:>18.2f}")
    
    # Total reconstruction
    print("-" * 70)
    t_recon_total = timer.get_total('total_reconstruction')
    t_recon_per_ckpt = t_recon_total / args.numframe * 1000 if args.numframe > 0 else 0
    print(f"  {'Total Reconstruction':<35} {t_recon_total:>12.4f} {t_recon_per_ckpt:>18.2f}")
    
    # Grand total
    t_grand_total = t_decode_total + t_recon_total
    t_grand_per_ckpt = t_grand_total / args.numframe * 1000 if args.numframe > 0 else 0
    print("-" * 70)
    print(f"  {'GRAND TOTAL':<35} {t_grand_total:>12.4f} {t_grand_per_ckpt:>18.2f}")
    print("=" * 70)
    
    print(f"\n=== 복원 완료 ===")
    print(f"복원된 체크포인트: {outdir}")
    print(f"Grid 크기: [1, R={R}, X={X}, Y={Y}, U={U}, V={V}]")
    print(f"처리된 체크포인트 수: {timer.get_count('checkpoint_reconstruction')}")

