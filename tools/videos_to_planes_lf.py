"""
Light Field Feature Plane Decompression Tool

This script decompresses Light Field feature planes (6 planes: xy, uv, xu, xv, yu, yv)
from video format back to checkpoint format.

Unlike NeRF-based decompression, Light Field doesn't have density grids.
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


def untile_image(image, h, w, ndim):
    """
    Untile a 2D image back to multiple feature channels.
    
    Args:
        image: tensor of shape [H, W] (tiled image)
        h: height of each tile
        w: width of each tile
        ndim: number of channels (tiles)
    
    Returns:
        features: tensor of shape [1, ndim, h, w]
    """
    features = torch.zeros(1, ndim, h, w)
    
    x, y = 0, 0
    for i in range(ndim):
        if y + w >= image.size(1):
            y = 0
            x = x + h
        assert x + h < image.size(0), f"untile_image: too many feature maps. x={x}, h={h}, image.size(0)={image.size(0)}"
        
        features[0, i, :, :] = image[x:x+h, y:y+w]
        y = y + w
    
    return features


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', required=True,
                        help='compressed files directory path')

    parser.add_argument('--model_template', type=str, default='fine_last_0.tar',
                        help='model template checkpoint for structure reference')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames')

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
    meta_file = os.path.join(args.dir, 'planes_frame_meta.nf')
    assert os.path.isfile(meta_file), f"Metadata file not found: {meta_file}"
    meta = torch.load(meta_file, weights_only=False)
    
    low_bound, high_bound = meta['bounds']
    plane_sizes = meta['plane_size']
    qp = meta.get('qp', 0)
    
    # Get plane names (default to LF planes if not in meta)
    LF_PLANE_NAMES = meta.get('plane_names', ['xy', 'uv', 'xu', 'xv', 'yu', 'yv'])
    
    print(f"Metadata loaded:")
    print(f"  Bounds: [{low_bound}, {high_bound}]")
    print(f"  QP: {qp}")
    print(f"  Planes: {LF_PLANE_NAMES}")
    print(f"  Plane sizes: {plane_sizes}")

    name = args.dir.split('/')[-2]
    wandbrun = None
    if WANDB_AVAILABLE and not args.no_wandb:
        wandbrun = wandb.init(
            project="TeTriRF_LF",
            resume="allow",
            id='compressionLF_' + name + '_' + args.codec,
        )

    # Initialize timer
    timer = Timer()

    # Decode videos to PNG frames using ffmpeg
    filename = '/dev/shm/videos_to_planes_lf.sh'
    with open(filename, 'w') as f:
        f.write(f'cd {args.dir}\n')
        for p in LF_PLANE_NAMES:
            if args.codec == 'mpg2':
                f.write(f"ffmpeg -y -i {p}_planes.mpg -pix_fmt gray16be {p}_planes_frame_%d_out.png\n")
            else:
                f.write(f"ffmpeg -y -i {p}_planes.mp4 -pix_fmt gray16be {p}_planes_frame_%d_out.png\n")
    
    print("Decoding videos to frames...")
    timer.start('video_decoding')
    os.system(f"bash {filename}")
    t_decode = timer.stop('video_decoding')
    print(f"  Video decoding time: {t_decode*1000:.2f} ms ({t_decode:.4f} s)")

    # Process each frame
    print("\nRestoring frames from decoded images...")
    timer.start('total_reconstruction')
    
    for frameid in tqdm(range(0, args.numframe)):
        
        timer.start('frame_reconstruction')
        
        # Create a copy of the template checkpoint
        frame_ckpt = copy.deepcopy(ckpt)
        
        success = True
        
        # Update model_kwargs world_size based on restored plane sizes
        # This is critical because Progressive Growing may have changed the world_size during training
        if 'model_kwargs' in frame_ckpt and 'world_size' in frame_ckpt['model_kwargs']:
            # Get world_size from plane_sizes metadata
            # xy_plane shape: [1, R, X, Y] -> world_size[0], world_size[1]
            # uv_plane shape: [1, R, U, V] -> world_size[2], world_size[3]
            xy_size = plane_sizes.get('xy_plane', [1, 16, 12, 12])
            uv_size = plane_sizes.get('uv_plane', [1, 16, 240, 120])
            
            new_world_size = [xy_size[2], xy_size[3], uv_size[2], uv_size[3]]
            old_world_size = frame_ckpt['model_kwargs']['world_size']
            
            if old_world_size != new_world_size:
                tqdm.write(f"  Updating model_kwargs world_size: {old_world_size} -> {new_world_size}")
                frame_ckpt['model_kwargs']['world_size'] = new_world_size
        
        # Restore feature planes
        timer.start('plane_restore')
        for p in LF_PLANE_NAMES:
            key = f'{p}_plane'
            img_path = os.path.join(args.dir, f"{p}_planes_frame_{frameid+1}_out.png")
            
            quant_img = cv2.imread(img_path, -1)
            
            if quant_img is None:
                tqdm.write(f"Warning: Could not read {img_path}, skipping frame {frameid}")
                success = False
                break

            # Get plane size from metadata
            if key not in plane_sizes:
                tqdm.write(f"Warning: {key} not in plane_sizes metadata")
                success = False
                break
                
            plane_size = plane_sizes[key]
            # plane_size format: [batch, channels, height, width] = [1, R, H, W]
            
            # Untile image to get feature plane
            plane = untile_image(
                torch.tensor(quant_img.astype(np.float32)) / int(2**16 - 1), 
                plane_size[2],  # height
                plane_size[3],  # width
                plane_size[1]   # num channels (R)
            )

            # Dequantize: convert from [0,1] back to original range
            plane = plane * (high_bound - low_bound) + low_bound

            # Update checkpoint
            state_dict_key = 'k0.' + key
            if state_dict_key not in frame_ckpt['model_state_dict']:
                tqdm.write(f"Warning: {state_dict_key} not in model_state_dict")
                success = False
                break
                
            frame_ckpt['model_state_dict'][state_dict_key] = plane.clone()
            
            tqdm.write(f"  Restored {key}: shape={list(plane.size())}, min={plane.min().item():.4f}, max={plane.max().item():.4f}")
        
        t_plane = timer.stop('plane_restore')

        if not success:
            continue

        # Save restored checkpoint
        timer.start('checkpoint_save')
        output_path = os.path.join(outdir, f'fine_last_{frameid}.tar')
        torch.save(frame_ckpt, output_path)
        t_save = timer.stop('checkpoint_save')
        
        t_frame = timer.stop('frame_reconstruction')
        tqdm.write(f"Saved restored model: {output_path} (plane: {t_plane*1000:.1f}ms, save: {t_save*1000:.1f}ms, total: {t_frame*1000:.1f}ms)")
    
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
    print(f"  {'Stage':<35} {'Total (s)':>12} {'Avg/Frame (ms)':>18}")
    print("-" * 70)
    
    # Video decoding (all frames at once)
    t_decode_total = timer.get_total('video_decoding')
    t_decode_per_frame = t_decode_total / args.numframe * 1000 if args.numframe > 0 else 0
    print(f"  {'Video Decoding (ffmpeg)':<35} {t_decode_total:>12.4f} {t_decode_per_frame:>18.2f}")
    
    # Plane restoration
    t_plane_total = timer.get_total('plane_restore')
    t_plane_avg = timer.get_avg('plane_restore') * 1000
    print(f"  {'Plane Restoration (PNG->Tensor)':<35} {t_plane_total:>12.4f} {t_plane_avg:>18.2f}")
    
    # Checkpoint saving
    t_save_total = timer.get_total('checkpoint_save')
    t_save_avg = timer.get_avg('checkpoint_save') * 1000
    print(f"  {'Checkpoint Saving':<35} {t_save_total:>12.4f} {t_save_avg:>18.2f}")
    
    # Total reconstruction
    print("-" * 70)
    t_recon_total = timer.get_total('total_reconstruction')
    t_recon_per_frame = t_recon_total / args.numframe * 1000 if args.numframe > 0 else 0
    print(f"  {'Total Reconstruction':<35} {t_recon_total:>12.4f} {t_recon_per_frame:>18.2f}")
    
    # Grand total
    t_grand_total = t_decode_total + t_recon_total
    t_grand_per_frame = t_grand_total / args.numframe * 1000 if args.numframe > 0 else 0
    print("-" * 70)
    print(f"  {'GRAND TOTAL':<35} {t_grand_total:>12.4f} {t_grand_per_frame:>18.2f}")
    print("=" * 70)
    
    print(f"\n=== 복원 완료 ===")
    print(f"복원된 체크포인트: {outdir}")
    print(f"복원된 plane: {', '.join(LF_PLANE_NAMES)}")
    print(f"처리된 프레임 수: {timer.get_count('frame_reconstruction')}")

