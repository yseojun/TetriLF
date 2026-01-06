"""
Realtime Light Field Rendering Script

This script performs single-frame decompression and rendering for Light Field models.
It measures and reports timing for each stage:
- Decompression: Video frame extraction using ffmpeg
- Model Loading: Checkpoint reconstruction from decompressed planes
- Rendering: Light Field rendering using XYUV coordinates
"""

import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import tempfile

import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from lib import utils, dvgo
from lib.load_data import load_data
from lib.dvgo_video import RGB_Net, RGB_SH_Net, DirectVoxGO_Video

try:
    from mmengine import Config
except ImportError:
    try:
        from mmcv import Config
    except ImportError:
        raise ImportError("mmcv.Config 또는 mmengine.Config를 import할 수 없습니다. mmengine을 설치하세요: pip install mmengine")


class CUDATimer:
    """
    High-precision timer using CUDA events for GPU operations.
    Falls back to time.perf_counter() for CPU operations.
    """
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.timings = {}
        
    def start(self, name):
        """Start timing a named operation."""
        if self.use_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.timings[name] = {'start_event': start_event, 'end_event': end_event}
        else:
            self.timings[name] = {'start_time': time.perf_counter()}
    
    def stop(self, name):
        """Stop timing and return elapsed time in seconds."""
        if name not in self.timings:
            raise ValueError(f"Timer '{name}' was not started")
        
        if self.use_cuda:
            torch.cuda.synchronize()
            self.timings[name]['end_event'].record()
            torch.cuda.synchronize()
            elapsed_ms = self.timings[name]['start_event'].elapsed_time(self.timings[name]['end_event'])
            elapsed_s = elapsed_ms / 1000.0
        else:
            elapsed_s = time.perf_counter() - self.timings[name]['start_time']
        
        self.timings[name]['elapsed'] = elapsed_s
        return elapsed_s
    
    def get(self, name):
        """Get elapsed time for a named operation."""
        if name in self.timings and 'elapsed' in self.timings[name]:
            return self.timings[name]['elapsed']
        return None


class CPUTimer:
    """
    Simple CPU timer using time.perf_counter() for CPU-bound operations like ffmpeg.
    """
    def __init__(self):
        self.timings = {}
        
    def start(self, name):
        self.timings[name] = {'start_time': time.perf_counter()}
    
    def stop(self, name):
        if name not in self.timings:
            raise ValueError(f"Timer '{name}' was not started")
        elapsed_s = time.perf_counter() - self.timings[name]['start_time']
        self.timings[name]['elapsed'] = elapsed_s
        return elapsed_s
    
    def get(self, name):
        if name in self.timings and 'elapsed' in self.timings[name]:
            return self.timings[name]['elapsed']
        return None


def config_parser():
    '''Define command line arguments'''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    
    # Frame selection
    parser.add_argument('--frame_id', type=int, required=True,
                        help='Single frame ID to decompress and render')

    # Compression settings
    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')
    parser.add_argument("--qp", type=int, default=20,
                        help='QP value used during compression')
    parser.add_argument("--compressed_dir", type=str, default=None,
                        help='Compressed files directory (default: {logdir}/compressed_{qp})')

    # Rendering options
    parser.add_argument("--render_test", action='store_true', default=True)
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--dump_images", action='store_true', default=True)
    parser.add_argument("--eval_ssim", action='store_true', default=True)
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true', default=True)
    
    return parser


def seed_everything(seed):
    '''Seed everything for better reproducibility.'''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def decompress_single_frame(compressed_dir, frame_id, codec, temp_dir):
    """
    Decompress a single frame from compressed videos.
    
    Args:
        compressed_dir: directory containing compressed videos
        frame_id: frame index to extract (0-based)
        codec: 'h265' or 'mpg2'
        temp_dir: temporary directory for extracted frames
    
    Returns:
        dict: plane_name -> extracted image path
    """
    # Load metadata
    meta_file = os.path.join(compressed_dir, 'planes_frame_meta.nf')
    assert os.path.isfile(meta_file), f"Metadata file not found: {meta_file}"
    meta = torch.load(meta_file, weights_only=False)
    
    LF_PLANE_NAMES = meta.get('plane_names', ['xy', 'uv', 'xu', 'xv', 'yu', 'yv'])
    
    # Extract single frame from each video
    # ffmpeg frame numbers are 1-based in the output filename pattern
    # But select filter uses 0-based indexing
    extracted_paths = {}
    
    for p in LF_PLANE_NAMES:
        if codec == 'mpg2':
            video_file = os.path.join(compressed_dir, f'{p}_planes.mpg')
        else:
            video_file = os.path.join(compressed_dir, f'{p}_planes.mp4')
        
        if not os.path.isfile(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        output_path = os.path.join(temp_dir, f'{p}_frame_{frame_id}.png')
        
        # Use ffmpeg to extract single frame
        # select filter: eq(n, frame_id) where n is 0-based frame number
        cmd = f'ffmpeg -y -i "{video_file}" -vf "select=eq(n\\,{frame_id})" -vsync vfr -pix_fmt gray16be -frames:v 1 "{output_path}" 2>/dev/null'
        ret = os.system(cmd)
        
        if ret != 0 or not os.path.isfile(output_path):
            raise RuntimeError(f"Failed to extract frame {frame_id} from {video_file}")
        
        extracted_paths[p] = output_path
    
    return extracted_paths, meta


def reconstruct_model(extracted_paths, meta, template_ckpt, device):
    """
    Reconstruct model checkpoint from extracted plane images.
    
    Args:
        extracted_paths: dict of plane_name -> image path
        meta: metadata dict
        template_ckpt: template checkpoint for structure
        device: torch device
    
    Returns:
        reconstructed checkpoint dict
    """
    low_bound, high_bound = meta['bounds']
    plane_sizes = meta['plane_size']
    LF_PLANE_NAMES = meta.get('plane_names', ['xy', 'uv', 'xu', 'xv', 'yu', 'yv'])
    
    # Create a copy of the template checkpoint
    frame_ckpt = copy.deepcopy(template_ckpt)
    
    # Restore feature planes
    for p in LF_PLANE_NAMES:
        key = f'{p}_plane'
        img_path = extracted_paths[p]
        
        quant_img = cv2.imread(img_path, -1)
        
        if quant_img is None:
            raise RuntimeError(f"Could not read {img_path}")

        # Get plane size from metadata
        if key not in plane_sizes:
            raise RuntimeError(f"{key} not in plane_sizes metadata")
            
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
            raise RuntimeError(f"{state_dict_key} not in model_state_dict")
            
        frame_ckpt['model_state_dict'][state_dict_key] = plane.clone()

    return frame_ckpt


def load_lf_model_from_ckpt(ckpt, rgbnet_path, cfg, device):
    """
    Load a Light Field model from checkpoint dict.
    
    Args:
        ckpt: checkpoint dict
        rgbnet_path: path to RGBNet checkpoint
        cfg: config object
        device: torch device
    
    Returns:
        model: DirectVoxGO model
        rgbnet: RGB_Net or RGB_SH_Net
    """
    model_kwargs = ckpt['model_kwargs']
    model = dvgo.DirectVoxGO(**model_kwargs)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    
    # Load RGBNet
    rgbnet_ckpt = torch.load(rgbnet_path, map_location=device, weights_only=False)
    rgbnet_kwargs = rgbnet_ckpt['model_kwargs']
    
    rgb_model_type = cfg.fine_model_and_render.RGB_model if hasattr(cfg.fine_model_and_render, 'RGB_model') else 'MLP'
    
    if rgb_model_type == 'MLP':
        rgbnet = RGB_Net(
            dim0=rgbnet_kwargs['dim0'],
            rgbnet_width=rgbnet_kwargs['rgbnet_width'],
            rgbnet_depth=rgbnet_kwargs['rgbnet_depth']
        )
    elif rgb_model_type == 'SH':
        rgbnet = RGB_SH_Net(
            dim0=rgbnet_kwargs['dim0'],
            rgbnet_width=rgbnet_kwargs['rgbnet_width'],
            rgbnet_depth=rgbnet_kwargs['rgbnet_depth'],
            deg=rgbnet_kwargs.get('deg', 2)
        )
    else:
        raise ValueError(f"Unknown RGB model type: {rgb_model_type}")
    
    rgbnet.load_state_dict(rgbnet_ckpt['model_state_dict'])
    rgbnet = rgbnet.to(device)
    rgbnet.eval()
    
    return model, rgbnet


def find_rgbnet_file(logdir, frame_id, dynamic_rgbnet=True):
    """Find the appropriate RGBNet file for a given frame."""
    if not dynamic_rgbnet:
        rgbnet_file = os.path.join(logdir, 'rgbnet.tar')
        if os.path.isfile(rgbnet_file):
            return rgbnet_file
    
    # Find dynamic rgbnet files
    rgbnet_files = [f for f in os.listdir(logdir) if f.endswith('.tar') and 'rgbnet' in f and '_' in f]
    
    if len(rgbnet_files) == 0:
        # Fallback to static rgbnet
        rgbnet_file = os.path.join(logdir, 'rgbnet.tar')
        if os.path.isfile(rgbnet_file):
            return rgbnet_file
        raise FileNotFoundError(f"No RGBNet file found in {logdir}")
    
    # Find the matching rgbnet file for the frame_id
    for f in rgbnet_files:
        parts = f.replace('.tar', '').split('_')
        if len(parts) >= 3:
            beg = int(parts[1])
            eend = int(parts[2])
            if beg <= frame_id <= eend:
                return os.path.join(logdir, f)
    
    # If no exact match, use the one with largest range
    rgbnet_files.sort()
    return os.path.join(logdir, rgbnet_files[-1])


def load_everything_lf(args, cfg):
    '''Load Light Field data with XYUV coordinates.'''
    data_dict = load_data(cfg.data)

    # LF specific keys
    kept_keys = {'hwf', 'HW', 'i_train', 'i_val', 'i_test', 'irregular_shape',
                 'images', 'frame_ids', 'xyuv', 'xyuv_min', 'xyuv_max',
                 'masks', 'grid_size_x', 'grid_size_y', 'num_views'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # Convert to tensors
    if not data_dict['irregular_shape']:
        data_dict['images'] = torch.FloatTensor(data_dict['images'])
        data_dict['xyuv'] = torch.FloatTensor(data_dict['xyuv'])
    
    return data_dict


@torch.no_grad()
def render_viewpoints_lf(model, rgbnet, xyuv_coords, HW, gt_imgs=None, savedir=None, 
                         dump_images=False, eval_ssim=False, eval_lpips_alex=False, 
                         eval_lpips_vgg=False, frame_id=0, masks=None, device='cuda'):
    '''
    Render images for Light Field model using XYUV coordinates.
    '''
    rgbs = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    model.eval()
    rgbnet.eval()

    for i in tqdm(range(len(xyuv_coords)), desc=f'Rendering frame {frame_id}'):
        H, W = HW[i]
        xyuv = xyuv_coords[i]  # [H, W, 4]
        
        if isinstance(xyuv, np.ndarray):
            xyuv = torch.from_numpy(xyuv).float()
        
        xyuv = xyuv.to(device)
        
        # Flatten for batch processing
        xyuv_flat = xyuv.reshape(-1, 4)  # [H*W, 4]
        
        # Process in chunks to avoid OOM
        chunk_size = 150480
        rgb_chunks = []
        
        for j in range(0, xyuv_flat.shape[0], chunk_size):
            xyuv_chunk = xyuv_flat[j:j+chunk_size]
            rgb_chunk = model(xyuv_chunk, shared_rgbnet=rgbnet)
            rgb_chunks.append(rgb_chunk)
        
        rgb = torch.cat(rgb_chunks, dim=0)
        rgb = rgb.reshape(H, W, 3).cpu().numpy()
        
        rgbs.append(rgb)
        
        if i == 0:
            print('Testing', rgb.shape)

        if gt_imgs is not None:
            gt = gt_imgs[i]
            if isinstance(gt, torch.Tensor):
                gt = gt.cpu().numpy()
            
            if masks is not None and masks[i] is not None:
                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask_bool = mask[..., 0] > 0.5
                p = -10. * np.log10(np.mean(np.square(rgb[mask_bool] - gt[mask_bool])))
            else:
                p = -10. * np.log10(np.mean(np.square(rgb - gt)))
            
            psnrs.append(p)
            
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt, max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt, net_name='alex', device=device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt, net_name='vgg', device=device))

    res_psnr = {}
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        res_psnr = {'psnr': np.mean(psnrs)}
        
        if savedir is not None:
            with open(os.path.join(savedir, f'{frame_id}_psnr.txt'), 'w') as f:
                f.write('%f' % np.mean(psnrs))
        
        if eval_ssim:
            print('Testing ssim', np.mean(ssims), '(avg)')
            res_psnr['ssim'] = np.mean(ssims)
            if savedir is not None:
                with open(os.path.join(savedir, f'{frame_id}_ssim.txt'), 'w') as f:
                    f.write('%f' % np.mean(ssims))
        
        if eval_lpips_vgg:
            print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
            res_psnr['lpips'] = np.mean(lpips_vgg)
            if savedir is not None:
                with open(os.path.join(savedir, f'{frame_id}_lpips.txt'), 'w') as f:
                    f.write('%f' % np.mean(lpips_vgg))
        
        if eval_lpips_alex:
            print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None and dump_images:
        for i in trange(len(rgbs), desc='Saving images'):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, f'{frame_id}_{i}.png')
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)

    return rgbs, res_psnr


if __name__ == '__main__':

    # Load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # Set single frame
    cfg.data.frame_ids = [args.frame_id]

    print("=" * 60)
    print("  Realtime Light Field Rendering")
    print("=" * 60)
    print(f"  Frame ID: {args.frame_id}")
    print(f"  Codec: {args.codec}")
    print(f"  QP: {args.qp}")
    print(f"  Dataset Type: {cfg.data.dataset_type}")
    print("=" * 60)

    # Set default model configs
    if not hasattr(cfg.fine_model_and_render, 'dynamic_rgbnet'):
        cfg.fine_model_and_render.dynamic_rgbnet = True
    if not hasattr(cfg.fine_model_and_render, 'RGB_model'):
        cfg.fine_model_and_render.RGB_model = 'MLP'

    # Init environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything(args.seed)

    # Log directory and compressed directory
    logdir = os.path.join(cfg.basedir, cfg.expname)
    if args.compressed_dir is None:
        compressed_dir = os.path.join(logdir, f'compressed_{args.qp}')
    else:
        compressed_dir = args.compressed_dir

    print(f"\nLog directory: {logdir}")
    print(f"Compressed directory: {compressed_dir}")

    # Create temporary directory for decompressed frames
    temp_dir = tempfile.mkdtemp(prefix='lf_realtime_')
    print(f"Temporary directory: {temp_dir}")

    # Initialize timers
    # CPU timer for ffmpeg (external process)
    cpu_timer = CPUTimer()
    # CUDA timer for GPU operations (model loading, rendering)
    cuda_timer = CUDATimer(use_cuda=torch.cuda.is_available())
    
    # Timing storage
    timing = {}

    try:
        # ========================================
        # Stage 1: Decompression (CPU/ffmpeg)
        # ========================================
        print("\n" + "=" * 60)
        print("  Stage 1: Decompression (Video -> Planes)")
        print("=" * 60)
        
        cpu_timer.start('decompression')
        
        extracted_paths, meta = decompress_single_frame(
            compressed_dir, args.frame_id, args.codec, temp_dir
        )
        
        timing['1_decompression'] = cpu_timer.stop('decompression')
        
        print(f"  Extracted {len(extracted_paths)} planes")
        print(f"  Time: {timing['1_decompression']*1000:.2f} ms")

        # ========================================
        # Stage 2: Model Reconstruction (CPU + GPU transfer)
        # ========================================
        print("\n" + "=" * 60)
        print("  Stage 2: Model Reconstruction (Planes -> Model)")
        print("=" * 60)
        
        cuda_timer.start('model_loading')
        
        # Load template checkpoint
        template_path = os.path.join(logdir, f'fine_last_0.tar')
        if not os.path.isfile(template_path):
            # Try to find any fine_last checkpoint
            fine_files = [f for f in os.listdir(logdir) if f.startswith('fine_last_') and f.endswith('.tar')]
            if fine_files:
                template_path = os.path.join(logdir, sorted(fine_files)[0])
            else:
                raise FileNotFoundError(f"No template checkpoint found in {logdir}")
        
        print(f"  Loading template: {template_path}")
        template_ckpt = torch.load(template_path, map_location='cpu', weights_only=False)
        
        # Reconstruct model checkpoint
        frame_ckpt = reconstruct_model(extracted_paths, meta, template_ckpt, device)
        
        # Find RGBNet file
        rgbnet_path = find_rgbnet_file(compressed_dir, args.frame_id, cfg.fine_model_and_render.dynamic_rgbnet)
        print(f"  RGBNet path: {rgbnet_path}")
        
        # Load model
        model, rgbnet = load_lf_model_from_ckpt(frame_ckpt, rgbnet_path, cfg, device)
        
        timing['2_model_loading'] = cuda_timer.stop('model_loading')
        
        print(f"  Model loaded successfully")
        print(f"  Time: {timing['2_model_loading']*1000:.2f} ms")

        # ========================================
        # Stage 3: Data Loading (CPU)
        # ========================================
        print("\n" + "=" * 60)
        print("  Stage 3: Data Loading")
        print("=" * 60)
        
        cpu_timer.start('data_loading')
        
        data_dict = load_everything_lf(args=args, cfg=cfg)
        
        timing['3_data_loading'] = cpu_timer.stop('data_loading')
        
        print(f"  Data loaded")
        print(f"  Time: {timing['3_data_loading']*1000:.2f} ms")

        # ========================================
        # Stage 4: Rendering (GPU)
        # ========================================
        print("\n" + "=" * 60)
        print("  Stage 4: Rendering")
        print("=" * 60)
        
        # Warm-up run (optional, for more accurate timing)
        if torch.cuda.is_available():
            print("  Performing warm-up...")
            with torch.no_grad():
                dummy_input = torch.randn(1000, 4, device=device)
                _ = model(dummy_input, shared_rgbnet=rgbnet)
            torch.cuda.synchronize()
        
        cuda_timer.start('rendering')
        
        # Since we loaded only one frame, all data belongs to that frame
        # i_test and i_train are already indices within the loaded data
        testsavedir = os.path.join(logdir, f'render_realtime_{args.qp}')
        os.makedirs(testsavedir, exist_ok=True)
        
        results = {}
        
        # Render test set
        if args.render_test:
            i_test = data_dict['i_test']
            
            if len(i_test) == 0:
                print(f"No test views for frame {args.frame_id}")
            else:
                xyuv_test = data_dict['xyuv'][i_test]
                if isinstance(xyuv_test, torch.Tensor):
                    xyuv_test = xyuv_test.numpy()
                
                data_mask = None
                if data_dict['masks'] is not None:
                    data_mask = [data_dict['masks'][i] for i in i_test]
                
                rgbs, res_psnr = render_viewpoints_lf(
                    model=model,
                    rgbnet=rgbnet,
                    xyuv_coords=xyuv_test,
                    HW=data_dict['HW'][i_test],
                    gt_imgs=[data_dict['images'][i] for i in i_test],
                    savedir=testsavedir,
                    dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim,
                    eval_lpips_alex=args.eval_lpips_alex,
                    eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id=args.frame_id,
                    masks=data_mask,
                    device=device,
                )
                
                results['test'] = res_psnr
        
        # Render train set
        if args.render_train:
            i_train = data_dict['i_train']
            
            if len(i_train) == 0:
                print(f"No training views for frame {args.frame_id}")
            else:
                xyuv_train = data_dict['xyuv'][i_train]
                if isinstance(xyuv_train, torch.Tensor):
                    xyuv_train = xyuv_train.numpy()
                
                rgbs, res_psnr = render_viewpoints_lf(
                    model=model,
                    rgbnet=rgbnet,
                    xyuv_coords=xyuv_train,
                    HW=data_dict['HW'][i_train],
                    gt_imgs=[data_dict['images'][i] for i in i_train],
                    savedir=testsavedir,
                    dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim,
                    eval_lpips_alex=args.eval_lpips_alex,
                    eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id=args.frame_id,
                    device=device,
                )
                
                results['train'] = res_psnr
        
        timing['4_rendering'] = cuda_timer.stop('rendering')
        
        print(f"  Rendering complete")
        print(f"  Time: {timing['4_rendering']*1000:.2f} ms")

    finally:
        # Cleanup temporary files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("  TIMING SUMMARY")
    print("=" * 60)
    print(f"  {'Stage':<25} {'Time (ms)':>15} {'Time (s)':>12}")
    print("-" * 60)
    
    total_time = 0
    for stage, t in timing.items():
        print(f"  {stage:<25} {t*1000:>15.2f} {t:>12.4f}")
        total_time += t
    
    print("-" * 60)
    print(f"  {'TOTAL':<25} {total_time*1000:>15.2f} {total_time:>12.4f}")
    print("=" * 60)
    
    # Print quality metrics if available
    if results:
        print("\n" + "=" * 60)
        print("  QUALITY METRICS")
        print("=" * 60)
        for split, metrics in results.items():
            if metrics:
                print(f"  [{split.upper()}]")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
        print("=" * 60)
    
    print(f"\nOutput saved to: {testsavedir}")
    print('\nDone')

