"""
Light Field Rendering Script

This script renders Light Field models using XYUV coordinates.
Unlike NeRF-based rendering, Light Field doesn't use ray marching.
"""

import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import wandb
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo
from lib.load_data import load_data
from lib.dvgo_video import RGB_Net, RGB_SH_Net, DirectVoxGO_Video

import pandas as pd

try:
    from mmengine import Config
except ImportError:
    try:
        from mmcv import Config
    except ImportError:
        raise ImportError("mmcv.Config 또는 mmengine.Config를 import할 수 없습니다. mmengine을 설치하세요: pip install mmengine")


wandbrun = None


def config_parser():
    '''Define command line arguments'''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')

    parser.add_argument('--frame_ids', nargs='+', type=int, help='a list of frame IDs')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--dump_images", action='store_true', default=True)
    parser.add_argument("--eval_ssim", action='store_true', default=True)
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true', default=True)

    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--reald", action='store_true', 
                        help='use original data (True) or compressed data (False)')

    # logging options
    parser.add_argument("--i_print", type=int, default=500,
                        help='frequency of console printout and metric logging')
    
    return parser


def seed_everything(seed):
    '''Seed everything for better reproducibility.'''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def load_lf_model(ckpt_path, rgbnet_path, cfg, device):
    """
    Load a single-frame Light Field model from checkpoint.
    
    Args:
        ckpt_path: path to model checkpoint
        rgbnet_path: path to RGBNet checkpoint
        cfg: config object
        device: torch device
    
    Returns:
        model: DirectVoxGO model
        rgbnet: RGB_Net or RGB_SH_Net
    """
    print(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model_kwargs = ckpt['model_kwargs']
    model = dvgo.DirectVoxGO(**model_kwargs)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    
    # Load RGBNet
    print(f"Loading RGBNet from {rgbnet_path}")
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


@torch.no_grad()
def render_viewpoints_lf(model, rgbnet, xyuv_coords, HW, gt_imgs=None, savedir=None, 
                         dump_images=False, eval_ssim=False, eval_lpips_alex=False, 
                         eval_lpips_vgg=False, frame_id=0, masks=None, device='cuda'):
    '''
    Render images for Light Field model using XYUV coordinates.
    
    Args:
        model: DirectVoxGO Light Field model
        rgbnet: RGB network for color prediction
        xyuv_coords: [N_views, H, W, 4] XYUV coordinates
        HW: [N_views, 2] height and width
        gt_imgs: optional ground truth images for evaluation
        savedir: directory to save rendered images
        dump_images: whether to save images
        frame_id: frame id for logging
        masks: optional masks for evaluation
        device: torch device
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

    cfg.data.frame_ids = args.frame_ids

    print("################################")
    print("--- Light Field Rendering ---")
    print("--- Frame_ID:", args.frame_ids)
    print("--- Dataset Type:", cfg.data.dataset_type)
    print("################################")

    # Initialize wandb
    wandbrun = wandb.init(
        project="TeTriRF_LF",
        config={
            "configs": cfg,
            "args": args,
        },
        resume="allow",
        id='TestingLF_' + cfg.expname + f'_{args.qp}_{args.codec}',
    )

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

    # Load Light Field data
    data_dict = load_everything_lf(args=args, cfg=cfg)

    # Log directory
    logdir = os.path.join(cfg.basedir, cfg.expname)
    
    # Process each frame
    all_psnrs = []
    all_ssims = []
    all_lpips = []

    for frame_id in args.frame_ids:
        print(f"\n=== Processing Frame {frame_id} ===")
        
        # Determine checkpoint path
        if args.reald:
            ckpt_path = os.path.join(logdir, f"fine_last_{frame_id}.tar")
        else:
            ckpt_path = os.path.join(logdir, f"raw_out/fine_last_{frame_id}.tar")
        
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping frame {frame_id}")
            continue
        
        # Find RGBNet file
        try:
            if args.reald:
                rgbnet_dir = logdir
            else:
                rgbnet_dir = os.path.join(logdir, "raw_out")
            rgbnet_path = find_rgbnet_file(rgbnet_dir, frame_id, cfg.fine_model_and_render.dynamic_rgbnet)
        except FileNotFoundError as e:
            print(f"RGBNet not found: {e}, skipping frame {frame_id}")
            continue
        
        # Load model
        model, rgbnet = load_lf_model(ckpt_path, rgbnet_path, cfg, device)
        
        # Get frame-specific data indices
        frame_ids_tensor = data_dict['frame_ids']
        if isinstance(frame_ids_tensor, np.ndarray):
            frame_ids_tensor = torch.from_numpy(frame_ids_tensor)
        
        # Render train set
        if args.render_train:
            testsavedir = os.path.join(logdir, f'render_train')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            
            i_train = data_dict['i_train']
            id_mask = (frame_ids_tensor == frame_id).cpu().numpy()[i_train]
            t_train = np.array(i_train)[id_mask]
            
            if len(t_train) == 0:
                print(f"No training views for frame {frame_id}")
            else:
                xyuv_train = data_dict['xyuv'][t_train]
                if isinstance(xyuv_train, torch.Tensor):
                    xyuv_train = xyuv_train.numpy()
                
                rgbs, res_psnr = render_viewpoints_lf(
                    model=model,
                    rgbnet=rgbnet,
                    xyuv_coords=xyuv_train,
                    HW=data_dict['HW'][t_train],
                    gt_imgs=[data_dict['images'][i] for i in t_train],
                    savedir=testsavedir,
                    dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim,
                    eval_lpips_alex=args.eval_lpips_alex,
                    eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id=frame_id,
                    device=device,
                )

        # Render test set
        if args.render_test:
            testsavedir = os.path.join(logdir, f'render_test')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            
            i_test = data_dict['i_test']
            id_mask = (frame_ids_tensor == frame_id).cpu().numpy()[i_test]
            t_test = np.array(i_test)[id_mask]
            
            if len(t_test) == 0:
                print(f"No test views for frame {frame_id}")
            else:
                xyuv_test = data_dict['xyuv'][t_test]
                if isinstance(xyuv_test, torch.Tensor):
                    xyuv_test = xyuv_test.numpy()
                
                data_mask = None
                if data_dict['masks'] is not None:
                    data_mask = [data_dict['masks'][i] for i in t_test]
                
                rgbs, res_psnr = render_viewpoints_lf(
                    model=model,
                    rgbnet=rgbnet,
                    xyuv_coords=xyuv_test,
                    HW=data_dict['HW'][t_test],
                    gt_imgs=[data_dict['images'][i] for i in t_test],
                    savedir=testsavedir,
                    dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim,
                    eval_lpips_alex=args.eval_lpips_alex,
                    eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id=frame_id,
                    masks=data_mask,
                    device=device,
                )
                
                # Log to wandb
                if len(res_psnr) > 0:
                    rgb_map = [wandb.Image(utils.to8b(i), caption=f"render test rgb {frame_id}") for i in rgbs]
                    
                    psnr_val = res_psnr.get('psnr', 0)
                    all_psnrs.append(psnr_val)
                    
                    log_dict = {
                        'psnr': psnr_val,
                        'frame_id': frame_id,
                    }
                    if 'ssim' in res_psnr:
                        log_dict['ssim'] = res_psnr['ssim']
                        all_ssims.append(res_psnr['ssim'])
                    if 'lpips' in res_psnr:
                        log_dict['lpips(vgg)'] = res_psnr['lpips']
                        all_lpips.append(res_psnr['lpips'])
                    
                    if frame_id % 10 == 0:
                        log_dict['render_rgb'] = rgb_map
                    
                    wandbrun.log(log_dict)

        # Clean up
        del model, rgbnet
        torch.cuda.empty_cache()

    # Final summary logging
    if len(all_psnrs) > 0:
        avg_psnr = np.mean(all_psnrs)
        print(f"\n{'='*30}")
        print(f"Final Average PSNR: {avg_psnr:.4f}")
        
        final_log = {'test_psnr': avg_psnr}
        
        if len(all_ssims) > 0:
            avg_ssim = np.mean(all_ssims)
            print(f"Final Average SSIM: {avg_ssim:.4f}")
            final_log['test_ssim'] = avg_ssim
            
        if len(all_lpips) > 0:
            avg_lpips = np.mean(all_lpips)
            print(f"Final Average LPIPS: {avg_lpips:.4f}")
            final_log['test_lpips'] = avg_lpips
            
        print(f"{'='*30}\n")
        wandbrun.log(final_log)

    print('\nDone')

