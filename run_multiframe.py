import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import wandb
import imageio
import numpy as np
import ipdb

# mmcv 버전 호환성 처리
try:
    from mmengine import Config
except ImportError:
    try:
        from mmcv import Config
    except ImportError:
        raise ImportError("mmcv.Config 또는 mmengine.Config를 import할 수 없습니다. mmengine을 설치하세요: pip install mmengine")

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dvgo_video
from lib.load_data import load_data
from torch.utils.data import DataLoader

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='0'

wandbrun=None

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')

    # training mode
    parser.add_argument("--training_mode", type=int, default=0)
    parser.add_argument('--frame_ids', nargs='+', type=int, help='a list of ID')


    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')

    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints_lf(model, xyuv_coords, HW, gt_imgs=None, savedir=None, dump_images=False,
                         eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
                         frame_id=0, shared_rgbnet=None):
    '''Render images for Light Field model using XYUV coordinates.
    
    Args:
        model: DirectVoxGO model for a specific frame
        xyuv_coords: [N_views, H, W, 4] XYUV coordinates
        HW: [N_views, 2] height and width
        gt_imgs: optional ground truth images for evaluation
        savedir: directory to save rendered images
        dump_images: whether to save images
        frame_id: frame id for logging
        shared_rgbnet: shared RGB network
    '''
    rgbs = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    forward_times = []

    device = next(model.parameters()).device

    for i in tqdm(range(len(xyuv_coords)), desc=f'Rendering frame {frame_id}'):
        H, W = HW[i]
        xyuv = torch.from_numpy(xyuv_coords[i]).float().to(device)  # [H, W, 4]
        
        # Forward pass through model with CUDA timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        rgb = model(xyuv, shared_rgbnet=shared_rgbnet)  # [H, W, 3]
        end_event.record()
        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event)  # milliseconds
        forward_times.append(forward_time)
        
        rgb = rgb.cpu().numpy()
        
        rgbs.append(rgb)
        
        if i == 0:
            print('Testing', rgb.shape)

        if gt_imgs is not None:
            gt = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb - gt)))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt, max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt, net_name='alex', device=device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt, net_name='vgg', device=device))

    res_psnr = None
    avg_forward_time = np.mean(forward_times) if forward_times else 0
    
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        res_psnr = np.mean(psnrs)
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
    
    print(f'Testing forward time: {avg_forward_time:.2f}ms (avg)')

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.jpg'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)

    return rgbs, res_psnr, avg_forward_time


def seed_everything():
    '''Seed everything for better reproducibility.
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything_lf(args, cfg):
    '''Load Light Field data with XYUV coordinates.
    '''
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


def scene_rep_reconstruction_lf(args, cfg, cfg_model, cfg_train, xyuv_min, xyuv_max, data_dict, stage):
    """Light Field 학습을 위한 scene reconstruction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract data
    HW = data_dict['HW']
    i_train = data_dict['i_train']
    i_val = data_dict['i_val']
    i_test = data_dict['i_test']
    images = data_dict['images']
    xyuv_coords = data_dict['xyuv']
    frame_ids = data_dict['frame_ids']

    frame_ids_cpu = frame_ids.cpu()
    unique_frame_ids = torch.unique(frame_ids, sorted=True).cpu().numpy().tolist()

    # Create model
    model = dvgo_video.DirectVoxGO_Video(
        frameids=unique_frame_ids,
        xyuv_min=xyuv_min,
        xyuv_max=xyuv_max,
        cfg=cfg
    )

    ret = model.load_checkpoints()

    if not cfg.fine_model_and_render.dynamic_rgbnet and args.training_mode > 0:
        cfg.fine_train.lrate_rgbnet = 0

    model.set_fixedframe(ret)
    model = model.cuda()
    
    # Create optimizer
    optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(model, cfg_train, global_step=0)

    # Gather training XYUV coordinates
    def gather_training_xyuv():
        """XYUV 좌표와 RGB 타겟 수집"""
        xyuv_tr_s = []
        rgb_tr_s = []
        frame_id_s = []

        training_fids = unique_frame_ids

        for fid in training_fids:
            if fid in model.fixed_frame:
                continue
            
            # 해당 프레임의 train 인덱스 찾기
            id_mask = (frame_ids_cpu == fid)
            train_mask = np.zeros(len(frame_ids_cpu), dtype=bool)
            train_mask[i_train] = True
            combined_mask = id_mask.numpy() & train_mask
            t_train = np.where(combined_mask)[0]
            
            if len(t_train) == 0:
                continue
            
            # 각 뷰의 데이터 flatten
            for idx in t_train:
                img = images[idx]  # [H, W, 3]
                xyuv = xyuv_coords[idx]  # [H, W, 4]
                
                # Flatten
                img_flat = img.reshape(-1, 3)  # [H*W, 3]
                xyuv_flat = xyuv.reshape(-1, 4)  # [H*W, 4]
                
                if cfg.data.load2gpu_on_the_fly:
                    xyuv_tr_s.append(xyuv_flat.cpu())
                    rgb_tr_s.append(img_flat.cpu())
                else:
                    xyuv_tr_s.append(xyuv_flat.to(device))
                    rgb_tr_s.append(img_flat.to(device))
                
                frame_id_s.append(torch.ones(img_flat.shape[0], dtype=torch.long) * fid)
        
        if len(xyuv_tr_s) == 0:
            raise ValueError("No training data found!")
        
        print(f'Gathered {len(xyuv_tr_s)} views for training, first view size: {xyuv_tr_s[0].shape}')
        
        # Batch sampler
        index_generator = dvgo.batch_indices_generator_MF(rgb_tr_s, cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        
        return xyuv_tr_s, rgb_tr_s, frame_id_s, batch_index_sampler

    xyuv_tr, rgb_tr, frame_id_tr, batch_index_sampler = gather_training_xyuv()

    # Training loop
    torch.cuda.empty_cache()
    psnr_lst = []
    forward_time_lst = []
    time0 = time.time()
    global_step = -1

    psnr_res = []
    time_res = []

    for global_step in trange(1, 1 + cfg_train.N_iters):

        # Progressive scaling (world_size 기반)
        # bilinear interpolation을 위해 최소 크기 2 보장
        MIN_GRID_SIZE = 2
        if args.training_mode != -1:
            final_world_size = list(cfg.data.world_size)
            
            if global_step in cfg_train.pg_scale:
                n_rest_scales = len(cfg_train.pg_scale) - cfg_train.pg_scale.index(global_step) - 1
                scale_factor = 2 ** n_rest_scales
                cur_world_size = [max(w // scale_factor, MIN_GRID_SIZE) for w in final_world_size]
                print(f'pg_scale: scaling to world_size={cur_world_size}')
                model.scale_volume_grid(cur_world_size)
                optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(model, cfg_train, global_step=0)
                torch.cuda.empty_cache()

            if global_step in cfg_train.pg_scale2:
                print('**Second Level PG****')
                n_rest_scales = len(cfg_train.pg_scale2) - cfg_train.pg_scale2.index(global_step) - 1
                scale_factor = 2 ** n_rest_scales
                cur_world_size = [max(w // scale_factor, MIN_GRID_SIZE) for w in final_world_size]
                print(f'pg_scale2: scaling to world_size={cur_world_size}')
                model.scale_volume_grid(cur_world_size)
                optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(model, cfg_train, global_step=0)
                torch.cuda.empty_cache()

        # Sample batch
        camera_id, sel_i = batch_index_sampler()
        sel_i = torch.from_numpy(sel_i)
        
        while sel_i.size(0) == 0:
            print('while loop: empty batch')
            camera_id, sel_i = batch_index_sampler()

        frameids = frame_id_tr[camera_id][sel_i]
        target = rgb_tr[camera_id][sel_i]
        xyuv = xyuv_tr[camera_id][sel_i]

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            xyuv = xyuv.to(device)

        # Forward pass with CUDA timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        render_result = model(
            xyuv, frame_ids=frameids,
            global_step=global_step, mode='feat'
        )
        end_event.record()
        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event)  # milliseconds

        # Loss computation
        optimizer.zero_grad(set_to_none=True)
        rgb_pred = render_result['rgb_marched']
        loss = cfg_train.weight_main * F.mse_loss(rgb_pred, target)
        psnr = utils.mse2psnr(loss.detach())
            
        l1loss = torch.tensor(0.0)
        # if stage == 'fine':
        #     l1loss = model.compute_k0_l1_loss(frameids)
        #     loss += cfg.fine_train.weight_l1_loss * l1loss

        loss.backward()
        optimizer.step()
        psnr_lst.append(psnr.item())
        forward_time_lst.append(forward_time)

        # Update learning rate
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_factor

        if global_step % 1000 == 0:
            psnr_res.append(np.mean(psnr_lst))
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            time_res.append(eps_time_str)

        # Logging
        if global_step % args.i_print == 0 or global_step == 1:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            avg_forward_time = np.mean(forward_time_lst) if forward_time_lst else 0
            iter_psnr = np.mean(psnr_lst)
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / L1 Loss: {l1loss.item():.9f} / PSNR: {iter_psnr:5.2f} / '
                       f'Forward: {avg_forward_time:.2f}ms / Eps: {eps_time_str}')
            if wandbrun is not None:
                wandbrun.log({
                    "train_psnr": iter_psnr,
                    "loss": loss.item(),
                    "l1_loss": l1loss.item(),
                    "forward_time_ms": avg_forward_time,
                    "global_step": global_step,
                }, step=global_step)
            psnr_lst = []
            forward_time_lst = []

        # Train PSNR logging to wandb every 10000 iters
        if global_step % 10000 == 0:
            train_psnr_avg = np.mean(psnr_res) if psnr_res else psnr.item()
            if wandbrun is not None:
                wandbrun.log({
                    "train_psnr": train_psnr_avg,
                    "global_step": global_step,
                })
            print(f'[iter {global_step}] Train PSNR: {train_psnr_avg:.2f}')

        # Test evaluation every 10000 iters
        if (global_step % 10000 == 0):
            all_test_psnrs = []
            all_test_forward_times = []
            
            for frameid in model.dvgos.keys():
                if int(frameid) in model.fixed_frame:
                    continue
              
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{frameid}')
                os.makedirs(testsavedir, exist_ok=True)

                # 해당 프레임의 테스트 인덱스 찾기
                frame_index = torch.nonzero(data_dict['frame_ids'] == int(frameid)).squeeze(1).cpu().numpy()
                test_indices = np.intersect1d(data_dict['i_test'], frame_index).copy()

                if len(test_indices) == 0:
                    print(f'No test views for frame {frameid}')
                    continue

                rgbs, res_psnr, test_forward_time = render_viewpoints_lf(
                    model=model.dvgos[frameid],
                    xyuv_coords=xyuv_coords[test_indices].numpy(),
                    HW=data_dict['HW'][test_indices],
                    gt_imgs=[images[i].cpu().numpy() for i in test_indices],
                    savedir=testsavedir, 
                    dump_images=True,
                    eval_ssim=args.eval_ssim, 
                    eval_lpips_alex=args.eval_lpips_alex, 
                    eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id=frameid,
                    shared_rgbnet=model.rgbnet,
                )
                print('iter:', global_step, 'test_psnr:', res_psnr, 'test_forward_time:', f'{test_forward_time:.2f}ms')
                
                # 평균 계산을 위해 수집
                if res_psnr is not None:
                    all_test_psnrs.append(res_psnr)
                all_test_forward_times.append(test_forward_time)

                if wandbrun is not None:
                    rgb_map = [wandb.Image(utils.to8b(i), caption=f"test rgb {frameid}") for i in rgbs]
                    wandbrun.log({
                        f"test_rgb_frame_{frameid}": rgb_map,
                        f"test_psnr_frame_{frameid}": res_psnr,
                        f"test_forward_time_frame_{frameid}": test_forward_time,
                        "global_step": global_step,
                    })
            
            # 전체 프레임 평균 로깅
            if len(all_test_psnrs) > 0 and wandbrun is not None:
                avg_test_psnr = np.mean(all_test_psnrs)
                avg_test_forward_time = np.mean(all_test_forward_times)
                print(f'[iter {global_step}] Average test PSNR: {avg_test_psnr:.2f}, Forward time: {avg_test_forward_time:.2f}ms')
                wandbrun.log({
                    "test_psnr_avg": avg_test_psnr,
                    "test_forward_time_avg": avg_test_forward_time,
                    "global_step": global_step,
                })

    if global_step != -1:
        model.save_checkpoints()


def train_lf(args, cfg, data_dict):
    """Light Field training entry point."""
    print('train_lf: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # XYUV bounds from data
    xyuv_min_fine = torch.tensor(data_dict['xyuv_min'])
    xyuv_max_fine = torch.tensor(data_dict['xyuv_max'])
    
    scene_rep_reconstruction_lf(
        args=args, cfg=cfg,
        cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
        xyuv_min=xyuv_min_fine, xyuv_max=xyuv_max_fine,
        data_dict=data_dict, stage='fine'
    )
    
    eps_fine = time.time() - eps_time
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train_lf: fine detail reconstruction in', eps_time_str)


# ================== Original NeRF functions (kept for compatibility) ==================

@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False, 
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,  **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))


    res_psnr = None
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        res_psnr = np.mean(psnrs)
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')


    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.jpg'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = utils.to8b(1 - depths[i] / np.max(depths[i]))
            filename = os.path.join(savedir, '{:03d}_depth.jpg'.format(i))
            if rgb8.shape[-1]<3:
                rgb8 = np.repeat(rgb8, 3, axis=-1)
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps, res_psnr


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
                'i_train', 'i_val', 'i_test', 'irregular_shape',
                'poses', 'render_poses', 'images', 'frame_ids','masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.frame_ids = args.frame_ids

    print("################################")
    print("--- Frame_ID:", cfg.data.frame_ids)
    print("--- training_mode:", args.training_mode)
    print("--- dataset_type:", cfg.data.dataset_type)
    print("################################")

    if args.training_mode > 0 and cfg.data.ndc:
        cfg.fine_train.lrate_rgbnet /= 5.0
        cfg.fine_train.weight_tv_density = 0
        cfg.fine_train.weight_tv_k0 = 0

    # wandb
    wandb_dir = "/data/ysj/result/tetrirf"
    os.makedirs(wandb_dir, exist_ok=True)
    wandbrun = wandb.init(
        project="TeTriRF",
        dir=wandb_dir,
        config={
            "configs": cfg,
            "args": args,
        },
        resume="allow",
        id=cfg.expname,
    )

    # init environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # Check dataset type and load accordingly
    if cfg.data.dataset_type == 'LF':
        # Light Field mode
        data_dict = load_everything_lf(args=args, cfg=cfg)
        
        if not args.render_only:
            train_lf(args, cfg, data_dict)
    else:
        # Original NeRF mode
        data_dict = load_everything(args=args, cfg=cfg)
        
        if not args.render_only:
            # Original train function would go here
            # For now, raise error since we're focusing on LF
            raise NotImplementedError("Original NeRF training not fully implemented in this version")

    print('Done')
