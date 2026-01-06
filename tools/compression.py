import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import cv2

from multiprocessing import Pool
from functools import partial



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
        plane_name: name of the plane (xy, xz, yz)
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


def save_density_visualization(density_image, frameid, vis_dir):
    """
    Save density visualization images.
    
    Args:
        density_image: tensor of shape [H, W]
        frameid: frame index
        vis_dir: directory to save visualization images
    """
    os.makedirs(vis_dir, exist_ok=True)
    
    img = density_image.cpu().numpy()
    
    # 8-bit grayscale
    img_8b = to8b(img)
    cv2.imwrite(os.path.join(vis_dir, f'density_frame_{frameid+1}.png'), img_8b)
    
    # Colormap version
    img_color = apply_colormap(img)
    cv2.imwrite(os.path.join(vis_dir, f'density_color_frame_{frameid+1}.png'), img_color)




def tile_maker(feat_plane, h = 2160, w= 3840):
    image = torch.zeros(h,w)
    #image = torch.zeros(1440,2560)

    h,w = list(feat_plane.size())[-2:]


    x,y = 0,0
    for i in range(feat_plane.size(1)):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "Tile_maker: not enough space"

        #ipdb.set_trace()

        image[x:x+h,y:y+w] = feat_plane[0,i,:,:]
        y = y + w

    return image

def density_quantize(density,nbits):

    nbits = 2**nbits-1
    data = density.clone()
    
    data[data<-5] = -5
    data[data>30] = 30

    data = data +5
    data = data /(30+5)
    

    data = torch.round(data *nbits)/nbits

    return data

def density_dequantize(density):
    
    data = density *(30+5)
    data = data-5


    return data




def make_density_image(density_grid, nbits, act_shift=0):



    data = density_quantize(density_grid, nbits)
    

    res = tile_maker(data[0])

    return res



if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True,
                        help='config file path')

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


    thresh = 0


    bound_thres = 20
    low_bound = -bound_thres
    high_bound = bound_thres
    nbits = 2**16-1

    if args.logdir[-1] =='/':
        args.logdir = args.logdir[:-1]
    name = args.logdir.split('/')[-1]

    # 16비트 PNG 이미지 저장 폴더 (인코딩 전 원본)
    frames_dir = os.path.join(args.logdir, f'frames_{args.qp}')
    # 압축된 비디오 저장 폴더
    compressed_dir = os.path.join(args.logdir, f'compressed_{args.qp}')

    for frameid in tqdm(range(0, args.numframe)):


        if os.path.isfile(os.path.join(frames_dir, f'density_frame_{frameid+1}.png')):
            continue


        tmp_file = os.path.join(args.logdir, f'fine_last_{frameid}.tar')

        assert os.path.isfile(tmp_file), "Checkpoint not found."

        tqdm.write(f"Loading Checkpoint {tmp_file}")
        ckpt = torch.load(tmp_file, map_location='cpu', weights_only=False)
        

        density = ckpt['model_state_dict']['density.grid'].clone()
        volume_size = list(density.size())[-3:]


        voxel_size_ratio = ckpt['model_kwargs']['voxel_size_ratio']



        masks = None
        if 'act_shift' in ckpt['model_state_dict']:

            alpha = 1- (torch.exp(density+ckpt['model_state_dict']['act_shift'])+1)**(-voxel_size_ratio)
            alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)

            mask = alpha<1e-4
            #mask = density.reshape(density.size(0),-1).clone()
            #mask = torch.nn.functional.softplus(mask + ckpt['model_state_dict']['act_shift'].cpu()) >0.4
            #mask = mask.sum(dim=1)
            #mask = mask<=thresh

            density[mask] = -5

         

            feature_alpha = F.interpolate(alpha, size=tuple(np.array(volume_size)*3), mode='trilinear', align_corners=True)
            mask_fg = feature_alpha>=1e-4

            #mask_fg = ~mask
            #mask_fg = mask_fg.reshape(mask_fg.size(0),1,1,1,1).repeat(1,1,voxel_size,voxel_size,voxel_size)
            #mask_fg = zero_unpads(merge_volume(mask_fg,grid_size),volume_size)



            # mask projection
            masks = {}
            masks['xy'] = mask_fg.sum(axis=4)
            masks['xz'] = mask_fg.sum(axis=3)
            masks['yz'] = mask_fg.sum(axis=2)


        planes = {}


        for key in ckpt['model_state_dict'].keys():
            if 'k0' in key and 'plane' in key and 'residual' not in key:
                data = ckpt['model_state_dict'][key]
                planes[key.split('.')[-1]]= data

        # 디버그: 어떤 plane들이 로드되었는지 출력
        tqdm.write(f"Loaded planes: {list(planes.keys())}")
        
        plane_data = []
        ratios = []
        tpsnr = []
        for p in ['xy','xz','yz']:
            plane_size = list(planes[f"{p}_plane"].size())[-1:-3:-1]

            # 디버그: feature plane 값 범위 출력
            raw_plane = planes[f"{p}_plane"]
            tqdm.write(f"  {p}_plane: shape={list(raw_plane.size())}, min={raw_plane.min().item():.4f}, max={raw_plane.max().item():.4f}, mean={raw_plane.mean().item():.4f}")
            
            if masks is not None:

                cur_mask = masks[p].repeat(1,planes[f"{p}_plane"].size(1),1,1)

                planes[f"{p}_plane"][cur_mask<1] = 0
                #torch.median(planes[f"{p}_plane"])


                #sanity check
                ra = planes[f"{p}_plane"].abs()
                ra = ra[cur_mask>=1].abs()
                assert (ra<bound_thres).sum()/ra.size(0) >0.95, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
                ratios.append(((ra<bound_thres).sum()/ra.size(0)).item())
            else:
                ra = planes[f"{p}_plane"].abs()
                ra = ra.reshape(-1)
                assert (ra<bound_thres).sum()/ra.size(0) >0.95, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
                ratios.append(((ra<bound_thres).sum()/ra.size(0)).item())

            feat = (planes[f"{p}_plane"] - low_bound)/(high_bound-low_bound)
            
            feat = torch.round(feat *nbits)/nbits

            
            feat[feat<0]=0
            feat[feat>1.0] = 1.0

            # 디버그: 정규화 후 값 범위 출력
            tqdm.write(f"  {p}_plane normalized: min={feat.min().item():.4f}, max={feat.max().item():.4f}, mean={feat.mean().item():.4f}")

            plane_data.append(feat)

            gt_feat = (planes[f"{p}_plane"]- low_bound)/(high_bound-low_bound)

            tpsnr.append(psnr(gt_feat,feat).item())


        tqdm.write(f"ratio: {np.mean(ratios)*100}%   PSNR:{np.mean(tpsnr)} qp:{args.qp}")
        os.makedirs(frames_dir, exist_ok=True)

        # Create visualization directory if needed
        vis_dir = None
        if args.save_vis or args.save_channels:
            vis_dir = os.path.join(args.logdir, f'visualization_{args.qp}')
            os.makedirs(vis_dir, exist_ok=True)

  
        imgs = {}
        plane_sizes ={}
        for ind,plane in zip(['xy','xz','yz'],plane_data):
            img = tile_maker(plane)
            imgs[f'{ind}_plane'] = img
            plane_sizes[f'{ind}_plane'] = plane.size()

            cv2.imwrite(os.path.join(frames_dir, f'{ind}_planes_frame_{frameid+1}.png'),to16b(img.cpu().numpy()))

            # Save visualization images
            if args.save_vis or args.save_channels:
                save_feature_visualization(plane, ind, frameid, vis_dir, save_channels=args.save_channels)

        
        #density_image = make_density_image(ckpt['model_state_dict']['density.grid'], 16)
        density_image = make_density_image(density, 16)
        #ipdb.set_trace()
        cv2.imwrite(os.path.join(frames_dir, f'density_frame_{frameid+1}.png'),to16b(density_image.cpu().numpy()))

        # Save density visualization
        if args.save_vis or args.save_channels:
            save_density_visualization(density_image, frameid, vis_dir)

        # density 크기 정보도 저장
        density_size = list(density.size())
        
        torch.save({'plane_size':plane_sizes, 
                    'density_size': density_size,
                    'bounds': (low_bound,high_bound),
                    'nbits':nbits,
                    'qp': args.qp}, 
                    os.path.join(frames_dir, f'planes_frame_meta.nf'))
    
    #parallel_process(args.numframe, args, thresh, bound_thres, low_bound, high_bound, nbits, args.qp)

    # 비디오 압축을 위한 스크립트 생성
    os.makedirs(compressed_dir, exist_ok=True)
    
    filename = '/dev/shm/planes_to_videos.sh'
    with open(filename,'w') as f:
        # frames_dir에서 이미지를 읽어 compressed_dir에 비디오 저장
        for p in ['xy','xz','yz']:
            if args.codec =='h265':
                f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/{p}_planes_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {compressed_dir}/{p}_planes.mp4\n")
            elif args.codec =='mpg2':
                f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/{p}_planes_frame_%d.png -c:v mpeg2video -color_range pc -qscale:v {args.qp} {compressed_dir}/{p}_planes.mpg\n")
        if args.codec =='h265':
            f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/density_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {compressed_dir}/density_planes.mp4\n")
        elif args.codec =='mpg2':    
            f.write(f"ffmpeg -y -framerate 30 -i {frames_dir}/density_frame_%d.png -c:v mpeg2video -color_range pc -qscale:v {args.qp} {compressed_dir}/density_planes.mpg\n")
       
        # 메타데이터와 네트워크 파일을 compressed_dir에 복사
        f.write(f'cp {frames_dir}/planes_frame_meta.nf {compressed_dir}/\n')
        f.write(f'cp {args.logdir}/rgbnet* {compressed_dir}/ 2>/dev/null || true\n')
        f.write(f'cp {args.logdir}/config.py {compressed_dir}/ 2>/dev/null || true\n')
        
        # 압축 파일 생성
        f.write(f'cd {args.logdir}\n')
        f.write(f'zip -r compressed.zip compressed_{args.qp}\n')

    os.system(f"bash {filename}")
    
    print(f"\n=== 완료 ===")
    print(f"프레임 이미지 (16-bit PNG): {frames_dir}")
    print(f"압축 비디오: {compressed_dir}")







