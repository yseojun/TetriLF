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

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_val = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_val

def untile_image(image,h,w,ndim):

    features = torch.zeros(1,ndim,h,w)

    x,y = 0,0
    for i in range(ndim):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "untile_image: too many feature maps"

        features[0,i,:,:] = image[x:x+h,y:y+w]
        y = y + w

    return features


def tile_maker(feat_plane, h = 2160, w= 3840):
    image = torch.zeros(h,w)

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

def make_density_image(density_grid, nbits, h=4320,w=7680):
    #image = torch.zeros(1080,1920)

    data = density_grid +5
    data[data<0] = 0

    data = data / 30
    data[data>1.0] = 1.0

    data = torch.round(data *nbits)/nbits
    
    #res = tile_maker(data,h=1080,w=1920)
    res = tile_maker(data, h=h,w=w)



    return res

if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', required=True,
                        help='compressed files directory path')

    parser.add_argument('--model_template', type=str, default='fine_last_0.tar',
                        help='model template')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames')

    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')

    parser.add_argument("--no_wandb", action='store_true',
                        help='disable wandb logging')

    args = parser.parse_args()


    outdir = os.path.join(args.dir,'../raw_out')

    os.makedirs(outdir, exist_ok=True)


    ckpt = torch.load(os.path.join(args.dir, '..', args.model_template), weights_only=False)

    # Load metadata from planes_frame_meta.nf
    meta_file = os.path.join(args.dir, 'planes_frame_meta.nf')
    assert os.path.isfile(meta_file), f"Metadata file not found: {meta_file}"
    meta = torch.load(meta_file, weights_only=False)
    
    low_bound, high_bound = meta['bounds']
    plane_sizes = meta['plane_size']
    density_size = meta.get('density_size', None)
    qp = meta.get('qp', 0)

    name = args.dir.split('/')[-2]
    wandbrun = None
    if not args.no_wandb:
        wandbrun = wandb.init(
            # set the wandb project where this run will be logged
            project="TeTriRF",
        
            # track hyperparameters and run metadata
            resume = "allow",
            id = 'compressionV7_'+name+'_'+args.codec,
        )


    # Decode videos to PNG frames using ffmpeg
    filename = '/dev/shm/videos_to_planes.sh'
    with open(filename,'w') as f:
        f.write(f'cd {args.dir}\n')
        for p in ['xy','xz','yz']:
            if args.codec =='mpg2':
                f.write(f"ffmpeg -y -i {p}_planes.mpg  -pix_fmt gray16be {p}_planes_frame_%d_out.png\n")
            else:
                f.write(f"ffmpeg -y -i {p}_planes.mp4  -pix_fmt gray16be {p}_planes_frame_%d_out.png\n")
                
        if args.codec =='mpg2':
            f.write(f"ffmpeg -y -i density_planes.mpg  -pix_fmt gray16be  density_frame_%d_out.png\n")
        else:
            f.write(f"ffmpeg -y -i density_planes.mp4  -pix_fmt gray16be  density_frame_%d_out.png\n")
    os.system(f"bash {filename}")

    # Process each frame
    for frameid in tqdm(range(0, args.numframe)):

        # Restore feature planes
        for p in ['xy', 'xz', 'yz']:
            key = f'{p}_plane'
            quant_img = cv2.imread(os.path.join(args.dir, f"{p}_planes_frame_{frameid+1}_out.png"), -1)
            
            if quant_img is None:
                tqdm.write(f"Warning: Could not read {p}_planes_frame_{frameid+1}_out.png, skipping frame {frameid}")
                continue

            # Untile image to get feature plane
            plane = untile_image(
                torch.tensor(quant_img.astype(np.float32)) / int(2**16-1), 
                plane_sizes[key][2],  # height
                plane_sizes[key][3],  # width
                plane_sizes[key][1]   # num channels
            )

            # Dequantize: convert from [0,1] back to original range
            plane = plane * (high_bound - low_bound) + low_bound

            assert 'k0.' + key in ckpt['model_state_dict'], f'Wrong plane name: k0.{key}'
            ckpt['model_state_dict']['k0.' + key] = plane.clone().cuda()

        # Restore density
        quant_img = cv2.imread(os.path.join(args.dir, f"density_frame_{frameid+1}_out.png"), -1)
        
        if quant_img is None:
            tqdm.write(f"Warning: Could not read density_frame_{frameid+1}_out.png, skipping frame {frameid}")
            continue

        # Get density size from metadata or from checkpoint
        # density.grid shape: [batch=1, channel=1, D, H, W]
        if density_size is not None:
            d_depth = density_size[2]   # D (number of depth slices = number of tiles)
            d_height = density_size[3]  # H (height of each slice)
            d_width = density_size[4]   # W (width of each slice)
        else:
            # Fallback: get from checkpoint
            orig_density = ckpt['model_state_dict']['density.grid']
            d_depth = orig_density.size(2)
            d_height = orig_density.size(3)
            d_width = orig_density.size(4)

        # untile_image returns [1, ndim, h, w] where ndim=D, h=H, w=W
        density_plane = untile_image(
            torch.tensor(quant_img.astype(np.float32)) / int(2**16-1), 
            d_height,  # h
            d_width,   # w
            d_depth    # ndim (number of depth slices)
        )

        # Dequantize density: convert from [0,1] back to original range [-5, 30]
        density_plane = density_plane * (30 + 5) - 5

        ckpt['model_state_dict']['density.grid'] = density_plane.clone().cuda().unsqueeze(0)

        # Save restored checkpoint
        torch.save(ckpt, os.path.join(outdir, f'fine_last_{frameid}.tar'))
        tqdm.write(f"Saved restored model: fine_last_{frameid}.tar")

    print(f"\nRestoration complete! Output saved to: {outdir}")