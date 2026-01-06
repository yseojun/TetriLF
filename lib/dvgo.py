import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_scatter import segment_coo
import ipdb
from . import grid
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)


'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyuv_min, xyuv_max,
                 world_size,  # world_size 필수
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 k0_type='PlaneGrid',
                 k0_config={},
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4, rgb_model='MLP',
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyuv_min', torch.Tensor(xyuv_min))
        self.register_buffer('xyuv_max', torch.Tensor(xyuv_max))
        self.fast_color_thres = fast_color_thres
        self.rgb_model = rgb_model

        # world_size 설정
        self._set_grid_resolution(world_size)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit

        self.k0_dim = rgbnet_dim
        
        self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyuv_min=self.xyuv_min, xyuv_max=self.xyuv_max, config=self.k0_config)

        print('dvgo: feature voxel grid', self.k0)

    def _set_grid_resolution(self, world_size):
        """world_size 설정"""
        if isinstance(world_size, (list, tuple)):
            self.world_size = torch.LongTensor(world_size)
        else:
            self.world_size = world_size.clone()
        print('dvgo: world_size', self.world_size.tolist())

    def get_kwargs(self):
        return {
            'xyuv_min': self.xyuv_min.cpu().numpy(),
            'xyuv_max': self.xyuv_max.cpu().numpy(),
            'world_size': self.world_size.cpu().tolist(),
            'fast_color_thres': self.fast_color_thres,
            'k0_type': self.k0_type,
            'k0_config': self.k0_config,
            'rgb_model': self.rgb_model,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, new_world_size):
        """볼륨 그리드 스케일 조정 (world_size 기반)"""
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(new_world_size)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.k0.scale_volume_grid(self.world_size)

        print('dvgo: scale_volume_grid finish')

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        # 4D의 경우 4개의 weight 전달
        self.k0.total_variation_add_grad(w, w, w, w, dense_mode)

    def forward(self, xyuv, shared_rgbnet=None, share_grid=None, global_step=None, mode='feat', **render_kwargs):
        '''Light Field rendering
        @xyuv: [*, 4] the xyuv coordinates (X_cam, Y_cam, U_img, V_img)
        '''
        # 입력 형태 유연화: [N, 4], [B, N, 4], [B, H, W, 4] 등 지원
        assert xyuv.shape[-1] == 4, f'XYUV coordinates must have 4 dimensions, got shape {xyuv.shape}'
        
        original_shape = xyuv.shape[:-1]

        k0 = self.k0(xyuv)

        if share_grid is not None:
            k0 = k0 + share_grid(xyuv)

        assert shared_rgbnet is not None, 'shared_rgbnet is None'

        rgb = shared_rgbnet(k0)
        rgb = torch.sigmoid(rgb)
        
        return rgb



''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    #assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

def get_training_rays_multi_frame(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, frame_ids, model, masks, render_kwargs, flatten=False):
    #print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW) and len(rgb_tr_ori)==len(frame_ids)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = []
    rays_o_tr = []
    rays_d_tr = []
    viewdirs_tr = []
    imsz = []
    frame_ids_tr = []
    top = 0
    for ind,c2w, img, (H, W), K, id in zip(range(len(train_poses)), train_poses, rgb_tr_ori, HW, Ks, frame_ids):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)


        if not flatten:
            for i in range(0, img.shape[0], CHUNK):
                mask[i:i+CHUNK] = model.hit_coarse_geo(
                        rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)

        if masks is not None:
            mask = mask & (masks[ind][...,0]>0.5)

        n = mask.sum()
        #print('mask percetage:', float(n)*100/mask.numel())
        rgb_tr.append(img[mask])
        rays_o_tr.append(rays_o[mask].to(DEVICE))
        rays_d_tr.append(rays_d[mask].to(DEVICE))
        viewdirs_tr.append(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        frame_ids_tr.append(torch.ones(n,device = 'cpu')*id)


    eps_time = time.time() - eps_time
    #print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_ids_tr

def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

def batch_indices_generator_MF(rgb_tr_s, BS):
    # torch.randperm on cuda produce incorrect results in my machine

    while True:
        camera_id = random.randint(0, len(rgb_tr_s)-1)
        N = len(rgb_tr_s[camera_id])

        #random_indices = torch.randint(0, N, (BS,), device ='cpu')
        random_indices = np.random.randint(0, N, BS)
        #idx = random_indices.long()
        yield camera_id, random_indices
