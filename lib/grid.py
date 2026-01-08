import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

def create_grid(type, **kwargs):
    return PlaneGrid(**kwargs)

class PlaneGrid(nn.Module):
    """
    4D Plane Grid for Light Field representation.
    Uses 6 planes: XY (camera), UV (image), XU, XV, YU, YV
    Coordinates: [X_cam, Y_cam, U_img, V_img]
    """
    def __init__(self, channels, world_size, xyuv_min, xyuv_max, config, residual_mode=False):
        super(PlaneGrid, self).__init__()
        if 'factor' in config:
            self.scale = config['factor']
        else:
            self.scale = 1
            
        self.channels = channels
        self.config = config
        self.residual_mode = residual_mode
        self.register_buffer('xyuv_min', torch.Tensor(xyuv_min))
        self.register_buffer('xyuv_max', torch.Tensor(xyuv_max))
        
        # 4D world_size: [X, Y, U, V]
        # world_size는 config에서 지정한 최종 크기 그대로 사용
        if len(world_size) == 4:
            X, Y, U, V = world_size
        else:
            # Fallback for 3D (legacy compatibility)
            X, Y, U = world_size
            V = U
        
        self.world_size = torch.tensor([X, Y, U, V])
        
        # 6 planes for 4D light field
        # channels divided by 6 (one for each plane)
        R = self.channels // 6
        if R < 1:
            R = 1
        
        # Camera planes
        self.xy_plane = nn.Parameter(torch.randn([1, R, X, Y]) * 0.1)  # Camera position
        # Image planes  
        self.uv_plane = nn.Parameter(torch.randn([1, R, U, V]) * 0.1)  # Image coordinates
        # Cross planes (disparity information)
        self.xu_plane = nn.Parameter(torch.randn([1, R, X, U]) * 0.1)
        self.xv_plane = nn.Parameter(torch.randn([1, R, X, V]) * 0.1)
        self.yu_plane = nn.Parameter(torch.randn([1, R, Y, U]) * 0.1)
        self.yv_plane = nn.Parameter(torch.randn([1, R, Y, V]) * 0.1)
        
        # Update actual channels
        self.channels = R * 6
        
        print(f'PlaneGrid: 4D LF mode, world_size={self.world_size.tolist()}, R={R}, total_channels={self.channels}')

    def compute_planes_feat(self, ind_norm):
        """
        Compute features from 6 planes.
        ind_norm: [1, 1, N, 4] normalized coordinates (x, y, u, v)
        """
        # Extract coordinates for each plane combination
        # grid_sample expects [..., 2] for 2D sampling (y, x order)
        
        # XY plane: sample with (y, x) = (ind_norm[1], ind_norm[0])
        xy_feat = F.grid_sample(self.xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
        
        # UV plane: sample with (v, u) = (ind_norm[3], ind_norm[2])
        uv_feat = F.grid_sample(self.uv_plane, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
        
        # XU plane: sample with (u, x) = (ind_norm[2], ind_norm[0])
        xu_feat = F.grid_sample(self.xu_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
        
        # XV plane: sample with (v, x) = (ind_norm[3], ind_norm[0])
        xv_feat = F.grid_sample(self.xv_plane, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
        
        # YU plane: sample with (u, y) = (ind_norm[2], ind_norm[1])
        yu_feat = F.grid_sample(self.yu_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
        
        # YV plane: sample with (v, y) = (ind_norm[3], ind_norm[1])
        yv_feat = F.grid_sample(self.yv_plane, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T

        # Aggregate all plane features
        feat = torch.cat([
            xy_feat,
            uv_feat,
            xu_feat,
            xv_feat,
            yu_feat,
            yv_feat
        ], dim=-1)

        return feat       

    def forward(self, xyuv, dir=None, center=None):
        '''
        xyuv: global 4D coordinates to query [*, 4] where 4 = (X_cam, Y_cam, U_img, V_img)
        '''
        shape = xyuv.shape[:-1]
        xyuv = xyuv.reshape(1, 1, -1, 4)
        
        # Normalize to [-1, 1]
        ind_norm = (xyuv - self.xyuv_min) / (self.xyuv_max - self.xyuv_min) * 2 - 1
       
        if self.channels > 1:
            out = self.compute_planes_feat(ind_norm)
            out = out.reshape(*shape, self.channels)
        else:
            raise Exception("channels must be > 1")
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        if len(new_world_size) == 4:
            X, Y, U, V = new_world_size
        else:
            X, Y, U = new_world_size
            V = U
        
        self.world_size = torch.tensor([X, Y, U, V])
        
        if self.residual_mode:
            # Residual mode scaling (if needed)
            pass
        else:
            self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X, Y], mode='bilinear', align_corners=True))
            self.uv_plane = nn.Parameter(F.interpolate(self.uv_plane.data, size=[U, V], mode='bilinear', align_corners=True))
            self.xu_plane = nn.Parameter(F.interpolate(self.xu_plane.data, size=[X, U], mode='bilinear', align_corners=True))
            self.xv_plane = nn.Parameter(F.interpolate(self.xv_plane.data, size=[X, V], mode='bilinear', align_corners=True))
            self.yu_plane = nn.Parameter(F.interpolate(self.yu_plane.data, size=[Y, U], mode='bilinear', align_corners=True))
            self.yv_plane = nn.Parameter(F.interpolate(self.yv_plane.data, size=[Y, V], mode='bilinear', align_corners=True))

    def scale_volume_grid_value(self, new_world_size):
        if self.channels == 0:
            return
        if len(new_world_size) == 4:
            X, Y, U, V = new_world_size
        else:
            X, Y, U = new_world_size
            V = U
    
        xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X, Y], mode='bilinear', align_corners=True), requires_grad=False)
        uv_plane = nn.Parameter(F.interpolate(self.uv_plane.data, size=[U, V], mode='bilinear', align_corners=True), requires_grad=False)
        xu_plane = nn.Parameter(F.interpolate(self.xu_plane.data, size=[X, U], mode='bilinear', align_corners=True), requires_grad=False)
        xv_plane = nn.Parameter(F.interpolate(self.xv_plane.data, size=[X, V], mode='bilinear', align_corners=True), requires_grad=False)
        yu_plane = nn.Parameter(F.interpolate(self.yu_plane.data, size=[Y, U], mode='bilinear', align_corners=True), requires_grad=False)
        yv_plane = nn.Parameter(F.interpolate(self.yv_plane.data, size=[Y, V], mode='bilinear', align_corners=True), requires_grad=False)

        return xy_plane, uv_plane, xu_plane, xv_plane, yu_plane, yv_plane

    def total_variation_add_grad(self, wx, wy, wu, wv, dense_mode):
        '''Add gradients by total variation loss in-place'''
        loss = 0
        # XY plane TV
        loss += wx * F.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum')
        loss += wy * F.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum')
        # UV plane TV
        loss += wu * F.smooth_l1_loss(self.uv_plane[:,:,1:], self.uv_plane[:,:,:-1], reduction='sum')
        loss += wv * F.smooth_l1_loss(self.uv_plane[:,:,:,1:], self.uv_plane[:,:,:,:-1], reduction='sum')
        # XU plane TV
        loss += wx * F.smooth_l1_loss(self.xu_plane[:,:,1:], self.xu_plane[:,:,:-1], reduction='sum')
        loss += wu * F.smooth_l1_loss(self.xu_plane[:,:,:,1:], self.xu_plane[:,:,:,:-1], reduction='sum')
        # XV plane TV
        loss += wx * F.smooth_l1_loss(self.xv_plane[:,:,1:], self.xv_plane[:,:,:-1], reduction='sum')
        loss += wv * F.smooth_l1_loss(self.xv_plane[:,:,:,1:], self.xv_plane[:,:,:,:-1], reduction='sum')
        # YU plane TV
        loss += wy * F.smooth_l1_loss(self.yu_plane[:,:,1:], self.yu_plane[:,:,:-1], reduction='sum')
        loss += wu * F.smooth_l1_loss(self.yu_plane[:,:,:,1:], self.yu_plane[:,:,:,:-1], reduction='sum')
        # YV plane TV
        loss += wy * F.smooth_l1_loss(self.yv_plane[:,:,1:], self.yv_plane[:,:,:-1], reduction='sum')
        loss += wv * F.smooth_l1_loss(self.yv_plane[:,:,:,1:], self.yv_plane[:,:,:,:-1], reduction='sum')
        
        loss /= 12
        loss.backward()

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.channels // 6}'
