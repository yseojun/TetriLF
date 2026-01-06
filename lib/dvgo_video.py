import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo
import ipdb
from . import grid
from torch.utils.cpp_extension import load
import copy
from .dvgo import DirectVoxGO
from .sh import eval_sh

class RGB_Net(torch.nn.Module):
    def __init__(self, dim0=None, rgbnet_width=None, rgbnet_depth=None):
        super(RGB_Net, self).__init__()
        self.rgbnet = None

        if dim0 is not None and rgbnet_width is not None and rgbnet_depth is not None:
            self.set_params(dim0, rgbnet_width, rgbnet_depth)

    def set_params(self, dim0, rgbnet_width, rgbnet_depth):
        
        if self.rgbnet is None:
            self.dim0 = dim0
            self.rgbnet_width = rgbnet_width
            self.rgbnet_depth = rgbnet_depth
            self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('***** rgb_net_ reset   *******')
        else:
            if self.dim0 != dim0 or self.rgbnet_width != rgbnet_width or self.rgbnet_depth != rgbnet_depth:
                ipdb.set_trace()
                raise Exception("Inconsistant parameters!")

        return lambda x: self.forward(x)

    def forward(self, x):
        if self.rgbnet is None:
            raise Exception("call set_params() first!")
        return self.rgbnet(x)

    def get_kwargs(self):
        return {
            'dim0': self.dim0,
            'rgbnet_width': self.rgbnet_width,
            'rgbnet_depth': self.rgbnet_depth
        }

class RGB_SH_Net(torch.nn.Module):
    def __init__(self, dim0=None, rgbnet_width=None, rgbnet_depth=None, deg=3):
        super(RGB_SH_Net, self).__init__()
        self.rgbnet = None
        self.deg = deg

        if dim0 is not None and rgbnet_width is not None and rgbnet_depth is not None:
            self.set_params(dim0, rgbnet_width, rgbnet_depth, out_dim=3*(self.deg+1)**2)

    def set_params(self, dim0, rgbnet_width, rgbnet_depth, out_dim):
        
        if self.rgbnet is None:
            self.dim0 = dim0
            self.rgbnet_width = rgbnet_width
            self.rgbnet_depth = rgbnet_depth
            self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, out_dim),
                )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('***** rgb_net_SH reset   *******')
        else:
            if self.dim0 != dim0 or self.rgbnet_width != rgbnet_width or self.rgbnet_depth != rgbnet_depth:
                ipdb.set_trace()
                raise Exception("Inconsistant parameters!")

        return lambda x: self.forward(x)

    def forward(self, x, dirs):
        if self.rgbnet is None:
            raise Exception("call set_params() first!")
        coeffs = self.rgbnet(x)
        coeffs = coeffs.reshape(x.size(0), 3, -1)
        return torch.sigmoid(eval_sh(self.deg, coeffs, dirs))

    def get_kwargs(self):
        return {
            'dim0': self.dim0,
            'rgbnet_width': self.rgbnet_width,
            'rgbnet_depth': self.rgbnet_depth,
            'deg': self.deg,
        }

class DirectVoxGO_Video(torch.nn.Module):
    """
    Multi-frame Light Field model using XYUV coordinates.
    """
    def __init__(self, frameids, xyuv_min, xyuv_max, cfg=None):
        super(DirectVoxGO_Video, self).__init__()

        self.xyuv_min = xyuv_min
        self.xyuv_max = xyuv_max
        self.frameids = frameids
        self.cfg = cfg
        self.dvgos = nn.ModuleDict()
        self.viewbase_pe = cfg.fine_model_and_render.viewbase_pe
        self.share_grid = None
        self.fixed_frame = []

        self.initial_models()

    def get_kwargs(self):
        return {
            'frameids': self.frameids,
            'xyuv_min': self.xyuv_min.cpu().numpy() if torch.is_tensor(self.xyuv_min) else self.xyuv_min,
            'xyuv_max': self.xyuv_max.cpu().numpy() if torch.is_tensor(self.xyuv_max) else self.xyuv_max,
            'viewbase_pe': self.viewbase_pe,
        }

    def initial_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_kwargs = copy.deepcopy(self.cfg.fine_model_and_render)
        model_kwargs.pop('num_voxels', None)  # num_voxels 제거 (사용하지 않음)
        model_kwargs.pop('num_voxels_base', None)
        cfg_train = self.cfg.fine_train

        coarse_ckpt_path = None
        
        # world_size를 config에서 가져옴 (필수, 최종 크기)
        final_world_size = list(self.cfg.data.world_size)
        
        # pg_scale이 있으면 초기 world_size를 줄임 (최종 크기에서 시작해서 점점 늘림)
        # bilinear interpolation을 위해 최소 크기 2 보장
        MIN_GRID_SIZE = 2
        if len(cfg_train.pg_scale):
            scale_factor = 2 ** len(cfg_train.pg_scale)
            world_size = [max(w // scale_factor, MIN_GRID_SIZE) for w in final_world_size]
            print(f'dvgo_video: initial world_size={world_size}, final_world_size={final_world_size}')
        else:
            world_size = final_world_size

        for frameid in self.frameids:
            coarse_ckpt_path = os.path.join(self.cfg.basedir, self.cfg.expname, f'coarse_last_{frameid}.tar')
            if not os.path.isfile(coarse_ckpt_path):
                coarse_ckpt_path = None
            frameid_str = str(frameid)
            print(f'model create: frame{frameid_str}')

            # Light Field 모드: world_size 직접 전달
            self.dvgos[frameid_str] = DirectVoxGO(
                xyuv_min=self.xyuv_min, xyuv_max=self.xyuv_max,
                world_size=world_size,
                mask_cache_path=coarse_ckpt_path, 
                rgb_model=self.cfg.fine_model_and_render.RGB_model,
                **model_kwargs)
            
            self.dvgos[frameid_str] = self.dvgos[frameid_str].to(device)

        # share_grid 생성
        self.share_grid = grid.create_grid(
            self.cfg.fine_model_and_render.k0_type, 
            channels=self.cfg.fine_model_and_render.rgbnet_dim, 
            world_size=world_size,
            xyuv_min=self.xyuv_min, 
            xyuv_max=self.xyuv_max, 
            config=self.cfg.fine_model_and_render.k0_config)
        self.share_grid = self.share_grid.to(device)

        # RGBNet 생성 - Light Field에서는 viewbase_pe 불필요
        if self.cfg.fine_model_and_render.RGB_model == 'MLP':
            # Light Field: plane features만 사용 (viewdir encoding 제거)
            dim0 = self.cfg.fine_model_and_render.rgbnet_dim
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            self.rgbnet = RGB_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth)
        elif self.cfg.fine_model_and_render.RGB_model == 'SH':
            dim0 = self.cfg.fine_model_and_render.rgbnet_dim
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            self.rgbnet = RGB_SH_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth, deg=2)
            
        print('*** models creation completed.', self.frameids)

    def load_checkpoints(self):
        cfg = self.cfg
        ret = []

        for frameid in self.frameids:
            frameid_str = str(frameid)
            last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last_{frameid_str}.tar')
            if not os.path.isfile(last_ckpt_path):
                print(f"Frame {frameid_str}'s checkpoint doesn't exist")
                continue
            ckpt = torch.load(last_ckpt_path, weights_only=False)

            model_kwargs = ckpt['model_kwargs']
            self.dvgos[frameid_str] = DirectVoxGO(**model_kwargs)
            
            self.dvgos[frameid_str].load_state_dict(ckpt['model_state_dict'], strict=True)
            self.dvgos[frameid_str] = self.dvgos[frameid_str].cuda()
            print(f"Frame {frameid_str}'s checkpoint loaded.")
            ret.append(int(frameid))
            break

        beg = self.frameids[0]
        eend = self.frameids[-1]

        if self.cfg.fine_model_and_render.dynamic_rgbnet:
            rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
        else:
            rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet.tar')
        
        if not os.path.isfile(rgbnet_file):
            rgbnet_files = [f for f in os.listdir(os.path.join(cfg.basedir, cfg.expname)) if f.endswith('.tar') and 'rgbnet' in f]
            if len(rgbnet_files) > 0:
                beg = -1
                eend = -1
                for f in rgbnet_files:
                    beg = max(beg, int(f.split('_')[1]))
                    eend = max(eend, int(f.split('_')[2].split('.')[0]))
                rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
            
        if os.path.isfile(rgbnet_file):
            checkpoint = torch.load(rgbnet_file, weights_only=False)
            self.rgbnet.load_state_dict(checkpoint['model_state_dict'])
            print('load RGBNet', rgbnet_file)

        # share_grid 로드
        share_grid_file = os.path.join(cfg.basedir, cfg.expname, 'share_grid.tar')
        if os.path.isfile(share_grid_file):
            checkpoint = torch.load(share_grid_file, weights_only=False)
            self.share_grid.load_state_dict(checkpoint['model_state_dict'])
            print('load share_grid', share_grid_file)

        return ret

    def save_checkpoints(self):
        cfg = self.cfg

        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid_str = str(frameid)
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last_{frameid_str}.tar')
            ckpt = {
                'model_state_dict': self.dvgos[frameid_str].state_dict(),
                'model_kwargs': self.dvgos[frameid_str].get_kwargs(),
            }
            torch.save(ckpt, ckpt_path)
            print(f"Frame {frameid_str}'s checkpoint saved to {ckpt_path}")

        beg = self.frameids[0]
        eend = self.frameids[-1]

        if self.cfg.fine_model_and_render.dynamic_rgbnet:
            rgbnet_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
        else:
            rgbnet_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'rgbnet.tar')
        rgbnet_ckpt = {
            'model_state_dict': self.rgbnet.state_dict(),
            'model_kwargs': self.rgbnet.get_kwargs(),
        }
        torch.save(rgbnet_ckpt, rgbnet_ckpt_path)
        print(f"RGBNet checkpoint saved to {rgbnet_ckpt_path}")

        # share_grid 저장
        share_grid_path = os.path.join(cfg.basedir, cfg.expname, 'share_grid.tar')
        share_grid_ckpt = {
            'model_state_dict': self.share_grid.state_dict(),
        }
        torch.save(share_grid_ckpt, share_grid_path)
        print(f"share_grid checkpoint saved to {share_grid_path}")

    def set_fixedframe(self, ids):
        """Light Field 모드에서는 density가 없으므로 plane만 초기화"""
        self.fixed_frame = ids
        
        if len(ids) > 0:
            frameid = -1
            for fid in self.frameids:
                if fid not in self.fixed_frame:
                    frameid = fid
                    break
            assert frameid != -1
            source_id = ids[0]
            
            # 6개의 plane 스케일 값 가져오기
            planes = self.dvgos[str(source_id)].k0.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)
            
            for fid in self.frameids:
                if fid in self.fixed_frame:
                    continue
                device = self.dvgos[str(fid)].k0.xy_plane.device
                if self.cfg.fine_train.initialize_feature:
                    self.dvgos[str(fid)].k0.xy_plane = nn.Parameter(planes[0].clone()).to(device)
                    self.dvgos[str(fid)].k0.uv_plane = nn.Parameter(planes[1].clone()).to(device)
                    self.dvgos[str(fid)].k0.xu_plane = nn.Parameter(planes[2].clone()).to(device)
                    self.dvgos[str(fid)].k0.xv_plane = nn.Parameter(planes[3].clone()).to(device)
                    self.dvgos[str(fid)].k0.yu_plane = nn.Parameter(planes[4].clone()).to(device)
                    self.dvgos[str(fid)].k0.yv_plane = nn.Parameter(planes[5].clone()).to(device)

                print(f'Initialize frame:{fid}')

    def forward(self, xyuv, frame_ids, global_step=None, mode='feat', **render_kwargs):
        """
        Light Field forward pass.
        @xyuv: [*, 4] XYUV coordinates
        @frame_ids: frame id tensor
        """
        # find unique frame ids
        frame_ids_unique = torch.unique(frame_ids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1

        frameid = frame_ids_unique[0]

        ret_frame = self.dvgos[str(frameid)](
            xyuv, shared_rgbnet=self.rgbnet, 
            share_grid=self.share_grid,
            global_step=global_step, mode=mode, **render_kwargs)

        return {'rgb_marched': ret_frame}

    def scale_volume_grid(self, scale):
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid_str = str(frameid)
            self.dvgos[frameid_str].scale_volume_grid(scale)
        
        if self.share_grid is not None:
            self.share_grid.scale_volume_grid(scale)

    def density_total_variation_add_grad(self, weight, dense_mode, frameids):
        # Light Field 모드에서는 density가 없으므로 pass
        pass

    def k0_total_variation_add_grad(self, weight, dense_mode, frameids):
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        frameid = frame_ids_unique[0]
        frameid_str = str(frameid)
        self.dvgos[frameid_str].k0_total_variation_add_grad(weight, dense_mode)

    def compute_k0_l1_loss(self, frameids):
        """인접 프레임 간 plane L1 loss 계산"""
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        loss = 0
        N = 0
        frameid = frame_ids_unique[0]
        
        if str(frameid-1) in self.dvgos:
            frameid2 = str(frameid-1)
            cur_k0 = self.dvgos[str(frameid)].k0
            prev_k0 = self.dvgos[frameid2].k0
            
            if cur_k0.xy_plane.size() != prev_k0.xy_plane.size():
                planes = prev_k0.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)
                loss += F.l1_loss(cur_k0.xy_plane, planes[0])
                loss += F.l1_loss(cur_k0.uv_plane, planes[1])
                loss += F.l1_loss(cur_k0.xu_plane, planes[2])
                loss += F.l1_loss(cur_k0.xv_plane, planes[3])
                loss += F.l1_loss(cur_k0.yu_plane, planes[4])
                loss += F.l1_loss(cur_k0.yv_plane, planes[5])
                N += 6
            else:
                loss += F.l1_loss(cur_k0.xy_plane, prev_k0.xy_plane)
                loss += F.l1_loss(cur_k0.uv_plane, prev_k0.uv_plane)
                loss += F.l1_loss(cur_k0.xu_plane, prev_k0.xu_plane)
                loss += F.l1_loss(cur_k0.xv_plane, prev_k0.xv_plane)
                loss += F.l1_loss(cur_k0.yu_plane, prev_k0.yu_plane)
                loss += F.l1_loss(cur_k0.yv_plane, prev_k0.yv_plane)
                N += 6
                
        if str(frameid+1) in self.dvgos:
            frameid2 = str(frameid+1)
            cur_k0 = self.dvgos[str(frameid)].k0
            next_k0 = self.dvgos[frameid2].k0
            
            loss += F.l1_loss(cur_k0.xy_plane, next_k0.xy_plane)
            loss += F.l1_loss(cur_k0.uv_plane, next_k0.uv_plane)
            loss += F.l1_loss(cur_k0.xu_plane, next_k0.xu_plane)
            loss += F.l1_loss(cur_k0.xv_plane, next_k0.xv_plane)
            loss += F.l1_loss(cur_k0.yu_plane, next_k0.yu_plane)
            loss += F.l1_loss(cur_k0.yv_plane, next_k0.yv_plane)
            N += 6
            
        if N == 0:
            return loss
        return loss / N

    def update_occupancy_cache(self):
        # Light Field 모드에서는 occupancy cache가 없음
        return 0.0
