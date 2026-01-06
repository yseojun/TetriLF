"""
Light Field Dataset Loader for UVST-based training.
Supports folder structure: basedir/yy_xx/frame_XXXXX.png
"""
import os
import re
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset


class LF_Dataset(Dataset):
    """
    Light Field Dataset for UVST coordinate learning.
    
    Expected folder structure:
        basedir/
            00_00/
                frame_00001.png
                frame_00002.png
                ...
            00_01/
                ...
            yy_xx/
                ...
    
    Where yy = grid_y index, xx = grid_x index
    """
    def __init__(self, basedir, frameids=[], test_views=[], 
                 grid_size_x=None, grid_size_y=None,
                 uv_scale=1.0, st_scale=0.25):
        """
        Args:
            basedir: Path to dataset root
            frameids: List of frame indices to use (e.g., [1, 2, 3, ...])
            test_views: List of view indices for testing
            grid_size_x: Number of views in x direction (auto-detect if None)
            grid_size_y: Number of views in y direction (auto-detect if None)
            uv_scale: Scale factor for UV coordinates
            st_scale: Scale factor for ST coordinates
        """
        super().__init__()
        self.basedir = basedir
        self.frameids = frameids
        self.test_views = test_views
        self.uv_scale = uv_scale
        self.st_scale = st_scale
        
        # Scan directory to find view folders
        view_dirs = sorted([d for d in os.listdir(basedir) 
                           if os.path.isdir(os.path.join(basedir, d)) and '_' in d])
        self.view_dirs = view_dirs
        
        # Auto-detect grid size
        if grid_size_x is None or grid_size_y is None:
            max_y, max_x = 0, 0
            for vd in view_dirs:
                parts = vd.split('_')
                if len(parts) == 2:
                    y, x = int(parts[0]), int(parts[1])
                    max_y = max(max_y, y)
                    max_x = max(max_x, x)
            self.grid_size_y = max_y + 1
            self.grid_size_x = max_x + 1
        else:
            self.grid_size_x = grid_size_x
            self.grid_size_y = grid_size_y
        
        self.num_views = len(view_dirs)
        
        # Get image size and detect frame naming pattern from first view
        first_view = view_dirs[0]
        first_view_path = os.path.join(basedir, first_view)
        
        # Detect frame file pattern and first frame number
        self.frame_pattern, self.first_frame_num = self._detect_frame_pattern(first_view_path)
        
        frames = sorted([f for f in os.listdir(first_view_path) 
                        if f.endswith('.png') or f.endswith('.jpg')])
        first_img = imageio.imread(os.path.join(first_view_path, frames[0]))
        self.H, self.W = first_img.shape[:2]
        
        # Build view info mapping
        self.view_info = {}  # view_idx -> (grid_y, grid_x, view_dir)
        for idx, vd in enumerate(view_dirs):
            parts = vd.split('_')
            grid_y, grid_x = int(parts[0]), int(parts[1])
            self.view_info[idx] = (grid_y, grid_x, vd)
        
        print(f"| LF_Dataset initialized")
        print(f"| basedir: {basedir}")
        print(f"| grid_size: {self.grid_size_y}x{self.grid_size_x} ({self.num_views} views)")
        print(f"| image size: {self.H}x{self.W}")
        print(f"| frame_pattern: {self.frame_pattern}, first_frame: {self.first_frame_num}")
        print(f"| frame_ids: {frameids}")
    
    def _detect_frame_pattern(self, view_path):
        """Detect frame naming pattern and first frame number."""
        files = sorted([f for f in os.listdir(view_path) 
                       if f.endswith('.png') or f.endswith('.jpg')])
        
        if len(files) == 0:
            return 'frame_{:05d}.png', 1
        
        first_file = files[0]
        
        # Try to extract number from filename
        # Pattern: frame_XXXXX.png or XXXXX.png
        match = re.search(r'(\d+)', first_file)
        if match:
            first_num = int(match.group(1))
            num_digits = len(match.group(1))
            
            # Determine pattern
            if first_file.startswith('frame_'):
                ext = os.path.splitext(first_file)[1]
                pattern = f'frame_{{:0{num_digits}d}}{ext}'
            else:
                ext = os.path.splitext(first_file)[1]
                pattern = f'{{:0{num_digits}d}}{ext}'
            
            return pattern, first_num
        
        return 'frame_{:05d}.png', 1
    
    def get_xyuv(self, grid_x, grid_y):
        """
        Generate XYUV coordinates for a given view position.
        Returns: xyuv [H, W, 4] array
        """
        aspect = self.W / self.H
        
        # Normalize grid coordinates to [-1, 1] (Camera Plane: X, Y)
        if self.grid_size_x > 1:
            norm_x = 2.0 * (grid_x / (self.grid_size_x - 1)) - 1
        else:
            norm_x = 0.0
        if self.grid_size_y > 1:
            norm_y = 2.0 * (grid_y / (self.grid_size_y - 1)) - 1
        else:
            norm_y = 0.0
        
        # Create UV meshgrid (Image Plane: U, V)
        u = np.linspace(-1, 1, self.W, dtype=np.float32)
        v = np.linspace(1, -1, self.H, dtype=np.float32) / aspect
        vu = np.meshgrid(u, v)
        
        u_grid = vu[0] * self.uv_scale
        v_grid = vu[1] * self.uv_scale
        
        # X, Y are constant for each view (camera position)
        x_cam = np.ones_like(u_grid) * norm_x * self.st_scale
        y_cam = np.ones_like(v_grid) * norm_y * self.st_scale
        
        # Stack as [H, W, 4] in order (X_cam, Y_cam, U_img, V_img)
        xyuv = np.stack([x_cam, y_cam, u_grid, v_grid], axis=-1)
        
        return xyuv
    
    def get_xyuv_bounds(self):
        """Calculate XYUV min/max bounds."""
        aspect = self.W / self.H
        
        x_min = -1.0 * self.st_scale
        x_max = 1.0 * self.st_scale
        y_min = -1.0 * self.st_scale
        y_max = 1.0 * self.st_scale
        u_min = -1.0 * self.uv_scale
        u_max = 1.0 * self.uv_scale
        v_min = -1.0 / aspect * self.uv_scale
        v_max = 1.0 / aspect * self.uv_scale
        
        xyuv_min = np.array([x_min, y_min, u_min, v_min], dtype=np.float32)
        xyuv_max = np.array([x_max, y_max, u_max, v_max], dtype=np.float32)
        
        return xyuv_min, xyuv_max
    
    def _get_frame_filename(self, frame_id):
        """
        Get actual filename for a frame_id.
        frame_id is 0-indexed from user, but files may start from 1.
        """
        # Map user frame_id (0-indexed) to actual file number
        actual_frame_num = self.first_frame_num + frame_id
        return self.frame_pattern.format(actual_frame_num)
    
    def _find_frame_file(self, view_path, frame_id):
        """Find frame file using detected pattern."""
        # First, try using the detected pattern
        filename = self._get_frame_filename(frame_id)
        test_path = os.path.join(view_path, filename)
        if os.path.exists(test_path):
            return test_path
        
        # Fallback: try multiple naming conventions
        possible_names = [
            f"frame_{frame_id:05d}.png",
            f"{frame_id:05d}.png",
            f"frame_{frame_id:04d}.png",
            f"{frame_id:04d}.png",
            f"frame_{frame_id:03d}.png",
            f"{frame_id:03d}.png",
            f"{frame_id:02d}.png",
            f"{frame_id}.png",
            # Also try jpg format
            f"frame_{frame_id:05d}.jpg",
            f"{frame_id:05d}.jpg",
            f"frame_{frame_id:04d}.jpg",
            f"{frame_id:04d}.jpg",
            f"frame_{frame_id:03d}.jpg",
            f"{frame_id:03d}.jpg",
            f"{frame_id:02d}.jpg",
            f"{frame_id}.jpg",
        ]
        
        for fname in possible_names:
            test_path = os.path.join(view_path, fname)
            if os.path.exists(test_path):
                return test_path
        return None
    
    def read_frame(self, frame_id):
        """
        Read all views for a given frame.
        Returns: images [N_views, H, W, 3], xyuv_coords [N_views, H, W, 4], view_indices
        """
        images = []
        xyuv_coords = []
        view_indices = []
        
        for view_idx, (grid_y, grid_x, view_dir) in self.view_info.items():
            view_path = os.path.join(self.basedir, view_dir)
            frame_path = self._find_frame_file(view_path, frame_id)
            
            if frame_path is None:
                print(f"Warning: frame {frame_id} not found in {view_path}")
                continue
            
            # Read image
            img = imageio.imread(frame_path)
            img = img[..., :3]  # Remove alpha if present
            img = (img / 255.0).astype(np.float32)
            
            # Generate XYUV coordinates
            xyuv = self.get_xyuv(grid_x, grid_y)
            
            images.append(img)
            xyuv_coords.append(xyuv)
            view_indices.append(view_idx)
        
        if len(images) == 0:
            raise ValueError(f"No images found for frame {frame_id}")
        
        images = np.stack(images, axis=0)
        xyuv_coords = np.stack(xyuv_coords, axis=0)
        
        return images, xyuv_coords, view_indices
    
    def __len__(self):
        return len(self.frameids)
    
    def __getitem__(self, idx):
        """
        Get data for one frame (all views).
        Called by DataLoader - enables parallel loading.
        
        Returns:
            images: [N_views, H, W, 3] tensor
            xyuv: [N_views, H, W, 4] tensor
            frame_id: int
        """
        frame_id = self.frameids[idx]
        images, xyuv_coords, view_indices = self.read_frame(frame_id)
        
        images = torch.from_numpy(images).float()
        xyuv_coords = torch.from_numpy(xyuv_coords).float()
        
        print(f'** LF frame {frame_id} loaded: {len(view_indices)} views')
        
        return images, xyuv_coords, frame_id
