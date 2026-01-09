"""
4D Grid Sample with CUDA acceleration

Provides quadrilinear interpolation for 4D grids using custom CUDA kernels.
This extends PyTorch's F.grid_sample (which only supports 2D/3D) to 4D.
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

# JIT compile the CUDA extension
_cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')
_grid_sample_4d = None

def _get_extension():
    """Lazily load the CUDA extension."""
    global _grid_sample_4d
    if _grid_sample_4d is None:
        _grid_sample_4d = load(
            name='grid_sample_4d',
            sources=[
                os.path.join(_cuda_dir, 'grid_sample_4d.cpp'),
                os.path.join(_cuda_dir, 'grid_sample_4d_kernel.cu'),
            ],
            verbose=False
        )
    return _grid_sample_4d


class GridSample4DFunction(Function):
    """
    Autograd Function for 4D grid sampling.
    
    Performs quadrilinear interpolation on a 6D input tensor 
    [N, C, X, Y, U, V] using 4D normalized coordinates.
    """
    
    @staticmethod
    def forward(ctx, input, grid, align_corners=True, padding_mode=0):
        """
        Forward pass of 4D grid sampling.
        
        Args:
            input: [N, C, X, Y, U, V] - 6D input tensor
            grid: [N, N_pts, 4] - normalized coordinates in [-1, 1]
                  The 4 coordinates are (x, y, u, v)
            align_corners: if True, corner pixels are aligned (default: True)
            padding_mode: 0 for zeros, 1 for border (default: 0)
        
        Returns:
            output: [N, C, N_pts] - sampled values
        """
        ext = _get_extension()
        output = ext.forward(input, grid, align_corners, padding_mode)
        
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of 4D grid sampling.
        
        Args:
            grad_output: [N, C, N_pts] - gradient from upstream
        
        Returns:
            grad_input: [N, C, X, Y, U, V] - gradient w.r.t. input
            None for other inputs (grid, align_corners, padding_mode)
        """
        input, grid = ctx.saved_tensors
        align_corners = ctx.align_corners
        padding_mode = ctx.padding_mode
        
        ext = _get_extension()
        grad_input = ext.backward(
            grad_output.contiguous(), 
            input, 
            grid, 
            align_corners, 
            padding_mode
        )
        
        # Return gradients for: input, grid, align_corners, padding_mode
        # Only input gradient is computed; grid gradient is not needed for our use case
        return grad_input, None, None, None


def grid_sample_4d(input, grid, align_corners=True, padding_mode='zeros'):
    """
    4D grid sampling with quadrilinear interpolation.
    
    Samples values from a 6D input tensor using 4D normalized coordinates.
    This extends PyTorch's F.grid_sample to 4D.
    
    Args:
        input: torch.Tensor of shape [N, C, X, Y, U, V]
            The 6D input tensor to sample from.
        grid: torch.Tensor of shape [N, N_pts, 4]
            Normalized coordinates in [-1, 1] for sampling.
            The 4 coordinates are (x, y, u, v).
        align_corners: bool, optional (default: True)
            If True, the corner pixels of input are aligned with
            the corner coordinates [-1, 1].
        padding_mode: str, optional (default: 'zeros')
            Padding mode for out-of-bound values.
            Options: 'zeros' or 'border'
    
    Returns:
        output: torch.Tensor of shape [N, C, N_pts]
            Sampled values from input at the grid coordinates.
    
    Example:
        >>> input = torch.randn(1, 16, 8, 8, 64, 64, device='cuda')  # [N, C, X, Y, U, V]
        >>> grid = torch.randn(1, 1000, 4, device='cuda')  # [N, N_pts, 4] in [-1, 1]
        >>> output = grid_sample_4d(input, grid)  # [1, 16, 1000]
    """
    # Convert padding_mode string to int
    if isinstance(padding_mode, str):
        padding_mode_int = {'zeros': 0, 'border': 1}.get(padding_mode, 0)
    else:
        padding_mode_int = padding_mode
    
    return GridSample4DFunction.apply(input, grid, align_corners, padding_mode_int)


class GridSample4D(nn.Module):
    """
    Module wrapper for 4D grid sampling.
    
    Args:
        align_corners: bool, optional (default: True)
        padding_mode: str, optional (default: 'zeros')
    """
    
    def __init__(self, align_corners=True, padding_mode='zeros'):
        super().__init__()
        self.align_corners = align_corners
        self.padding_mode = padding_mode
    
    def forward(self, input, grid):
        """
        Args:
            input: [N, C, X, Y, U, V]
            grid: [N, N_pts, 4]
        
        Returns:
            output: [N, C, N_pts]
        """
        return grid_sample_4d(input, grid, self.align_corners, self.padding_mode)
    
    def extra_repr(self):
        return f'align_corners={self.align_corners}, padding_mode={self.padding_mode}'


class Interpolate4DFunction(Function):
    """
    Autograd Function for 4D interpolation (quadrilinear).
    
    Resizes a 6D input tensor [N, C, X, Y, U, V] to new spatial dimensions.
    """
    
    @staticmethod
    def forward(ctx, input, size, align_corners=True):
        """
        Forward pass of 4D interpolation.
        
        Args:
            input: [N, C, X, Y, U, V] - 6D input tensor
            size: tuple/list of 4 ints [X_out, Y_out, U_out, V_out]
            align_corners: if True, corner pixels are aligned (default: True)
        
        Returns:
            output: [N, C, X_out, Y_out, U_out, V_out] - interpolated tensor
        """
        ext = _get_extension()
        output = ext.interpolate_forward(input, list(size), align_corners)
        
        ctx.save_for_backward(torch.tensor(input.shape[2:]))  # Save input spatial size
        ctx.align_corners = align_corners
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of 4D interpolation.
        
        Args:
            grad_output: [N, C, X_out, Y_out, U_out, V_out]
        
        Returns:
            grad_input: [N, C, X_in, Y_in, U_in, V_in]
            None for size, align_corners
        """
        input_size_tensor, = ctx.saved_tensors
        input_size = input_size_tensor.tolist()
        align_corners = ctx.align_corners
        
        ext = _get_extension()
        grad_input = ext.interpolate_backward(
            grad_output.contiguous(),
            input_size,
            align_corners
        )
        
        return grad_input, None, None


def interpolate_4d(input, size, align_corners=True):
    """
    4D interpolation with quadrilinear method.
    
    Resizes a 6D input tensor to new spatial dimensions.
    Similar to F.interpolate but extended to 4D spatial dimensions.
    
    Args:
        input: torch.Tensor of shape [N, C, X, Y, U, V]
            The 6D input tensor to resize.
        size: tuple or list of 4 ints [X_out, Y_out, U_out, V_out]
            Target output spatial size.
        align_corners: bool, optional (default: True)
            If True, the corner pixels of input and output tensors are aligned.
    
    Returns:
        output: torch.Tensor of shape [N, C, X_out, Y_out, U_out, V_out]
            Resized tensor.
    
    Example:
        >>> input = torch.randn(1, 16, 4, 4, 32, 32, device='cuda')
        >>> output = interpolate_4d(input, size=[8, 8, 64, 64])  # [1, 16, 8, 8, 64, 64]
    """
    return Interpolate4DFunction.apply(input, size, align_corners)


class Interpolate4D(nn.Module):
    """
    Module wrapper for 4D interpolation.
    
    Args:
        size: tuple of 4 ints [X, Y, U, V] - target output size
        align_corners: bool, optional (default: True)
    """
    
    def __init__(self, size, align_corners=True):
        super().__init__()
        self.size = size
        self.align_corners = align_corners
    
    def forward(self, input):
        """
        Args:
            input: [N, C, X_in, Y_in, U_in, V_in]
        
        Returns:
            output: [N, C, X_out, Y_out, U_out, V_out]
        """
        return interpolate_4d(input, self.size, self.align_corners)
    
    def extra_repr(self):
        return f'size={self.size}, align_corners={self.align_corners}'

