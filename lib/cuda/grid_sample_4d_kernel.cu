#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/*
    4D Grid Sample (Quadrilinear Interpolation) CUDA Kernel
    
    Input tensor: [N, C, D1, D2, D3, D4] = [batch, channels, X, Y, U, V]
    Grid tensor: [N, O1, O2, O3, O4, 4] = [batch, output spatial dims..., 4D coords]
    Output tensor: [N, C, O1, O2, O3, O4]
    
    For our use case (point queries):
    Input: [1, C, X, Y, U, V]
    Grid: [1, 1, 1, 1, N_pts, 4]
    Output: [1, C, 1, 1, 1, N_pts]
*/

// Helper function to compute grid index with boundary check
template <typename scalar_t>
__device__ __forceinline__ scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int size,
    bool align_corners) {
    if (align_corners) {
        // coord: [-1, 1] -> [0, size-1]
        return ((coord + 1) / 2) * (size - 1);
    } else {
        // coord: [-1, 1] -> [-0.5, size-0.5]
        return ((coord + 1) * size - 1) / 2;
    }
}

// Clamp index to valid range
__device__ __forceinline__ int clamp_index(int idx, int size) {
    return max(0, min(idx, size - 1));
}

// Check if index is within bounds
__device__ __forceinline__ bool within_bounds(int idx, int size) {
    return idx >= 0 && idx < size;
}

/*
    Forward Kernel: Quadrilinear interpolation
    Samples from a 6D input tensor (N, C, X, Y, U, V) using 4D coordinates
*/
template <typename scalar_t>
__global__ void grid_sample_4d_forward_kernel(
    const scalar_t* __restrict__ input,     // [N, C, X, Y, U, V]
    const scalar_t* __restrict__ grid,      // [N, N_pts, 4] (simplified for point queries)
    scalar_t* __restrict__ output,          // [N, C, N_pts]
    const int N,
    const int C,
    const int inp_X,
    const int inp_Y,
    const int inp_U,
    const int inp_V,
    const int n_pts,
    const bool align_corners,
    const int padding_mode) {  // 0: zeros, 1: border, 2: reflection
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = N * n_pts;
    
    if (idx >= total_threads) return;
    
    const int n = idx / n_pts;
    const int pt = idx % n_pts;
    
    // Get 4D coordinates from grid
    const int grid_offset = n * n_pts * 4 + pt * 4;
    const scalar_t x = grid[grid_offset + 0];
    const scalar_t y = grid[grid_offset + 1];
    const scalar_t u = grid[grid_offset + 2];
    const scalar_t v = grid[grid_offset + 3];
    
    // Convert normalized coordinates to input indices
    scalar_t ix = grid_sampler_compute_source_index(x, inp_X, align_corners);
    scalar_t iy = grid_sampler_compute_source_index(y, inp_Y, align_corners);
    scalar_t iu = grid_sampler_compute_source_index(u, inp_U, align_corners);
    scalar_t iv = grid_sampler_compute_source_index(v, inp_V, align_corners);
    
    // Get corner indices for 4D interpolation (16 corners in 4D hypercube)
    int ix0 = static_cast<int>(floorf(ix));
    int iy0 = static_cast<int>(floorf(iy));
    int iu0 = static_cast<int>(floorf(iu));
    int iv0 = static_cast<int>(floorf(iv));
    
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iu1 = iu0 + 1;
    int iv1 = iv0 + 1;
    
    // Compute interpolation weights
    scalar_t wx1 = ix - ix0;
    scalar_t wy1 = iy - iy0;
    scalar_t wu1 = iu - iu0;
    scalar_t wv1 = iv - iv0;
    
    scalar_t wx0 = 1 - wx1;
    scalar_t wy0 = 1 - wy1;
    scalar_t wu0 = 1 - wu1;
    scalar_t wv0 = 1 - wv1;
    
    // Input strides
    const int stride_n = C * inp_X * inp_Y * inp_U * inp_V;
    const int stride_c = inp_X * inp_Y * inp_U * inp_V;
    const int stride_x = inp_Y * inp_U * inp_V;
    const int stride_y = inp_U * inp_V;
    const int stride_u = inp_V;
    const int stride_v = 1;
    
    // Helper lambda to get input value with boundary handling
    auto get_value = [&](int c, int x_idx, int y_idx, int u_idx, int v_idx) -> scalar_t {
        if (padding_mode == 0) {  // zeros padding
            if (!within_bounds(x_idx, inp_X) || !within_bounds(y_idx, inp_Y) ||
                !within_bounds(u_idx, inp_U) || !within_bounds(v_idx, inp_V)) {
                return static_cast<scalar_t>(0);
            }
        } else {  // border padding
            x_idx = clamp_index(x_idx, inp_X);
            y_idx = clamp_index(y_idx, inp_Y);
            u_idx = clamp_index(u_idx, inp_U);
            v_idx = clamp_index(v_idx, inp_V);
        }
        
        const int input_offset = n * stride_n + c * stride_c + 
                                 x_idx * stride_x + y_idx * stride_y + 
                                 u_idx * stride_u + v_idx * stride_v;
        return input[input_offset];
    };
    
    // Iterate over channels
    for (int c = 0; c < C; c++) {
        scalar_t result = static_cast<scalar_t>(0);
        
        // 16-point quadrilinear interpolation (2^4 corners)
        // (x0, y0, u0, v0)
        result += wx0 * wy0 * wu0 * wv0 * get_value(c, ix0, iy0, iu0, iv0);
        // (x0, y0, u0, v1)
        result += wx0 * wy0 * wu0 * wv1 * get_value(c, ix0, iy0, iu0, iv1);
        // (x0, y0, u1, v0)
        result += wx0 * wy0 * wu1 * wv0 * get_value(c, ix0, iy0, iu1, iv0);
        // (x0, y0, u1, v1)
        result += wx0 * wy0 * wu1 * wv1 * get_value(c, ix0, iy0, iu1, iv1);
        // (x0, y1, u0, v0)
        result += wx0 * wy1 * wu0 * wv0 * get_value(c, ix0, iy1, iu0, iv0);
        // (x0, y1, u0, v1)
        result += wx0 * wy1 * wu0 * wv1 * get_value(c, ix0, iy1, iu0, iv1);
        // (x0, y1, u1, v0)
        result += wx0 * wy1 * wu1 * wv0 * get_value(c, ix0, iy1, iu1, iv0);
        // (x0, y1, u1, v1)
        result += wx0 * wy1 * wu1 * wv1 * get_value(c, ix0, iy1, iu1, iv1);
        // (x1, y0, u0, v0)
        result += wx1 * wy0 * wu0 * wv0 * get_value(c, ix1, iy0, iu0, iv0);
        // (x1, y0, u0, v1)
        result += wx1 * wy0 * wu0 * wv1 * get_value(c, ix1, iy0, iu0, iv1);
        // (x1, y0, u1, v0)
        result += wx1 * wy0 * wu1 * wv0 * get_value(c, ix1, iy0, iu1, iv0);
        // (x1, y0, u1, v1)
        result += wx1 * wy0 * wu1 * wv1 * get_value(c, ix1, iy0, iu1, iv1);
        // (x1, y1, u0, v0)
        result += wx1 * wy1 * wu0 * wv0 * get_value(c, ix1, iy1, iu0, iv0);
        // (x1, y1, u0, v1)
        result += wx1 * wy1 * wu0 * wv1 * get_value(c, ix1, iy1, iu0, iv1);
        // (x1, y1, u1, v0)
        result += wx1 * wy1 * wu1 * wv0 * get_value(c, ix1, iy1, iu1, iv0);
        // (x1, y1, u1, v1)
        result += wx1 * wy1 * wu1 * wv1 * get_value(c, ix1, iy1, iu1, iv1);
        
        // Write output: [N, C, N_pts]
        const int output_offset = n * C * n_pts + c * n_pts + pt;
        output[output_offset] = result;
    }
}

/*
    Backward Kernel: Compute gradients for input tensor
    Uses atomic operations to accumulate gradients
*/
template <typename scalar_t>
__global__ void grid_sample_4d_backward_kernel(
    const scalar_t* __restrict__ grad_output,  // [N, C, N_pts]
    const scalar_t* __restrict__ input,        // [N, C, X, Y, U, V]
    const scalar_t* __restrict__ grid,         // [N, N_pts, 4]
    scalar_t* __restrict__ grad_input,         // [N, C, X, Y, U, V]
    const int N,
    const int C,
    const int inp_X,
    const int inp_Y,
    const int inp_U,
    const int inp_V,
    const int n_pts,
    const bool align_corners,
    const int padding_mode) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = N * n_pts;
    
    if (idx >= total_threads) return;
    
    const int n = idx / n_pts;
    const int pt = idx % n_pts;
    
    // Get 4D coordinates from grid
    const int grid_offset = n * n_pts * 4 + pt * 4;
    const scalar_t x = grid[grid_offset + 0];
    const scalar_t y = grid[grid_offset + 1];
    const scalar_t u = grid[grid_offset + 2];
    const scalar_t v = grid[grid_offset + 3];
    
    // Convert normalized coordinates to input indices
    scalar_t ix = grid_sampler_compute_source_index(x, inp_X, align_corners);
    scalar_t iy = grid_sampler_compute_source_index(y, inp_Y, align_corners);
    scalar_t iu = grid_sampler_compute_source_index(u, inp_U, align_corners);
    scalar_t iv = grid_sampler_compute_source_index(v, inp_V, align_corners);
    
    // Get corner indices
    int ix0 = static_cast<int>(floorf(ix));
    int iy0 = static_cast<int>(floorf(iy));
    int iu0 = static_cast<int>(floorf(iu));
    int iv0 = static_cast<int>(floorf(iv));
    
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iu1 = iu0 + 1;
    int iv1 = iv0 + 1;
    
    // Compute interpolation weights
    scalar_t wx1 = ix - ix0;
    scalar_t wy1 = iy - iy0;
    scalar_t wu1 = iu - iu0;
    scalar_t wv1 = iv - iv0;
    
    scalar_t wx0 = 1 - wx1;
    scalar_t wy0 = 1 - wy1;
    scalar_t wu0 = 1 - wu1;
    scalar_t wv0 = 1 - wv1;
    
    // Input strides
    const int stride_n = C * inp_X * inp_Y * inp_U * inp_V;
    const int stride_c = inp_X * inp_Y * inp_U * inp_V;
    const int stride_x = inp_Y * inp_U * inp_V;
    const int stride_y = inp_U * inp_V;
    const int stride_u = inp_V;
    const int stride_v = 1;
    
    // Helper to add gradient with boundary check
    auto add_grad = [&](int c, int x_idx, int y_idx, int u_idx, int v_idx, scalar_t weight, scalar_t grad_val) {
        if (padding_mode == 0) {  // zeros padding - skip out of bounds
            if (!within_bounds(x_idx, inp_X) || !within_bounds(y_idx, inp_Y) ||
                !within_bounds(u_idx, inp_U) || !within_bounds(v_idx, inp_V)) {
                return;
            }
        } else {  // border padding
            x_idx = clamp_index(x_idx, inp_X);
            y_idx = clamp_index(y_idx, inp_Y);
            u_idx = clamp_index(u_idx, inp_U);
            v_idx = clamp_index(v_idx, inp_V);
        }
        
        const int input_offset = n * stride_n + c * stride_c + 
                                 x_idx * stride_x + y_idx * stride_y + 
                                 u_idx * stride_u + v_idx * stride_v;
        atomicAdd(&grad_input[input_offset], weight * grad_val);
    };
    
    // Iterate over channels
    for (int c = 0; c < C; c++) {
        const int grad_out_offset = n * C * n_pts + c * n_pts + pt;
        const scalar_t grad_val = grad_output[grad_out_offset];
        
        // Distribute gradient to 16 corners
        add_grad(c, ix0, iy0, iu0, iv0, wx0 * wy0 * wu0 * wv0, grad_val);
        add_grad(c, ix0, iy0, iu0, iv1, wx0 * wy0 * wu0 * wv1, grad_val);
        add_grad(c, ix0, iy0, iu1, iv0, wx0 * wy0 * wu1 * wv0, grad_val);
        add_grad(c, ix0, iy0, iu1, iv1, wx0 * wy0 * wu1 * wv1, grad_val);
        add_grad(c, ix0, iy1, iu0, iv0, wx0 * wy1 * wu0 * wv0, grad_val);
        add_grad(c, ix0, iy1, iu0, iv1, wx0 * wy1 * wu0 * wv1, grad_val);
        add_grad(c, ix0, iy1, iu1, iv0, wx0 * wy1 * wu1 * wv0, grad_val);
        add_grad(c, ix0, iy1, iu1, iv1, wx0 * wy1 * wu1 * wv1, grad_val);
        add_grad(c, ix1, iy0, iu0, iv0, wx1 * wy0 * wu0 * wv0, grad_val);
        add_grad(c, ix1, iy0, iu0, iv1, wx1 * wy0 * wu0 * wv1, grad_val);
        add_grad(c, ix1, iy0, iu1, iv0, wx1 * wy0 * wu1 * wv0, grad_val);
        add_grad(c, ix1, iy0, iu1, iv1, wx1 * wy0 * wu1 * wv1, grad_val);
        add_grad(c, ix1, iy1, iu0, iv0, wx1 * wy1 * wu0 * wv0, grad_val);
        add_grad(c, ix1, iy1, iu0, iv1, wx1 * wy1 * wu0 * wv1, grad_val);
        add_grad(c, ix1, iy1, iu1, iv0, wx1 * wy1 * wu1 * wv0, grad_val);
        add_grad(c, ix1, iy1, iu1, iv1, wx1 * wy1 * wu1 * wv1, grad_val);
    }
}

// C++ wrapper functions

torch::Tensor grid_sample_4d_forward_cuda(
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners,
    int padding_mode) {
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int inp_X = input.size(2);
    const int inp_Y = input.size(3);
    const int inp_U = input.size(4);
    const int inp_V = input.size(5);
    const int n_pts = grid.size(1);
    
    auto output = torch::zeros({N, C, n_pts}, input.options());
    
    const int threads = 256;
    const int total = N * n_pts;
    const int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sample_4d_forward_cuda", ([&] {
        grid_sample_4d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            grid.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, inp_X, inp_Y, inp_U, inp_V, n_pts,
            align_corners, padding_mode);
    }));
    
    return output;
}

torch::Tensor grid_sample_4d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners,
    int padding_mode) {
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int inp_X = input.size(2);
    const int inp_Y = input.size(3);
    const int inp_U = input.size(4);
    const int inp_V = input.size(5);
    const int n_pts = grid.size(1);
    
    auto grad_input = torch::zeros_like(input);
    
    const int threads = 256;
    const int total = N * n_pts;
    const int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sample_4d_backward_cuda", ([&] {
        grid_sample_4d_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            grid.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            N, C, inp_X, inp_Y, inp_U, inp_V, n_pts,
            align_corners, padding_mode);
    }));
    
    return grad_input;
}

/*
    4D Interpolate (Quadrilinear) CUDA Kernel
    Resizes a 6D tensor [N, C, X, Y, U, V] to new spatial dimensions
    
    Input: [N, C, X_in, Y_in, U_in, V_in]
    Output: [N, C, X_out, Y_out, U_out, V_out]
*/
template <typename scalar_t>
__global__ void interpolate_4d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N,
    const int C,
    const int inp_X, const int inp_Y, const int inp_U, const int inp_V,
    const int out_X, const int out_Y, const int out_U, const int out_V,
    const bool align_corners) {
    
    // Total output elements per batch*channel
    const int out_spatial = out_X * out_Y * out_U * out_V;
    const int total_elements = N * C * out_spatial;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Compute indices
    const int n = idx / (C * out_spatial);
    const int c = (idx / out_spatial) % C;
    const int spatial_idx = idx % out_spatial;
    
    const int ox = spatial_idx / (out_Y * out_U * out_V);
    const int oy = (spatial_idx / (out_U * out_V)) % out_Y;
    const int ou = (spatial_idx / out_V) % out_U;
    const int ov = spatial_idx % out_V;
    
    // Compute source coordinates
    scalar_t sx, sy, su, sv;
    if (align_corners) {
        // Map [0, out_size-1] -> [0, inp_size-1]
        sx = (out_X > 1) ? (static_cast<scalar_t>(ox) * (inp_X - 1) / (out_X - 1)) : 0;
        sy = (out_Y > 1) ? (static_cast<scalar_t>(oy) * (inp_Y - 1) / (out_Y - 1)) : 0;
        su = (out_U > 1) ? (static_cast<scalar_t>(ou) * (inp_U - 1) / (out_U - 1)) : 0;
        sv = (out_V > 1) ? (static_cast<scalar_t>(ov) * (inp_V - 1) / (out_V - 1)) : 0;
    } else {
        // Map output center to input center
        sx = (static_cast<scalar_t>(ox) + 0.5) * inp_X / out_X - 0.5;
        sy = (static_cast<scalar_t>(oy) + 0.5) * inp_Y / out_Y - 0.5;
        su = (static_cast<scalar_t>(ou) + 0.5) * inp_U / out_U - 0.5;
        sv = (static_cast<scalar_t>(ov) + 0.5) * inp_V / out_V - 0.5;
    }
    
    // Get floor indices
    int ix0 = static_cast<int>(floorf(sx));
    int iy0 = static_cast<int>(floorf(sy));
    int iu0 = static_cast<int>(floorf(su));
    int iv0 = static_cast<int>(floorf(sv));
    
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iu1 = iu0 + 1;
    int iv1 = iv0 + 1;
    
    // Compute weights
    scalar_t wx1 = sx - ix0;
    scalar_t wy1 = sy - iy0;
    scalar_t wu1 = su - iu0;
    scalar_t wv1 = sv - iv0;
    
    scalar_t wx0 = 1 - wx1;
    scalar_t wy0 = 1 - wy1;
    scalar_t wu0 = 1 - wu1;
    scalar_t wv0 = 1 - wv1;
    
    // Clamp indices
    ix0 = max(0, min(ix0, inp_X - 1));
    iy0 = max(0, min(iy0, inp_Y - 1));
    iu0 = max(0, min(iu0, inp_U - 1));
    iv0 = max(0, min(iv0, inp_V - 1));
    ix1 = max(0, min(ix1, inp_X - 1));
    iy1 = max(0, min(iy1, inp_Y - 1));
    iu1 = max(0, min(iu1, inp_U - 1));
    iv1 = max(0, min(iv1, inp_V - 1));
    
    // Input strides
    const int inp_stride_n = C * inp_X * inp_Y * inp_U * inp_V;
    const int inp_stride_c = inp_X * inp_Y * inp_U * inp_V;
    const int inp_stride_x = inp_Y * inp_U * inp_V;
    const int inp_stride_y = inp_U * inp_V;
    const int inp_stride_u = inp_V;
    
    // Base offset for this (n, c)
    const int base_offset = n * inp_stride_n + c * inp_stride_c;
    
    // Helper to get input value
    auto get_val = [&](int x, int y, int u, int v) -> scalar_t {
        return input[base_offset + x * inp_stride_x + y * inp_stride_y + u * inp_stride_u + v];
    };
    
    // Quadrilinear interpolation (16 corners)
    scalar_t result = static_cast<scalar_t>(0);
    result += wx0 * wy0 * wu0 * wv0 * get_val(ix0, iy0, iu0, iv0);
    result += wx0 * wy0 * wu0 * wv1 * get_val(ix0, iy0, iu0, iv1);
    result += wx0 * wy0 * wu1 * wv0 * get_val(ix0, iy0, iu1, iv0);
    result += wx0 * wy0 * wu1 * wv1 * get_val(ix0, iy0, iu1, iv1);
    result += wx0 * wy1 * wu0 * wv0 * get_val(ix0, iy1, iu0, iv0);
    result += wx0 * wy1 * wu0 * wv1 * get_val(ix0, iy1, iu0, iv1);
    result += wx0 * wy1 * wu1 * wv0 * get_val(ix0, iy1, iu1, iv0);
    result += wx0 * wy1 * wu1 * wv1 * get_val(ix0, iy1, iu1, iv1);
    result += wx1 * wy0 * wu0 * wv0 * get_val(ix1, iy0, iu0, iv0);
    result += wx1 * wy0 * wu0 * wv1 * get_val(ix1, iy0, iu0, iv1);
    result += wx1 * wy0 * wu1 * wv0 * get_val(ix1, iy0, iu1, iv0);
    result += wx1 * wy0 * wu1 * wv1 * get_val(ix1, iy0, iu1, iv1);
    result += wx1 * wy1 * wu0 * wv0 * get_val(ix1, iy1, iu0, iv0);
    result += wx1 * wy1 * wu0 * wv1 * get_val(ix1, iy1, iu0, iv1);
    result += wx1 * wy1 * wu1 * wv0 * get_val(ix1, iy1, iu1, iv0);
    result += wx1 * wy1 * wu1 * wv1 * get_val(ix1, iy1, iu1, iv1);
    
    output[idx] = result;
}

/*
    Backward kernel for 4D interpolate
    Distributes gradients from output back to input using atomic operations
*/
template <typename scalar_t>
__global__ void interpolate_4d_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    const int N,
    const int C,
    const int inp_X, const int inp_Y, const int inp_U, const int inp_V,
    const int out_X, const int out_Y, const int out_U, const int out_V,
    const bool align_corners) {
    
    const int out_spatial = out_X * out_Y * out_U * out_V;
    const int total_elements = N * C * out_spatial;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    const int n = idx / (C * out_spatial);
    const int c = (idx / out_spatial) % C;
    const int spatial_idx = idx % out_spatial;
    
    const int ox = spatial_idx / (out_Y * out_U * out_V);
    const int oy = (spatial_idx / (out_U * out_V)) % out_Y;
    const int ou = (spatial_idx / out_V) % out_U;
    const int ov = spatial_idx % out_V;
    
    // Compute source coordinates (same as forward)
    scalar_t sx, sy, su, sv;
    if (align_corners) {
        sx = (out_X > 1) ? (static_cast<scalar_t>(ox) * (inp_X - 1) / (out_X - 1)) : 0;
        sy = (out_Y > 1) ? (static_cast<scalar_t>(oy) * (inp_Y - 1) / (out_Y - 1)) : 0;
        su = (out_U > 1) ? (static_cast<scalar_t>(ou) * (inp_U - 1) / (out_U - 1)) : 0;
        sv = (out_V > 1) ? (static_cast<scalar_t>(ov) * (inp_V - 1) / (out_V - 1)) : 0;
    } else {
        sx = (static_cast<scalar_t>(ox) + 0.5) * inp_X / out_X - 0.5;
        sy = (static_cast<scalar_t>(oy) + 0.5) * inp_Y / out_Y - 0.5;
        su = (static_cast<scalar_t>(ou) + 0.5) * inp_U / out_U - 0.5;
        sv = (static_cast<scalar_t>(ov) + 0.5) * inp_V / out_V - 0.5;
    }
    
    int ix0 = static_cast<int>(floorf(sx));
    int iy0 = static_cast<int>(floorf(sy));
    int iu0 = static_cast<int>(floorf(su));
    int iv0 = static_cast<int>(floorf(sv));
    
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iu1 = iu0 + 1;
    int iv1 = iv0 + 1;
    
    scalar_t wx1 = sx - ix0;
    scalar_t wy1 = sy - iy0;
    scalar_t wu1 = su - iu0;
    scalar_t wv1 = sv - iv0;
    
    scalar_t wx0 = 1 - wx1;
    scalar_t wy0 = 1 - wy1;
    scalar_t wu0 = 1 - wu1;
    scalar_t wv0 = 1 - wv1;
    
    // Clamp
    ix0 = max(0, min(ix0, inp_X - 1));
    iy0 = max(0, min(iy0, inp_Y - 1));
    iu0 = max(0, min(iu0, inp_U - 1));
    iv0 = max(0, min(iv0, inp_V - 1));
    ix1 = max(0, min(ix1, inp_X - 1));
    iy1 = max(0, min(iy1, inp_Y - 1));
    iu1 = max(0, min(iu1, inp_U - 1));
    iv1 = max(0, min(iv1, inp_V - 1));
    
    const int inp_stride_n = C * inp_X * inp_Y * inp_U * inp_V;
    const int inp_stride_c = inp_X * inp_Y * inp_U * inp_V;
    const int inp_stride_x = inp_Y * inp_U * inp_V;
    const int inp_stride_y = inp_U * inp_V;
    const int inp_stride_u = inp_V;
    
    const int base_offset = n * inp_stride_n + c * inp_stride_c;
    const scalar_t grad_val = grad_output[idx];
    
    auto add_grad = [&](int x, int y, int u, int v, scalar_t weight) {
        const int offset = base_offset + x * inp_stride_x + y * inp_stride_y + u * inp_stride_u + v;
        atomicAdd(&grad_input[offset], weight * grad_val);
    };
    
    // Distribute gradient to 16 corners
    add_grad(ix0, iy0, iu0, iv0, wx0 * wy0 * wu0 * wv0);
    add_grad(ix0, iy0, iu0, iv1, wx0 * wy0 * wu0 * wv1);
    add_grad(ix0, iy0, iu1, iv0, wx0 * wy0 * wu1 * wv0);
    add_grad(ix0, iy0, iu1, iv1, wx0 * wy0 * wu1 * wv1);
    add_grad(ix0, iy1, iu0, iv0, wx0 * wy1 * wu0 * wv0);
    add_grad(ix0, iy1, iu0, iv1, wx0 * wy1 * wu0 * wv1);
    add_grad(ix0, iy1, iu1, iv0, wx0 * wy1 * wu1 * wv0);
    add_grad(ix0, iy1, iu1, iv1, wx0 * wy1 * wu1 * wv1);
    add_grad(ix1, iy0, iu0, iv0, wx1 * wy0 * wu0 * wv0);
    add_grad(ix1, iy0, iu0, iv1, wx1 * wy0 * wu0 * wv1);
    add_grad(ix1, iy0, iu1, iv0, wx1 * wy0 * wu1 * wv0);
    add_grad(ix1, iy0, iu1, iv1, wx1 * wy0 * wu1 * wv1);
    add_grad(ix1, iy1, iu0, iv0, wx1 * wy1 * wu0 * wv0);
    add_grad(ix1, iy1, iu0, iv1, wx1 * wy1 * wu0 * wv1);
    add_grad(ix1, iy1, iu1, iv0, wx1 * wy1 * wu1 * wv0);
    add_grad(ix1, iy1, iu1, iv1, wx1 * wy1 * wu1 * wv1);
}

// C++ wrapper for interpolate_4d forward
torch::Tensor interpolate_4d_forward_cuda(
    torch::Tensor input,
    int out_X, int out_Y, int out_U, int out_V,
    bool align_corners) {
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int inp_X = input.size(2);
    const int inp_Y = input.size(3);
    const int inp_U = input.size(4);
    const int inp_V = input.size(5);
    
    auto output = torch::empty({N, C, out_X, out_Y, out_U, out_V}, input.options());
    
    const int threads = 256;
    const int total = N * C * out_X * out_Y * out_U * out_V;
    const int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "interpolate_4d_forward_cuda", ([&] {
        interpolate_4d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, inp_X, inp_Y, inp_U, inp_V,
            out_X, out_Y, out_U, out_V,
            align_corners);
    }));
    
    return output;
}

// C++ wrapper for interpolate_4d backward
torch::Tensor interpolate_4d_backward_cuda(
    torch::Tensor grad_output,
    int inp_X, int inp_Y, int inp_U, int inp_V,
    bool align_corners) {
    
    const int N = grad_output.size(0);
    const int C = grad_output.size(1);
    const int out_X = grad_output.size(2);
    const int out_Y = grad_output.size(3);
    const int out_U = grad_output.size(4);
    const int out_V = grad_output.size(5);
    
    auto grad_input = torch::zeros({N, C, inp_X, inp_Y, inp_U, inp_V}, grad_output.options());
    
    const int threads = 256;
    const int total = N * C * out_X * out_Y * out_U * out_V;
    const int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "interpolate_4d_backward_cuda", ([&] {
        interpolate_4d_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            N, C, inp_X, inp_Y, inp_U, inp_V,
            out_X, out_Y, out_U, out_V,
            align_corners);
    }));
    
    return grad_input;
}
