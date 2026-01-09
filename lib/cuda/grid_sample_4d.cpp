#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor grid_sample_4d_forward_cuda(
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners,
    int padding_mode);

torch::Tensor grid_sample_4d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners,
    int padding_mode);

torch::Tensor interpolate_4d_forward_cuda(
    torch::Tensor input,
    int out_X, int out_Y, int out_U, int out_V,
    bool align_corners);

torch::Tensor interpolate_4d_backward_cuda(
    torch::Tensor grad_output,
    int inp_X, int inp_Y, int inp_U, int inp_V,
    bool align_corners);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor grid_sample_4d_forward(
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners,
    int padding_mode) {
    /*
    4D Grid Sample Forward Pass
    
    Args:
        input: [N, C, X, Y, U, V] - 6D input tensor
        grid: [N, N_pts, 4] - normalized coordinates in [-1, 1]
              The 4 coordinates are (x, y, u, v)
        align_corners: if True, the corner pixels are aligned
        padding_mode: 0 for zeros, 1 for border
    
    Returns:
        output: [N, C, N_pts] - sampled values
    */
    CHECK_INPUT(input);
    CHECK_INPUT(grid);
    
    TORCH_CHECK(input.dim() == 6, "input must be 6D tensor [N, C, X, Y, U, V]");
    TORCH_CHECK(grid.dim() == 3, "grid must be 3D tensor [N, N_pts, 4]");
    TORCH_CHECK(grid.size(2) == 4, "grid last dimension must be 4 (x, y, u, v coordinates)");
    TORCH_CHECK(input.size(0) == grid.size(0), "input and grid must have same batch size");
    
    return grid_sample_4d_forward_cuda(input, grid, align_corners, padding_mode);
}

torch::Tensor grid_sample_4d_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners,
    int padding_mode) {
    /*
    4D Grid Sample Backward Pass
    
    Args:
        grad_output: [N, C, N_pts] - gradient from upstream
        input: [N, C, X, Y, U, V] - original input tensor
        grid: [N, N_pts, 4] - normalized coordinates
        align_corners: if True, the corner pixels are aligned
        padding_mode: 0 for zeros, 1 for border
    
    Returns:
        grad_input: [N, C, X, Y, U, V] - gradient w.r.t. input
    */
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(grid);
    
    return grid_sample_4d_backward_cuda(grad_output, input, grid, align_corners, padding_mode);
}

torch::Tensor interpolate_4d_forward(
    torch::Tensor input,
    std::vector<int64_t> size,
    bool align_corners) {
    /*
    4D Interpolate Forward Pass (Quadrilinear)
    
    Args:
        input: [N, C, X, Y, U, V] - 6D input tensor
        size: [X_out, Y_out, U_out, V_out] - target output size
        align_corners: if True, corner pixels are aligned
    
    Returns:
        output: [N, C, X_out, Y_out, U_out, V_out] - interpolated tensor
    */
    CHECK_INPUT(input);
    
    TORCH_CHECK(input.dim() == 6, "input must be 6D tensor [N, C, X, Y, U, V]");
    TORCH_CHECK(size.size() == 4, "size must have 4 elements [X, Y, U, V]");
    
    return interpolate_4d_forward_cuda(
        input,
        static_cast<int>(size[0]), static_cast<int>(size[1]),
        static_cast<int>(size[2]), static_cast<int>(size[3]),
        align_corners);
}

torch::Tensor interpolate_4d_backward(
    torch::Tensor grad_output,
    std::vector<int64_t> input_size,
    bool align_corners) {
    /*
    4D Interpolate Backward Pass
    
    Args:
        grad_output: [N, C, X_out, Y_out, U_out, V_out] - gradient from upstream
        input_size: [X_in, Y_in, U_in, V_in] - original input spatial size
        align_corners: if True, corner pixels are aligned
    
    Returns:
        grad_input: [N, C, X_in, Y_in, U_in, V_in] - gradient w.r.t. input
    */
    CHECK_INPUT(grad_output);
    
    TORCH_CHECK(grad_output.dim() == 6, "grad_output must be 6D tensor");
    TORCH_CHECK(input_size.size() == 4, "input_size must have 4 elements [X, Y, U, V]");
    
    return interpolate_4d_backward_cuda(
        grad_output,
        static_cast<int>(input_size[0]), static_cast<int>(input_size[1]),
        static_cast<int>(input_size[2]), static_cast<int>(input_size[3]),
        align_corners);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grid_sample_4d_forward, "4D Grid Sample forward (CUDA)",
          py::arg("input"), py::arg("grid"), py::arg("align_corners") = true, py::arg("padding_mode") = 0);
    m.def("backward", &grid_sample_4d_backward, "4D Grid Sample backward (CUDA)",
          py::arg("grad_output"), py::arg("input"), py::arg("grid"), py::arg("align_corners") = true, py::arg("padding_mode") = 0);
    m.def("interpolate_forward", &interpolate_4d_forward, "4D Interpolate forward (CUDA)",
          py::arg("input"), py::arg("size"), py::arg("align_corners") = true);
    m.def("interpolate_backward", &interpolate_4d_backward, "4D Interpolate backward (CUDA)",
          py::arg("grad_output"), py::arg("input_size"), py::arg("align_corners") = true);
}

