extern "C" __global__ void get_target_kernel(
    const float *cm, const float *target_coords, int *output, 
    int cm_size, int target_coords_size, int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cm_size) return;

    bool match = true;
    for (int d = 0; d < dim; ++d) {
        if (cm[idx * dim + d] != target_coords[idx * dim + d]) {
            match = false;
            break;
        }
    }
    if (match) {
        output[idx] = 1;
        break;
    }
}
