#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {
    // TODO: fill in your implementation here ...
    // Each thread initializes one element of flags_d to 0
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tot_size) flags_d[i] = 0;
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    // TODO: fill in your implementation here ...
    // Each thread marks the start of a new segment by writing 1 to the corresponding position in flags_d
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < mat_rows) {
        int start = (r == 0) ? 0 : mat_shp_sc_d[r - 1];
        flags_d[start] = 1;
    }
}

__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    // TODO: fill in your implementation here ...
    // Each thread computes the product of a matrix value and the corresponding vector value
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tot_size) {
        int col = mat_inds[i];
        tmp_pairs[i] = mat_vals[i] * vct[col];
    }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    // TODO: fill in your implementation here ...
    // Each thread selects the last element of each segment and writes it to the result vector
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < mat_rows) {
        int end = mat_shp_sc_d[r] - 1;
        res_vct_d[r] = (end >= 0) ? tmp_scan[end] : 0.0f;
    }
}

#endif // SPMV_MUL_KERNELS
