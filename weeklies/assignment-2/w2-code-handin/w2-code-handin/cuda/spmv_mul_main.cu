#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "host_skel.cuh"
#include "spmv_mul_kernels.cuh"


/**
 * mat_width   is the number of rows of the matrix (input)
 * vct_size    is the input vector size (input)
 * rands       is a temporary array of randoms of size mat_width (output)
 * int Result  is the total number of elements of to-be-constructed matrix
 */
int getTotSizeAndRands(int mat_width, int vct_size, int* rands) {
    int tot_size = 0;

    for(int i=0; i<mat_width; i++) {
        int r = 1 + (rand() % (vct_size-1));
        rands[i] = r;
        tot_size += r ;
    }

    return tot_size;
}

/**
 * vct_size   is the size of the vector (input)
 * width      is the number of rows of the matrix (input)
 * tot_size   is the total number of elements of the matrix (input)
 * rands      is an array of randoms in [0,mat_width) (input)
 * vct        is the input vector (output)
 * flags      is the flag array of the matrix, i.e.,
 *               each segment denotes a row of the matrix. (output)
 * mat_vals   is the matrix values in flat form (output)
 * mat_inds   is the matrix-column indices in flat form (output),
 *               i.e., each column index corresponds to a matrix value,
 *               and `zip mat_inds mat_vals' is the sparse matrix.
 */
void mkVactDataAndFlags(
        int     vct_size,  int   mat_width,
        int     tot_size,  int*  rands,
        float*  vct,       int*  flags,
        float*  mat_vals,  int*  mat_inds
) {
    for(int i=0; i<vct_size; i++) {
        vct[i] = (rand() / (float)RAND_MAX)*5.0;
    }
    int ind = 0;
    for(int i=0; i<tot_size; i+=rands[ind], ind++ ) {
        for(int j=0; j < rands[ind]; j++) {
            flags[i+j] = j == 0;
            mat_inds[i+j] = j;
            mat_vals[i+j] = (rand() / (float)RAND_MAX)*10.0;
        }
    }
}


void seqSparseMatVctMult( int    tot_size, int  mat_width,
                          float* vct,      int* flags,
                          float* mat_vals, int* mat_inds,
                          float* res_vct
) {
    int   ind   = 0;
    float accum = mat_vals[0]*vct[mat_inds[0]];
    for(int i=1; i<tot_size; i++) {
        if (flags[i] != 0) {
            res_vct[ind] = accum;
            accum = mat_vals[i]*vct[mat_inds[i]];
            ind ++;
        } else {
            accum += mat_vals[i]*vct[mat_inds[i]];
        }
    }
    res_vct[ind] = accum;
}

int SparseMatVctMult(int block_size, int mat_rows, int vct_size) {
    uint64_t elapsed;
    struct timeval t_start, t_end, t_diff;

    int* rands   = (int*)malloc(mat_rows*sizeof(int));
    int tot_size = getTotSizeAndRands(mat_rows, vct_size, rands);

    printf("Vect_size: %d, tot_size: %d mat_rows: %d\n", vct_size, tot_size, mat_rows);

    float *vct, *mat_vals, *vct_res1, *vct_res2;
    int   *flags, *mat_inds;
    vct       = (float*)malloc(vct_size*sizeof(float));
    flags     = (int  *)malloc(tot_size*sizeof(int  ));
    mat_inds  = (int  *)malloc(tot_size*sizeof(int  ));
    mat_vals  = (float*)malloc(tot_size*sizeof(float));

    vct_res1  = (float*)malloc(mat_rows*sizeof(float));
    vct_res2  = (float*)malloc(mat_rows*sizeof(float));

    mkVactDataAndFlags( vct_size, mat_rows, tot_size, rands,
                        vct,      flags,    mat_vals, mat_inds );

    { // CPU sequential execution
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_CPU; i++) {
            seqSparseMatVctMult( tot_size, mat_rows, vct,
                                 flags, mat_vals, mat_inds, vct_res1 );
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        printf("CPU Sparse Matrix-Vector Multiplication runs in: %lu microsecs\n", elapsed);
    }

    { // GPU execution
        int   *mat_inds_d, *mat_shp_d, *mat_shp_sc_d, *d_tmp_int;
        float *vct_d,   *mat_vals_d, *res_vct_d, *d_tmp_float;
        char  *flags_d, *d_tmp_flag;

        cudaMalloc((void**)&mat_shp_d,    mat_rows*sizeof(int ));
        cudaMalloc((void**)&mat_shp_sc_d, mat_rows*sizeof(int ));
        cudaMalloc((void**)&flags_d,      tot_size*sizeof(char));
        cudaMalloc((void**)&mat_inds_d,   tot_size*sizeof(int ));

        cudaMalloc((void**)&mat_vals_d, tot_size*sizeof(float));
        cudaMalloc((void**)&vct_d,      vct_size*sizeof(float));
        cudaMalloc((void**)&res_vct_d,  mat_rows*sizeof(float));

        cudaMalloc((void**)&d_tmp_int,   MAX_BLOCK*sizeof(int));
        cudaMalloc((void**)&d_tmp_float, MAX_BLOCK*sizeof(float));
        cudaMalloc((void**)&d_tmp_flag,  MAX_BLOCK*sizeof(char));

        // other temporaries:
        float *tmp_pairs, *tmp_scan;
        int   *tmp_inds;

        cudaMalloc((void**)&tmp_pairs, tot_size*sizeof(float));
        cudaMalloc((void**)&tmp_scan,  tot_size*sizeof(float));
        cudaMalloc((void**)&tmp_inds,  tot_size*sizeof(int  ));

        CUDASSERT(cudaPeekAtLastError());

        ////////////////////////////////////////////////////////////
        // TODO: Weekly Assignment 2, Task 4
        ////////////////////////////////////////////////////////////
        //       a) Fill in a meaningful implementation for
        //          `num_blocks` and `num_blocks_shp` below.
        //          The former denotes the number of blocks required for the execution
        //            of `replicate0` and `mult_pairs` kernels, and the latter
        //            corresponds to the `mkFlags` and `select_last_in_sgm` kernels.
        //       b) Implement said four kernels in file `spMV-Mul-kernels.cuh`.
        //
        // The total number of non-zero elements in the matrix is `tot_size`.
        // The number of rows in the matrix is `mat_rows`.
        // The block size of the CUDA block is `block_size`
        uint32_t num_blocks     = (tot_size  + block_size - 1) / block_size;  // TODO: replace this dummy value.
        uint32_t num_blocks_shp = (mat_rows  + block_size - 1) / block_size;  // TODO: replace this dummy value.

        { // copy-in stage
            cudaMemcpy(mat_shp_d,  rands,    mat_rows*sizeof(int),   cudaMemcpyHostToDevice);
            cudaMemcpy(mat_inds_d, mat_inds, tot_size*sizeof(int),   cudaMemcpyHostToDevice);
            cudaMemcpy(mat_vals_d, mat_vals, tot_size*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(vct_d,      vct,      vct_size*sizeof(float), cudaMemcpyHostToDevice);
            CUDASSERT(cudaPeekAtLastError());
        }

        { // dry run to manifest the allocations in memory
            scanInc< Add<int> > ( block_size, mat_rows, mat_shp_sc_d, mat_shp_d, d_tmp_int );
            replicate0<<< num_blocks, block_size >>> ( tot_size, flags_d );
            mkFlags<<< num_blocks_shp, block_size >>> ( mat_rows, mat_shp_sc_d, flags_d );
            mult_pairs<<< num_blocks, block_size >>>(mat_inds_d, mat_vals_d, vct_d, tot_size, tmp_pairs);
            sgmScanInc< Add<float> > ( block_size, tot_size, tmp_scan, flags_d, tmp_pairs, d_tmp_float, d_tmp_flag );
            select_last_in_sgm<<< num_blocks_shp, block_size >>>(mat_rows, mat_shp_sc_d, tmp_scan, res_vct_d);
        }

        cudaMemset(res_vct_d, 0, mat_rows*sizeof(float));
        CUDASSERT(cudaPeekAtLastError());

        gettimeofday(&t_start, NULL);

        for(uint32_t k=0; k<RUNS_GPU; k++)
        {
            { // Make flag array
                // 1. scan the shape array
                scanInc< Add<int> > ( block_size, mat_rows, mat_shp_sc_d, mat_shp_d, d_tmp_int );

                // 2. create an array of zeros
                replicate0<<< num_blocks, block_size >>> ( tot_size, flags_d );

                // 3. scatter the flag array
                mkFlags<<< num_blocks_shp, block_size >>> ( mat_rows, mat_shp_sc_d, flags_d );
            }

            { // Kernels
                // 4. mutiply each matrix element with its corresponding
                //    vector element, i.e., the flat map from the Haskell code
                mult_pairs<<< num_blocks, block_size >>>(mat_inds_d, mat_vals_d, vct_d, tot_size, tmp_pairs);

                // 5. perform a segmented scan, such as the last element
                //    of each segment has the value of the corresponding
                //    result-vector index.
                sgmScanInc< Add<float> > ( block_size, tot_size, tmp_scan, flags_d, tmp_pairs, d_tmp_float, d_tmp_flag );

                // 6. select the end-of-segment element
                select_last_in_sgm<<< num_blocks_shp, block_size >>>(mat_rows, mat_shp_sc_d, tmp_scan, res_vct_d);
            }
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);

        CUDASSERT(cudaPeekAtLastError());

        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        printf("GPU Sparse Matrix-Vector Multiplication runs in: %lu microsecs\n", elapsed);

        { // copy out stage:
            CUDASSERT(cudaMemcpy(vct_res2, res_vct_d, mat_rows*sizeof(float),
                        cudaMemcpyDeviceToHost));
        }

        cudaFree(mat_shp_d);  cudaFree(mat_shp_sc_d);
        cudaFree(flags_d);    cudaFree(mat_inds_d);  cudaFree(vct_d);
        cudaFree(mat_vals_d); cudaFree(res_vct_d);   cudaFree(tmp_pairs);
        cudaFree(tmp_scan);   cudaFree(tmp_inds);
        cudaFree(d_tmp_int);  cudaFree(d_tmp_float); cudaFree(d_tmp_flag);
        CUDACHECK(cudaPeekAtLastError());
    }

    {// validation
        bool valid = true;
        for(int i=0; i<mat_rows; i++) {
            float res1 = fabs(vct_res1[i]);
            float res2 = fabs(vct_res2[i]);
            if(fabs(res1 - res2) > 0.1) {
                printf("ERROR at result vector index %d (cpu,gpu): (%6.5f, %6.5f)\n", i, res1, res2);
                valid = false;
                break;
            }
        }
        if(valid) printf("Sparse Mat-Vect Mult VALID   RESULT\n");
        else      printf("Sparse Mat-Vect Mult INVALID RESULT\n");
    }

    { // free space
        free(rands);
        free(vct);
        free(flags);
        free(mat_inds);
        free(mat_vals);
        free(vct_res1);
        free(vct_res2);
    }
    return 1;
}

int main(int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s <num-rows-matrix> <vct-size> <CUDA-block-size>\n", argv[0]);
        exit(1);
    }

    initHwd();

    const uint32_t matrix_row_num = atoi(argv[1]);  // 11033;
    const uint32_t vct_size       = atoi(argv[2]);  // 2076;
    const uint32_t block_size     = atoi(argv[3]);  // 256;

    printf("Testing Sparse-MatVec Mul with num-rows-matrix: %d, vct-size: %d, block size: %d\n\n"
          , matrix_row_num, vct_size, block_size);

    SparseMatVctMult(block_size, matrix_row_num, vct_size);
}
