#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "host_skel.cuh"


void initArray(int32_t* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i] = (rand() % M) - R;
    }
}

/**
 * Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel
 */
int bandwidthMemcpy( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                   , const size_t   N     // length of the input array
                   , int* d_in            // device input  of length N
                   , int* d_out           // device result of length N
) {
    // dry run to exercise the d_out allocation!
    const uint32_t num_blocks = (N + B - 1) / B;
    naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);

    double gigaBytesPerSec;
    uint64_t elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_GPU; i++) {
            naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);

        CUDASSERT(cudaPeekAtLastError());

        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int) * 1.0e-3f / elapsed;
        printf("Naive Memcpy GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    return 0;
}

/**
 * Generic reduce for both the commutative and non-commutative case
 */
template<class OP>
int testGenRed( const uint32_t        B     // desired CUDA block size ( <= 1024, multiple of 32)
              , const size_t          N     // length of the input array
              , typename OP::InpElTp* h_in  // host input    of size: N * sizeof(int)
              , typename OP::InpElTp* d_in  // device input  of size: N * sizeof(ElTp)
              , bool optimized              // optimized or naive?
) {
    typename OP::RedElTp* d_out;
    // for the optimized version "mapReduce" is enough MAX_BLOCK instead of 2*N
    if(optimized) cudaMalloc((void**)&d_out, MAX_BLOCK*sizeof(typename OP::RedElTp));
    else          cudaMalloc((void**)&d_out, 2*N*sizeof(typename OP::RedElTp));
    CUDASSERT(cudaPeekAtLastError());

    // dry run to exercise the d_out allocation!
    if (optimized) mapReduce<OP>  (B, N, d_out, d_in);
    else           mapredNaive<OP>(B, N, d_out, d_in);

    double gigaBytesPerSec;
    uint64_t elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_GPU; i++) {
            if(optimized) mapReduce< OP >( B, N, d_out, d_in );
            else          mapredNaive<OP>( B, N, d_out, d_in );
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);

        CUDASSERT(cudaPeekAtLastError());

        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = N * sizeof(typename OP::InpElTp) * 1.0e-3f / elapsed;
        printf("Reduce GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }


    typename OP::RedElTp cpu_res;
    { // sequential computation
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_CPU; i++) {
            cpu_res = OP::identity();
            for(uint32_t i=0; i<N; i++) {
                typename OP::InpElTp elm = h_in[i];
                volatile typename OP::RedElTp red = OP::mapFun(elm);
                cpu_res = OP::apply(cpu_res, red);
            }
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        double gigaBytesPerSec = N * sizeof(typename OP::InpElTp) * 1.0e-3f / elapsed;
        printf("Reduce CPU Sequential runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }

    { // Validation
        typename OP::RedElTp gpu_res;
        CUDASSERT(cudaMemcpy(&gpu_res, d_out, sizeof(typename OP::RedElTp),
                    cudaMemcpyDeviceToHost));
        if (!OP::equals(gpu_res, cpu_res)) {
            printf("INVALID, EXITING!!!\n");
            exit(1);
        }
        printf("Reduce: VALID result!\n\n");
    }
    cudaFree(d_out);

    return 0;
}

int scanIncAddI32( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                 , const size_t   N     // length of the input array
                 , int* h_in            // host input    of size: N * sizeof(int)
                 , int* d_in            // device input  of size: N * sizeof(ElTp)
                 , int* d_out           // device result of size: N * sizeof(int)
) {
    const size_t mem_size = N * sizeof(int);
    int* d_tmp;
    int* h_out = (int*)malloc(mem_size);
    int* h_ref = (int*)malloc(mem_size);
    cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(int));
    cudaMemset(d_out, 0, N*sizeof(int));

    // dry run to exercise d_tmp allocation
    scanInc< Add<int> > ( B, N, d_out, d_in, d_tmp );
    CUDASSERT(cudaPeekAtLastError());

    // time the GPU computation
    uint64_t elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int i=0; i<RUNS_GPU; i++) {
        scanInc<Add<int>> ( B, N, d_out, d_in, d_tmp );
    }
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);

    CUDASSERT(cudaPeekAtLastError());

    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
    double gigaBytesPerSec = N  * (2*sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
    printf("Scan Inclusive AddI32 GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
          , elapsed, gigaBytesPerSec);


    { // sequential computation
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_CPU; i++) {
            int acc = 0;
            for(uint32_t i=0; i<N; i++) {
                acc += h_in[i];
                h_ref[i] = acc;
            }
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        double gigaBytesPerSec = N * (sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
        printf("Scan Inclusive AddI32 CPU Sequential runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }

    { // Validation
        CUDASSERT(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));
        for(uint32_t i = 0; i<N; i++) {
            if(h_out[i] != h_ref[i]) {
                printf("!!!INVALID!!!: Scan Inclusive AddI32 at index %d, dev-val: %d, host-val: %d\n"
                      , i, h_out[i], h_ref[i]);
                exit(1);
            }
        }
        printf("Scan Inclusive AddI32: VALID result!\n\n");
    }

    free(h_out);
    free(h_ref);
    cudaFree(d_tmp);

    return 0;
}

int sgmScanIncAddI32( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                    , const size_t   N     // length of the input array
                    , int* h_inp           // host input    of size: N * sizeof(int)
                    , int* d_inp           // device input  of size: N * sizeof(int)
                    , int* d_out           // device result of size: N * sizeof(int)
) {
    const size_t mem_size = N * sizeof(int);
    int*  d_tmp_vals;
    char* d_tmp_flag;
    char* d_inp_flag;
    int*  h_out = (int*)malloc(mem_size);
    int*  h_ref = (int*)malloc(mem_size);
    char* h_inp_flag = (char*)malloc(N);
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(int ));
    cudaMalloc((void**)&d_tmp_flag, MAX_BLOCK*sizeof(char));
    cudaMalloc((void**)&d_inp_flag, N * sizeof(char));
    cudaMemset(d_out, 0, N*sizeof(int));

    { // init flag array
        for(uint32_t i=0; i<N; i++) {
            h_inp_flag[i] = 0;
        }
        uint32_t tot_len = 0;
        bool done = false;
        while (!done) {
            h_inp_flag[tot_len] = 1;
            uint32_t s = (rand() % 100) + 1;
            tot_len += s;
            if (tot_len >= N) done = true;
        }
    }
    // copy flag array to GPU
    cudaMemcpy(d_inp_flag, h_inp_flag, N*sizeof(char), cudaMemcpyHostToDevice);

    // dry run to exercise d_tmp allocation
    sgmScanInc< Add<int> >( B, N, d_out, d_inp_flag, d_inp, d_tmp_vals, d_tmp_flag );
    CUDASSERT(cudaPeekAtLastError());

    // measure the time taken by the GPU execution
    uint64_t elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int i=0; i<RUNS_GPU; i++) {
        sgmScanInc< Add<int> >( B, N, d_out, d_inp_flag, d_inp, d_tmp_vals, d_tmp_flag );
    }
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);

    CUDASSERT(cudaPeekAtLastError());

    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
    double gigaBytesPerSec = N  * (2*sizeof(int) + sizeof(int) + 2*sizeof(char)) * 1.0e-3f / elapsed;
    printf("SgmScan Inclusive AddI32 GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
          , elapsed, gigaBytesPerSec);


    { // sequential computation
        typedef ValFlg<int> FVTup;
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_CPU; i++) {
            FVTup acc = LiftOP< Add<int> >::identity();
            for(uint32_t i=0; i<N; i++) {
                FVTup elm(h_inp_flag[i], h_inp[i]);
                acc = LiftOP< Add<int> >::apply(acc, elm);
                h_ref[i] = acc.v;
            }
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        double gigaBytesPerSec = N * (sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
        printf("SgmScan Inclusive AddI32 CPU Sequential runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }

    { // Validation
        CUDASSERT(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));
        for(uint32_t i = 0; i<N; i++) {
            if(h_out[i] != h_ref[i]) {
                printf("!!!INVALID!!!: SgmScan Inclusive AddI32 at index %d, dev-val: %d, host-val: %d\n"
                      , i, h_out[i], h_ref[i]);
                exit(1);
            }
        }
        printf("SgmScan Inclusive AddI32: VALID result!\n\n");
    }

    free(h_out);
    free(h_ref);
    free(h_inp_flag);
    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    cudaFree(d_inp_flag);

    return 0;
}

int main(int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s <array-length> <block-size>\n", argv[0]);
        exit(1);
    }

    initHwd();

    const uint32_t N = atoi(argv[1]);
    const uint32_t B = atoi(argv[2]);

    printf("Testing parallel basic blocks\nN = %d\nblock size = %d\n"
           "ELEMS_PER_THREAD = %d\n\n",
           N, B, ELEMS_PER_THREAD);

    const size_t mem_size = N*sizeof(int);
    int* h_in    = (int*) malloc(mem_size);
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in ,   mem_size);
    cudaMalloc((void**)&d_out,   mem_size);

    initArray(h_in, N, 13);
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    CUDASSERT(cudaPeekAtLastError());

    // computing a "realistic/achievable" bandwidth figure
    bandwidthMemcpy(B, N, d_in, d_out);

    { // test reduce
        printf("Testing Naive Reduce with Int32 Addition Operator:\n");
        testGenRed< Add<int> >(B, N, h_in, d_in, false);

        printf("Testing Optimized Reduce with Int32 Addition Operator:\n");
        testGenRed< Add<int> >(B, N, h_in, d_in, true);

        printf("Testing Naive Reduce with MSSP Operator:\n");
        testGenRed< Mssp     >(B, N, h_in, d_in, false);

        printf("Testing Optimized Reduce with MSSP Operator:\n");
        testGenRed< Mssp     >(B, N, h_in, d_in, true);
    }

    { // inclusive scan and segmented scan with int addition
        scanIncAddI32   (B, N, h_in, d_in, d_out);
        sgmScanIncAddI32(B, N, h_in, d_in, d_out);
    }


    // cleanup memory
    free(h_in);
    cudaFree(d_in );
    cudaFree(d_out);
}
