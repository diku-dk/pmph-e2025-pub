#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 300
#define EPS 1e-6f

// ===== CUDA kernel: element-wise vector add (flat map) =====
__global__ void addKernel(const float* A, const float* B, float* C, unsigned int N) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        C[gid] = A[gid] + B[gid];
    }
}

// ===== CPU reference implementation & single-run timing =====
static void addCPU(const float* A, const float* B, float* C, unsigned int N) {
    for (unsigned int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    unsigned int N;

    {
      if (argc != 2) {
        printf("Num Args is: %d instead of 1. Exiting!\n", argc);
        exit(1);
      }
      N = (unsigned int)atoi(argv[1]);
      printf("Vector length N             : %u\n", N);

      const unsigned int maxN = 500000000u;
      if (N == 0 || N > maxN) {
        printf("N is invalid; maximal value is %u. Exiting!\n", maxN);
        exit(2);
      }
    }

    cudaSetDevice(0);

    // ===== Host memory allocation & initialization =====
    const unsigned int mem_size = N * sizeof(float);

    float* h_A     = (float*)malloc(mem_size);
    float* h_B     = (float*)malloc(mem_size);
    float* h_C_cpu = (float*)malloc(mem_size);
    float* h_C_gpu = (float*)malloc(mem_size);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
      printf("Host allocation failed. Exiting!\n");
      exit(3);
    }

    for (unsigned int i = 0; i < N; ++i) {
        h_A[i] = (float)(i % 1000) * 0.001f; 
        h_B[i] = (float)((i * 7) % 1000) * 0.002f;
    }

    // ===== CPU reference run & timing =====
    double cpu_ms = 0.0;
    {
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);
        addCPU(h_A, h_B, h_C_cpu, N);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        cpu_ms = (1.0 * (t_diff.tv_sec*1e3 + t_diff.tv_usec/1e3));
        printf("CPU time (one run)          : %.3f ms\n", cpu_ms);
    }

    // ===== Device memory allocation & host->device copy =====
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaMalloc((void**)&d_A, mem_size);
    cudaMalloc((void**)&d_B, mem_size);
    cudaMalloc((void**)&d_C, mem_size);

    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);

    // ===== Kernel configuration & warm-up =====
    const unsigned int BLOCK = 256u;
    const unsigned int GRID  = (N + BLOCK - 1) / BLOCK;
    printf("Block size (threads)        : %u\n", BLOCK);
    printf("Grid size (blocks)          : %u\n", GRID);
    printf("GPU kernel runs averaged    : %d\n", GPU_RUNS);

    addKernel<<<GRID, BLOCK>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    double avg_us;
    {
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);
        for (int r = 0; r < GPU_RUNS; ++r) {
            addKernel<<<GRID, BLOCK>>>(d_A, d_B, d_C, N);
        }
        cudaDeviceSynchronize(); 
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        avg_us = (1.0 * (t_diff.tv_sec*1e6 + t_diff.tv_usec)) / GPU_RUNS;
        printf("GPU avg kernel time         : %.3f us (excludes alloc & memcpy)\n", avg_us);
    }

    gpuAssert(cudaPeekAtLastError());

    cudaMemcpy(h_C_gpu, d_C, mem_size, cudaMemcpyDeviceToHost);

    // ===== Result validation =====
    {
        int ok = 1;
        for (unsigned int i = 0; i < N; ++i) {
            float a = h_C_cpu[i];
            float b = h_C_gpu[i];
            if (fabsf(a - b) > EPS) { ok = 0; break; }
        }
        printf("Validation (abs eps=%.0e): %s\n", (double)EPS, ok ? "VALID" : "INVALID");
    }

    // ===== Performance reporting: speedup & throughput =====
    {
        double gpu_ms = avg_us / 1000.0;
        double speedup = (gpu_ms > 0.0) ? (cpu_ms / gpu_ms) : 0.0;

        const double total_bytes = 3.0 * (double)N * 4.0;
        const double secs = avg_us * 1e-6;
        const double gbps = (total_bytes / secs) / 1e9;

        printf("Speedup (CPU/GPU)           : %.3f x\n", speedup);
        printf("GPU throughput              : %.3f GB/s    (12*N bytes / kernel)\n", gbps);
    }

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
