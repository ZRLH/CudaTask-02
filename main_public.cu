//#define RUNNING_MODE -1
#define RUNNING_MODE 110     //  ** Ultra version **   1.04 sec on my computer with (900L,10it) Modified from the 11th RUNNING_MODE in main.cu (Current file is main_public.cu, not main.cu)

#if RUNNING_MODE == 110
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>



#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define IT_MAX 10
//#define IT_MAX 100
#define L 900
//#define L 384

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 32       // BLOCK size setting

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call)                                                 \
{                                                                              \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(err));                                      \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}


// Initialization on the CPU
void init(double *a) {
    int i, j, k;

#pragma omp parallel for private(i, j, k) shared(a) collapse(3)
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            for (k = 0; k < nz; k++) {
                int idx = i * ny * nz + j * nz + k;
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[idx] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a[idx] = 0;
            }
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // Use the result of this round of calculation as the left variable of the next round of calculation

    // get j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // return unneeded threads
        return;

    int j = thread_global_id / nz;
    int k = thread_global_id % nz;

    d_al = a[(1 - 1) * ny * nz + j * nz + k];

    // Each thread processes a whole x-direction data line serially
    for (int i = 1; i < nx - 1; ++i) {
        d_al = (d_al + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // Use the result of this round of calculation as the left variable of the next round of calculation

    // get i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // return unneeded threads
        return;

    int i = thread_global_id / nz;
    int k = thread_global_id % nz;

    d_al = a[i * ny * nz + (1 - 1) * nz + k];

    // Each thread processes a whole y-direction data line serially
    for (int j = 1; j < ny - 1; ++j) {
        d_al = (d_al + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

// print debug
#define DEBUG_PRINT(info, bIdx, tIdx)                   \
{                                           \
    if (blockIdx.x == (bIdx) && threadIdx.x == (tIdx))\
        printf("blockIdx: %d, threadIdx: %d: %s\n", (bIdx), (tIdx), (info));               \
}                                           \


//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
// shard memory block size per block: 33 * 32 double, the shard block will also for block binary algorithm reduce
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    __shared__ double sh_data[(BLOCK_SIZE + 1) * BLOCK_SIZE];  // 33 * 32 double shared memory
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double tmp;
    double s_eps = 0.;
    double d_al, d_ac, d_ar;  // left, center, right values

    // Total number of (i, j) lines in the grid
    int ij_total = nx * ny;

    // Decode i and j from the global thread ID; ordering affects performance but not correctness
    int i = thread_global_id / ny;
    int j = thread_global_id % ny;

    // Determine how many lines this block will process
    int line_per_block;
    if (blockIdx.x == ij_total / BLOCK_SIZE) {
        line_per_block = ij_total % BLOCK_SIZE; // Last block may have fewer lines (remainder)
    } else {
        line_per_block = BLOCK_SIZE;           // Other blocks handle BLOCK_SIZE lines each
    }

    // Compute how many elements to load in the first partial round (for alignment)
    int first_round_size = nx % blockDim.x;
    int round_times      = nx / blockDim.x;  // Number of full rounds along x
    bool flag            = true;

    __syncthreads();

    int i_round = 0;  // Current round index

    // Handle the special first round when only 1 or 2 elements need to be loaded
    // We use d_al and d_ac for left and center, then process a[i+1] -> a[i]
    if (first_round_size == 1) {
        // Load only one element per line for the first round
        d_al = a[i * ny * nz + j * nz + 0]; // leftmost element a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // next element a[1]

        // In parallel, each thread loads BLOCK_SIZE elements per line into shared memory
        for (int line = 0; line < line_per_block; ++line) {
            int idx = blockIdx.x * blockDim.x + line;
            sh_data[line * (BLOCK_SIZE+1) + tid_in_block + 1] =
                    a[idx * nz + tid_in_block + 1];
            __syncthreads();
        }

        // Each thread performs serial computation on its shared-memory segment
        for (int x = 1; x < BLOCK_SIZE; ++x) {
            d_ar = sh_data[tid_in_block * (BLOCK_SIZE+1) + x + 1];
            tmp  = (d_al + d_ar) * 0.5;
            sh_data[tid_in_block * (BLOCK_SIZE+1) + x] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));
            d_al  = tmp;
            d_ac  = d_ar;
        }

        // Write back results for each line in the block (excluding thread 0)
        for (int line = 0; line < line_per_block; ++line) {
            if (tid_in_block > 0) {
                int idx = blockIdx.x * blockDim.x + line;
                a[idx * nz + tid_in_block] =
                        sh_data[line * (BLOCK_SIZE+1) + tid_in_block];
            }
            __syncthreads();
        }

        flag = false;
        ++i_round; // After first_round_size==1, we need two passes
    }
    if (first_round_size == 2) {
        // Similar handling when first_round_size == 2
        // Load two elements per line for the first round
        d_al = a[i * ny * nz + j * nz + 0];
        d_ac = a[i * ny * nz + j * nz + 1];

        for (int line = 0; line < line_per_block; ++line) {
            int idx = blockIdx.x * blockDim.x + line;
            sh_data[line * (BLOCK_SIZE+1) + tid_in_block + 1] =
                    a[idx * nz + tid_in_block + 2];
            __syncthreads();
        }

        // Serial compute for BLOCK_SIZE values
        for (int x = 0; x < BLOCK_SIZE; ++x) {
            d_ar = sh_data[tid_in_block * (BLOCK_SIZE+1) + x + 1];
            tmp  = (d_al + d_ar) * 0.5;
            sh_data[tid_in_block * (BLOCK_SIZE+1) + x] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));
            d_al  = tmp;
            d_ac  = d_ar;
        }

        // Write back BLOCK_SIZE results per line
        for (int line = 0; line < line_per_block; ++line) {
            int idx = blockIdx.x * blockDim.x + line;
            a[idx * nz + tid_in_block + 1] =
                    sh_data[line * (BLOCK_SIZE+1) + tid_in_block];
            __syncthreads();
        }

        flag = false;
        ++i_round;
    } else if (first_round_size == 0) {
        // Handle when nx is an exact multiple of BLOCK_SIZE
        // Load BLOCK_SIZE-2 elements per line for alignment
        int frs = BLOCK_SIZE;
        d_al = a[i * ny * nz + j * nz + 0];
        d_ac = a[i * ny * nz + j * nz + 1];

        for (int line = 0; line < line_per_block; ++line) {
            if (tid_in_block < frs-2) {
                int idx = blockIdx.x * blockDim.x + line;
                sh_data[line * (BLOCK_SIZE+1) + (frs-2 - tid_in_block -1)] =
                        a[idx * nz + tid_in_block + 2];
            }
            __syncthreads();
        }

        // Serial compute for frs-2 elements
        for (int x = frs-2; x > 0; --x) {
            d_ar = sh_data[tid_in_block * (BLOCK_SIZE+1) + x - 1];
            tmp  = (d_al + d_ar) * 0.5;
            sh_data[tid_in_block * (BLOCK_SIZE+1) + x] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));
            d_al  = tmp;
            d_ac  = d_ar;
        }

        // Write back frs-2 results per line
        for (int line = 0; line < line_per_block; ++line) {
            if (tid_in_block < frs-2) {
                int idx = blockIdx.x * blockDim.x + line;
                a[idx * nz + tid_in_block + 1] =
                        sh_data[line * (BLOCK_SIZE+1) + (frs-2 - tid_in_block)];
            }
            __syncthreads();
        }

        flag = true;
        i_round = 1;
        first_round_size = 0;
    } else {
        // Handle when first_round_size >= 3
        d_al = a[i * ny * nz + j * nz + 0];
        d_ac = a[i * ny * nz + j * nz + 1];

        for (int line = 0; line < line_per_block; ++line) {
            if (tid_in_block < first_round_size-2) {
                int idx = blockIdx.x * blockDim.x + line;
                sh_data[line * (BLOCK_SIZE+1) + (first_round_size-2 - tid_in_block -1)] =
                        a[idx * nz + tid_in_block + 2];
            }
            __syncthreads();
        }

        for (int x = first_round_size-2; x > 0; --x) {
            d_ar = sh_data[tid_in_block * (BLOCK_SIZE+1) + x - 1];
            tmp  = (d_al + d_ar) * 0.5;
            sh_data[tid_in_block * (BLOCK_SIZE+1) + x] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));
            d_al  = tmp;
            d_ac  = d_ar;
        }

        for (int line = 0; line < line_per_block; ++line) {
            if (tid_in_block < first_round_size-2) {
                int idx = blockIdx.x * blockDim.x + line;
                a[idx * nz + tid_in_block + 1] =
                        sh_data[line * (BLOCK_SIZE+1) + (first_round_size-2 - tid_in_block)];
            }
            __syncthreads();
        }

        flag = true;
        i_round = 0;
    }

    __syncthreads();

    // Perform remaining rounds: load, compute, write back for round_times
    for (; i_round < round_times; ++i_round) {
        if (flag) {
            // Forward pass: load BLOCK_SIZE, compute, write back
            for (int line = 0; line < line_per_block; ++line) {
                int idx = blockIdx.x * blockDim.x + line;
                sh_data[line * (BLOCK_SIZE+1) + tid_in_block + 1] =
                        a[idx * nz + i_round*BLOCK_SIZE + first_round_size + tid_in_block];
                __syncthreads();
            }

            for (int x = 0; x < BLOCK_SIZE; ++x) {
                d_ar = sh_data[tid_in_block * (BLOCK_SIZE+1) + x + 1];
                tmp  = (d_al + d_ar) * 0.5;
                sh_data[tid_in_block * (BLOCK_SIZE+1) + x] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));
                d_al  = tmp;
                d_ac  = d_ar;
            }

            for (int line = 0; line < line_per_block; ++line) {
                int idx = blockIdx.x * blockDim.x + line;
                a[idx * nz + i_round*BLOCK_SIZE + first_round_size + tid_in_block - 1] =
                        sh_data[line * (BLOCK_SIZE+1) + tid_in_block];
                __syncthreads();
            }
            flag = false;
        } else {
            // Reverse pass: symmetrical operations in reverse
            for (int line = 0; line < line_per_block; ++line) {
                int idx = blockIdx.x * blockDim.x + line;
                sh_data[line * (BLOCK_SIZE+1) + (BLOCK_SIZE - tid_in_block -1)] =
                        a[idx * nz + i_round*BLOCK_SIZE + first_round_size + tid_in_block];
                __syncthreads();
            }

            for (int x = BLOCK_SIZE; x > 0; --x) {
                d_ar = sh_data[tid_in_block * (BLOCK_SIZE+1) + x -1];
                tmp  = (d_al + d_ar) * 0.5;
                sh_data[tid_in_block * (BLOCK_SIZE+1) + x] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));
                d_al  = tmp;
                d_ac  = d_ar;
            }

            for (int line = 0; line < line_per_block; ++line) {
                int idx = blockIdx.x * blockDim.x + line;
                a[idx * nz + i_round*BLOCK_SIZE + first_round_size + tid_in_block - 1] =
                        sh_data[line * (BLOCK_SIZE+1) + (BLOCK_SIZE - tid_in_block)];
                __syncthreads();
            }
            flag = true;
        }
    }

    // Write each thread's local max error into shared memory, zero out extras
    if (thread_global_id >= ij_total)
        sh_data[tid_in_block] = 0.;
    else
        sh_data[tid_in_block] = s_eps;

    __syncthreads();

    // Block-wide reduction (binary tree)
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // Thread 0 writes back the block's max error
    if (tid_in_block == 0)
        d_eps[blockIdx.x] = sh_data[0];
}

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared Mem per Block: %d bytes\n", prop.sharedMemPerBlock);  // limit in block
    printf("Shared Mem per SM   : %d bytes\n", prop.sharedMemPerMultiprocessor); // limit in SM
}

int main(int argc, char *argv[]) {
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;

    print_gpu_info();

    // Initialization parameters
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // Allocate host memory. The three-dimensional array is stored by pulling it into a line.
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize Data
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // The number of threads in each block
    int total_lines_x = ny * nz;    // x direction line number
    int total_lines_y = nx * nz;    // y direction line number
    int total_lines_z = nx * ny;    // z direction line number
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // Calculate the number of blocks
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // Calculate the number of blocks
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // Calculate the number of blocks
    //**********************************************************
    //**********************************************************

    // Allocating device memory
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // Store the maximum value in each block

    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    int num_items = blocks_z;   // parameter for thrust::reduce() function

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 32 threads per block


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 32 threads per block


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();


        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 32 threads per block; 33 doubles per thread


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // At this time, d_eps stores the maximum value in each block
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
        eps = thrust::reduce(dev_ptr, dev_ptr + num_items, 0., thrust::maximum<double>());

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // Checking convergence conditions
        if (eps < maxeps) {
            printf("Convergence reached at iteration %d\n", it);
            break;
        }

        check_time_e = omp_get_wtime();

        sum_time_a += check_time_b - check_time_a;
        sum_time_b += check_time_c - check_time_b;
        sum_time_c += check_time_d - check_time_c;
        sum_time_d += check_time_e - check_time_d;
    }

    endt = omp_get_wtime();


    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));


    // Freeing up memory
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));

#ifdef RUNNING_MODE
    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
#endif
    printf(" ADI Benchmark (1) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    if (L == 900 && IT_MAX == 10)
        printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 390 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2570170E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 389 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2557102E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 388 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2543966E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 387 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2530762E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 386 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2517489E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 385 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2504148E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 384 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 383 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2477255E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 382 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2463703E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 381 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2450079E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 380 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2436384E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 379 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 7.2422616E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 384 && IT_MAX == 500)
        printf(" Verification    =       %12s\n", (fabs(eps - 1.2671190E-02) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "UNKNOWN" : "UNKNOWN"));


    printf(" threads_per_block: %d\n", threads_per_block);


    printf(" sum_time_a: %lf\n", sum_time_a);
    printf(" sum_time_b: %lf\n", sum_time_b);
    printf(" sum_time_c: %lf\n", sum_time_c);
    printf(" sum_time_d: %lf\n", sum_time_d);

    printf(" sum_time_a+b+c+d: %lf\n", sum_time_a + sum_time_b + sum_time_c + sum_time_d);

    printf(" END OF ADI Benchmark\n");
    return 0;
}

#endif

#if RUNNING_MODE == 210
// nothing exists ...

#endif
