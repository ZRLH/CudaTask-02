#define RUNNING_MODE -1     // 停用
//#define RUNNING_MODE 0      // 过于拉胯, 辣眼睛
//#define RUNNING_MODE 1
//#define RUNNING_MODE 2      // 测试 核函数 内的 cub scan 函数
//#define RUNNING_MODE 3      // 用于测试 scan 效果, 使用主机端 scan 函数
//#define RUNNING_MODE 4      // cub 版本
//#define RUNNING_MODE 5 //2025/4/13日最快的计算方式, 每个block内32个线程, 每个线程单线程计算x方向一条线
//#define RUNNING_MODE 6      // 全村的希望 (两个方向 x y 已经实现) (x y z 均已实现)
//#define RUNNING_MODE 7      // 优化速度版, 准备交作业 //目前结果正确, 但是xy方向还没有优化成只两个全局内存读写操作(目前需要三个操作)
                        // warp 版 900*10 => 1.51s
//#define RUNNING_MODE 8      // 原子操作版, 用作对比 900*10 => 1.57s
//#define RUNNING_MODE 9      // thrust 替代 CUB
//#define RUNNING_MODE 10     // 尝试优化 k 方向        尝试采用 transition     // 原型机已实现！！！  超级快  Ultra级别    1.04秒
//#define RUNNING_MODE 11     // 对 Ultra 级别进行代码级优化  // 阶段性结论：i, j方向计算已经趋于最优解 (可能就是最优解)      易读性为0    表面上算法进行了简化，但是实际上速度并没有一丝加快
                                // 目前还剩下 a b 没有优化成两个读取
// 设置 RUNNING MODE ****************

// 笔记：将所有 /2. 改为 * 0.5 会更快

#if RUNNING_MODE == 0
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

// OpenMP will automatically use all available processors
// if we don't explicitly set the number of threads

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define IT_MAX 10
#define L 900

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 128

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

// 在 CPU 上进行初始化
void init(double *a)
{
    int i, j, k;

#pragma omp parallel for private(i,j,k) shared(a) collapse(3)
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

void compute_i_direction(double *a)
{
    int i, j, k;

    for (i = 1; i < nx - 1; i++) {
        for (j = 1; j < ny - 1; j++) {
            for (k = 1; k < nz - 1; k++) {
                int idx = i * ny * nz + j * nz + k;
                a[idx] = (a[(i-1)*ny*nz + j*nz + k] + a[(i+1)*ny*nz + j*nz + k]) / 2;
            }
        }
    }
}

// i方向核函数
__global__ void compute_i_direction_kernel(double *a)
{
    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // 每个线程块处理一行(j,k)上的所有i
    // 注意：需要按照i的顺序更新，因为存在数据依赖

    // 使用共享内存保存当前行
    extern __shared__ double s_a[];

    // 加载整行数据到共享内存
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }
    // 加载边界值
    if (threadIdx.x == 0) {
        s_a[0] = a[0 * ny * nz + j * nz + k];
        s_a[nx - 1] = a[(nx - 1) * ny * nz + j * nz + k];
    }

    __syncthreads();

    // 串行更新，确保正确的依赖关系
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[i] = (s_a[i-1] + s_a[i+1]) / 2.0;
        }
        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    // 写回全局内存
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[i];
        }
    }
}

// 计算第二个方向 (j 方向)
void compute_j_direction(double *a)
{
    int i, j, k;

    for (i = 1; i < nx - 1; i++) {
        for (j = 1; j < ny - 1; j++) {
            for (k = 1; k < nz - 1; k++) {
                int idx = i * ny * nz + j * nz + k;
                a[idx] = (a[i*ny*nz + (j-1)*nz + k] + a[i*ny*nz + (j+1)*nz + k]) / 2;
            }
        }
    }
}

// j方向核函数
__global__ void compute_j_direction_kernel(double *a)
{
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // 每个线程块处理一列(i,k)上的所有j
    // 注意：需要按照j的顺序更新，因为存在数据依赖

    // 使用共享内存保存当前列
    extern __shared__ double s_a[];

    // 加载整列数据到共享内存
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }
    // 加载边界值
    if (threadIdx.x == 0) {
        s_a[0] = a[i * ny * nz + 0 * nz + k];
        s_a[ny - 1] = a[i * ny * nz + (ny - 1) * nz + k];
    }

    __syncthreads();

    // 串行更新，确保正确的依赖关系
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[j] = (s_a[j-1] + s_a[j+1]) / 2.0;
        }
        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    // 写回全局内存
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// 计算第三个方向 (k 方向) 并计算 eps - CPU版本
void compute_k_direction_and_eps(double *a, double *eps_ptr)
{
    int i, j, k;
    double eps = 0.0;

    // 当计算最大值时需要小心处理
#pragma omp parallel
    {
        double local_eps = 0.0;

#pragma omp for private(i,j,k) collapse(3)
        for (i = 1; i < nx - 1; i++) {
            for (j = 1; j < ny - 1; j++) {
                for (k = 1; k < nz - 1; k++) {
                    int idx = i * ny * nz + j * nz + k;
                    double tmp1 = (a[i*ny*nz + j*nz + (k-1)] + a[i*ny*nz + j*nz + (k+1)]) / 2;
                    double tmp2 = fabs(a[idx] - tmp1);
                    local_eps = Max(local_eps, tmp2);
                    a[idx] = tmp1;
                }
            }
        }

        // 临界区用于安全地更新全局 eps 值
#pragma omp critical
        {
            eps = Max(eps, local_eps);
        }
    }

    *eps_ptr = eps;
}

// k方向核函数，包含eps计算
__global__ void compute_k_direction_kernel(double *a, double *d_eps)
{
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // 每个线程块处理一个(i,j)上的所有k点

    // 使用共享内存保存当前"柱"和局部最大误差
    extern __shared__ double shared[];
    double* s_a = shared;                        // 用于存储k方向数据
    double* s_eps = &shared[nz];                // 用于存储局部误差

    s_eps[threadIdx.x] = 0.0;  // 初始化局部误差

    // 加载整个k方向数据到共享内存
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }
    // 加载边界值
    if (threadIdx.x == 0) {
        s_a[0] = a[i * ny * nz + j * nz + 0];
        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
    }

    __syncthreads();

    // 串行更新k方向并计算局部最大误差
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k-1] + s_a[k+1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // 更新局部最大误差
        }
        __syncthreads();
    }

    // 写回全局内存
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // 使用原子操作更新全局最大误差
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int*)d_eps, __double_as_longlong(s_eps[0]));
    }
}

// 辅助函数：将不同表示形式的双精度数转换
__device__ double __longlong_as_double(unsigned long long int x) {
    return __longlong_as_double(x);
}

int main(int argc, char *argv[])
{
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;

    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配内存
    a = (double*)malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_eps, sizeof(double)));

    printf("Starting computation...\n");

    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++)
    {
        // Copy current data from host to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

        // 初始化误差值为0
        double h_eps = 0.0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        // 定义线程块和网格配置
        dim3 block(BLOCK_SIZE, 1, 1);
        dim3 grid_i(1, ny - 2, nz - 2);      // 每个线程块处理一行的所有i
        dim3 grid_j(nx - 2, 1, nz - 2);      // 每个线程块处理一列的所有j
        dim3 grid_k(nx - 2, ny - 2, 1);      // 每个线程块处理一个"柱"的所有k

        // 共享内存大小
        size_t shared_mem_size_i = nx * sizeof(double);                  // 整行
        size_t shared_mem_size_j = ny * sizeof(double);                  // 整列
        size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // 整"柱"加上临时存储

        double check_time_a = omp_get_wtime();

        // i方向更新
        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        double check_time_b = omp_get_wtime();

        // j方向更新
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        double check_time_c = omp_get_wtime();

        // k方向更新
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        double check_time_d = omp_get_wtime();

        // 获取eps值
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        // 拷贝最终结果回主机
        CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
        if (eps < maxeps) {
            printf("Convergence reached at iteration %d\n", it);
            break;
        }

        double check_time_e = omp_get_wtime();

        sum_time_a += check_time_b - check_time_a;
        sum_time_b += check_time_c - check_time_b;
        sum_time_c += check_time_d - check_time_c;
        sum_time_d += check_time_e - check_time_d;
    }

    endt = omp_get_wtime();

    // 释放内存
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));

    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
    printf(" ADI Benchmark (0) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.2lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    if (L == 900 && IT_MAX == 10)
        printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 384 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "UNKNOWN" : "UNKNOWN"));


    printf(" sum_time_a: %lf\n", sum_time_a);
    printf(" sum_time_b: %lf\n", sum_time_b);
    printf(" sum_time_c: %lf\n", sum_time_c);
    printf(" sum_time_d: %lf\n", sum_time_d);

    printf(" sum_time_a+b+c+d: %lf\n", sum_time_a + sum_time_b + sum_time_c + sum_time_d);


    printf(" END OF ADI Benchmark\n");
    return 0;
}

#endif

#if RUNNING_MODE == 1
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define L 900
#define IT_MAX 10

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 32

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

// 在 CPU 上进行初始化
void init(double *a)
{
    int i, j, k;

#pragma omp parallel for private(i,j,k) shared(a) collapse(3)
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

// i方向核函数
__global__ void compute_i_direction_kernel(double *a)
{
    unsigned long long time_checker_1 = clock64();      // 开始计时

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前行
    extern __shared__ double s_a[];


    // 加载整行数据到共享内存
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    unsigned long long time_checker_2 = clock64();      // 23


    // 串行更新，确保正确的依赖关系
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[i] = (s_a[i-1] + s_a[i+1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // 写回全局内存
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[i];
        }
    }
    __syncthreads();                                    // 19
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        printf("time_checker_1: %lld    time_checker_2: %lld    time_checker_3: %lld\n",
               time_checker_2 - time_checker_1,
               time_checker_3 - time_checker_2,
               clock64() - time_checker_3);
}


// j方向核函数
__global__ void compute_j_direction_kernel(double *a)
{
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前列
    extern __shared__ double s_a[];

    // 加载整列数据到共享内存
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新，确保正确的依赖关系
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[j] = (s_a[j-1] + s_a[j+1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    // 写回全局内存
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k方向核函数，包含eps计算
__global__ void compute_k_direction_kernel(double *a, double *d_eps)
{
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // 使用共享内存保存当前"柱"和局部最大误差
    extern __shared__ double shared[];
    double* s_a = shared;                        // 用于存储k方向数据
    double* s_eps = &shared[nz];                // 用于存储局部误差

    s_eps[threadIdx.x] = 0.0;  // 初始化局部误差

//    // 加载整个k方向数据到共享内存
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // 加载边界值
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // 加载整个k方向数据到共享内存
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新k方向并计算局部最大误差
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k-1] + s_a[k+1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // 更新局部最大误差
        }
//        __syncthreads();
    }

    // 写回全局内存
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // 使用原子操作更新全局最大误差
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int*)d_eps, __double_as_longlong(s_eps[0]));
    }
}

// 辅助函数：将不同表示形式的双精度数转换
__device__ double __longlong_as_double(unsigned long long int x) {
    return __longlong_as_double(x);
}

int main(int argc, char *argv[])
{
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;


    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double*)malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_eps, sizeof(double)));

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // 定义线程块和网格配置 - 移到循环外部以减少重复设置
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // 每个线程块处理一行的所有i
    dim3 grid_j(nx - 2, 1, nz - 2);      // 每个线程块处理一列的所有j
    dim3 grid_k(nx - 2, ny - 2, 1);      // 每个线程块处理一个"柱"的所有k

    // 共享内存大小 - 同样移到循环外部
    size_t shared_mem_size_i = nx * sizeof(double);                  // 整行
    size_t shared_mem_size_j = ny * sizeof(double);                  // 整列
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // 整"柱"加上临时存储

    for (it = 1; it <= itmax; it++)
    {
        // 优化点2: 不再每次迭代重新拷贝完整数据

        check_time_a = omp_get_wtime();

        // 初始化误差值为0
        double h_eps = 0.0;
        // ***** 耗时很少 *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));



        // i方向更新
        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);

        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();

        // j方向更新
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        // k方向更新
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();


        // 优化点3: 只传输eps值而不是整个数组
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));

    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
    printf(" ADI Benchmark (1) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    if (L == 900 && IT_MAX == 10)
        printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 384 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "UNKNOWN" : "UNKNOWN"));


    printf(" sum_time_a: %lf\n", sum_time_a);
    printf(" sum_time_b: %lf\n", sum_time_b);
    printf(" sum_time_c: %lf\n", sum_time_c);
    printf(" sum_time_d: %lf\n", sum_time_d);

    printf(" sum_time_a+b+c+d: %lf\n", sum_time_a + sum_time_b + sum_time_c + sum_time_d);

    printf(" END OF ADI Benchmark\n");
    return 0;
}

#endif

#if RUNNING_MODE == 2
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

#define NX 900
#define NY 900
#define NZ 900

// 自定义操作符，示例中为：result = (prev_value)/2 + current_input
struct MyCustomOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

// 每个线程块负责处理一条数据线（即固定 (j,k) 下 x 方向上的元素）
__global__ void scan_x_direction_kernel(double *a)
{
    // 设定：沿 x 方向扫描，对固定的 j, k 进行操作
    // 可用 gridDim.x = NY * NZ，每个块对应一个 (j, k) 组合
    int lineId = blockIdx.x;
    int j = lineId / NZ;      // j 方向索引
    int k = lineId % NZ;      // k 方向索引

    // 每条数据线的起始地址，数据存放顺序：a[i * (NY*NZ) + j*NZ + k]
    // 因为 i 变化最快在 x 方向，保证连续访问
    double *line = a + j * NZ + k;

    // 声明共享内存数组，长度为 NX
    extern __shared__ double s_line[];  // 大小应设为 NX * sizeof(double)

    // 每个线程根据线程索引读取数据
    // 假设 blockDim.x >= NX（例如设置为 1024），每个线程读取一个数据
    int tid = threadIdx.x;
    if (tid < NX)
    {
        // 注意：全局数组中 x 方向间隔为 (NY*NZ)，因为 i 从 0 到 NX-1
        s_line[tid] = line[tid * (NY * NZ)];
    }
    __syncthreads();

    // 利用 CUB BlockScan 在块内做扫描
    typedef cub::BlockScan<double, NX> BlockScan; // 1024 为 blockDim.x
    __shared__ typename BlockScan::TempStorage temp_storage;

    double scan_result = 0.0;
    // 每个线程调用 BlockScan 的 InclusiveScan
    BlockScan(temp_storage).InclusiveScan(s_line[tid], scan_result, MyCustomOp());
    __syncthreads();

    // 将计算结果写回全局内存
    if (tid < NX)
    {
        line[tid * (NY * NZ)] = scan_result;
    }
}

int main()
{
    size_t total_elems = NX * NY * NZ;
    size_t total_bytes = total_elems * sizeof(double);

    // 分配和初始化数据（这里只做简单初始化）
    double *h_a = (double*) malloc(total_bytes);
    double *h_b = (double*) malloc(NX * sizeof(double));
    for (size_t i = 0; i < total_elems; i++) {
        h_a[i] = 1.0;  // 或其它初始化值
    }
    for (int i = 0; i < NX; i++)
        h_b[i] = 1.0;

    double *d_a;
    cudaMalloc(&d_a, total_bytes);
    cudaMemcpy(d_a, h_a, total_bytes, cudaMemcpyHostToDevice);

    // 每个 block 处理一条数据线，共有 NY * NZ 条数据线
    int num_blocks = NY * NZ;
    int threads_per_block = 1024;  // 要确保 threads_per_block >= NX (900)
    size_t shared_mem_bytes = NX * sizeof(double);

    // 启动 kernel
    scan_x_direction_kernel<<<num_blocks, threads_per_block, shared_mem_bytes>>>(d_a);
    cudaDeviceSynchronize();

    // 将结果复制回主机进行验证（这里只验证部分数据以保证正确性）
    cudaMemcpy(h_a, d_a, total_bytes, cudaMemcpyDeviceToHost);

    // 示例：输出第一个数据线的前几个结果
    int line_index = 0;  // 对应于 j = 0, k = 0
    printf("Results for first line (j=0, k=0):\n");

    // 用串行方式计算一个结果
    for (int i = 0; i < NX; i++) {
        if (i == 0)
            h_b[0] = h_b[0];
        else {
            h_b[i] = h_b[i - 1] / 2 + h_b[i];
        }
    }

    for (int i = 0; i < NX; i++) {
        // 注意：间隔为 NY*NZ
        printf("h_a[%d] = %f   h_b[%d] = %f\n", i, h_a[i * (NY * NZ)], i, h_b[i]);
    }

    // 清理
    free(h_a);
    cudaFree(d_a);

    return 0;
}

#endif

#if RUNNING_MODE == 3
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

// 自定义操作符：在扫描时，将累积值先除以 2，再加上当前的输入
struct MyScanOp {
    __device__ double operator()(double a, double b) const {
//        return a / 2.0 + b;
        return a + b;
    }
};

int main() {
    const int L = 20;
    const int num_items = L * L * L;

    // 主机输入数据：假设 old_a 数组中，第 0 个元素直接作为初始值，
    // 后续元素参与扫描（例如 old_a[1] ... old_a[899]）。
    double h_input[num_items];
    for (int i = 0; i < num_items; i++) {
        h_input[i] = static_cast<double>(i + 1); // 例如初始化为 1, 2, 3, …, 900
    }

    // 设备内存分配
    double *d_input, *d_output;
    cudaMalloc(&d_input, num_items * sizeof(double));
    cudaMalloc(&d_output, num_items * sizeof(double));
    cudaMemcpy(d_input, h_input, num_items * sizeof(double), cudaMemcpyHostToDevice);

    // 临时存储指针与大小
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 第一次调用获取临时存储空间大小
    cub::DeviceScan::InclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_input, d_output,
            MyScanOp(),
            num_items
    );

    // 分配临时存储空间
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 执行 inclusive scan（前缀扫描），应用自定义操作符
    cub::DeviceScan::InclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_input, d_output,
            MyScanOp(),
            num_items
    );

    // 将结果复制回主机查看
    double h_output[num_items];
    cudaMemcpy(h_output, d_output, num_items * sizeof(double), cudaMemcpyDeviceToHost);

    // 输出部分结果用于检查
    std::cout << "First 10 scan results:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "output[" << i << "] = " << h_output[i] << std::endl;
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);

    return 0;
}

#endif

#if RUNNING_MODE == 4
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define Max(a, b) ((a) > (b) ? (a) : (b))

//#define L 900

//#define nx L
//#define ny L
//#define nz L

#define nx 900
#define ny 900
#define nz 900

//#define nx 32
//#define ny 32
//#define nz 32

#define IT_MAX 10

#define BLOCK_SIZE 64       // 默认 32, 目前 64 表现最佳

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


// 在 CPU 上进行初始化
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
































//// 用 CUB 的 BlockScan 替代串行更新的核函数版本
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
//    int j = blockIdx.y + 1;
//    int k = blockIdx.z + 1;
//
//    // 共享内存用于存放当前数据线，共 nx 个元素
//    extern __shared__ double s_a[];
//
//    // 加载整行数据到共享内存
//    // 为方便，这里每个线程负责加载一个数据（要求 blockDim.x >= nx）
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // 数据全局索引： i * ny * nz + j * nz + k  (i从0到nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
////    // 加载整行数据到共享内存 (使用串行方式读取,以便于 debug) *********
////    for (int i = 0; i < nx; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            s_a[i] = a[i * ny * nz + j * nz + k] / 2.;
////        }
////    }
//
//    __syncthreads();
//
////    // 第一次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0) {
////        printf("第一次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // 预先添加上 s_a[0] 的值
//    }
//    __syncthreads();
//
//    // 利用 CUB BlockScan 在块内对 s_a[] 做前缀扫描
//    // 这里我们要“扫描”的对象是 s_a 数组中的每个元素，采用自定义操作符 MyOp
//    typedef cub::BlockScan<double, nx> BlockScan;       // 这里 <> 中的第二个参数是当前 block 的大小, 很关键!!!
//    __shared__ typename BlockScan::TempStorage temp_storage;
//
//
//    if (threadIdx.x == 0 && blockIdx.x == 0)
//        printf("TempStorage size: %llu\n", (unsigned long long) sizeof(BlockScan::TempStorage));
//
//    double scan_val = 0.0;
//    // 调用 InclusiveScan：
//    // 得到结果：result[i] = s_a[0] op s_a[1] op ... op s_a[i]
//    // 这里不可以这样, 必须要 block 内全部线程都执行 BlockScan
////    if (tid >= 2 && tid < nx)
////        BlockScan(temp_storage).InclusiveScan(s_a[tid], scan_val, MyOp());
////    __syncthreads();
//    // 因此我们将 tid_0 tid_1 线程传入的 input值设定成 0
//
////    double input = (tid >= 2 && tid < nx) ? s_a[tid] : 0.0;
//
//    // 第二次查看 s_a 数组内容
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("第二次\n");
//        for (int ii = 0; ii < nx; ++ii)
//            printf("%.2lf ", s_a[ii]);
//        printf("\n");
//    }
//
//    double input;
//    if (tid >= 2 && tid < nx)
//        input = s_a[tid];
//    else
//        input = 0.;
//
//
//    __syncthreads();
//
//    BlockScan(temp_storage).InclusiveScan(input, scan_val, MyOp());
//
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
////        printf(" OK aga");
//
//    __syncthreads();
//
//
//
//    // 注意！！ 这里不需要写回 共享内存, 可以直接将结果写入全局内存
//    // 将扫描结果写回共享内存（扫描结果应对应转换后数据）
//    if (tid >= 2 && tid < nx) {
//        s_a[tid] = scan_val;
//    }
//    __syncthreads();
//
//    // 第三次查看 s_a 数组内容
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("第三次\n");
//        for (int ii = 0; ii < nx; ++ii)
//            printf("%.2lf ", s_a[ii]);
//        printf("\n");
//    }
//
//    // 此处若需要再做后处理（例如除以某个系数恢复原变量），可以在这里添加
//    // 例如： new_value = (转换结果) / (2^i) 之类的逆变换
//
//    // 将更新后的共享内存结果写回全局内存（只更新内部数据）
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // 错位存储
//    }
//    // 边界可以保持不变或其他处理
//
////    // 用串行方式写回全局内存, 确保正确无误
////    for (int i = 1; i < nx - 1; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            a[i * ny * nz + j * nz + k] = s_a[i];
////        }
////    }
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (计算结果正确)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//    // 共享内存用于存放当前数据线，共 nx 个元素
//    extern __shared__ double s_a[];
//
//    // 并行算法加载整行数据到共享内存   这里每个线程负责加载一个数据（要求 blockDim.x >= nx）
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // 数据全局索引： i * ny * nz + j * nz + k  (i从0到nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // 第一次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第一次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // 第二次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第二次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//    // 串行更新
//    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
//        for (int i = 1; i < nx - 1; i++) {
//            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
//        }
//    }
//    __syncthreads();
//
////    // 第三次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第三次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid];
//    }
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (在采用串行计算成功运行的基础上尝试重新使用并行计算) (计算结果错误, 原因: myop 不满足交换律)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//
//    // 共享内存用于存放当前数据线，共 nx 个元素
//    extern __shared__ double s_a[];
//
//    // 并行算法加载整行数据到共享内存   这里每个线程负责加载一个数据（要求 blockDim.x >= nx）
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // 数据全局索引： i * ny * nz + j * nz + k  (i从0到nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // 第一次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第一次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // 第二次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第二次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // 串行更新
////    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
////        for (int i = 1; i < nx - 1; i++) {
////            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
////        }
//////        __syncthreads(); // 确保其他线程能看到更新后的值
////    }
////    __syncthreads();
//
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // 预先添加上 s_a[0] 的值
//    }
//    __syncthreads();
//    typedef cub::BlockScan<double, nx> BlockScan;       // 这里 <> 中的第二个参数是当前 block 的大小, 很关键!!!
//    __shared__ typename BlockScan::TempStorage temp_storage;
//
//    double scan_val = 0.0;
//    double input = (tid >= 2 && tid < nx) ? s_a[tid] : 0.;
//    __syncthreads();
//    BlockScan(temp_storage).InclusiveScan(input, scan_val, MyOp());
//    __syncthreads();
//
//
//
//
//
//
//
//
//
//
//
//
//
////    // 第三次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第三次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//
//    // 将共享内存结果写回全局内存
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // 错位存储
//    }
//}



//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确)
 __global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本     (目前最快) (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;
    int k = thread_global_id % nz;

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;

//        int idx_center = i * ny * nz + j * nz + k;
//        int idx_left = (i - 1) * ny * nz + j * nz + k;
//        int idx_right = (i + 1) * ny * nz + j * nz + k;
//
//        a[idx_center] = (a[idx_left] + a[idx_right]) / 2.0;
    }
}













// i方向核函数
__global__ void compute_i_direction_kernel(double *a) {
    unsigned long long time_checker_1 = clock64();      // 开始计时

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前行
    extern __shared__ double s_a[];


    // 加载整行数据到共享内存
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    unsigned long long time_checker_2 = clock64();      // 23


    // 串行更新，确保正确的依赖关系
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // 写回全局内存
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[i];
        }
    }
    __syncthreads();                                    // 19
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 &&
        blockIdx.z == 0)
        printf("time_checker_1: %lld    time_checker_2: %lld    time_checker_3: %lld\n",
               time_checker_2 - time_checker_1,
               time_checker_3 - time_checker_2,
               clock64() - time_checker_3);
}


// j方向核函数
__global__ void compute_j_direction_kernel(double *a) {
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前列
    extern __shared__ double s_a[];

    // 加载整列数据到共享内存
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新，确保正确的依赖关系
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[j] = (s_a[j - 1] + s_a[j + 1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    // 写回全局内存
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k方向核函数，包含eps计算
__global__ void compute_k_direction_kernel(double *a, double *d_eps) {
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // 使用共享内存保存当前"柱"和局部最大误差
    extern __shared__ double shared[];
    double *s_a = shared;                        // 用于存储k方向数据
    double *s_eps = &shared[nz];                // 用于存储局部误差

    s_eps[threadIdx.x] = 0.0;  // 初始化局部误差

//    // 加载整个k方向数据到共享内存
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // 加载边界值
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // 加载整个k方向数据到共享内存
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新k方向并计算局部最大误差
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k - 1] + s_a[k + 1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // 更新局部最大误差
        }
//        __syncthreads();
    }

    // 写回全局内存
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // 使用原子操作更新全局最大误差
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}

// 辅助函数：将不同表示形式的双精度数转换
__device__ double __longlong_as_double(unsigned long long int x) {
    return __longlong_as_double(x);
}

int main(int argc, char *argv[]) {
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;


    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // 定义线程块和网格配置 - 移到循环外部以减少重复设置
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // 每个线程块处理一行的所有i
    dim3 grid_j(nx - 2, 1, nz - 2);      // 每个线程块处理一列的所有j
    dim3 grid_k(nx - 2, ny - 2, 1);      // 每个线程块处理一个"柱"的所有k

    dim3 my_block(1, 1, 1);     // 每个块内仅仅 1 个线程
    dim3 my_grid(1, ny, nz);    // 总共 ny * nz 个线程



    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // 每个块内的线程数量
    int total_lines = ny * nz;      // x 方向的总线数
    int blocks = (total_lines + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************



    // 共享内存大小 - 同样移到循环外部
    size_t shared_mem_size_i = nx * sizeof(double);                  // 整行
    size_t shared_mem_size_j = ny * sizeof(double);                  // 整列
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // 整"柱"加上临时存储

    for (it = 1; it <= itmax; it++) {
        // 优化点2: 不再每次迭代重新拷贝完整数据

        check_time_a = omp_get_wtime();

        // 初始化误差值为0
        double h_eps = 0.0;
        // ***** 耗时很少 *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        check_time_b = omp_get_wtime();


        if (false)
            // i方向更新
            compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        else // 使用 cub 版本的 i 方向更新
//            compute_i_direction_kernel_cub<<<my_grid, my_block>>>(d_A); // 每个块内 1 个线程
            compute_i_direction_kernel_32_thread<<<blocks, threads_per_block>>>(d_A); // 每个块内 32 个线程

//        dim3 grid_i(1, ny, nz);
//        dim3 block(900, 1, 1); // 至少需要 >= 900 线程
//        size_t shared_mem_size_i = nx * sizeof(double);  // 900 * sizeof(double)
//        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_c = omp_get_wtime();

        // j方向更新
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // k方向更新
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());



        // 优化点3: 只传输eps值而不是整个数组
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));

    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
    printf(" ADI Benchmark (1) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
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

#if RUNNING_MODE == 5
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define Max(a, b) ((a) > (b) ? (a) : (b))

//#define L 900

//#define nx L
//#define ny L
//#define nz L

#define nx 900
#define ny 900
#define nz 900

//#define nx 32
//#define ny 32
//#define nz 32

#define IT_MAX 10

#define BLOCK_SIZE 32

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


// 在 CPU 上进行初始化
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
































//// 用 CUB 的 BlockScan 替代串行更新的核函数版本
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
//    int j = blockIdx.y + 1;
//    int k = blockIdx.z + 1;
//
//    // 共享内存用于存放当前数据线，共 nx 个元素
//    extern __shared__ double s_a[];
//
//    // 加载整行数据到共享内存
//    // 为方便，这里每个线程负责加载一个数据（要求 blockDim.x >= nx）
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // 数据全局索引： i * ny * nz + j * nz + k  (i从0到nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
////    // 加载整行数据到共享内存 (使用串行方式读取,以便于 debug) *********
////    for (int i = 0; i < nx; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            s_a[i] = a[i * ny * nz + j * nz + k] / 2.;
////        }
////    }
//
//    __syncthreads();
//
////    // 第一次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0) {
////        printf("第一次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // 预先添加上 s_a[0] 的值
//    }
//    __syncthreads();
//
//    // 利用 CUB BlockScan 在块内对 s_a[] 做前缀扫描
//    // 这里我们要“扫描”的对象是 s_a 数组中的每个元素，采用自定义操作符 MyOp
//    typedef cub::BlockScan<double, nx> BlockScan;       // 这里 <> 中的第二个参数是当前 block 的大小, 很关键!!!
//    __shared__ typename BlockScan::TempStorage temp_storage;
//
//
//    if (threadIdx.x == 0 && blockIdx.x == 0)
//        printf("TempStorage size: %llu\n", (unsigned long long) sizeof(BlockScan::TempStorage));
//
//    double scan_val = 0.0;
//    // 调用 InclusiveScan：
//    // 得到结果：result[i] = s_a[0] op s_a[1] op ... op s_a[i]
//    // 这里不可以这样, 必须要 block 内全部线程都执行 BlockScan
////    if (tid >= 2 && tid < nx)
////        BlockScan(temp_storage).InclusiveScan(s_a[tid], scan_val, MyOp());
////    __syncthreads();
//    // 因此我们将 tid_0 tid_1 线程传入的 input值设定成 0
//
////    double input = (tid >= 2 && tid < nx) ? s_a[tid] : 0.0;
//
//    // 第二次查看 s_a 数组内容
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("第二次\n");
//        for (int ii = 0; ii < nx; ++ii)
//            printf("%.2lf ", s_a[ii]);
//        printf("\n");
//    }
//
//    double input;
//    if (tid >= 2 && tid < nx)
//        input = s_a[tid];
//    else
//        input = 0.;
//
//
//    __syncthreads();
//
//    BlockScan(temp_storage).InclusiveScan(input, scan_val, MyOp());
//
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
////        printf(" OK aga");
//
//    __syncthreads();
//
//
//
//    // 注意！！ 这里不需要写回 共享内存, 可以直接将结果写入全局内存
//    // 将扫描结果写回共享内存（扫描结果应对应转换后数据）
//    if (tid >= 2 && tid < nx) {
//        s_a[tid] = scan_val;
//    }
//    __syncthreads();
//
//    // 第三次查看 s_a 数组内容
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("第三次\n");
//        for (int ii = 0; ii < nx; ++ii)
//            printf("%.2lf ", s_a[ii]);
//        printf("\n");
//    }
//
//    // 此处若需要再做后处理（例如除以某个系数恢复原变量），可以在这里添加
//    // 例如： new_value = (转换结果) / (2^i) 之类的逆变换
//
//    // 将更新后的共享内存结果写回全局内存（只更新内部数据）
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // 错位存储
//    }
//    // 边界可以保持不变或其他处理
//
////    // 用串行方式写回全局内存, 确保正确无误
////    for (int i = 1; i < nx - 1; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            a[i * ny * nz + j * nz + k] = s_a[i];
////        }
////    }
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (计算结果正确)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//    // 共享内存用于存放当前数据线，共 nx 个元素
//    extern __shared__ double s_a[];
//
//    // 并行算法加载整行数据到共享内存   这里每个线程负责加载一个数据（要求 blockDim.x >= nx）
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // 数据全局索引： i * ny * nz + j * nz + k  (i从0到nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // 第一次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第一次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // 第二次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第二次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//    // 串行更新
//    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
//        for (int i = 1; i < nx - 1; i++) {
//            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
//        }
//    }
//    __syncthreads();
//
////    // 第三次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第三次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid];
//    }
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (在采用串行计算成功运行的基础上尝试重新使用并行计算) (计算结果错误, 原因: myop 不满足交换律)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//
//    // 共享内存用于存放当前数据线，共 nx 个元素
//    extern __shared__ double s_a[];
//
//    // 并行算法加载整行数据到共享内存   这里每个线程负责加载一个数据（要求 blockDim.x >= nx）
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // 数据全局索引： i * ny * nz + j * nz + k  (i从0到nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // 第一次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第一次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // 第二次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第二次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // 串行更新
////    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
////        for (int i = 1; i < nx - 1; i++) {
////            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
////        }
//////        __syncthreads(); // 确保其他线程能看到更新后的值
////    }
////    __syncthreads();
//
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // 预先添加上 s_a[0] 的值
//    }
//    __syncthreads();
//    typedef cub::BlockScan<double, nx> BlockScan;       // 这里 <> 中的第二个参数是当前 block 的大小, 很关键!!!
//    __shared__ typename BlockScan::TempStorage temp_storage;
//
//    double scan_val = 0.0;
//    double input = (tid >= 2 && tid < nx) ? s_a[tid] : 0.;
//    __syncthreads();
//    BlockScan(temp_storage).InclusiveScan(input, scan_val, MyOp());
//    __syncthreads();
//
//
//
//
//
//
//
//
//
//
//
//
//
////    // 第三次查看 s_a 数组内容
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("第三次\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//
//    // 将共享内存结果写回全局内存
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // 错位存储
//    }
//}



//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确)
 __global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本     (目前最快) (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;
    int k = thread_global_id % nz;

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;

//        int idx_center = i * ny * nz + j * nz + k;
//        int idx_left = (i - 1) * ny * nz + j * nz + k;
//        int idx_right = (i + 1) * ny * nz + j * nz + k;
//
//        a[idx_center] = (a[idx_left] + a[idx_right]) / 2.0;
    }
}













// i方向核函数
__global__ void compute_i_direction_kernel(double *a) {
    unsigned long long time_checker_1 = clock64();      // 开始计时

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前行
    extern __shared__ double s_a[];


    // 加载整行数据到共享内存
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    unsigned long long time_checker_2 = clock64();      // 23


    // 串行更新，确保正确的依赖关系
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // 写回全局内存
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[i];
        }
    }
    __syncthreads();                                    // 19
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 &&
        blockIdx.z == 0)
        printf("time_checker_1: %lld    time_checker_2: %lld    time_checker_3: %lld\n",
               time_checker_2 - time_checker_1,
               time_checker_3 - time_checker_2,
               clock64() - time_checker_3);
}


// j方向核函数
__global__ void compute_j_direction_kernel(double *a) {
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前列
    extern __shared__ double s_a[];

    // 加载整列数据到共享内存
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新，确保正确的依赖关系
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[j] = (s_a[j - 1] + s_a[j + 1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    // 写回全局内存
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k方向核函数，包含eps计算
__global__ void compute_k_direction_kernel(double *a, double *d_eps) {
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // 使用共享内存保存当前"柱"和局部最大误差
    extern __shared__ double shared[];
    double *s_a = shared;                        // 用于存储k方向数据
    double *s_eps = &shared[nz];                // 用于存储局部误差

    s_eps[threadIdx.x] = 0.0;  // 初始化局部误差

//    // 加载整个k方向数据到共享内存
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // 加载边界值
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // 加载整个k方向数据到共享内存
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新k方向并计算局部最大误差
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k - 1] + s_a[k + 1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // 更新局部最大误差
        }
//        __syncthreads();
    }

    // 写回全局内存
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // 使用原子操作更新全局最大误差
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}

// 辅助函数：将不同表示形式的双精度数转换
__device__ double __longlong_as_double(unsigned long long int x) {
    return __longlong_as_double(x);
}

int main(int argc, char *argv[]) {
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;


    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // 定义线程块和网格配置 - 移到循环外部以减少重复设置
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // 每个线程块处理一行的所有i
    dim3 grid_j(nx - 2, 1, nz - 2);      // 每个线程块处理一列的所有j
    dim3 grid_k(nx - 2, ny - 2, 1);      // 每个线程块处理一个"柱"的所有k

    dim3 my_block(1, 1, 1);     // 每个块内仅仅 1 个线程
    dim3 my_grid(1, ny, nz);    // 总共 ny * nz 个线程



    //**********************************************************
    //**********************************************************
    int threads_per_block = 32;     // 每个块内 32 个线程
    int total_lines = ny * nz;      // x 方向的总线数
    int blocks = (total_lines + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************



    // 共享内存大小 - 同样移到循环外部
    size_t shared_mem_size_i = nx * sizeof(double);                  // 整行
    size_t shared_mem_size_j = ny * sizeof(double);                  // 整列
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // 整"柱"加上临时存储

    for (it = 1; it <= itmax; it++) {
        // 优化点2: 不再每次迭代重新拷贝完整数据

        check_time_a = omp_get_wtime();

        // 初始化误差值为0
        double h_eps = 0.0;
        // ***** 耗时很少 *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        check_time_b = omp_get_wtime();


        if (false)
            // i方向更新
            compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        else // 使用 cub 版本的 i 方向更新
//            compute_i_direction_kernel_cub<<<my_grid, my_block>>>(d_A); // 每个块内 1 个线程
            compute_i_direction_kernel_32_thread<<<blocks, threads_per_block>>>(d_A); // 每个块内 32 个线程

//        dim3 grid_i(1, ny, nz);
//        dim3 block(900, 1, 1); // 至少需要 >= 900 线程
//        size_t shared_mem_size_i = nx * sizeof(double);  // 900 * sizeof(double)
//        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_c = omp_get_wtime();

        // j方向更新
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // k方向更新
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());



        // 优化点3: 只传输eps值而不是整个数组
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));

    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
    printf(" ADI Benchmark (1) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
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

#if RUNNING_MODE == 6
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define Max(a, b) ((a) > (b) ? (a) : (b))

//#define L 900

//#define nx L
//#define ny L
//#define nz L

#define nx 900
#define ny 900
#define nz 900

//#define nx 32
//#define ny 32
//#define nz 32

#define IT_MAX 10

#define BLOCK_SIZE 32       // 默认 32, 目前 64 表现最佳

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


// 在 CPU 上进行初始化
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

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本   (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确) (目前最快)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;

    }
}

// 同理, 仿照 compute_i_direction_kernel_32_thread 写出 compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    // 解码出 i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // 如果有空线程, 直接返回即可
        return;

    int i = thread_global_id / nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！

    // 每个线程串行处理一整条 y 方向数据线
    for (int j = 1; j < ny - 1; ++j) {
        a[i * ny * nz + j * nz + k] = (a[i * ny * nz + (j - 1) * nz + k] + a[i * ny * nz + (j + 1) * nz + k]) / 2.0;
    }
}



// 同理, 仿照 compute_i_direction_kernel_32_thread 写出 compute_j_direction_kernel_32_thread
// 使用共享内存版本, 底下再写一个使用寄存器版本
// 因为这些数据仅在线程本地使用, 不需要跨线程, 因此不需要使用共享内存
// ！！！有问题
__global__ void compute_k_direction_kernel_32_thread_shared_memory(double *a, double* d_eps) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    extern __shared__ double shared[];
    double *s_eps = &shared[blockDim.x * 0];
    double *d_al  = &shared[blockDim.x * 1];
    double *d_ac  = &shared[blockDim.x * 2];
    double *d_ar  = &shared[blockDim.x * 3];

    // 解码出 i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可
        return;


    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！


    s_eps[tid_in_block] = 0.; // 初始化局部误差
    d_al[tid_in_block] = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac[tid_in_block] = a[i * ny * nz + j * nz + 1];   // k = 1

    // 每个线程串行处理一整条 z 方向数据线
    for (int k = 1; k < nz - 1; ++k) {
        d_ar[tid_in_block] = a[i * ny * nz + j * nz + (k + 1)];

        // 计算局部误差值
        // 在 900 次调用的情况下, 使用宏 Max 会比 fmax 内置函数更高效, 原因是不需要函数调用的额外开销
//        s_eps[tid_in_block] = Max(s_eps[tid_in_block], fabs((d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0 - d_ac[tid_in_block]));
        s_eps[tid_in_block] = fmax(s_eps[tid_in_block], fabs((d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0 - d_ac[tid_in_block]));


        // 优化的事情就交给寄存器吧！但愿寄存器能存储这个除法的值
        a[i * ny * nz + j * nz + k] = (d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0;

        // 更新 d_al 和 d_ac
        d_al[tid_in_block] = (d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0;
        d_ac[tid_in_block] = d_ar[tid_in_block];
    }

    __syncthreads();
    double max_eps = s_eps[0];
    // 首先通过串行操作在 block 内将最大值保存在 s_eps[0] 中
    if (threadIdx.x == 0) {
        for (int k = 1; k < nz; ++k) {
            if (s_eps[k] > max_eps) {
                max_eps = s_eps[k];
            }
        }
    }
    __syncthreads();

    // 然后 0 线程将局部误差存储到全局内存中去
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}



__global__ void compute_k_direction_kernel_32_thread(double *a, double* d_eps) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double s_eps = 0.;  // 初始化局部误差
    double d_al;
    double d_ac;
    double d_ar;
    double tmp;
    double tmp2;

    // 解码出 i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可
        return;


    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // 每个线程串行处理一整条 z 方向数据线
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];

        // 计算局部误差值
        // 在 900 次调用的情况下, 使用宏 Max 会比 fmax 内置函数更高效, 原因是不需要函数调用的额外开销
//        s_eps[tid_in_block] = Max(s_eps[tid_in_block], fabs((d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0 - d_ac[tid_in_block]));
        tmp = (d_al + d_ar) / 2.0;
        // 优化的事情就交给寄存器吧！但愿寄存器能存储这个除法的值
        a[i * ny * nz + j * nz + k] = tmp;

        tmp2 = Max(s_eps, fabs(d_ac - tmp));
        s_eps = tmp2;

        // 更新 d_al 和 d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // 将局部误差存储到全局内存中去
    atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
}



// i方向核函数
__global__ void compute_i_direction_kernel(double *a) {
    unsigned long long time_checker_1 = clock64();      // 开始计时

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前行
    extern __shared__ double s_a[];


    // 加载整行数据到共享内存
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    unsigned long long time_checker_2 = clock64();      // 23


    // 串行更新，确保正确的依赖关系
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // 写回全局内存
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[i];
        }
    }
    __syncthreads();                                    // 19
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 &&
        blockIdx.z == 0)
        printf("time_checker_1: %lld    time_checker_2: %lld    time_checker_3: %lld\n",
               time_checker_2 - time_checker_1,
               time_checker_3 - time_checker_2,
               clock64() - time_checker_3);
}


// j方向核函数
__global__ void compute_j_direction_kernel(double *a) {
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // 使用共享内存保存当前列
    extern __shared__ double s_a[];

    // 加载整列数据到共享内存
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新，确保正确的依赖关系
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
            s_a[j] = (s_a[j - 1] + s_a[j + 1]) / 2.0;
        }
//        __syncthreads(); // 确保其他线程能看到更新后的值
    }

    // 写回全局内存
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k方向核函数，包含eps计算
__global__ void compute_k_direction_kernel(double *a, double *d_eps) {
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // 使用共享内存保存当前"柱"和局部最大误差
    extern __shared__ double shared[];
    double *s_a = shared;                        // 用于存储k方向数据
    double *s_eps = &shared[nz];                // 用于存储局部误差

    s_eps[threadIdx.x] = 0.0;  // 初始化局部误差

//    // 加载整个k方向数据到共享内存
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // 加载边界值
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // 加载整个k方向数据到共享内存
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // 需要

    // 串行更新k方向并计算局部最大误差
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k - 1] + s_a[k + 1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // 更新局部最大误差
        }
//        __syncthreads();
    }

    // 写回全局内存
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // 使用原子操作更新全局最大误差
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}

// 辅助函数：将不同表示形式的双精度数转换
__device__ double __longlong_as_double(unsigned long long int x) {
    return __longlong_as_double(x);
}

int main(int argc, char *argv[]) {
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;


    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // 定义线程块和网格配置 - 移到循环外部以减少重复设置
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // 每个线程块处理一行的所有i
    dim3 grid_j(nx - 2, 1, nz - 2);      // 每个线程块处理一列的所有j
    dim3 grid_k(nx - 2, ny - 2, 1);      // 每个线程块处理一个"柱"的所有k

    dim3 my_block(1, 1, 1);     // 每个块内仅仅 1 个线程
    dim3 my_grid(1, ny, nz);    // 总共 ny * nz 个线程



    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // 每个块内的线程数量
    int total_lines_x = ny * nz;    // x 方向的总线数
    int total_lines_y = nx * nz;    // y 方向的总线数
    int total_lines_z = nx * ny;    // z 方向的总线数
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************



    // 共享内存大小 - 同样移到循环外部
    size_t shared_mem_size_i = nx * sizeof(double);                  // 整行
    size_t shared_mem_size_j = ny * sizeof(double);                  // 整列
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // 整"柱"加上临时存储

    for (it = 1; it <= itmax; it++) {
        // 优化点2: 不再每次迭代重新拷贝完整数据

        check_time_a = omp_get_wtime();

        // 初始化误差值为0
        double h_eps = 0.0;
        // ***** 耗时很少 *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        check_time_b = omp_get_wtime();


        if (false)
            // i方向更新
            compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        else // 使用 cub 版本的 i 方向更新
//            compute_i_direction_kernel_cub<<<my_grid, my_block>>>(d_A); // 每个块内 1 个线程
            compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 每个块内 32 个线程



        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_c = omp_get_wtime();

        if (false)
            // j方向更新
            compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        else
            compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 每个块内 32 个线程

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        if (false)
            // k方向更新
            compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        else
//            compute_k_direction_kernel_32_thread_shared_memory<<<blocks_z, threads_per_block, threads_per_block * 4 * sizeof(double)>>>(d_A, d_eps); // 每个块内 32 个线程
            compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());



        // 优化点3: 只传输eps值而不是整个数组
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));

    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
    printf(" ADI Benchmark (1) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
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

#if RUNNING_MODE == 7
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/reduce.h>
//#include <cub/block/block_scan.cuh>   // 未使用

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define MAX(a, b) ((a) > (b) ? (a) : (b))

//#define nx 900
//#define ny 900
//#define nz 900

//#define nx 32
//#define ny 32
//#define nz 32

#define IT_MAX 10
//#define IT_MAX 100
#define L 900
//#define L 384

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 32       // 默认 32, 目前 64 表现最佳

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


// 在 CPU 上进行初始化
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

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本   (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确) (目前最快)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;

    }
}

// 同理, 仿照 compute_i_direction_kernel_32_thread 写出 compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    // 解码出 i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // 如果有空线程, 直接返回即可
        return;

    int i = thread_global_id / nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！

    // 每个线程串行处理一整条 y 方向数据线
    for (int j = 1; j < ny - 1; ++j) {
        a[i * ny * nz + j * nz + k] = (a[i * ny * nz + (j - 1) * nz + k] + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
    }
}

//// 二分规约
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
////        tmp = (d_al + d_ar) / 2.0;   // 以防古老编译器不进行优化
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//}


// warp 内规约
__inline__ __device__
double warp_reduce_max(double val) {
    // 使用 warp shuffle 操作做规约
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// 为 block_reduce 准备的 shared memory
    __shared__ double sh_data[BLOCK_SIZE];

    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double tmp;
    double s_eps = 0.;  // 初始化局部误差
    double d_al, d_ac, d_ar;  // left  center  right

    // 解码出 i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
        return;

    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // 每个线程串行处理一整条 z 方向数据线
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // 计算局部误差值
//        tmp = (d_al + d_ar) / 2.0;   // 以防古老编译器不进行优化
        tmp = (d_al + d_ar) * 0.5;  // 手动优化
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // 更新 d_al 和 d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // 使用 warp 内规约
    double warp_max = warp_reduce_max(s_eps);
    if (threadIdx.x == 0)
        d_eps[blockIdx.x] = warp_max;

//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // 第 i 个块
//    int tid = threadIdx.x; // 第 idx 个进程
//    int idx_thread_global = tid + i * blockDim.x; // 相对于 A 数组（即 block_max）的地址位置
//    // 将块内所有值存入shared mem中
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // 让每个线程不超出全局线程
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // 对于最后一个 block 的超出部分我们填充进一个极小值
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // 二分法块内归约
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // 很重要，用MAX而不是max
//        }
//        __syncthreads();
//    }
//    // 第 0 个线程将当前最大值写入全局内存
//    if (tid == 0) {
//        B[i] = sdata[0];
//    }
//}

int main(int argc, char *argv[]) {
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps, *d_out;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;

    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // 每个块内的线程数量
    int total_lines_x = ny * nz;    // x 方向的总线数
    int total_lines_y = nx * nz;    // y 方向的总线数
    int total_lines_z = nx * ny;    // z 方向的总线数
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // 存放每个block内最大值
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_out, sizeof(double)));  // 存放全局规约(global_reduce)结果

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // 为 CUB 预分配内存
    // d_eps: 输入数据在设备上的指针
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int num_items = blocks_z;   // 等于block的总数量

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_eps, d_out, num_items);
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));


    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // 初始化误差值为0  (使用block_reduce + global_reduce时不需要初始化误差)
//        double h_eps = 0.0;
//        // ***** 耗时很少 *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));



        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// 做 block_reduce 需要用到 shared memory
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double)>>>(d_A, d_eps); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // 此时d_eps中存储每个block内的最大值
        // global_reduce 1
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_eps, d_out, num_items);


        // 优化点3: 只传输eps值而不是整个数组
        // 将d_out全局最大规约值存入eps变量
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));

    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
    printf(" ADI Benchmark (1) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    if (L == 900 && IT_MAX == 10)
        printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 384 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
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

#if RUNNING_MODE == 8
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
//#include <cub/block/block_scan.cuh>   // 未使用

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define MAX(a, b) ((a) > (b) ? (a) : (b))

//#define nx 900
//#define ny 900
//#define nz 900

//#define nx 32
//#define ny 32
//#define nz 32

#define IT_MAX 10
//#define IT_MAX 100
#define L 900
//#define L 384

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 32       // 默认 32, 目前 64 表现最佳

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


// 在 CPU 上进行初始化
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

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本   (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确) (目前最快)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;

    }
}

// 同理, 仿照 compute_i_direction_kernel_32_thread 写出 compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    // 解码出 i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // 如果有空线程, 直接返回即可
        return;

    int i = thread_global_id / nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！

    // 每个线程串行处理一整条 y 方向数据线
    for (int j = 1; j < ny - 1; ++j) {
        a[i * ny * nz + j * nz + k] = (a[i * ny * nz + (j - 1) * nz + k] + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
    }
}

//// 二分规约
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
////        tmp = (d_al + d_ar) / 2.0;   // 以防古老编译器不进行优化
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//}


// warp 内规约
__inline__ __device__
double warp_reduce_max(double val) {
    // 使用 warp shuffle 操作做规约
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double tmp;
    double s_eps = 0.;  // 初始化局部误差
    double d_al, d_ac, d_ar;  // left  center  right

    // 解码出 i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
        return;

    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // 每个线程串行处理一整条 z 方向数据线
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // 计算局部误差值
        tmp = (d_al + d_ar) * 0.5;  // 手动优化
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // 更新 d_al 和 d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // 将局部误差存储到全局内存中去   // 原子操作性能较低
    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
    // 其实一点都不慢！！！
    atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // 第 i 个块
//    int tid = threadIdx.x; // 第 idx 个进程
//    int idx_thread_global = tid + i * blockDim.x; // 相对于 A 数组（即 block_max）的地址位置
//    // 将块内所有值存入shared mem中
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // 让每个线程不超出全局线程
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // 对于最后一个 block 的超出部分我们填充进一个极小值
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // 二分法块内归约
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // 很重要，用MAX而不是max
//        }
//        __syncthreads();
//    }
//    // 第 0 个线程将当前最大值写入全局内存
//    if (tid == 0) {
//        B[i] = sdata[0];
//    }
//}

int main(int argc, char *argv[]) {
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps, *d_out;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;

    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // 每个块内的线程数量
    int total_lines_x = ny * nz;    // x 方向的总线数
    int total_lines_y = nx * nz;    // y 方向的总线数
    int total_lines_z = nx * ny;    // z 方向的总线数
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));  // 存放每个block内最大值

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

        // 初始化误差值为0  (使用block_reduce + global_reduce时不需要初始化误差)
        double h_eps = 0.0;
        // ***** 耗时很少 *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// 做 block_reduce 需要用到 shared memory
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();


        // 优化点3: 只传输eps值而不是整个数组
        // 将d_out全局最大规约值存入eps变量
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
    free(a);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_eps));

    printf(" using RUNNING_MODE: %d\n", RUNNING_MODE);
    printf(" ADI Benchmark (1) Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", it < itmax ? it : itmax);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    if (L == 900 && IT_MAX == 10)
        printf(" Verification    =       %12s\n", (fabs(eps - 9.0896725E-01) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    else if (L == 384 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
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

#if RUNNING_MODE == 9
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define MAX(a, b) ((a) > (b) ? (a) : (b))

//#define nx 900
//#define ny 900
//#define nz 900

//#define nx 32
//#define ny 32
//#define nz 32

#define IT_MAX 10
//#define IT_MAX 500
#define L 900
//#define L 384

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 32       // 默认 32, 目前 64 表现最佳

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


// 在 CPU 上进行初始化
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

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本   (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确) (目前最快)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // 将本轮计算结果作为下轮计算的左变量

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！

    d_al = a[(1 - 1) * ny * nz + j * nz + k];

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {
        d_al = (d_al + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

// 同理, 仿照 compute_i_direction_kernel_32_thread 写出 compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double d_al; // 将本轮计算结果作为下轮计算的左变量

    // 解码出 i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // 如果有空线程, 直接返回即可
        return;

    int i = thread_global_id / nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！

    d_al = a[i * ny * nz + (1 - 1) * nz + k];

    // 每个线程串行处理一整条 y 方向数据线
    for (int j = 1; j < ny - 1; ++j) {
        d_al = (d_al + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

//// 二分规约 -- 无 tmp  (慢于普通二分规约)
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    register double s_eps = 0.;  // 初始化局部误差
//    register double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    if (thread_global_id >= nx * ny)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
//        a[i * ny * nz + j * nz + k] = (d_al + d_ar) * 0.5;
//        s_eps = MAX(s_eps, fabs(d_ac - (d_al + d_ar) * 0.5));
//
//        // 更新 d_al 和 d_ac
//        d_al = (d_al + d_ar) * 0.5;
//        d_ac = d_ar;
//    }
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//}

// 二分规约
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// 为 block_reduce 准备的 shared memory
    __shared__ double sh_data[BLOCK_SIZE];
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double tmp;
    double s_eps = 0.;  // 初始化局部误差
    double d_al, d_ac, d_ar;  // left  center  right

    // 解码出 i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
        return;

    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // 每个线程串行处理一整条 z 方向数据线
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // 计算局部误差值
        tmp = (d_al + d_ar) * 0.5;  // 手动优化
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // 更新 d_al 和 d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // 首先将各个线程的 s_eps 写入共享内存
    sh_data[tid_in_block] = s_eps;
    __syncthreads();

    // 使用 二分 块内规约
    // 将局部误差存储到全局内存中去   // 原子操作性能较低
    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
    // (这里我们使用 лекция 上讲的 вариант 4)
    // 首先我们知道做 block_reduce 需要用到shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
    if (tid_in_block == 0) // 写回block_reduce结果
        d_eps[blockIdx.x] = sh_data[0];
}

//// 二分规约   clock debug 版
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    unsigned long long time_checker_1 = clock64();
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    __syncthreads();
//
//
//    unsigned long long time_checker_2 = clock64();
//
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    unsigned long long time_checker_3 = clock64();
//    __syncthreads();
//
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//
//
//    unsigned long long time_checker_4 = clock64();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//
//
//
//    unsigned long long time_checker_5 = clock64();
//
//
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//
//
//    if (blockIdx.x == 0 && threadIdx.x == 0)
//        printf("time_checker_1: %lld    time_checker_2: %lld    time_checker_3: %lld    time_checker_4: %lld    time_checker_5: %lld\n",
//            time_checker_2 - time_checker_1,
//            time_checker_3 - time_checker_2,
//            time_checker_4 - time_checker_3,
//            time_checker_5 - time_checker_4,
//            clock64() - time_checker_5);
//}


//// warp 内规约的辅助函数
//__inline__ __device__
//double warp_reduce_max(double val) {
//    // 使用 warp shuffle 操作做规约
//    for (int offset = 16; offset > 0; offset /= 2)
//        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
//    return val;
//}
//// warp 内规约
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
////        tmp = (d_al + d_ar) / 2.0;   // 以防古老编译器不进行优化
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // 使用 warp 内规约
//    double warp_max = warp_reduce_max(s_eps);
//    if (threadIdx.x == 0)
//        d_eps[blockIdx.x] = warp_max;
//
////    // 首先将各个线程的 s_eps 写入共享内存
////    sh_data[tid_in_block] = s_eps;
////    __syncthreads();
////
////    // 使用 二分 块内规约
////    // 将局部误差存储到全局内存中去   // 原子操作性能较低
////    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
////    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
////    // (这里我们使用 лекция 上讲的 вариант 4)
////    // 首先我们知道做 block_reduce 需要用到shared memory
////    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
////        if (tid_in_block < s)
////            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
////        __syncthreads();
////    }
////    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
////    if (tid_in_block == 0) // 写回block_reduce结果
////        d_eps[blockIdx.x] = sh_data[0];
//}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // 第 i 个块
//    int tid = threadIdx.x; // 第 idx 个进程
//    int idx_thread_global = tid + i * blockDim.x; // 相对于 A 数组（即 block_max）的地址位置
//    // 将块内所有值存入shared mem中
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // 让每个线程不超出全局线程
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // 对于最后一个 block 的超出部分我们填充进一个极小值
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // 二分法块内归约
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // 很重要，用MAX而不是max
//        }
//        __syncthreads();
//    }
//    // 第 0 个线程将当前最大值写入全局内存
//    if (tid == 0) {
//        B[i] = sdata[0];
//    }
//}

int main(int argc, char *argv[]) {
    double maxeps, eps;
    double *a;
    double *d_A, *d_eps;
    int it, itmax;
    double startt, endt;
    double sum_time_a = 0, sum_time_b = 0, sum_time_c = 0, sum_time_d = 0;
    double check_time_a, check_time_b, check_time_c, check_time_d, check_time_e;

    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // 每个块内的线程数量
    int total_lines_x = ny * nz;    // x 方向的总线数
    int total_lines_y = nx * nz;    // y 方向的总线数
    int total_lines_z = nx * ny;    // z 方向的总线数
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // 存放每个block内最大值

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // 为 CUB 预分配内存
    // d_eps: 输入数据在设备上的指针
    int num_items = blocks_z;   // 等于block的总数量

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // 初始化误差值为0  (使用block_reduce + global_reduce时不需要初始化误差)
//        double h_eps = 0.0;
//        // ***** 耗时很少 *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// 二分规约需要 shared memory
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double) >>>(d_A, d_eps); // 每个块内 32 个线程
//        //// warp 内规约不需要 shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // 此时d_eps中存储每个block内的最大值
        // global_reduce 2  thrust  // 灰常方便！！！
        // 创建一个 Thrust device_ptr 包装器，指向 d_eps（device 上的 block 局部最大误差数组）
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
        // 使用 thrust::reduce 找出最大值（初始值必须是 -∞）
        eps = thrust::reduce(dev_ptr, dev_ptr + num_items, 0., thrust::maximum<double>());

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
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
    else if (L == 384 && IT_MAX == 100)
        printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
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

#if RUNNING_MODE == 10
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define IT_MAX 10
//#define IT_MAX 500
#define L 900
//#define L 384

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 32       // 默认 32, 目前 64 表现最佳

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


// 在 CPU 上进行初始化
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

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本   (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确) (目前最快)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // 将本轮计算结果作为下轮计算的左变量

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！

    d_al = a[(1 - 1) * ny * nz + j * nz + k];

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {
        d_al = (d_al + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

// 同理, 仿照 compute_i_direction_kernel_32_thread 写出 compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double d_al; // 将本轮计算结果作为下轮计算的左变量

    // 解码出 i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // 如果有空线程, 直接返回即可
        return;

    int i = thread_global_id / nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！

    d_al = a[i * ny * nz + (1 - 1) * nz + k];

    // 每个线程串行处理一整条 y 方向数据线
    for (int j = 1; j < ny - 1; ++j) {
        d_al = (d_al + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

//// 二分规约 -- 无 tmp  (慢于普通二分规约)
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    register double s_eps = 0.;  // 初始化局部误差
//    register double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    if (thread_global_id >= nx * ny)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
//        a[i * ny * nz + j * nz + k] = (d_al + d_ar) * 0.5;
//        s_eps = MAX(s_eps, fabs(d_ac - (d_al + d_ar) * 0.5));
//
//        // 更新 d_al 和 d_ac
//        d_al = (d_al + d_ar) * 0.5;
//        d_ac = d_ar;
//    }
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//}


// 自定义打印 debug
#define DEBUG_PRINT(info, bIdx, tIdx)                   \
{                                           \
    if (blockIdx.x == (bIdx) && threadIdx.x == (tIdx))\
        printf("blockIdx: %d, threadIdx: %d: %s\n", (bIdx), (tIdx), (info));               \
}                                           \

//#define DEBUG_PRINT_GLOBAL(info)                   \
//{                                           \
//    \
//        printf("blockIdx: %d, threadIdx: %d: %s\n", blockIdx.x, threadIdx.x, (info));               \
//}   \

#if true
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
// 在二分规约方案的基础上，使用并行加载内存的方式      这种情况下，每个 block 的共享内存大小：33
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// 为 block_reduce 准备的 shared memory
    __shared__ double sh_data[(BLOCK_SIZE + 1) * BLOCK_SIZE];  // 33 * 32 个共享内存单位
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double tmp;
    double s_eps = 0.;  // 初始化局部误差
    double d_al, d_ac, d_ar;  // left  center  right

    // 解码出 i, j
    int ij_total = nx * ny;

    // 不存在空线程, 空线程同样将充当worker
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;

    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！



    // 获取当前block中的line数量
    int line_per_block;
    if (blockIdx.x == ij_total / 32) {
        line_per_block = ij_total % 32; // 最后一个block中line的数量为 total % 32
    } else {
        line_per_block = 32; //  其他block中line数量为 32
    }

    // 同一个block中全部线程协同将block内所有的line中的32项数据导入进shared memory中
    // 不过，对于第一或者第二项，需要单独考虑
    // 首先，我们计算出一条线上的总长度nx除以32的余数：这是第一轮对于一条line而言搬运的数据量，之后的数据将能被对齐
    int first_round_size = nx % 32;
    int round_times = nx / 32;  // 总 i_round 数量
    int flag = 1;



    __syncthreads();

    int i_round = 0;    // 当前的完整轮 idx

    // 当第一轮搬运1或2个数据时，由于我们独特的计算结构，因此需要做特殊的处理
    // 我们的计算结构： d_al d_ac 将单独存储，然后读取a[i+1]，写入a[i]
    if (first_round_size == 1) {


//        DEBUG_PRINT("A-1", 0,0);
//        DEBUG_PRINT("A-1", 29,29);
//        DEBUG_PRINT("A-1", 30,30);
//        DEBUG_PRINT("A-1", 31,31);



        // 第一轮只读一个元素
        // 每个线程记录自己的 d_al d_ac
        // 在这种情况下，(ny,nz) 将会是线程的标识符
        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取32项
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 解析出i_tid的i和j
            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
            int i_i = i_thread_global_id / ny;
            int i_j = i_thread_global_id % ny;
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ];

            __syncthreads();
        }
        // sh_data[tid_in_block + 1] = a[i * ny * nz + j * nz + tid_in_block];

        // 每条线程开始在shared memory中独立处理自己的串行计算
        for (int i_i = 1; i_i < BLOCK_SIZE; ++i_i) {    // 31项，比正常的32项少一项
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 当前情况下第二轮只需写回 31 项，而不是 32 项
            // 由于我们不需要写回的是sh_data[0]，因此 tid_in_block 休息
//            if (tid_in_block == 0)
//                break;
            if (tid_in_block > 0) {
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
            }

            __syncthreads();
        }

        flag = -1;
        ++i_round; // 首先，在这种情况下最初轮次需要两轮

    } else if (first_round_size == 2) {


//        DEBUG_PRINT("A-2", 0,0);
//        DEBUG_PRINT("A-2", 29,29);
//        DEBUG_PRINT("A-2", 30,30);
//        DEBUG_PRINT("A-2", 31,31);



        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取32项
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 解析出i_tid的i和j
            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
            int i_i = i_thread_global_id / ny;
            int i_j = i_thread_global_id % ny;
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 2    ];

            __syncthreads();
        }

        // 每条线程开始在shared memory中独立处理自己的串行计算
        for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32项，正常
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 当前情况下第二轮写回 32 项
            // 解析出i_tid的i和j
            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
            int i_i = i_thread_global_id / ny;
            int i_j = i_thread_global_id % ny;
            a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取

            __syncthreads();
        }

        flag = -1;
        ++i_round; // 首先，在这种情况下最初轮次需要两轮

    } else if (first_round_size == 0) {

//        DEBUG_PRINT("A-0", 0,0);
//        DEBUG_PRINT("A-0", 29,29);
//        DEBUG_PRINT("A-0", 30,30);
//        DEBUG_PRINT("A-0", 31,31);

        first_round_size = 32;

        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取 first_round_size - 2 项
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 为block内每个line读取 first_round_size - 2 项
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < first_round_size - 2) {
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
            }

            __syncthreads();
        }                                                   //*** 这段代码没问题



        // 每条线程开始在shared memory中独立处理自己的串行计算
        // 只计算 first_round_size - 2 项
        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 项
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** 这段代码没问题




        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 为block内每个line写回 first_round_size - 2 项
            if (tid_in_block < first_round_size - 2) {      // 0~3
                // 当前情况下第一轮写回 first_round_size - 2 项
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取
            }

            __syncthreads();
        }

        flag = 1;
        i_round = 1;
        first_round_size = 0; // 需要设置回 0

    } else { // first_round_size 为 3 4 5 6...

//        DEBUG_PRINT("A-3", 0,0);
//        DEBUG_PRINT("A-3", 29,29);
//        DEBUG_PRINT("A-3", 30,30);
//        DEBUG_PRINT("A-3", 31,31);

        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取 first_round_size - 2 项
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 为block内每个line读取 first_round_size - 2 项
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < first_round_size - 2) {
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
            }

            __syncthreads();
        }                                                   //*** 这段代码没问题



        // 每条线程开始在shared memory中独立处理自己的串行计算
        // 只计算 first_round_size - 2 项
        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 项
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** 这段代码没问题




        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 为block内每个line写回 first_round_size - 2 项
            if (tid_in_block < first_round_size - 2) {      // 0~3
                // 当前情况下第一轮写回 first_round_size - 2 项
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取
            }

            __syncthreads();
        }

        flag = 1;
        i_round = 0;// 在这种情况下最初轮次需要一轮，因此不需要改变 i_round 值
    }                                                       //*** 这段代码没问题

    __syncthreads();



    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // 迭代开始
    // 然后进行 round_times 轮迭代
    // 此时 d_al  d_ac 已经准备就绪
    // 在每一轮中读取 32 项，写入 32 项
    // 1. 载入shared memory; 2. 按照方向计算; 3. 数据写回全局内存
    for ( ; i_round < round_times; ++i_round) {
        if (flag == 1) {        /// flag = 1 ==> 正向
            // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
            // 为每条line读取32项
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + tid_in_block + 1] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block];

                __syncthreads();
            }

            // 每条线程在shared memory中独立处理自己的串行计算
            for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32项，正常
                d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
                // 计算局部误差
                tmp = (d_al + d_ar) * 0.5;
                // 写回shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // 更新 d_al 和 d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // 并行地运行line_per_block将解写回内存
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
                // 当前情况下第二轮写回 32 项
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取

                __syncthreads();
            }                                               //*** 这段代码没问题

        } else {         /// flag == -1   ==> 逆向
            // 对称情形

            // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
            // 为每条line读取32项
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {  // line 的读取顺序不重要，不需要改变
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block]; // 我们让右侧a的idx保持不变

                __syncthreads();
            }

            // 每条线程开始在shared memory中独立处理自己的串行计算
            // 只计算 first_round_size - 2 项
            for (int i_i = BLOCK_SIZE; i_i > 0; --i_i) {    // 32 项，存进内存的作为基准
                d_ar = sh_data[tid_in_block * 33 + i_i - 1]; // 此时 flag 等于 1
                // 计算局部误差
                tmp = (d_al + d_ar) * 0.5;
                // 写回shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // 更新 d_al 和 d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // 并行地运行line_per_block将解写回内存
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回，line 的读取顺序不重要，不需要改变
                // 写回 32 项
                // 解析出i_tid的i和j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[    i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // 左侧的a的idx保持不变

                __syncthreads();
            }                                               //*** 这段代码没问题

        }
        flag *= -1;
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    /// 到此为止，共享内存作为计算的中间媒介的作用已经结束了
    //！而我们已经将所需的计算结果存储进了 a[] 全局内存中，因此共享内存现在可以退化成一个比之前大很多倍(32*33 vs 32)，但仅仅执行之前的功能的共享内存
    // 首先将各个线程的 s_eps 写入共享内存

    if (thread_global_id >= ij_total)  // 如果线程大于维数，则将误差设置为 0.
        sh_data[tid_in_block] = 0.;
    else
        sh_data[tid_in_block] = s_eps; // 共 32 项: 0~32

//    sh_data[tid_in_block] = s_eps;    // 由于在前面的串行计算过程中， >- ij_total 的线程也执行了计算，因此 s_eps 不可信
    __syncthreads();

    // 使用 二分 块内规约
    // 将局部误差存储到全局内存中去   // 原子操作性能较低
    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
    // (这里我们使用 лекция 上讲的 вариант 4)
    // 首先我们知道做 block_reduce 需要用到shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {     // blockDim = 32; // blockDim 必须是 2 的倍数！！！
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
    if (tid_in_block == 0) // 写回block_reduce结果
        d_eps[blockIdx.x] = sh_data[0];
}
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#else






// 二分规约
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// 为 block_reduce 准备的 shared memory
    __shared__ double sh_data[BLOCK_SIZE];
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double tmp;
    double s_eps = 0.;  // 初始化局部误差
    double d_al, d_ac, d_ar;  // left  center  right

    // 解码出 i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
        return;

    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // 每个线程串行处理一整条 z 方向数据线
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // 计算局部误差值
        tmp = (d_al + d_ar) * 0.5;  // 手动优化
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // 更新 d_al 和 d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // 首先将各个线程的 s_eps 写入共享内存
    sh_data[tid_in_block] = s_eps;
    __syncthreads();

    // 使用 二分 块内规约
    // 将局部误差存储到全局内存中去   // 原子操作性能较低
    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
    // (这里我们使用 лекция 上讲的 вариант 4)
    // 首先我们知道做 block_reduce 需要用到shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
    if (tid_in_block == 0) // 写回block_reduce结果
        d_eps[blockIdx.x] = sh_data[0];
}
#endif


//// 二分规约   clock debug 版
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    unsigned long long time_checker_1 = clock64();
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    __syncthreads();
//
//
//    unsigned long long time_checker_2 = clock64();
//
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    unsigned long long time_checker_3 = clock64();
//    __syncthreads();
//
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//
//
//    unsigned long long time_checker_4 = clock64();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//
//
//
//    unsigned long long time_checker_5 = clock64();
//
//
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//
//
//    if (blockIdx.x == 0 && threadIdx.x == 0)
//        printf("time_checker_1: %lld    time_checker_2: %lld    time_checker_3: %lld    time_checker_4: %lld    time_checker_5: %lld\n",
//            time_checker_2 - time_checker_1,
//            time_checker_3 - time_checker_2,
//            time_checker_4 - time_checker_3,
//            time_checker_5 - time_checker_4,
//            clock64() - time_checker_5);
//}


//// warp 内规约的辅助函数
//__inline__ __device__
//double warp_reduce_max(double val) {
//    // 使用 warp shuffle 操作做规约
//    for (int offset = 16; offset > 0; offset /= 2)
//        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
//    return val;
//}
//// warp 内规约
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
////        tmp = (d_al + d_ar) / 2.0;   // 以防古老编译器不进行优化
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // 使用 warp 内规约
//    double warp_max = warp_reduce_max(s_eps);
//    if (threadIdx.x == 0)
//        d_eps[blockIdx.x] = warp_max;
//
////    // 首先将各个线程的 s_eps 写入共享内存
////    sh_data[tid_in_block] = s_eps;
////    __syncthreads();
////
////    // 使用 二分 块内规约
////    // 将局部误差存储到全局内存中去   // 原子操作性能较低
////    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
////    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
////    // (这里我们使用 лекция 上讲的 вариант 4)
////    // 首先我们知道做 block_reduce 需要用到shared memory
////    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
////        if (tid_in_block < s)
////            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
////        __syncthreads();
////    }
////    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
////    if (tid_in_block == 0) // 写回block_reduce结果
////        d_eps[blockIdx.x] = sh_data[0];
//}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // 第 i 个块
//    int tid = threadIdx.x; // 第 idx 个进程
//    int idx_thread_global = tid + i * blockDim.x; // 相对于 A 数组（即 block_max）的地址位置
//    // 将块内所有值存入shared mem中
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // 让每个线程不超出全局线程
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // 对于最后一个 block 的超出部分我们填充进一个极小值
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // 二分法块内归约
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // 很重要，用MAX而不是max
//        }
//        __syncthreads();
//    }
//    // 第 0 个线程将当前最大值写入全局内存
//    if (tid == 0) {
//        B[i] = sdata[0];
//    }
//}


void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared Mem per Block: %d bytes\n", prop.sharedMemPerBlock);  // 每个 block 限制
    printf("Shared Mem per SM   : %d bytes\n", prop.sharedMemPerMultiprocessor); // 每个 SM 限制
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

    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // 每个块内的线程数量
    int total_lines_x = ny * nz;    // x 方向的总线数
    int total_lines_y = nx * nz;    // y 方向的总线数
    int total_lines_z = nx * ny;    // z 方向的总线数
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // 存放每个block内最大值

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // 为 CUB 预分配内存
    // d_eps: 输入数据在设备上的指针
    int num_items = blocks_z;   // 等于block的总数量

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // 初始化误差值为0  (使用block_reduce + global_reduce时不需要初始化误差)
//        double h_eps = 0.0;
//        // ***** 耗时很少 *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// 二分规约需要 shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double) >>>(d_A, d_eps); // 每个块内 32 个线程  每个block 32个double
//        //// warp 内规约不需要 shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程
        //// 并行加载数据进 shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * (threads_per_block + 1) * sizeof(double) >>>(d_A, d_eps); // 每个块内 32 个线程  每个线程 33个double
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程  每个线程 33个double


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // 此时d_eps中存储每个block内的最大值
        // global_reduce 2  thrust  // 灰常方便！！！
        // 创建一个 Thrust device_ptr 包装器，指向 d_eps（device 上的 block 局部最大误差数组）
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
        // 使用 thrust::reduce 找出最大值（初始值必须是 -∞）
        eps = thrust::reduce(dev_ptr, dev_ptr + num_items, 0., thrust::maximum<double>());

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
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

#if RUNNING_MODE == 11
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

// 用于块内扫描的自定义操作符，注意要求该操作具有结合性
struct MyOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define IT_MAX 10
//#define IT_MAX 500
#define L 900
//#define L 384

#define nx L
#define ny L
#define nz L

#define BLOCK_SIZE 32       // 默认 32, 目前 64 表现最佳

/// CUDA error checking macro
#define CHECK_CUDA_ERROR(call)                                                 \
{                                                                              \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(err));                                      \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}


// 在 CPU 上进行初始化
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

//// 用 CUB 的 BlockScan 替代串行更新的核函数版本      (采用串行计算) (但是不将数据存储在 共享内存中) (每个block中1个线程)
//  (计算结果正确) (速度比较慢)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // 每个块处理一条 x 方向的数据线，假设网格的 blockIdx.y, blockIdx.z 分别对应 j, k 坐标
    // 在当前情况下, 每个块只需要一个线程即可
    int j = blockIdx.y;
    int k = blockIdx.z;

    // 串行更新
    if (threadIdx.x == 0) { // 只让一个线程执行更新，避免竞争
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// 用 CUB 的 BlockScan 替代串行更新的核函数版本   (采用串行计算) (但是不将数据存储在 共享内存中)(每个block中32个线程)
//  (计算结果正确) (目前最快)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // 将本轮计算结果作为下轮计算的左变量

    // 解码出 j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // 有必要添加这句
        return;

    int j = thread_global_id / nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 j k 顺序不会影响结果, 但非常影响程序速度！！！

    d_al = a[(1 - 1) * ny * nz + j * nz + k];

    // 每个线程串行处理一整条 x 方向数据线
    for (int i = 1; i < nx - 1; ++i) {
        d_al = (d_al + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

// 同理, 仿照 compute_i_direction_kernel_32_thread 写出 compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double d_al; // 将本轮计算结果作为下轮计算的左变量

    // 解码出 i, k
    int ik_total = nx * nz;
    if (thread_global_id >= ik_total)  // 如果有空线程, 直接返回即可
        return;

    int i = thread_global_id / nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！
    int k = thread_global_id % nz;  // 不正确的 i k 顺序不会影响结果, 但非常影响程序速度！！！

    d_al = a[i * ny * nz + (1 - 1) * nz + k];

    // 每个线程串行处理一整条 y 方向数据线
    for (int j = 1; j < ny - 1; ++j) {
        d_al = (d_al + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

//// 二分规约 -- 无 tmp  (慢于普通二分规约)
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    register double s_eps = 0.;  // 初始化局部误差
//    register double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    if (thread_global_id >= nx * ny)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
//        a[i * ny * nz + j * nz + k] = (d_al + d_ar) * 0.5;
//        s_eps = MAX(s_eps, fabs(d_ac - (d_al + d_ar) * 0.5));
//
//        // 更新 d_al 和 d_ac
//        d_al = (d_al + d_ar) * 0.5;
//        d_ac = d_ar;
//    }
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//}


// 自定义打印 debug
#define DEBUG_PRINT(info, bIdx, tIdx)                   \
{                                           \
    if (blockIdx.x == (bIdx) && threadIdx.x == (tIdx))\
        printf("blockIdx: %d, threadIdx: %d: %s\n", (bIdx), (tIdx), (info));               \
}                                           \

//#define DEBUG_PRINT_GLOBAL(info)                   \
//{                                           \
//    \
//        printf("blockIdx: %d, threadIdx: %d: %s\n", blockIdx.x, threadIdx.x, (info));               \
//}   \

#if true
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
// 在二分规约方案的基础上，使用并行加载内存的方式      这种情况下，每个 block 的共享内存大小：33
// 优化前：1.067秒
// 最大幅度优化性能，代码可读性为 0，仅适合外星人和ChatGPT阅读
// 运行时间无变化 ... 仅仅大幅度降低了程序易读性 ...    1.067秒  1.068秒
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// 为 block_reduce 准备的 shared memory
    __shared__ double sh_data[(BLOCK_SIZE + 1) * BLOCK_SIZE];  // 33 * 32 个共享内存单位
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int block_Idx_mul_Dim = blockIdx.x * blockDim.x; // 提前计算出 blockIdx.x * blockDim.x
    int thread_global_id = block_Idx_mul_Dim + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double tmp;
    double s_eps = 0.;  // 初始化局部误差
    double d_al, d_ac, d_ar;  // left  center  right

    // 解码出 i, j
    int ij_total = nx * ny;


    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！



    // 获取当前block中的line数量
    int line_per_block;
    if (blockIdx.x == ij_total / 32) {
        line_per_block = ij_total % 32; // 最后一个block中line的数量为 total % 32
    } else {
        line_per_block = 32; //  其他block中line数量为 32
    }

    // 同一个block中全部线程协同将block内所有的line中的32项数据导入进shared memory中
    // 不过，对于第一或者第二项，需要单独考虑
    // 首先，我们计算出一条线上的总长度nx除以32的余数：这是第一轮对于一条line而言搬运的数据量，之后的数据将能被对齐
    int first_round_size = nx % 32;
    int round_times = nx / 32;  // 总 i_round 数量
    int flag = 1;

//    DEBUG_PRINT("A-0", 0,0);
//    DEBUG_PRINT("A-0", 29,29);
//    DEBUG_PRINT("A-0", 30,30);
//    DEBUG_PRINT("A-0", 31,31);

    __syncthreads();

    int i_round = 0;    // 当前的完整轮 idx

    // 当第一轮搬运1或2个数据时，由于我们独特的计算结构，因此需要做特殊的处理
    // 我们的计算结构： d_al d_ac 将单独存储，然后读取a[i+1]，写入a[i]
    if (first_round_size == 1) {

        // 第一轮只读一个元素
        // 每个线程记录自己的 d_al d_ac
        // 在这种情况下，(ny,nz) 将会是线程的标识符
        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取32项
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 解析出i_tid的i和j
//            int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ];

//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1    ];
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ];

            __syncthreads();
        }

        // 每条线程开始在shared memory中独立处理自己的串行计算
        for (int i_i = 1; i_i < BLOCK_SIZE; ++i_i) {    // 31项，比正常的32项少一项
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 当前情况下第二轮只需写回 31 项，而不是 32 项
            // 由于我们不需要写回的是sh_data[0]，因此 tid_in_block 休息
//            if (tid_in_block == 0)
//                break;
            if (tid_in_block > 0) {
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
                a[(block_Idx_mul_Dim + i_tid) * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
            }

            __syncthreads();
        }

        flag = -1;
        ++i_round; // 首先，在这种情况下最初轮次需要两轮

    } else if (first_round_size == 2) {



        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取32项
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 解析出i_tid的i和j
//            int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;

//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 2    ];
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 2    ];

            __syncthreads();
        }

        // 每条线程开始在shared memory中独立处理自己的串行计算
        for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32项，正常
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 当前情况下第二轮写回 32 项
            // 解析出i_tid的i和j
//            int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;

//            a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
            a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取

            __syncthreads();
        }

        flag = -1;
        ++i_round; // 首先，在这种情况下最初轮次需要两轮

    } else if (first_round_size == 0) {



//        first_round_size = 32;    // 已经将后面代码中的 first_round_size 替换为了 32，并对 first_round_size - 2 改为了 30

        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取 first_round_size - 2 项
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 为block内每个line读取 first_round_size - 2 项
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < 30) {
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;

//                sh_data[i_tid * 33 + (30 - tid_in_block - 1)] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 2];
                sh_data[i_tid * 33 + (30 - tid_in_block - 1)] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 2    ];

            }

            __syncthreads();
        }                                                   //*** 这段代码没问题



        // 每条线程开始在shared memory中独立处理自己的串行计算
        // 只计算 first_round_size - 2 项
        for (int i_i = 30; i_i > 0; --i_i) {    // first_round_size - 2 项
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** 这段代码没问题




        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 为block内每个line写回 first_round_size - 2 项
            if (tid_in_block < 30) {      // 0~3
                // 当前情况下第一轮写回 first_round_size - 2 项
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (30 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取
                a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + (30 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取

            }

            __syncthreads();
        }

        flag = 1;
        i_round = 1;
//        first_round_size = 0; // 需要设置回 0

    } else { // first_round_size 为 3 4 5 6...

//        DEBUG_PRINT("A-3", 0,0);
//        DEBUG_PRINT("A-3", 29,29);
//        DEBUG_PRINT("A-3", 30,30);
//        DEBUG_PRINT("A-3", 31,31);

        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取 first_round_size - 2 项
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 为block内每个line读取 first_round_size - 2 项
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < first_round_size - 2) {
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;
//
//                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 2];
                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 2    ];

            }

            __syncthreads();
        }                                                   //*** 这段代码没问题



        // 每条线程开始在shared memory中独立处理自己的串行计算
        // 只计算 first_round_size - 2 项
        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 项
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // 计算局部误差
            tmp = (d_al + d_ar) * 0.5;
            // 写回shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // 更新 d_al 和 d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** 这段代码没问题




        // 并行地运行line_per_block将解写回内存
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
            // 为block内每个line写回 first_round_size - 2 项
            if (tid_in_block < first_round_size - 2) {      // 0~3
                // 当前情况下第一轮写回 first_round_size - 2 项
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取
                a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取

            }

            __syncthreads();
        }

        flag = 1;
        i_round = 0;// 在这种情况下最初轮次需要一轮，因此不需要改变 i_round 值
    }                                                       //*** 这段代码没问题

    __syncthreads();



    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // 迭代开始
    // 然后进行 round_times 轮迭代
    // 此时 d_al  d_ac 已经准备就绪
    // 在每一轮中读取 32 项，写入 32 项
    // 1. 载入shared memory; 2. 按照方向计算; 3. 数据写回全局内存
    for ( ; i_round < round_times; ++i_round) {
        if (flag == 1) {        /// flag = 1 ==> 正向
            // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
            // 为每条line读取32项
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;
//
//                sh_data[i_tid * 33 + tid_in_block + 1] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block];
                sh_data[i_tid * 33 + tid_in_block + 1] = a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block    ];

                __syncthreads();
            }

            // 每条线程在shared memory中独立处理自己的串行计算
            for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32项，正常
                d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
                // 计算局部误差
                tmp = (d_al + d_ar) * 0.5;
                // 写回shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // 更新 d_al 和 d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // 并行地运行line_per_block将解写回内存
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
                // 当前情况下第二轮写回 32 项
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block - 1] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
                a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取

                __syncthreads();
            }                                               //*** 这段代码没问题


            flag = -1;
        } else {         /// flag == -1   ==> 逆向
            // 对称情形

            // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
            // 为每条line读取32项
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {  // line 的读取顺序不重要，不需要改变
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block]; // 我们让右侧a的idx保持不变
                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block    ]; // 我们让右侧a的idx保持不变

                __syncthreads();
            }

            // 每条线程开始在shared memory中独立处理自己的串行计算
            // 只计算 first_round_size - 2 项
            for (int i_i = BLOCK_SIZE; i_i > 0; --i_i) {    // 32 项，存进内存的作为基准
                d_ar = sh_data[tid_in_block * 33 + i_i - 1]; // 此时 flag 等于 1
                // 计算局部误差
                tmp = (d_al + d_ar) * 0.5;
                // 写回shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // 更新 d_al 和 d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // 并行地运行line_per_block将解写回内存
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回，line 的读取顺序不重要，不需要改变
                // 写回 32 项
                // 解析出i_tid的i和j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // 左侧的a的idx保持不变
                a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // 左侧的a的idx保持不变

                __syncthreads();
            }                                               //*** 这段代码没问题


            flag = 1;
        }
//        flag *= -1;   // 显然，直接赋值比这更快
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    /// 到此为止，共享内存作为计算的中间媒介的作用已经结束了
    //！而我们已经将所需的计算结果存储进了 a[] 全局内存中，因此共享内存现在可以退化成一个比之前大很多倍(32*33 vs 32)，但仅仅执行之前的功能的共享内存
    // 首先将各个线程的 s_eps 写入共享内存

    if (thread_global_id >= ij_total)  // 如果线程大于维数，则将误差设置为 0.
        sh_data[tid_in_block] = 0.;
    else
        sh_data[tid_in_block] = s_eps; // 共 32 项: 0~32

//    sh_data[tid_in_block] = s_eps;    // 由于在前面的串行计算过程中， >- ij_total 的线程也执行了计算，因此 s_eps 不可信
    __syncthreads();

    // 使用 二分 块内规约
    // 将局部误差存储到全局内存中去   // 原子操作性能较低
    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
    // (这里我们使用 лекция 上讲的 вариант 4)
    // 首先我们知道做 block_reduce 需要用到shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {     // blockDim = 32; // blockDim 必须是 2 的倍数！！！
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
    if (tid_in_block == 0) // 写回block_reduce结果
        d_eps[blockIdx.x] = sh_data[0];
}







// 副本 并行加载内存算法
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[(BLOCK_SIZE + 1) * BLOCK_SIZE];  // 33 * 32 个共享内存单位
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//
//    // 不存在空线程, 空线程同样将充当worker
////    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
////        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//
//    // 获取当前block中的line数量
//    int line_per_block;
//    if (blockIdx.x == ij_total / 32) {
//        line_per_block = ij_total % 32; // 最后一个block中line的数量为 total % 32
//    } else {
//        line_per_block = 32; //  其他block中line数量为 32
//    }
//
//    // 同一个block中全部线程协同将block内所有的line中的32项数据导入进shared memory中
//    // 不过，对于第一或者第二项，需要单独考虑
//    // 首先，我们计算出一条线上的总长度nx除以32的余数：这是第一轮对于一条line而言搬运的数据量，之后的数据将能被对齐
//    int first_round_size = nx % 32;
//    int round_times = nx / 32;  // 总 i_round 数量
//    int flag = 1;
//
//
//
//    __syncthreads();
//
//    int i_round = 0;    // 当前的完整轮 idx
//
//    // 当第一轮搬运1或2个数据时，由于我们独特的计算结构，因此需要做特殊的处理
//    // 我们的计算结构： d_al d_ac 将单独存储，然后读取a[i+1]，写入a[i]
//    if (first_round_size == 1) {
//
//
////        DEBUG_PRINT("A-1", 0,0);
////        DEBUG_PRINT("A-1", 29,29);
////        DEBUG_PRINT("A-1", 30,30);
////        DEBUG_PRINT("A-1", 31,31);
//
//
//
//        // 第一轮只读一个元素
//        // 每个线程记录自己的 d_al d_ac
//        // 在这种情况下，(ny,nz) 将会是线程的标识符
//        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
//        // 为每条line读取32项
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // 解析出i_tid的i和j
//            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ];
//
//            __syncthreads();
//        }
//        // sh_data[tid_in_block + 1] = a[i * ny * nz + j * nz + tid_in_block];
//
//        // 每条线程开始在shared memory中独立处理自己的串行计算
//        for (int i_i = 1; i_i < BLOCK_SIZE; ++i_i) {    // 31项，比正常的32项少一项
//            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
//            // 计算局部误差
//            tmp = (d_al + d_ar) * 0.5;
//            // 写回shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // 更新 d_al 和 d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }
//
//        // 并行地运行line_per_block将解写回内存
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
//            // 当前情况下第二轮只需写回 31 项，而不是 32 项
//            // 由于我们不需要写回的是sh_data[0]，因此 tid_in_block 休息
////            if (tid_in_block == 0)
////                break;
//            if (tid_in_block > 0) {
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
//            }
//
//            __syncthreads();
//        }
//
//        flag = -1;
//        ++i_round; // 首先，在这种情况下最初轮次需要两轮
//
//    } else if (first_round_size == 2) {
//
//
////        DEBUG_PRINT("A-2", 0,0);
////        DEBUG_PRINT("A-2", 29,29);
////        DEBUG_PRINT("A-2", 30,30);
////        DEBUG_PRINT("A-2", 31,31);
//
//
//
//        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
//        // 为每条line读取32项
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // 解析出i_tid的i和j
//            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 2    ];
//
//            __syncthreads();
//        }
//
//        // 每条线程开始在shared memory中独立处理自己的串行计算
//        for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32项，正常
//            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
//            // 计算局部误差
//            tmp = (d_al + d_ar) * 0.5;
//            // 写回shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // 更新 d_al 和 d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }
//
//        // 并行地运行line_per_block将解写回内存
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
//            // 当前情况下第二轮写回 32 项
//            // 解析出i_tid的i和j
//            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
//
//            __syncthreads();
//        }
//
//        flag = -1;
//        ++i_round; // 首先，在这种情况下最初轮次需要两轮
//
//    } else if (first_round_size == 0) {
//
////        DEBUG_PRINT("A-0", 0,0);
////        DEBUG_PRINT("A-0", 29,29);
////        DEBUG_PRINT("A-0", 30,30);
////        DEBUG_PRINT("A-0", 31,31);
//
//        first_round_size = 32;
//
//        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // first_round_size - 2
//
//        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
//        // 为每条line读取 first_round_size - 2 项
////        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // 为block内每个line读取 first_round_size - 2 项
////            if (tid_in_block >= first_round_size - 2)
////                break;
//            if (tid_in_block < first_round_size - 2) {
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
//            }
//
//            __syncthreads();
//        }                                                   //*** 这段代码没问题
//
//
//
//        // 每条线程开始在shared memory中独立处理自己的串行计算
//        // 只计算 first_round_size - 2 项
//        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 项
//            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
//            // 计算局部误差
//            tmp = (d_al + d_ar) * 0.5;
//            // 写回shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // 更新 d_al 和 d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }                                                   //*** 这段代码没问题
//
//
//
//
//        // 并行地运行line_per_block将解写回内存
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
//            // 为block内每个line写回 first_round_size - 2 项
//            if (tid_in_block < first_round_size - 2) {      // 0~3
//                // 当前情况下第一轮写回 first_round_size - 2 项
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取
//            }
//
//            __syncthreads();
//        }
//
//        flag = 1;
//        i_round = 1;
//        first_round_size = 0; // 需要设置回 0
//
//    } else { // first_round_size 为 3 4 5 6...
//
////        DEBUG_PRINT("A-3", 0,0);
////        DEBUG_PRINT("A-3", 29,29);
////        DEBUG_PRINT("A-3", 30,30);
////        DEBUG_PRINT("A-3", 31,31);
//
//        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // first_round_size - 2
//
//        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
//        // 为每条line读取 first_round_size - 2 项
////        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // 为block内每个line读取 first_round_size - 2 项
////            if (tid_in_block >= first_round_size - 2)
////                break;
//            if (tid_in_block < first_round_size - 2) {
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
//            }
//
//            __syncthreads();
//        }                                                   //*** 这段代码没问题
//
//
//
//        // 每条线程开始在shared memory中独立处理自己的串行计算
//        // 只计算 first_round_size - 2 项
//        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 项
//            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
//            // 计算局部误差
//            tmp = (d_al + d_ar) * 0.5;
//            // 写回shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // 更新 d_al 和 d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }                                                   //*** 这段代码没问题
//
//
//
//
//        // 并行地运行line_per_block将解写回内存
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
//            // 为block内每个line写回 first_round_size - 2 项
//            if (tid_in_block < first_round_size - 2) {      // 0~3
//                // 当前情况下第一轮写回 first_round_size - 2 项
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // 第0项已经由于线程返回而不会被读取
//            }
//
//            __syncthreads();
//        }
//
//        flag = 1;
//        i_round = 0;// 在这种情况下最初轮次需要一轮，因此不需要改变 i_round 值
//    }                                                       //*** 这段代码没问题
//
//    __syncthreads();
//
//
//
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    // 迭代开始
//    // 然后进行 round_times 轮迭代
//    // 此时 d_al  d_ac 已经准备就绪
//    // 在每一轮中读取 32 项，写入 32 项
//    // 1. 载入shared memory; 2. 按照方向计算; 3. 数据写回全局内存
//    for ( ; i_round < round_times; ++i_round) {
//        if (flag == 1) {        /// flag = 1 ==> 正向
//            // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
//            // 为每条line读取32项
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + tid_in_block + 1] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block];
//
//                __syncthreads();
//            }
//
//            // 每条线程在shared memory中独立处理自己的串行计算
//            for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32项，正常
//                d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // 此时 flag 等于 1
//                // 计算局部误差
//                tmp = (d_al + d_ar) * 0.5;
//                // 写回shared memory
//                sh_data[tid_in_block * 33 + i_i] = tmp;
//                s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//                // 更新 d_al 和 d_ac
//                d_al = tmp;
//                d_ac = d_ar;
//            }
//
//            // 并行地运行line_per_block将解写回内存
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回
//                // 当前情况下第二轮写回 32 项
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1] = sh_data[i_tid * 33 + tid_in_block]; // 第0项已经由于线程返回而不会被读取
//
//                __syncthreads();
//            }                                               //*** 这段代码没问题
//
//        } else {         /// flag == -1   ==> 逆向
//            // 对称情形
//
//            // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
//            // 为每条line读取32项
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {  // line 的读取顺序不重要，不需要改变
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block]; // 我们让右侧a的idx保持不变
//
//                __syncthreads();
//            }
//
//            // 每条线程开始在shared memory中独立处理自己的串行计算
//            // 只计算 first_round_size - 2 项
//            for (int i_i = BLOCK_SIZE; i_i > 0; --i_i) {    // 32 项，存进内存的作为基准
//                d_ar = sh_data[tid_in_block * 33 + i_i - 1]; // 此时 flag 等于 1
//                // 计算局部误差
//                tmp = (d_al + d_ar) * 0.5;
//                // 写回shared memory
//                sh_data[tid_in_block * 33 + i_i] = tmp;
//                s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//                // 更新 d_al 和 d_ac
//                d_al = tmp;
//                d_ac = d_ar;
//            }
//
//            // 并行地运行line_per_block将解写回内存
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // 对block内所有line执行写回，line 的读取顺序不重要，不需要改变
//                // 写回 32 项
//                // 解析出i_tid的i和j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[    i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // 左侧的a的idx保持不变
//
//                __syncthreads();
//            }                                               //*** 这段代码没问题
//
//        }
//        flag *= -1;
//    }
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//    /// 到此为止，共享内存作为计算的中间媒介的作用已经结束了
//    //！而我们已经将所需的计算结果存储进了 a[] 全局内存中，因此共享内存现在可以退化成一个比之前大很多倍(32*33 vs 32)，但仅仅执行之前的功能的共享内存
//    // 首先将各个线程的 s_eps 写入共享内存
//
//    if (thread_global_id >= ij_total)  // 如果线程大于维数，则将误差设置为 0.
//        sh_data[tid_in_block] = 0.;
//    else
//        sh_data[tid_in_block] = s_eps; // 共 32 项: 0~32
//
////    sh_data[tid_in_block] = s_eps;    // 由于在前面的串行计算过程中， >- ij_total 的线程也执行了计算，因此 s_eps 不可信
//    __syncthreads();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {     // blockDim = 32; // blockDim 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//}



//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#else






// 二分规约
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// 为 block_reduce 准备的 shared memory
    __shared__ double sh_data[BLOCK_SIZE];
    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向

    double tmp;
    double s_eps = 0.;  // 初始化局部误差
    double d_al, d_ac, d_ar;  // left  center  right

    // 解码出 i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
        return;

    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // 每个线程串行处理一整条 z 方向数据线
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // 计算局部误差值
        tmp = (d_al + d_ar) * 0.5;  // 手动优化
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // 更新 d_al 和 d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // 首先将各个线程的 s_eps 写入共享内存
    sh_data[tid_in_block] = s_eps;
    __syncthreads();

    // 使用 二分 块内规约
    // 将局部误差存储到全局内存中去   // 原子操作性能较低
    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
    // (这里我们使用 лекция 上讲的 вариант 4)
    // 首先我们知道做 block_reduce 需要用到shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
    if (tid_in_block == 0) // 写回block_reduce结果
        d_eps[blockIdx.x] = sh_data[0];
}
#endif


//// 二分规约   clock debug 版
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// 为 block_reduce 准备的 shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    unsigned long long time_checker_1 = clock64();
//
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    __syncthreads();
//
//
//    unsigned long long time_checker_2 = clock64();
//
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    unsigned long long time_checker_3 = clock64();
//    __syncthreads();
//
//
//    // 首先将各个线程的 s_eps 写入共享内存
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//
//
//    unsigned long long time_checker_4 = clock64();
//
//    // 使用 二分 块内规约
//    // 将局部误差存储到全局内存中去   // 原子操作性能较低
//    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
//    // (这里我们使用 лекция 上讲的 вариант 4)
//    // 首先我们知道做 block_reduce 需要用到shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//
//
//
//    unsigned long long time_checker_5 = clock64();
//
//
//    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
//    if (tid_in_block == 0) // 写回block_reduce结果
//        d_eps[blockIdx.x] = sh_data[0];
//
//
//    if (blockIdx.x == 0 && threadIdx.x == 0)
//        printf("time_checker_1: %lld    time_checker_2: %lld    time_checker_3: %lld    time_checker_4: %lld    time_checker_5: %lld\n",
//            time_checker_2 - time_checker_1,
//            time_checker_3 - time_checker_2,
//            time_checker_4 - time_checker_3,
//            time_checker_5 - time_checker_4,
//            clock64() - time_checker_5);
//}


//// warp 内规约的辅助函数
//__inline__ __device__
//double warp_reduce_max(double val) {
//    // 使用 warp shuffle 操作做规约
//    for (int offset = 16; offset > 0; offset /= 2)
//        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
//    return val;
//}
//// warp 内规约
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    int tid_in_block = threadIdx.x;     // 由于我们使用 1 维 block, 因此只有 x 方向
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // 同样, 使用 1 维 grid, 只有 x 方向
//
//    double tmp;
//    double s_eps = 0.;  // 初始化局部误差
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // 解码出 i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // 如果有空线程, 直接返回即可 (在我的电脑上去掉这个代码, 程序运行时间没有任何区别)
//        return;
//
//    // 因为上面 ij_total = nx * ny, 因此应该是 /ny; %ny
//    int i = thread_global_id / ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//    int j = thread_global_id % ny;  // 不正确的 i j 顺序不会影响结果, 但非常影响程序速度！！！
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // 每个线程串行处理一整条 z 方向数据线
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // 计算局部误差值
////        tmp = (d_al + d_ar) / 2.0;   // 以防古老编译器不进行优化
//        tmp = (d_al + d_ar) * 0.5;  // 手动优化
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // 更新 d_al 和 d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // 使用 warp 内规约
//    double warp_max = warp_reduce_max(s_eps);
//    if (threadIdx.x == 0)
//        d_eps[blockIdx.x] = warp_max;
//
////    // 首先将各个线程的 s_eps 写入共享内存
////    sh_data[tid_in_block] = s_eps;
////    __syncthreads();
////
////    // 使用 二分 块内规约
////    // 将局部误差存储到全局内存中去   // 原子操作性能较低
////    // 太慢, 我们将其改为 1. block 内规约(block_reduce); 2. 全局内存内规约(global_reduce)
////    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // 一种特殊技巧, 但仅限于非负浮点数
////    // (这里我们使用 лекция 上讲的 вариант 4)
////    // 首先我们知道做 block_reduce 需要用到shared memory
////    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime 必须是 2 的倍数！！！
////        if (tid_in_block < s)
////            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
////        __syncthreads();
////    }
////    // 不仅仅是需要共享内存, 而且还需要block总数数量的额外全局内存, 用来存放每一个block内的局部规约值(global_reduce)
////    if (tid_in_block == 0) // 写回block_reduce结果
////        d_eps[blockIdx.x] = sh_data[0];
//}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // 第 i 个块
//    int tid = threadIdx.x; // 第 idx 个进程
//    int idx_thread_global = tid + i * blockDim.x; // 相对于 A 数组（即 block_max）的地址位置
//    // 将块内所有值存入shared mem中
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // 让每个线程不超出全局线程
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // 对于最后一个 block 的超出部分我们填充进一个极小值
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // 二分法块内归约
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // 很重要，用MAX而不是max
//        }
//        __syncthreads();
//    }
//    // 第 0 个线程将当前最大值写入全局内存
//    if (tid == 0) {
//        B[i] = sdata[0];
//    }
//}


void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared Mem per Block: %d bytes\n", prop.sharedMemPerBlock);  // 每个 block 限制
    printf("Shared Mem per SM   : %d bytes\n", prop.sharedMemPerMultiprocessor); // 每个 SM 限制
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

    // 初始化参数
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // 分配 host 内存   三维数组通过拉成一条线来存储
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化数据
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // 每个块内的线程数量
    int total_lines_x = ny * nz;    // x 方向的总线数
    int total_lines_y = nx * nz;    // y 方向的总线数
    int total_lines_z = nx * ny;    // z 方向的总线数
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // 计算此时所需的块数
    //**********************************************************
    //**********************************************************

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // 存放每个block内最大值

    // 优化点1: 仅在开始时传输数据到设备
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // 为 CUB 预分配内存
    // d_eps: 输入数据在设备上的指针
    int num_items = blocks_z;   // 等于block的总数量

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // 初始化误差值为0  (使用block_reduce + global_reduce时不需要初始化误差)
//        double h_eps = 0.0;
//        // ***** 耗时很少 *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// 二分规约需要 shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double) >>>(d_A, d_eps); // 每个块内 32 个线程  每个block 32个double
//        //// warp 内规约不需要 shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程
        //// 并行加载数据进 shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * (threads_per_block + 1) * sizeof(double) >>>(d_A, d_eps); // 每个块内 32 个线程  每个线程 33个double
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程  每个线程 33个double

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // 此时d_eps中存储每个block内的最大值
        // global_reduce 2  thrust  // 灰常方便！！！
        // 创建一个 Thrust device_ptr 包装器，指向 d_eps（device 上的 block 局部最大误差数组）
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
        // 使用 thrust::reduce 找出最大值（初始值必须是 -∞）
        eps = thrust::reduce(dev_ptr, dev_ptr + num_items, 0., thrust::maximum<double>());

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // 检查收敛条件
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

    // 优化点4: 只在计算完成后拷贝结果回主机
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // 释放内存
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
