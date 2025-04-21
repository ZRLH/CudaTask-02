#define RUNNING_MODE -1     // ͣ��
//#define RUNNING_MODE 0      // ��������, ���۾�
//#define RUNNING_MODE 1
//#define RUNNING_MODE 2      // ���� �˺��� �ڵ� cub scan ����
//#define RUNNING_MODE 3      // ���ڲ��� scan Ч��, ʹ�������� scan ����
//#define RUNNING_MODE 4      // cub �汾
//#define RUNNING_MODE 5 //2025/4/13�����ļ��㷽ʽ, ÿ��block��32���߳�, ÿ���̵߳��̼߳���x����һ����
//#define RUNNING_MODE 6      // ȫ���ϣ�� (�������� x y �Ѿ�ʵ��) (x y z ����ʵ��)
//#define RUNNING_MODE 7      // �Ż��ٶȰ�, ׼������ҵ //Ŀǰ�����ȷ, ����xy����û���Ż���ֻ����ȫ���ڴ��д����(Ŀǰ��Ҫ��������)
                        // warp �� 900*10 => 1.51s
//#define RUNNING_MODE 8      // ԭ�Ӳ�����, �����Ա� 900*10 => 1.57s
//#define RUNNING_MODE 9      // thrust ��� CUB
//#define RUNNING_MODE 10     // �����Ż� k ����        ���Բ��� transition     // ԭ�ͻ���ʵ�֣�����  ������  Ultra����    1.04��
//#define RUNNING_MODE 11     // �� Ultra ������д��뼶�Ż�  // �׶��Խ��ۣ�i, j��������Ѿ��������Ž� (���ܾ������Ž�)      �׶���Ϊ0    �������㷨�����˼򻯣�����ʵ�����ٶȲ�û��һ˿�ӿ�
                                // Ŀǰ��ʣ�� a b û���Ż���������ȡ
// ���� RUNNING MODE ****************

// �ʼǣ������� /2. ��Ϊ * 0.5 �����

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

// �� CPU �Ͻ��г�ʼ��
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

// i����˺���
__global__ void compute_i_direction_kernel(double *a)
{
    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // ÿ���߳̿鴦��һ��(j,k)�ϵ�����i
    // ע�⣺��Ҫ����i��˳����£���Ϊ������������

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];

    // �����������ݵ������ڴ�
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }
    // ���ر߽�ֵ
    if (threadIdx.x == 0) {
        s_a[0] = a[0 * ny * nz + j * nz + k];
        s_a[nx - 1] = a[(nx - 1) * ny * nz + j * nz + k];
    }

    __syncthreads();

    // ���и��£�ȷ����ȷ��������ϵ
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[i] = (s_a[i-1] + s_a[i+1]) / 2.0;
        }
        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    // д��ȫ���ڴ�
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[i];
        }
    }
}

// ����ڶ������� (j ����)
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

// j����˺���
__global__ void compute_j_direction_kernel(double *a)
{
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // ÿ���߳̿鴦��һ��(i,k)�ϵ�����j
    // ע�⣺��Ҫ����j��˳����£���Ϊ������������

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];

    // �����������ݵ������ڴ�
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }
    // ���ر߽�ֵ
    if (threadIdx.x == 0) {
        s_a[0] = a[i * ny * nz + 0 * nz + k];
        s_a[ny - 1] = a[i * ny * nz + (ny - 1) * nz + k];
    }

    __syncthreads();

    // ���и��£�ȷ����ȷ��������ϵ
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[j] = (s_a[j-1] + s_a[j+1]) / 2.0;
        }
        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    // д��ȫ���ڴ�
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// ������������� (k ����) ������ eps - CPU�汾
void compute_k_direction_and_eps(double *a, double *eps_ptr)
{
    int i, j, k;
    double eps = 0.0;

    // ���������ֵʱ��ҪС�Ĵ���
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

        // �ٽ������ڰ�ȫ�ظ���ȫ�� eps ֵ
#pragma omp critical
        {
            eps = Max(eps, local_eps);
        }
    }

    *eps_ptr = eps;
}

// k����˺���������eps����
__global__ void compute_k_direction_kernel(double *a, double *d_eps)
{
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // ÿ���߳̿鴦��һ��(i,j)�ϵ�����k��

    // ʹ�ù����ڴ汣�浱ǰ"��"�;ֲ�������
    extern __shared__ double shared[];
    double* s_a = shared;                        // ���ڴ洢k��������
    double* s_eps = &shared[nz];                // ���ڴ洢�ֲ����

    s_eps[threadIdx.x] = 0.0;  // ��ʼ���ֲ����

    // ��������k�������ݵ������ڴ�
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }
    // ���ر߽�ֵ
    if (threadIdx.x == 0) {
        s_a[0] = a[i * ny * nz + j * nz + 0];
        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
    }

    __syncthreads();

    // ���и���k���򲢼���ֲ�������
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k-1] + s_a[k+1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // ���¾ֲ�������
        }
        __syncthreads();
    }

    // д��ȫ���ڴ�
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // ʹ��ԭ�Ӳ�������ȫ��������
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int*)d_eps, __double_as_longlong(s_eps[0]));
    }
}

// ��������������ͬ��ʾ��ʽ��˫������ת��
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

    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // �����ڴ�
    a = (double*)malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_eps, sizeof(double)));

    printf("Starting computation...\n");

    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++)
    {
        // Copy current data from host to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

        // ��ʼ�����ֵΪ0
        double h_eps = 0.0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        // �����߳̿����������
        dim3 block(BLOCK_SIZE, 1, 1);
        dim3 grid_i(1, ny - 2, nz - 2);      // ÿ���߳̿鴦��һ�е�����i
        dim3 grid_j(nx - 2, 1, nz - 2);      // ÿ���߳̿鴦��һ�е�����j
        dim3 grid_k(nx - 2, ny - 2, 1);      // ÿ���߳̿鴦��һ��"��"������k

        // �����ڴ��С
        size_t shared_mem_size_i = nx * sizeof(double);                  // ����
        size_t shared_mem_size_j = ny * sizeof(double);                  // ����
        size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // ��"��"������ʱ�洢

        double check_time_a = omp_get_wtime();

        // i�������
        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        double check_time_b = omp_get_wtime();

        // j�������
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        double check_time_c = omp_get_wtime();

        // k�������
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        double check_time_d = omp_get_wtime();

        // ��ȡepsֵ
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        // �������ս��������
        CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �ͷ��ڴ�
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

// �� CPU �Ͻ��г�ʼ��
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

// i����˺���
__global__ void compute_i_direction_kernel(double *a)
{
    unsigned long long time_checker_1 = clock64();      // ��ʼ��ʱ

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];


    // �����������ݵ������ڴ�
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    unsigned long long time_checker_2 = clock64();      // 23


    // ���и��£�ȷ����ȷ��������ϵ
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[i] = (s_a[i-1] + s_a[i+1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // д��ȫ���ڴ�
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


// j����˺���
__global__ void compute_j_direction_kernel(double *a)
{
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];

    // �����������ݵ������ڴ�
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и��£�ȷ����ȷ��������ϵ
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[j] = (s_a[j-1] + s_a[j+1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    // д��ȫ���ڴ�
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k����˺���������eps����
__global__ void compute_k_direction_kernel(double *a, double *d_eps)
{
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // ʹ�ù����ڴ汣�浱ǰ"��"�;ֲ�������
    extern __shared__ double shared[];
    double* s_a = shared;                        // ���ڴ洢k��������
    double* s_eps = &shared[nz];                // ���ڴ洢�ֲ����

    s_eps[threadIdx.x] = 0.0;  // ��ʼ���ֲ����

//    // ��������k�������ݵ������ڴ�
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // ���ر߽�ֵ
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // ��������k�������ݵ������ڴ�
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и���k���򲢼���ֲ�������
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k-1] + s_a[k+1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // ���¾ֲ�������
        }
//        __syncthreads();
    }

    // д��ȫ���ڴ�
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // ʹ��ԭ�Ӳ�������ȫ��������
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int*)d_eps, __double_as_longlong(s_eps[0]));
    }
}

// ��������������ͬ��ʾ��ʽ��˫������ת��
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


    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double*)malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_eps, sizeof(double)));

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // �����߳̿���������� - �Ƶ�ѭ���ⲿ�Լ����ظ�����
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // ÿ���߳̿鴦��һ�е�����i
    dim3 grid_j(nx - 2, 1, nz - 2);      // ÿ���߳̿鴦��һ�е�����j
    dim3 grid_k(nx - 2, ny - 2, 1);      // ÿ���߳̿鴦��һ��"��"������k

    // �����ڴ��С - ͬ���Ƶ�ѭ���ⲿ
    size_t shared_mem_size_i = nx * sizeof(double);                  // ����
    size_t shared_mem_size_j = ny * sizeof(double);                  // ����
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // ��"��"������ʱ�洢

    for (it = 1; it <= itmax; it++)
    {
        // �Ż���2: ����ÿ�ε������¿�����������

        check_time_a = omp_get_wtime();

        // ��ʼ�����ֵΪ0
        double h_eps = 0.0;
        // ***** ��ʱ���� *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));



        // i�������
        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);

        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();

        // j�������
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        // k�������
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();


        // �Ż���3: ֻ����epsֵ��������������
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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

// �Զ����������ʾ����Ϊ��result = (prev_value)/2 + current_input
struct MyCustomOp {
    __device__ double operator()(double a, double b) const {
        return a / 2.0 + b;
    }
};

// ÿ���߳̿鸺����һ�������ߣ����̶� (j,k) �� x �����ϵ�Ԫ�أ�
__global__ void scan_x_direction_kernel(double *a)
{
    // �趨���� x ����ɨ�裬�Թ̶��� j, k ���в���
    // ���� gridDim.x = NY * NZ��ÿ�����Ӧһ�� (j, k) ���
    int lineId = blockIdx.x;
    int j = lineId / NZ;      // j ��������
    int k = lineId % NZ;      // k ��������

    // ÿ�������ߵ���ʼ��ַ�����ݴ��˳��a[i * (NY*NZ) + j*NZ + k]
    // ��Ϊ i �仯����� x ���򣬱�֤��������
    double *line = a + j * NZ + k;

    // ���������ڴ����飬����Ϊ NX
    extern __shared__ double s_line[];  // ��СӦ��Ϊ NX * sizeof(double)

    // ÿ���̸߳����߳�������ȡ����
    // ���� blockDim.x >= NX����������Ϊ 1024����ÿ���̶߳�ȡһ������
    int tid = threadIdx.x;
    if (tid < NX)
    {
        // ע�⣺ȫ�������� x ������Ϊ (NY*NZ)����Ϊ i �� 0 �� NX-1
        s_line[tid] = line[tid * (NY * NZ)];
    }
    __syncthreads();

    // ���� CUB BlockScan �ڿ�����ɨ��
    typedef cub::BlockScan<double, NX> BlockScan; // 1024 Ϊ blockDim.x
    __shared__ typename BlockScan::TempStorage temp_storage;

    double scan_result = 0.0;
    // ÿ���̵߳��� BlockScan �� InclusiveScan
    BlockScan(temp_storage).InclusiveScan(s_line[tid], scan_result, MyCustomOp());
    __syncthreads();

    // ��������д��ȫ���ڴ�
    if (tid < NX)
    {
        line[tid * (NY * NZ)] = scan_result;
    }
}

int main()
{
    size_t total_elems = NX * NY * NZ;
    size_t total_bytes = total_elems * sizeof(double);

    // ����ͳ�ʼ�����ݣ�����ֻ���򵥳�ʼ����
    double *h_a = (double*) malloc(total_bytes);
    double *h_b = (double*) malloc(NX * sizeof(double));
    for (size_t i = 0; i < total_elems; i++) {
        h_a[i] = 1.0;  // ��������ʼ��ֵ
    }
    for (int i = 0; i < NX; i++)
        h_b[i] = 1.0;

    double *d_a;
    cudaMalloc(&d_a, total_bytes);
    cudaMemcpy(d_a, h_a, total_bytes, cudaMemcpyHostToDevice);

    // ÿ�� block ����һ�������ߣ����� NY * NZ ��������
    int num_blocks = NY * NZ;
    int threads_per_block = 1024;  // Ҫȷ�� threads_per_block >= NX (900)
    size_t shared_mem_bytes = NX * sizeof(double);

    // ���� kernel
    scan_x_direction_kernel<<<num_blocks, threads_per_block, shared_mem_bytes>>>(d_a);
    cudaDeviceSynchronize();

    // ��������ƻ�����������֤������ֻ��֤���������Ա�֤��ȷ�ԣ�
    cudaMemcpy(h_a, d_a, total_bytes, cudaMemcpyDeviceToHost);

    // ʾ���������һ�������ߵ�ǰ�������
    int line_index = 0;  // ��Ӧ�� j = 0, k = 0
    printf("Results for first line (j=0, k=0):\n");

    // �ô��з�ʽ����һ�����
    for (int i = 0; i < NX; i++) {
        if (i == 0)
            h_b[0] = h_b[0];
        else {
            h_b[i] = h_b[i - 1] / 2 + h_b[i];
        }
    }

    for (int i = 0; i < NX; i++) {
        // ע�⣺���Ϊ NY*NZ
        printf("h_a[%d] = %f   h_b[%d] = %f\n", i, h_a[i * (NY * NZ)], i, h_b[i]);
    }

    // ����
    free(h_a);
    cudaFree(d_a);

    return 0;
}

#endif

#if RUNNING_MODE == 3
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

// �Զ������������ɨ��ʱ�����ۻ�ֵ�ȳ��� 2���ټ��ϵ�ǰ������
struct MyScanOp {
    __device__ double operator()(double a, double b) const {
//        return a / 2.0 + b;
        return a + b;
    }
};

int main() {
    const int L = 20;
    const int num_items = L * L * L;

    // �����������ݣ����� old_a �����У��� 0 ��Ԫ��ֱ����Ϊ��ʼֵ��
    // ����Ԫ�ز���ɨ�裨���� old_a[1] ... old_a[899]����
    double h_input[num_items];
    for (int i = 0; i < num_items; i++) {
        h_input[i] = static_cast<double>(i + 1); // �����ʼ��Ϊ 1, 2, 3, ��, 900
    }

    // �豸�ڴ����
    double *d_input, *d_output;
    cudaMalloc(&d_input, num_items * sizeof(double));
    cudaMalloc(&d_output, num_items * sizeof(double));
    cudaMemcpy(d_input, h_input, num_items * sizeof(double), cudaMemcpyHostToDevice);

    // ��ʱ�洢ָ�����С
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // ��һ�ε��û�ȡ��ʱ�洢�ռ��С
    cub::DeviceScan::InclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_input, d_output,
            MyScanOp(),
            num_items
    );

    // ������ʱ�洢�ռ�
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // ִ�� inclusive scan��ǰ׺ɨ�裩��Ӧ���Զ��������
    cub::DeviceScan::InclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_input, d_output,
            MyScanOp(),
            num_items
    );

    // ��������ƻ������鿴
    double h_output[num_items];
    cudaMemcpy(h_output, d_output, num_items * sizeof(double), cudaMemcpyDeviceToHost);

    // ������ֽ�����ڼ��
    std::cout << "First 10 scan results:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "output[" << i << "] = " << h_output[i] << std::endl;
    }

    // ����
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

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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

#define BLOCK_SIZE 64       // Ĭ�� 32, Ŀǰ 64 �������

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


// �� CPU �Ͻ��г�ʼ��
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
































//// �� CUB �� BlockScan ������и��µĺ˺����汾
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
//    int j = blockIdx.y + 1;
//    int k = blockIdx.z + 1;
//
//    // �����ڴ����ڴ�ŵ�ǰ�����ߣ��� nx ��Ԫ��
//    extern __shared__ double s_a[];
//
//    // �����������ݵ������ڴ�
//    // Ϊ���㣬����ÿ���̸߳������һ�����ݣ�Ҫ�� blockDim.x >= nx��
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // ����ȫ�������� i * ny * nz + j * nz + k  (i��0��nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
////    // �����������ݵ������ڴ� (ʹ�ô��з�ʽ��ȡ,�Ա��� debug) *********
////    for (int i = 0; i < nx; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            s_a[i] = a[i * ny * nz + j * nz + k] / 2.;
////        }
////    }
//
//    __syncthreads();
//
////    // ��һ�β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0) {
////        printf("��һ��\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // Ԥ������� s_a[0] ��ֵ
//    }
//    __syncthreads();
//
//    // ���� CUB BlockScan �ڿ��ڶ� s_a[] ��ǰ׺ɨ��
//    // ��������Ҫ��ɨ�衱�Ķ����� s_a �����е�ÿ��Ԫ�أ������Զ�������� MyOp
//    typedef cub::BlockScan<double, nx> BlockScan;       // ���� <> �еĵڶ��������ǵ�ǰ block �Ĵ�С, �ܹؼ�!!!
//    __shared__ typename BlockScan::TempStorage temp_storage;
//
//
//    if (threadIdx.x == 0 && blockIdx.x == 0)
//        printf("TempStorage size: %llu\n", (unsigned long long) sizeof(BlockScan::TempStorage));
//
//    double scan_val = 0.0;
//    // ���� InclusiveScan��
//    // �õ������result[i] = s_a[0] op s_a[1] op ... op s_a[i]
//    // ���ﲻ��������, ����Ҫ block ��ȫ���̶߳�ִ�� BlockScan
////    if (tid >= 2 && tid < nx)
////        BlockScan(temp_storage).InclusiveScan(s_a[tid], scan_val, MyOp());
////    __syncthreads();
//    // ������ǽ� tid_0 tid_1 �̴߳���� inputֵ�趨�� 0
//
////    double input = (tid >= 2 && tid < nx) ? s_a[tid] : 0.0;
//
//    // �ڶ��β鿴 s_a ��������
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("�ڶ���\n");
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
//    // ע�⣡�� ���ﲻ��Ҫд�� �����ڴ�, ����ֱ�ӽ����д��ȫ���ڴ�
//    // ��ɨ����д�ع����ڴ棨ɨ����Ӧ��Ӧת�������ݣ�
//    if (tid >= 2 && tid < nx) {
//        s_a[tid] = scan_val;
//    }
//    __syncthreads();
//
//    // �����β鿴 s_a ��������
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("������\n");
//        for (int ii = 0; ii < nx; ++ii)
//            printf("%.2lf ", s_a[ii]);
//        printf("\n");
//    }
//
//    // �˴�����Ҫ���������������ĳ��ϵ���ָ�ԭ���������������������
//    // ���磺 new_value = (ת�����) / (2^i) ֮�����任
//
//    // �����º�Ĺ����ڴ���д��ȫ���ڴ棨ֻ�����ڲ����ݣ�
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // ��λ�洢
//    }
//    // �߽���Ա��ֲ������������
//
////    // �ô��з�ʽд��ȫ���ڴ�, ȷ����ȷ����
////    for (int i = 1; i < nx - 1; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            a[i * ny * nz + j * nz + k] = s_a[i];
////        }
////    }
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (��������ȷ)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//    // �����ڴ����ڴ�ŵ�ǰ�����ߣ��� nx ��Ԫ��
//    extern __shared__ double s_a[];
//
//    // �����㷨�����������ݵ������ڴ�   ����ÿ���̸߳������һ�����ݣ�Ҫ�� blockDim.x >= nx��
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // ����ȫ�������� i * ny * nz + j * nz + k  (i��0��nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // ��һ�β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("��һ��\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // �ڶ��β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("�ڶ���\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//    // ���и���
//    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
//        for (int i = 1; i < nx - 1; i++) {
//            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
//        }
//    }
//    __syncthreads();
//
////    // �����β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("������\n");
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

// �� CUB �� BlockScan ������и��µĺ˺����汾      (�ڲ��ô��м���ɹ����еĻ����ϳ�������ʹ�ò��м���) (����������, ԭ��: myop �����㽻����)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//
//    // �����ڴ����ڴ�ŵ�ǰ�����ߣ��� nx ��Ԫ��
//    extern __shared__ double s_a[];
//
//    // �����㷨�����������ݵ������ڴ�   ����ÿ���̸߳������һ�����ݣ�Ҫ�� blockDim.x >= nx��
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // ����ȫ�������� i * ny * nz + j * nz + k  (i��0��nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // ��һ�β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("��һ��\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // �ڶ��β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("�ڶ���\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // ���и���
////    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
////        for (int i = 1; i < nx - 1; i++) {
////            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
////        }
//////        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
////    }
////    __syncthreads();
//
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // Ԥ������� s_a[0] ��ֵ
//    }
//    __syncthreads();
//    typedef cub::BlockScan<double, nx> BlockScan;       // ���� <> �еĵڶ��������ǵ�ǰ block �Ĵ�С, �ܹؼ�!!!
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
////    // �����β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("������\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//
//    // �������ڴ���д��ȫ���ڴ�
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // ��λ�洢
//    }
//}



//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ)
 __global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾     (Ŀǰ���) (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;
    int k = thread_global_id % nz;

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;

//        int idx_center = i * ny * nz + j * nz + k;
//        int idx_left = (i - 1) * ny * nz + j * nz + k;
//        int idx_right = (i + 1) * ny * nz + j * nz + k;
//
//        a[idx_center] = (a[idx_left] + a[idx_right]) / 2.0;
    }
}













// i����˺���
__global__ void compute_i_direction_kernel(double *a) {
    unsigned long long time_checker_1 = clock64();      // ��ʼ��ʱ

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];


    // �����������ݵ������ڴ�
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    unsigned long long time_checker_2 = clock64();      // 23


    // ���и��£�ȷ����ȷ��������ϵ
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // д��ȫ���ڴ�
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


// j����˺���
__global__ void compute_j_direction_kernel(double *a) {
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];

    // �����������ݵ������ڴ�
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и��£�ȷ����ȷ��������ϵ
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[j] = (s_a[j - 1] + s_a[j + 1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    // д��ȫ���ڴ�
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k����˺���������eps����
__global__ void compute_k_direction_kernel(double *a, double *d_eps) {
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // ʹ�ù����ڴ汣�浱ǰ"��"�;ֲ�������
    extern __shared__ double shared[];
    double *s_a = shared;                        // ���ڴ洢k��������
    double *s_eps = &shared[nz];                // ���ڴ洢�ֲ����

    s_eps[threadIdx.x] = 0.0;  // ��ʼ���ֲ����

//    // ��������k�������ݵ������ڴ�
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // ���ر߽�ֵ
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // ��������k�������ݵ������ڴ�
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и���k���򲢼���ֲ�������
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k - 1] + s_a[k + 1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // ���¾ֲ�������
        }
//        __syncthreads();
    }

    // д��ȫ���ڴ�
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // ʹ��ԭ�Ӳ�������ȫ��������
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}

// ��������������ͬ��ʾ��ʽ��˫������ת��
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


    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // �����߳̿���������� - �Ƶ�ѭ���ⲿ�Լ����ظ�����
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // ÿ���߳̿鴦��һ�е�����i
    dim3 grid_j(nx - 2, 1, nz - 2);      // ÿ���߳̿鴦��һ�е�����j
    dim3 grid_k(nx - 2, ny - 2, 1);      // ÿ���߳̿鴦��һ��"��"������k

    dim3 my_block(1, 1, 1);     // ÿ�����ڽ��� 1 ���߳�
    dim3 my_grid(1, ny, nz);    // �ܹ� ny * nz ���߳�



    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // ÿ�����ڵ��߳�����
    int total_lines = ny * nz;      // x �����������
    int blocks = (total_lines + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************



    // �����ڴ��С - ͬ���Ƶ�ѭ���ⲿ
    size_t shared_mem_size_i = nx * sizeof(double);                  // ����
    size_t shared_mem_size_j = ny * sizeof(double);                  // ����
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // ��"��"������ʱ�洢

    for (it = 1; it <= itmax; it++) {
        // �Ż���2: ����ÿ�ε������¿�����������

        check_time_a = omp_get_wtime();

        // ��ʼ�����ֵΪ0
        double h_eps = 0.0;
        // ***** ��ʱ���� *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        check_time_b = omp_get_wtime();


        if (false)
            // i�������
            compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        else // ʹ�� cub �汾�� i �������
//            compute_i_direction_kernel_cub<<<my_grid, my_block>>>(d_A); // ÿ������ 1 ���߳�
            compute_i_direction_kernel_32_thread<<<blocks, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�

//        dim3 grid_i(1, ny, nz);
//        dim3 block(900, 1, 1); // ������Ҫ >= 900 �߳�
//        size_t shared_mem_size_i = nx * sizeof(double);  // 900 * sizeof(double)
//        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_c = omp_get_wtime();

        // j�������
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // k�������
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());



        // �Ż���3: ֻ����epsֵ��������������
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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


// �� CPU �Ͻ��г�ʼ��
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
































//// �� CUB �� BlockScan ������и��µĺ˺����汾
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
//    int j = blockIdx.y + 1;
//    int k = blockIdx.z + 1;
//
//    // �����ڴ����ڴ�ŵ�ǰ�����ߣ��� nx ��Ԫ��
//    extern __shared__ double s_a[];
//
//    // �����������ݵ������ڴ�
//    // Ϊ���㣬����ÿ���̸߳������һ�����ݣ�Ҫ�� blockDim.x >= nx��
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // ����ȫ�������� i * ny * nz + j * nz + k  (i��0��nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
////    // �����������ݵ������ڴ� (ʹ�ô��з�ʽ��ȡ,�Ա��� debug) *********
////    for (int i = 0; i < nx; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            s_a[i] = a[i * ny * nz + j * nz + k] / 2.;
////        }
////    }
//
//    __syncthreads();
//
////    // ��һ�β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0) {
////        printf("��һ��\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // Ԥ������� s_a[0] ��ֵ
//    }
//    __syncthreads();
//
//    // ���� CUB BlockScan �ڿ��ڶ� s_a[] ��ǰ׺ɨ��
//    // ��������Ҫ��ɨ�衱�Ķ����� s_a �����е�ÿ��Ԫ�أ������Զ�������� MyOp
//    typedef cub::BlockScan<double, nx> BlockScan;       // ���� <> �еĵڶ��������ǵ�ǰ block �Ĵ�С, �ܹؼ�!!!
//    __shared__ typename BlockScan::TempStorage temp_storage;
//
//
//    if (threadIdx.x == 0 && blockIdx.x == 0)
//        printf("TempStorage size: %llu\n", (unsigned long long) sizeof(BlockScan::TempStorage));
//
//    double scan_val = 0.0;
//    // ���� InclusiveScan��
//    // �õ������result[i] = s_a[0] op s_a[1] op ... op s_a[i]
//    // ���ﲻ��������, ����Ҫ block ��ȫ���̶߳�ִ�� BlockScan
////    if (tid >= 2 && tid < nx)
////        BlockScan(temp_storage).InclusiveScan(s_a[tid], scan_val, MyOp());
////    __syncthreads();
//    // ������ǽ� tid_0 tid_1 �̴߳���� inputֵ�趨�� 0
//
////    double input = (tid >= 2 && tid < nx) ? s_a[tid] : 0.0;
//
//    // �ڶ��β鿴 s_a ��������
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("�ڶ���\n");
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
//    // ע�⣡�� ���ﲻ��Ҫд�� �����ڴ�, ����ֱ�ӽ����д��ȫ���ڴ�
//    // ��ɨ����д�ع����ڴ棨ɨ����Ӧ��Ӧת�������ݣ�
//    if (tid >= 2 && tid < nx) {
//        s_a[tid] = scan_val;
//    }
//    __syncthreads();
//
//    // �����β鿴 s_a ��������
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        printf("������\n");
//        for (int ii = 0; ii < nx; ++ii)
//            printf("%.2lf ", s_a[ii]);
//        printf("\n");
//    }
//
//    // �˴�����Ҫ���������������ĳ��ϵ���ָ�ԭ���������������������
//    // ���磺 new_value = (ת�����) / (2^i) ֮�����任
//
//    // �����º�Ĺ����ڴ���д��ȫ���ڴ棨ֻ�����ڲ����ݣ�
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // ��λ�洢
//    }
//    // �߽���Ա��ֲ������������
//
////    // �ô��з�ʽд��ȫ���ڴ�, ȷ����ȷ����
////    for (int i = 1; i < nx - 1; i++) {
////        if (threadIdx.x == (i - 1) % blockDim.x) {
////            a[i * ny * nz + j * nz + k] = s_a[i];
////        }
////    }
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (��������ȷ)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//    // �����ڴ����ڴ�ŵ�ǰ�����ߣ��� nx ��Ԫ��
//    extern __shared__ double s_a[];
//
//    // �����㷨�����������ݵ������ڴ�   ����ÿ���̸߳������һ�����ݣ�Ҫ�� blockDim.x >= nx��
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // ����ȫ�������� i * ny * nz + j * nz + k  (i��0��nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // ��һ�β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("��һ��\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // �ڶ��β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("�ڶ���\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//    // ���и���
//    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
//        for (int i = 1; i < nx - 1; i++) {
//            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
//        }
//    }
//    __syncthreads();
//
////    // �����β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("������\n");
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

// �� CUB �� BlockScan ������и��µĺ˺����汾      (�ڲ��ô��м���ɹ����еĻ����ϳ�������ʹ�ò��м���) (����������, ԭ��: myop �����㽻����)
//__global__ void compute_i_direction_kernel_cub(double *a) {
//    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
//    int j = blockIdx.y;
//    int k = blockIdx.z;
//
//
//    // �����ڴ����ڴ�ŵ�ǰ�����ߣ��� nx ��Ԫ��
//    extern __shared__ double s_a[];
//
//    // �����㷨�����������ݵ������ڴ�   ����ÿ���̸߳������һ�����ݣ�Ҫ�� blockDim.x >= nx��
//    int tid = threadIdx.x;
//    if (tid < nx) {     // tid --> i
//        // ����ȫ�������� i * ny * nz + j * nz + k  (i��0��nx-1)
//        s_a[tid] = a[tid * ny * nz + j * nz + k];
//    }
//
//    __syncthreads();
//
////    // ��һ�β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("��һ��\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // �ڶ��β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("�ڶ���\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
////    // ���и���
////    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
////        for (int i = 1; i < nx - 1; i++) {
////            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
////        }
//////        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
////    }
////    __syncthreads();
//
//
//    if (tid == 2) {
//        s_a[2] = s_a[2] + s_a[0]; // Ԥ������� s_a[0] ��ֵ
//    }
//    __syncthreads();
//    typedef cub::BlockScan<double, nx> BlockScan;       // ���� <> �еĵڶ��������ǵ�ǰ block �Ĵ�С, �ܹؼ�!!!
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
////    // �����β鿴 s_a ��������
////    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
////        printf("������\n");
////        for (int ii = 0; ii < nx; ++ii)
////            printf("%.2lf ", s_a[ii]);
////        printf("\n");
////    }
//
//
//
//    // �������ڴ���д��ȫ���ڴ�
//    if (tid >= 1 && tid < nx - 1) {
//        a[tid * ny * nz + j * nz + k] = s_a[tid + 1] / 2.0; // ��λ�洢
//    }
//}



//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ)
 __global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾     (Ŀǰ���) (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;
    int k = thread_global_id % nz;

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;

//        int idx_center = i * ny * nz + j * nz + k;
//        int idx_left = (i - 1) * ny * nz + j * nz + k;
//        int idx_right = (i + 1) * ny * nz + j * nz + k;
//
//        a[idx_center] = (a[idx_left] + a[idx_right]) / 2.0;
    }
}













// i����˺���
__global__ void compute_i_direction_kernel(double *a) {
    unsigned long long time_checker_1 = clock64();      // ��ʼ��ʱ

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];


    // �����������ݵ������ڴ�
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    unsigned long long time_checker_2 = clock64();      // 23


    // ���и��£�ȷ����ȷ��������ϵ
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // д��ȫ���ڴ�
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


// j����˺���
__global__ void compute_j_direction_kernel(double *a) {
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];

    // �����������ݵ������ڴ�
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и��£�ȷ����ȷ��������ϵ
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[j] = (s_a[j - 1] + s_a[j + 1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    // д��ȫ���ڴ�
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k����˺���������eps����
__global__ void compute_k_direction_kernel(double *a, double *d_eps) {
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // ʹ�ù����ڴ汣�浱ǰ"��"�;ֲ�������
    extern __shared__ double shared[];
    double *s_a = shared;                        // ���ڴ洢k��������
    double *s_eps = &shared[nz];                // ���ڴ洢�ֲ����

    s_eps[threadIdx.x] = 0.0;  // ��ʼ���ֲ����

//    // ��������k�������ݵ������ڴ�
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // ���ر߽�ֵ
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // ��������k�������ݵ������ڴ�
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и���k���򲢼���ֲ�������
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k - 1] + s_a[k + 1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // ���¾ֲ�������
        }
//        __syncthreads();
    }

    // д��ȫ���ڴ�
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // ʹ��ԭ�Ӳ�������ȫ��������
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}

// ��������������ͬ��ʾ��ʽ��˫������ת��
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


    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // �����߳̿���������� - �Ƶ�ѭ���ⲿ�Լ����ظ�����
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // ÿ���߳̿鴦��һ�е�����i
    dim3 grid_j(nx - 2, 1, nz - 2);      // ÿ���߳̿鴦��һ�е�����j
    dim3 grid_k(nx - 2, ny - 2, 1);      // ÿ���߳̿鴦��һ��"��"������k

    dim3 my_block(1, 1, 1);     // ÿ�����ڽ��� 1 ���߳�
    dim3 my_grid(1, ny, nz);    // �ܹ� ny * nz ���߳�



    //**********************************************************
    //**********************************************************
    int threads_per_block = 32;     // ÿ������ 32 ���߳�
    int total_lines = ny * nz;      // x �����������
    int blocks = (total_lines + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************



    // �����ڴ��С - ͬ���Ƶ�ѭ���ⲿ
    size_t shared_mem_size_i = nx * sizeof(double);                  // ����
    size_t shared_mem_size_j = ny * sizeof(double);                  // ����
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // ��"��"������ʱ�洢

    for (it = 1; it <= itmax; it++) {
        // �Ż���2: ����ÿ�ε������¿�����������

        check_time_a = omp_get_wtime();

        // ��ʼ�����ֵΪ0
        double h_eps = 0.0;
        // ***** ��ʱ���� *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        check_time_b = omp_get_wtime();


        if (false)
            // i�������
            compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        else // ʹ�� cub �汾�� i �������
//            compute_i_direction_kernel_cub<<<my_grid, my_block>>>(d_A); // ÿ������ 1 ���߳�
            compute_i_direction_kernel_32_thread<<<blocks, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�

//        dim3 grid_i(1, ny, nz);
//        dim3 block(900, 1, 1); // ������Ҫ >= 900 �߳�
//        size_t shared_mem_size_i = nx * sizeof(double);  // 900 * sizeof(double)
//        compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_c = omp_get_wtime();

        // j�������
        compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // k�������
        compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());



        // �Ż���3: ֻ����epsֵ��������������
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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

#define BLOCK_SIZE 32       // Ĭ�� 32, Ŀǰ 64 �������

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


// �� CPU �Ͻ��г�ʼ��
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

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾   (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ) (Ŀǰ���)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) / 2.0;

    }
}

// ͬ��, ���� compute_i_direction_kernel_32_thread д�� compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    // ����� i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;

    int i = thread_global_id / nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    // ÿ���̴߳��д���һ���� y ����������
    for (int j = 1; j < ny - 1; ++j) {
        a[i * ny * nz + j * nz + k] = (a[i * ny * nz + (j - 1) * nz + k] + a[i * ny * nz + (j + 1) * nz + k]) / 2.0;
    }
}



// ͬ��, ���� compute_i_direction_kernel_32_thread д�� compute_j_direction_kernel_32_thread
// ʹ�ù����ڴ�汾, ������дһ��ʹ�üĴ����汾
// ��Ϊ��Щ���ݽ����̱߳���ʹ��, ����Ҫ���߳�, ��˲���Ҫʹ�ù����ڴ�
// ������������
__global__ void compute_k_direction_kernel_32_thread_shared_memory(double *a, double* d_eps) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    extern __shared__ double shared[];
    double *s_eps = &shared[blockDim.x * 0];
    double *d_al  = &shared[blockDim.x * 1];
    double *d_ac  = &shared[blockDim.x * 2];
    double *d_ar  = &shared[blockDim.x * 3];

    // ����� i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;


    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����


    s_eps[tid_in_block] = 0.; // ��ʼ���ֲ����
    d_al[tid_in_block] = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac[tid_in_block] = a[i * ny * nz + j * nz + 1];   // k = 1

    // ÿ���̴߳��д���һ���� z ����������
    for (int k = 1; k < nz - 1; ++k) {
        d_ar[tid_in_block] = a[i * ny * nz + j * nz + (k + 1)];

        // ����ֲ����ֵ
        // �� 900 �ε��õ������, ʹ�ú� Max ��� fmax ���ú�������Ч, ԭ���ǲ���Ҫ�������õĶ��⿪��
//        s_eps[tid_in_block] = Max(s_eps[tid_in_block], fabs((d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0 - d_ac[tid_in_block]));
        s_eps[tid_in_block] = fmax(s_eps[tid_in_block], fabs((d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0 - d_ac[tid_in_block]));


        // �Ż�������ͽ����Ĵ����ɣ���Ը�Ĵ����ܴ洢���������ֵ
        a[i * ny * nz + j * nz + k] = (d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0;

        // ���� d_al �� d_ac
        d_al[tid_in_block] = (d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0;
        d_ac[tid_in_block] = d_ar[tid_in_block];
    }

    __syncthreads();
    double max_eps = s_eps[0];
    // ����ͨ�����в����� block �ڽ����ֵ������ s_eps[0] ��
    if (threadIdx.x == 0) {
        for (int k = 1; k < nz; ++k) {
            if (s_eps[k] > max_eps) {
                max_eps = s_eps[k];
            }
        }
    }
    __syncthreads();

    // Ȼ�� 0 �߳̽��ֲ����洢��ȫ���ڴ���ȥ
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}



__global__ void compute_k_direction_kernel_32_thread(double *a, double* d_eps) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al;
    double d_ac;
    double d_ar;
    double tmp;
    double tmp2;

    // ����� i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;


    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // ÿ���̴߳��д���һ���� z ����������
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];

        // ����ֲ����ֵ
        // �� 900 �ε��õ������, ʹ�ú� Max ��� fmax ���ú�������Ч, ԭ���ǲ���Ҫ�������õĶ��⿪��
//        s_eps[tid_in_block] = Max(s_eps[tid_in_block], fabs((d_al[tid_in_block] + d_ar[tid_in_block]) / 2.0 - d_ac[tid_in_block]));
        tmp = (d_al + d_ar) / 2.0;
        // �Ż�������ͽ����Ĵ����ɣ���Ը�Ĵ����ܴ洢���������ֵ
        a[i * ny * nz + j * nz + k] = tmp;

        tmp2 = Max(s_eps, fabs(d_ac - tmp));
        s_eps = tmp2;

        // ���� d_al �� d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // ���ֲ����洢��ȫ���ڴ���ȥ
    atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
}



// i����˺���
__global__ void compute_i_direction_kernel(double *a) {
    unsigned long long time_checker_1 = clock64();      // ��ʼ��ʱ

    int j = blockIdx.y + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];


    // �����������ݵ������ڴ�
    for (int i = 0; i < nx; i++) {
        if (threadIdx.x == (i - 1) % blockDim.x) {
            s_a[i] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    unsigned long long time_checker_2 = clock64();      // 23


    // ���и��£�ȷ����ȷ��������ϵ
    for (int i = 1; i < nx - 1; i++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[i] = (s_a[i - 1] + s_a[i + 1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    __syncthreads();
    unsigned long long time_checker_3 = clock64();      // 41

    // д��ȫ���ڴ�
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


// j����˺���
__global__ void compute_j_direction_kernel(double *a) {
    int i = blockIdx.x + 1;
    int k = blockIdx.z + 1;

    // ʹ�ù����ڴ汣�浱ǰ��
    extern __shared__ double s_a[];

    // �����������ݵ������ڴ�
    for (int j = 0; j < ny; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            s_a[j] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и��£�ȷ����ȷ��������ϵ
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
            s_a[j] = (s_a[j - 1] + s_a[j + 1]) / 2.0;
        }
//        __syncthreads(); // ȷ�������߳��ܿ������º��ֵ
    }

    // д��ȫ���ڴ�
    for (int j = 1; j < ny - 1; j++) {
        if (threadIdx.x == (j - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[j];
        }
    }
}

// k����˺���������eps����
__global__ void compute_k_direction_kernel(double *a, double *d_eps) {
    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    // ʹ�ù����ڴ汣�浱ǰ"��"�;ֲ�������
    extern __shared__ double shared[];
    double *s_a = shared;                        // ���ڴ洢k��������
    double *s_eps = &shared[nz];                // ���ڴ洢�ֲ����

    s_eps[threadIdx.x] = 0.0;  // ��ʼ���ֲ����

//    // ��������k�������ݵ������ڴ�
//    for (int k = 1; k < nz - 1; k++) {
//        if (threadIdx.x == (k - 1) % blockDim.x) {
//            s_a[k] = a[i * ny * nz + j * nz + k];
//        }
//    }
//    // ���ر߽�ֵ
//    if (threadIdx.x == 0) {
//        s_a[0] = a[i * ny * nz + j * nz + 0];
//        s_a[nz - 1] = a[i * ny * nz + j * nz + (nz - 1)];
//    }
    // ��������k�������ݵ������ڴ�
    for (int k = 0; k < nz; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            s_a[k] = a[i * ny * nz + j * nz + k];
        }
    }

    __syncthreads();    // ��Ҫ

    // ���и���k���򲢼���ֲ�������
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == 0) {
            double old_val = s_a[k];
            double new_val = (s_a[k - 1] + s_a[k + 1]) / 2.0;
            double local_error = fabs(new_val - old_val);
            s_a[k] = new_val;
            s_eps[0] = fmax(s_eps[0], local_error);  // ���¾ֲ�������
        }
//        __syncthreads();
    }

    // д��ȫ���ڴ�
    for (int k = 1; k < nz - 1; k++) {
        if (threadIdx.x == (k - 1) % blockDim.x) {
            a[i * ny * nz + j * nz + k] = s_a[k];
        }
    }

    // ʹ��ԭ�Ӳ�������ȫ��������
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long int *) d_eps, __double_as_longlong(s_eps[0]));
    }
}

// ��������������ͬ��ʾ��ʽ��˫������ת��
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


    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    // �����߳̿���������� - �Ƶ�ѭ���ⲿ�Լ����ظ�����
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_i(1, ny - 2, nz - 2);      // ÿ���߳̿鴦��һ�е�����i
    dim3 grid_j(nx - 2, 1, nz - 2);      // ÿ���߳̿鴦��һ�е�����j
    dim3 grid_k(nx - 2, ny - 2, 1);      // ÿ���߳̿鴦��һ��"��"������k

    dim3 my_block(1, 1, 1);     // ÿ�����ڽ��� 1 ���߳�
    dim3 my_grid(1, ny, nz);    // �ܹ� ny * nz ���߳�



    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // ÿ�����ڵ��߳�����
    int total_lines_x = ny * nz;    // x �����������
    int total_lines_y = nx * nz;    // y �����������
    int total_lines_z = nx * ny;    // z �����������
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************



    // �����ڴ��С - ͬ���Ƶ�ѭ���ⲿ
    size_t shared_mem_size_i = nx * sizeof(double);                  // ����
    size_t shared_mem_size_j = ny * sizeof(double);                  // ����
    size_t shared_mem_size_k = (nz + BLOCK_SIZE) * sizeof(double);   // ��"��"������ʱ�洢

    for (it = 1; it <= itmax; it++) {
        // �Ż���2: ����ÿ�ε������¿�����������

        check_time_a = omp_get_wtime();

        // ��ʼ�����ֵΪ0
        double h_eps = 0.0;
        // ***** ��ʱ���� *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));

        check_time_b = omp_get_wtime();


        if (false)
            // i�������
            compute_i_direction_kernel<<<grid_i, block, shared_mem_size_i>>>(d_A);
        else // ʹ�� cub �汾�� i �������
//            compute_i_direction_kernel_cub<<<my_grid, my_block>>>(d_A); // ÿ������ 1 ���߳�
            compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�



        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_c = omp_get_wtime();

        if (false)
            // j�������
            compute_j_direction_kernel<<<grid_j, block, shared_mem_size_j>>>(d_A);
        else
            compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        if (false)
            // k�������
            compute_k_direction_kernel<<<grid_k, block, shared_mem_size_k>>>(d_A, d_eps);
        else
//            compute_k_direction_kernel_32_thread_shared_memory<<<blocks_z, threads_per_block, threads_per_block * 4 * sizeof(double)>>>(d_A, d_eps); // ÿ������ 32 ���߳�
            compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());



        // �Ż���3: ֻ����epsֵ��������������
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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
//#include <cub/block/block_scan.cuh>   // δʹ��

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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

#define BLOCK_SIZE 32       // Ĭ�� 32, Ŀǰ 64 �������

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


// �� CPU �Ͻ��г�ʼ��
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

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾   (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ) (Ŀǰ���)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;

    }
}

// ͬ��, ���� compute_i_direction_kernel_32_thread д�� compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    // ����� i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;

    int i = thread_global_id / nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    // ÿ���̴߳��д���һ���� y ����������
    for (int j = 1; j < ny - 1; ++j) {
        a[i * ny * nz + j * nz + k] = (a[i * ny * nz + (j - 1) * nz + k] + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
    }
}

//// ���ֹ�Լ
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
////        tmp = (d_al + d_ar) / 2.0;   // �Է����ϱ������������Ż�
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
//        d_eps[blockIdx.x] = sh_data[0];
//}


// warp �ڹ�Լ
__inline__ __device__
double warp_reduce_max(double val) {
    // ʹ�� warp shuffle ��������Լ
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// Ϊ block_reduce ׼���� shared memory
    __shared__ double sh_data[BLOCK_SIZE];

    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double tmp;
    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al, d_ac, d_ar;  // left  center  right

    // ����� i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
        return;

    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // ÿ���̴߳��д���һ���� z ����������
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // ����ֲ����ֵ
//        tmp = (d_al + d_ar) / 2.0;   // �Է����ϱ������������Ż�
        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // ���� d_al �� d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // ʹ�� warp �ڹ�Լ
    double warp_max = warp_reduce_max(s_eps);
    if (threadIdx.x == 0)
        d_eps[blockIdx.x] = warp_max;

//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
//        d_eps[blockIdx.x] = sh_data[0];
}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // �� i ����
//    int tid = threadIdx.x; // �� idx ������
//    int idx_thread_global = tid + i * blockDim.x; // ����� A ���飨�� block_max���ĵ�ַλ��
//    // ����������ֵ����shared mem��
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // ��ÿ���̲߳�����ȫ���߳�
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // �������һ�� block �ĳ���������������һ����Сֵ
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // ���ַ����ڹ�Լ
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // ����Ҫ����MAX������max
//        }
//        __syncthreads();
//    }
//    // �� 0 ���߳̽���ǰ���ֵд��ȫ���ڴ�
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

    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // ÿ�����ڵ��߳�����
    int total_lines_x = ny * nz;    // x �����������
    int total_lines_y = nx * nz;    // y �����������
    int total_lines_z = nx * ny;    // z �����������
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // ���ÿ��block�����ֵ
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_out, sizeof(double)));  // ���ȫ�ֹ�Լ(global_reduce)���

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // Ϊ CUB Ԥ�����ڴ�
    // d_eps: �����������豸�ϵ�ָ��
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int num_items = blocks_z;   // ����block��������

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_eps, d_out, num_items);
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));


    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // ��ʼ�����ֵΪ0  (ʹ��block_reduce + global_reduceʱ����Ҫ��ʼ�����)
//        double h_eps = 0.0;
//        // ***** ��ʱ���� *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));



        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// �� block_reduce ��Ҫ�õ� shared memory
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double)>>>(d_A, d_eps); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // ��ʱd_eps�д洢ÿ��block�ڵ����ֵ
        // global_reduce 1
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_eps, d_out, num_items);


        // �Ż���3: ֻ����epsֵ��������������
        // ��d_outȫ������Լֵ����eps����
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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
//#include <cub/block/block_scan.cuh>   // δʹ��

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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

#define BLOCK_SIZE 32       // Ĭ�� 32, Ŀǰ 64 �������

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


// �� CPU �Ͻ��г�ʼ��
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

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾   (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ) (Ŀǰ���)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {

        a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;

    }
}

// ͬ��, ���� compute_i_direction_kernel_32_thread д�� compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    // ����� i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;

    int i = thread_global_id / nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    // ÿ���̴߳��д���һ���� y ����������
    for (int j = 1; j < ny - 1; ++j) {
        a[i * ny * nz + j * nz + k] = (a[i * ny * nz + (j - 1) * nz + k] + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
    }
}

//// ���ֹ�Լ
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
////        tmp = (d_al + d_ar) / 2.0;   // �Է����ϱ������������Ż�
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
//        d_eps[blockIdx.x] = sh_data[0];
//}


// warp �ڹ�Լ
__inline__ __device__
double warp_reduce_max(double val) {
    // ʹ�� warp shuffle ��������Լ
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double tmp;
    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al, d_ac, d_ar;  // left  center  right

    // ����� i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
        return;

    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // ÿ���̴߳��д���һ���� z ����������
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // ����ֲ����ֵ
        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // ���� d_al �� d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
    // ��ʵһ�㶼����������
    atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // �� i ����
//    int tid = threadIdx.x; // �� idx ������
//    int idx_thread_global = tid + i * blockDim.x; // ����� A ���飨�� block_max���ĵ�ַλ��
//    // ����������ֵ����shared mem��
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // ��ÿ���̲߳�����ȫ���߳�
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // �������һ�� block �ĳ���������������һ����Сֵ
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // ���ַ����ڹ�Լ
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // ����Ҫ����MAX������max
//        }
//        __syncthreads();
//    }
//    // �� 0 ���߳̽���ǰ���ֵд��ȫ���ڴ�
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

    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // ÿ�����ڵ��߳�����
    int total_lines_x = ny * nz;    // x �����������
    int total_lines_y = nx * nz;    // y �����������
    int total_lines_z = nx * ny;    // z �����������
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, sizeof(double)));  // ���ÿ��block�����ֵ

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

        // ��ʼ�����ֵΪ0  (ʹ��block_reduce + global_reduceʱ����Ҫ��ʼ�����)
        double h_eps = 0.0;
        // ***** ��ʱ���� *****
        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// �� block_reduce ��Ҫ�õ� shared memory
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();


        // �Ż���3: ֻ����epsֵ��������������
        // ��d_outȫ������Լֵ����eps����
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, d_eps, sizeof(double), cudaMemcpyDeviceToHost));
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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

#define BLOCK_SIZE 32       // Ĭ�� 32, Ŀǰ 64 �������

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


// �� CPU �Ͻ��г�ʼ��
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

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾   (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ) (Ŀǰ���)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // �����ּ�������Ϊ���ּ���������

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    d_al = a[(1 - 1) * ny * nz + j * nz + k];

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {
        d_al = (d_al + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

// ͬ��, ���� compute_i_direction_kernel_32_thread д�� compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double d_al; // �����ּ�������Ϊ���ּ���������

    // ����� i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;

    int i = thread_global_id / nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    d_al = a[i * ny * nz + (1 - 1) * nz + k];

    // ÿ���̴߳��д���һ���� y ����������
    for (int j = 1; j < ny - 1; ++j) {
        d_al = (d_al + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

//// ���ֹ�Լ -- �� tmp  (������ͨ���ֹ�Լ)
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    register double s_eps = 0.;  // ��ʼ���ֲ����
//    register double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    if (thread_global_id >= nx * ny)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
//        a[i * ny * nz + j * nz + k] = (d_al + d_ar) * 0.5;
//        s_eps = MAX(s_eps, fabs(d_ac - (d_al + d_ar) * 0.5));
//
//        // ���� d_al �� d_ac
//        d_al = (d_al + d_ar) * 0.5;
//        d_ac = d_ar;
//    }
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
//        d_eps[blockIdx.x] = sh_data[0];
//}

// ���ֹ�Լ
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// Ϊ block_reduce ׼���� shared memory
    __shared__ double sh_data[BLOCK_SIZE];
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double tmp;
    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al, d_ac, d_ar;  // left  center  right

    // ����� i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
        return;

    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // ÿ���̴߳��д���һ���� z ����������
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // ����ֲ����ֵ
        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // ���� d_al �� d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
    sh_data[tid_in_block] = s_eps;
    __syncthreads();

    // ʹ�� ���� ���ڹ�Լ
    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
    if (tid_in_block == 0) // д��block_reduce���
        d_eps[blockIdx.x] = sh_data[0];
}

//// ���ֹ�Լ   clock debug ��
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    unsigned long long time_checker_1 = clock64();
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
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
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    unsigned long long time_checker_3 = clock64();
//    __syncthreads();
//
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//
//
//    unsigned long long time_checker_4 = clock64();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
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
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
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


//// warp �ڹ�Լ�ĸ�������
//__inline__ __device__
//double warp_reduce_max(double val) {
//    // ʹ�� warp shuffle ��������Լ
//    for (int offset = 16; offset > 0; offset /= 2)
//        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
//    return val;
//}
//// warp �ڹ�Լ
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
////        tmp = (d_al + d_ar) / 2.0;   // �Է����ϱ������������Ż�
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // ʹ�� warp �ڹ�Լ
//    double warp_max = warp_reduce_max(s_eps);
//    if (threadIdx.x == 0)
//        d_eps[blockIdx.x] = warp_max;
//
////    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
////    sh_data[tid_in_block] = s_eps;
////    __syncthreads();
////
////    // ʹ�� ���� ���ڹ�Լ
////    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
////    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
////    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
////    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
////    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
////    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
////        if (tid_in_block < s)
////            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
////        __syncthreads();
////    }
////    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
////    if (tid_in_block == 0) // д��block_reduce���
////        d_eps[blockIdx.x] = sh_data[0];
//}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // �� i ����
//    int tid = threadIdx.x; // �� idx ������
//    int idx_thread_global = tid + i * blockDim.x; // ����� A ���飨�� block_max���ĵ�ַλ��
//    // ����������ֵ����shared mem��
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // ��ÿ���̲߳�����ȫ���߳�
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // �������һ�� block �ĳ���������������һ����Сֵ
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // ���ַ����ڹ�Լ
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // ����Ҫ����MAX������max
//        }
//        __syncthreads();
//    }
//    // �� 0 ���߳̽���ǰ���ֵд��ȫ���ڴ�
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

    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // ÿ�����ڵ��߳�����
    int total_lines_x = ny * nz;    // x �����������
    int total_lines_y = nx * nz;    // y �����������
    int total_lines_z = nx * ny;    // z �����������
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // ���ÿ��block�����ֵ

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // Ϊ CUB Ԥ�����ڴ�
    // d_eps: �����������豸�ϵ�ָ��
    int num_items = blocks_z;   // ����block��������

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // ��ʼ�����ֵΪ0  (ʹ��block_reduce + global_reduceʱ����Ҫ��ʼ�����)
//        double h_eps = 0.0;
//        // ***** ��ʱ���� *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// ���ֹ�Լ��Ҫ shared memory
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double) >>>(d_A, d_eps); // ÿ������ 32 ���߳�
//        //// warp �ڹ�Լ����Ҫ shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // ��ʱd_eps�д洢ÿ��block�ڵ����ֵ
        // global_reduce 2  thrust  // �ҳ����㣡����
        // ����һ�� Thrust device_ptr ��װ����ָ�� d_eps��device �ϵ� block �ֲ����������飩
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
        // ʹ�� thrust::reduce �ҳ����ֵ����ʼֵ������ -�ޣ�
        eps = thrust::reduce(dev_ptr, dev_ptr + num_items, 0., thrust::maximum<double>());

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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

#define BLOCK_SIZE 32       // Ĭ�� 32, Ŀǰ 64 �������

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


// �� CPU �Ͻ��г�ʼ��
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

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾   (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ) (Ŀǰ���)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // �����ּ�������Ϊ���ּ���������

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    d_al = a[(1 - 1) * ny * nz + j * nz + k];

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {
        d_al = (d_al + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

// ͬ��, ���� compute_i_direction_kernel_32_thread д�� compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double d_al; // �����ּ�������Ϊ���ּ���������

    // ����� i, k
    int ik_total = nx * nz;

    if (thread_global_id >= ik_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;

    int i = thread_global_id / nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    d_al = a[i * ny * nz + (1 - 1) * nz + k];

    // ÿ���̴߳��д���һ���� y ����������
    for (int j = 1; j < ny - 1; ++j) {
        d_al = (d_al + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

//// ���ֹ�Լ -- �� tmp  (������ͨ���ֹ�Լ)
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    register double s_eps = 0.;  // ��ʼ���ֲ����
//    register double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    if (thread_global_id >= nx * ny)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
//        a[i * ny * nz + j * nz + k] = (d_al + d_ar) * 0.5;
//        s_eps = MAX(s_eps, fabs(d_ac - (d_al + d_ar) * 0.5));
//
//        // ���� d_al �� d_ac
//        d_al = (d_al + d_ar) * 0.5;
//        d_ac = d_ar;
//    }
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
//        d_eps[blockIdx.x] = sh_data[0];
//}


// �Զ����ӡ debug
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
// �ڶ��ֹ�Լ�����Ļ����ϣ�ʹ�ò��м����ڴ�ķ�ʽ      ��������£�ÿ�� block �Ĺ����ڴ��С��33
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// Ϊ block_reduce ׼���� shared memory
    __shared__ double sh_data[(BLOCK_SIZE + 1) * BLOCK_SIZE];  // 33 * 32 �������ڴ浥λ
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double tmp;
    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al, d_ac, d_ar;  // left  center  right

    // ����� i, j
    int ij_total = nx * ny;

    // �����ڿ��߳�, ���߳�ͬ�����䵱worker
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;

    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����



    // ��ȡ��ǰblock�е�line����
    int line_per_block;
    if (blockIdx.x == ij_total / 32) {
        line_per_block = ij_total % 32; // ���һ��block��line������Ϊ total % 32
    } else {
        line_per_block = 32; //  ����block��line����Ϊ 32
    }

    // ͬһ��block��ȫ���߳�Эͬ��block�����е�line�е�32�����ݵ����shared memory��
    // ���������ڵ�һ���ߵڶ����Ҫ��������
    // ���ȣ����Ǽ����һ�����ϵ��ܳ���nx����32�����������ǵ�һ�ֶ���һ��line���԰��˵���������֮������ݽ��ܱ�����
    int first_round_size = nx % 32;
    int round_times = nx / 32;  // �� i_round ����
    int flag = 1;



    __syncthreads();

    int i_round = 0;    // ��ǰ�������� idx

    // ����һ�ְ���1��2������ʱ���������Ƕ��صļ���ṹ�������Ҫ������Ĵ���
    // ���ǵļ���ṹ�� d_al d_ac �������洢��Ȼ���ȡa[i+1]��д��a[i]
    if (first_round_size == 1) {


//        DEBUG_PRINT("A-1", 0,0);
//        DEBUG_PRINT("A-1", 29,29);
//        DEBUG_PRINT("A-1", 30,30);
//        DEBUG_PRINT("A-1", 31,31);



        // ��һ��ֻ��һ��Ԫ��
        // ÿ���̼߳�¼�Լ��� d_al d_ac
        // ����������£�(ny,nz) �������̵߳ı�ʶ��
        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ32��
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // ������i_tid��i��j
            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
            int i_i = i_thread_global_id / ny;
            int i_j = i_thread_global_id % ny;
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ];

            __syncthreads();
        }
        // sh_data[tid_in_block + 1] = a[i * ny * nz + j * nz + tid_in_block];

        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        for (int i_i = 1; i_i < BLOCK_SIZE; ++i_i) {    // 31���������32����һ��
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // ��ǰ����µڶ���ֻ��д�� 31 ������� 32 ��
            // �������ǲ���Ҫд�ص���sh_data[0]����� tid_in_block ��Ϣ
//            if (tid_in_block == 0)
//                break;
            if (tid_in_block > 0) {
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
            }

            __syncthreads();
        }

        flag = -1;
        ++i_round; // ���ȣ����������������ִ���Ҫ����

    } else if (first_round_size == 2) {


//        DEBUG_PRINT("A-2", 0,0);
//        DEBUG_PRINT("A-2", 29,29);
//        DEBUG_PRINT("A-2", 30,30);
//        DEBUG_PRINT("A-2", 31,31);



        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ32��
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // ������i_tid��i��j
            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
            int i_i = i_thread_global_id / ny;
            int i_j = i_thread_global_id % ny;
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 2    ];

            __syncthreads();
        }

        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32�����
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // ��ǰ����µڶ���д�� 32 ��
            // ������i_tid��i��j
            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
            int i_i = i_thread_global_id / ny;
            int i_j = i_thread_global_id % ny;
            a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ

            __syncthreads();
        }

        flag = -1;
        ++i_round; // ���ȣ����������������ִ���Ҫ����

    } else if (first_round_size == 0) {

//        DEBUG_PRINT("A-0", 0,0);
//        DEBUG_PRINT("A-0", 29,29);
//        DEBUG_PRINT("A-0", 30,30);
//        DEBUG_PRINT("A-0", 31,31);

        first_round_size = 32;

        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ first_round_size - 2 ��
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // Ϊblock��ÿ��line��ȡ first_round_size - 2 ��
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < first_round_size - 2) {
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
            }

            __syncthreads();
        }                                                   //*** ��δ���û����



        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        // ֻ���� first_round_size - 2 ��
        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 ��
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** ��δ���û����




        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // Ϊblock��ÿ��lineд�� first_round_size - 2 ��
            if (tid_in_block < first_round_size - 2) {      // 0~3
                // ��ǰ����µ�һ��д�� first_round_size - 2 ��
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
            }

            __syncthreads();
        }

        flag = 1;
        i_round = 1;
        first_round_size = 0; // ��Ҫ���û� 0

    } else { // first_round_size Ϊ 3 4 5 6...

//        DEBUG_PRINT("A-3", 0,0);
//        DEBUG_PRINT("A-3", 29,29);
//        DEBUG_PRINT("A-3", 30,30);
//        DEBUG_PRINT("A-3", 31,31);

        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ first_round_size - 2 ��
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // Ϊblock��ÿ��line��ȡ first_round_size - 2 ��
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < first_round_size - 2) {
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
            }

            __syncthreads();
        }                                                   //*** ��δ���û����



        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        // ֻ���� first_round_size - 2 ��
        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 ��
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** ��δ���û����




        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // Ϊblock��ÿ��lineд�� first_round_size - 2 ��
            if (tid_in_block < first_round_size - 2) {      // 0~3
                // ��ǰ����µ�һ��д�� first_round_size - 2 ��
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
            }

            __syncthreads();
        }

        flag = 1;
        i_round = 0;// ���������������ִ���Ҫһ�֣���˲���Ҫ�ı� i_round ֵ
    }                                                       //*** ��δ���û����

    __syncthreads();



    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // ������ʼ
    // Ȼ����� round_times �ֵ���
    // ��ʱ d_al  d_ac �Ѿ�׼������
    // ��ÿһ���ж�ȡ 32 �д�� 32 ��
    // 1. ����shared memory; 2. ���շ������; 3. ����д��ȫ���ڴ�
    for ( ; i_round < round_times; ++i_round) {
        if (flag == 1) {        /// flag = 1 ==> ����
            // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
            // Ϊÿ��line��ȡ32��
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + tid_in_block + 1] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block];

                __syncthreads();
            }

            // ÿ���߳���shared memory�ж��������Լ��Ĵ��м���
            for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32�����
                d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
                // ����ֲ����
                tmp = (d_al + d_ar) * 0.5;
                // д��shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // ���� d_al �� d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // ���е�����line_per_block����д���ڴ�
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
                // ��ǰ����µڶ���д�� 32 ��
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ

                __syncthreads();
            }                                               //*** ��δ���û����

        } else {         /// flag == -1   ==> ����
            // �Գ�����

            // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
            // Ϊÿ��line��ȡ32��
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {  // line �Ķ�ȡ˳����Ҫ������Ҫ�ı�
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block]; // �������Ҳ�a��idx���ֲ���

                __syncthreads();
            }

            // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
            // ֻ���� first_round_size - 2 ��
            for (int i_i = BLOCK_SIZE; i_i > 0; --i_i) {    // 32 �����ڴ����Ϊ��׼
                d_ar = sh_data[tid_in_block * 33 + i_i - 1]; // ��ʱ flag ���� 1
                // ����ֲ����
                tmp = (d_al + d_ar) * 0.5;
                // д��shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // ���� d_al �� d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // ���е�����line_per_block����д���ڴ�
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д�أ�line �Ķ�ȡ˳����Ҫ������Ҫ�ı�
                // д�� 32 ��
                // ������i_tid��i��j
                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
                int i_i = i_thread_global_id / ny;
                int i_j = i_thread_global_id % ny;
                a[    i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // ����a��idx���ֲ���

                __syncthreads();
            }                                               //*** ��δ���û����

        }
        flag *= -1;
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    /// ����Ϊֹ�������ڴ���Ϊ������м�ý��������Ѿ�������
    //���������Ѿ�������ļ������洢���� a[] ȫ���ڴ��У���˹����ڴ����ڿ����˻���һ����֮ǰ��ܶ౶(32*33 vs 32)��������ִ��֮ǰ�Ĺ��ܵĹ����ڴ�
    // ���Ƚ������̵߳� s_eps д�빲���ڴ�

    if (thread_global_id >= ij_total)  // ����̴߳���ά�������������Ϊ 0.
        sh_data[tid_in_block] = 0.;
    else
        sh_data[tid_in_block] = s_eps; // �� 32 ��: 0~32

//    sh_data[tid_in_block] = s_eps;    // ������ǰ��Ĵ��м�������У� >- ij_total ���߳�Ҳִ���˼��㣬��� s_eps ������
    __syncthreads();

    // ʹ�� ���� ���ڹ�Լ
    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {     // blockDim = 32; // blockDim ������ 2 �ı���������
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
    if (tid_in_block == 0) // д��block_reduce���
        d_eps[blockIdx.x] = sh_data[0];
}
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#else






// ���ֹ�Լ
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// Ϊ block_reduce ׼���� shared memory
    __shared__ double sh_data[BLOCK_SIZE];
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double tmp;
    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al, d_ac, d_ar;  // left  center  right

    // ����� i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
        return;

    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // ÿ���̴߳��д���һ���� z ����������
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // ����ֲ����ֵ
        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // ���� d_al �� d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
    sh_data[tid_in_block] = s_eps;
    __syncthreads();

    // ʹ�� ���� ���ڹ�Լ
    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
    if (tid_in_block == 0) // д��block_reduce���
        d_eps[blockIdx.x] = sh_data[0];
}
#endif


//// ���ֹ�Լ   clock debug ��
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    unsigned long long time_checker_1 = clock64();
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
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
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    unsigned long long time_checker_3 = clock64();
//    __syncthreads();
//
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//
//
//    unsigned long long time_checker_4 = clock64();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
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
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
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


//// warp �ڹ�Լ�ĸ�������
//__inline__ __device__
//double warp_reduce_max(double val) {
//    // ʹ�� warp shuffle ��������Լ
//    for (int offset = 16; offset > 0; offset /= 2)
//        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
//    return val;
//}
//// warp �ڹ�Լ
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
////        tmp = (d_al + d_ar) / 2.0;   // �Է����ϱ������������Ż�
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // ʹ�� warp �ڹ�Լ
//    double warp_max = warp_reduce_max(s_eps);
//    if (threadIdx.x == 0)
//        d_eps[blockIdx.x] = warp_max;
//
////    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
////    sh_data[tid_in_block] = s_eps;
////    __syncthreads();
////
////    // ʹ�� ���� ���ڹ�Լ
////    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
////    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
////    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
////    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
////    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
////    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
////        if (tid_in_block < s)
////            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
////        __syncthreads();
////    }
////    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
////    if (tid_in_block == 0) // д��block_reduce���
////        d_eps[blockIdx.x] = sh_data[0];
//}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // �� i ����
//    int tid = threadIdx.x; // �� idx ������
//    int idx_thread_global = tid + i * blockDim.x; // ����� A ���飨�� block_max���ĵ�ַλ��
//    // ����������ֵ����shared mem��
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // ��ÿ���̲߳�����ȫ���߳�
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // �������һ�� block �ĳ���������������һ����Сֵ
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // ���ַ����ڹ�Լ
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // ����Ҫ����MAX������max
//        }
//        __syncthreads();
//    }
//    // �� 0 ���߳̽���ǰ���ֵд��ȫ���ڴ�
//    if (tid == 0) {
//        B[i] = sdata[0];
//    }
//}


void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared Mem per Block: %d bytes\n", prop.sharedMemPerBlock);  // ÿ�� block ����
    printf("Shared Mem per SM   : %d bytes\n", prop.sharedMemPerMultiprocessor); // ÿ�� SM ����
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

    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // ÿ�����ڵ��߳�����
    int total_lines_x = ny * nz;    // x �����������
    int total_lines_y = nx * nz;    // y �����������
    int total_lines_z = nx * ny;    // z �����������
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // ���ÿ��block�����ֵ

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // Ϊ CUB Ԥ�����ڴ�
    // d_eps: �����������豸�ϵ�ָ��
    int num_items = blocks_z;   // ����block��������

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // ��ʼ�����ֵΪ0  (ʹ��block_reduce + global_reduceʱ����Ҫ��ʼ�����)
//        double h_eps = 0.0;
//        // ***** ��ʱ���� *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// ���ֹ�Լ��Ҫ shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double) >>>(d_A, d_eps); // ÿ������ 32 ���߳�  ÿ��block 32��double
//        //// warp �ڹ�Լ����Ҫ shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // ÿ������ 32 ���߳�
        //// ���м������ݽ� shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * (threads_per_block + 1) * sizeof(double) >>>(d_A, d_eps); // ÿ������ 32 ���߳�  ÿ���߳� 33��double
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // ÿ������ 32 ���߳�  ÿ���߳� 33��double


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // ��ʱd_eps�д洢ÿ��block�ڵ����ֵ
        // global_reduce 2  thrust  // �ҳ����㣡����
        // ����һ�� Thrust device_ptr ��װ����ָ�� d_eps��device �ϵ� block �ֲ����������飩
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
        // ʹ�� thrust::reduce �ҳ����ֵ����ʼֵ������ -�ޣ�
        eps = thrust::reduce(dev_ptr, dev_ptr + num_items, 0., thrust::maximum<double>());

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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

// ���ڿ���ɨ����Զ����������ע��Ҫ��ò������н����
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

#define BLOCK_SIZE 32       // Ĭ�� 32, Ŀǰ 64 �������

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


// �� CPU �Ͻ��г�ʼ��
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

//// �� CUB �� BlockScan ������и��µĺ˺����汾      (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���) (ÿ��block��1���߳�)
//  (��������ȷ) (�ٶȱȽ���)
__global__ void compute_i_direction_kernel_cub(double *a) {
    // ÿ���鴦��һ�� x ����������ߣ���������� blockIdx.y, blockIdx.z �ֱ��Ӧ j, k ����
    // �ڵ�ǰ�����, ÿ����ֻ��Ҫһ���̼߳���
    int j = blockIdx.y;
    int k = blockIdx.z;

    // ���и���
    if (threadIdx.x == 0) { // ֻ��һ���߳�ִ�и��£����⾺��
        for (int i = 1; i < nx - 1; i++) {
            a[i * ny * nz + j * nz + k] = (a[(i - 1) * ny * nz + j * nz + k] + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        }
    }
}
// �� CUB �� BlockScan ������и��µĺ˺����汾   (���ô��м���) (���ǲ������ݴ洢�� �����ڴ���)(ÿ��block��32���߳�)
//  (��������ȷ) (Ŀǰ���)
__global__ void compute_i_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;

    double d_al; // �����ּ�������Ϊ���ּ���������

    // ����� j, k
    int jk_total = ny * nz;
    if (thread_global_id >= jk_total)   // �б�Ҫ������
        return;

    int j = thread_global_id / nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� j k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    d_al = a[(1 - 1) * ny * nz + j * nz + k];

    // ÿ���̴߳��д���һ���� x ����������
    for (int i = 1; i < nx - 1; ++i) {
        d_al = (d_al + a[(i + 1) * ny * nz + j * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

// ͬ��, ���� compute_i_direction_kernel_32_thread д�� compute_j_direction_kernel_32_thread
__global__ void compute_j_direction_kernel_32_thread(double *a) {
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double d_al; // �����ּ�������Ϊ���ּ���������

    // ����� i, k
    int ik_total = nx * nz;
    if (thread_global_id >= ik_total)  // ����п��߳�, ֱ�ӷ��ؼ���
        return;

    int i = thread_global_id / nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int k = thread_global_id % nz;  // ����ȷ�� i k ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����

    d_al = a[i * ny * nz + (1 - 1) * nz + k];

    // ÿ���̴߳��д���һ���� y ����������
    for (int j = 1; j < ny - 1; ++j) {
        d_al = (d_al + a[i * ny * nz + (j + 1) * nz + k]) * 0.5;
        a[i * ny * nz + j * nz + k] = d_al;
    }
}

//// ���ֹ�Լ -- �� tmp  (������ͨ���ֹ�Լ)
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    register double s_eps = 0.;  // ��ʼ���ֲ����
//    register double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    if (thread_global_id >= nx * ny)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
//        a[i * ny * nz + j * nz + k] = (d_al + d_ar) * 0.5;
//        s_eps = MAX(s_eps, fabs(d_ac - (d_al + d_ar) * 0.5));
//
//        // ���� d_al �� d_ac
//        d_al = (d_al + d_ar) * 0.5;
//        d_ac = d_ar;
//    }
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
//        d_eps[blockIdx.x] = sh_data[0];
//}


// �Զ����ӡ debug
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
// �ڶ��ֹ�Լ�����Ļ����ϣ�ʹ�ò��м����ڴ�ķ�ʽ      ��������£�ÿ�� block �Ĺ����ڴ��С��33
// �Ż�ǰ��1.067��
// �������Ż����ܣ�����ɶ���Ϊ 0�����ʺ������˺�ChatGPT�Ķ�
// ����ʱ���ޱ仯 ... ��������Ƚ����˳����׶��� ...    1.067��  1.068��
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// Ϊ block_reduce ׼���� shared memory
    __shared__ double sh_data[(BLOCK_SIZE + 1) * BLOCK_SIZE];  // 33 * 32 �������ڴ浥λ
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int block_Idx_mul_Dim = blockIdx.x * blockDim.x; // ��ǰ����� blockIdx.x * blockDim.x
    int thread_global_id = block_Idx_mul_Dim + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double tmp;
    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al, d_ac, d_ar;  // left  center  right

    // ����� i, j
    int ij_total = nx * ny;


    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����



    // ��ȡ��ǰblock�е�line����
    int line_per_block;
    if (blockIdx.x == ij_total / 32) {
        line_per_block = ij_total % 32; // ���һ��block��line������Ϊ total % 32
    } else {
        line_per_block = 32; //  ����block��line����Ϊ 32
    }

    // ͬһ��block��ȫ���߳�Эͬ��block�����е�line�е�32�����ݵ����shared memory��
    // ���������ڵ�һ���ߵڶ����Ҫ��������
    // ���ȣ����Ǽ����һ�����ϵ��ܳ���nx����32�����������ǵ�һ�ֶ���һ��line���԰��˵���������֮������ݽ��ܱ�����
    int first_round_size = nx % 32;
    int round_times = nx / 32;  // �� i_round ����
    int flag = 1;

//    DEBUG_PRINT("A-0", 0,0);
//    DEBUG_PRINT("A-0", 29,29);
//    DEBUG_PRINT("A-0", 30,30);
//    DEBUG_PRINT("A-0", 31,31);

    __syncthreads();

    int i_round = 0;    // ��ǰ�������� idx

    // ����һ�ְ���1��2������ʱ���������Ƕ��صļ���ṹ�������Ҫ������Ĵ���
    // ���ǵļ���ṹ�� d_al d_ac �������洢��Ȼ���ȡa[i+1]��д��a[i]
    if (first_round_size == 1) {

        // ��һ��ֻ��һ��Ԫ��
        // ÿ���̼߳�¼�Լ��� d_al d_ac
        // ����������£�(ny,nz) �������̵߳ı�ʶ��
        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ32��
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // ������i_tid��i��j
//            int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ];

//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1    ];
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ];

            __syncthreads();
        }

        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        for (int i_i = 1; i_i < BLOCK_SIZE; ++i_i) {    // 31���������32����һ��
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // ��ǰ����µڶ���ֻ��д�� 31 ������� 32 ��
            // �������ǲ���Ҫд�ص���sh_data[0]����� tid_in_block ��Ϣ
//            if (tid_in_block == 0)
//                break;
            if (tid_in_block > 0) {
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
                a[(block_Idx_mul_Dim + i_tid) * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
            }

            __syncthreads();
        }

        flag = -1;
        ++i_round; // ���ȣ����������������ִ���Ҫ����

    } else if (first_round_size == 2) {



        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ32��
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // ������i_tid��i��j
//            int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;

//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 2    ];
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 2    ];

            __syncthreads();
        }

        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32�����
            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }

        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // ��ǰ����µڶ���д�� 32 ��
            // ������i_tid��i��j
//            int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;

//            a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
            a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ

            __syncthreads();
        }

        flag = -1;
        ++i_round; // ���ȣ����������������ִ���Ҫ����

    } else if (first_round_size == 0) {



//        first_round_size = 32;    // �Ѿ�����������е� first_round_size �滻Ϊ�� 32������ first_round_size - 2 ��Ϊ�� 30

        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ first_round_size - 2 ��
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // Ϊblock��ÿ��line��ȡ first_round_size - 2 ��
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < 30) {
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;

//                sh_data[i_tid * 33 + (30 - tid_in_block - 1)] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 2];
                sh_data[i_tid * 33 + (30 - tid_in_block - 1)] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 2    ];

            }

            __syncthreads();
        }                                                   //*** ��δ���û����



        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        // ֻ���� first_round_size - 2 ��
        for (int i_i = 30; i_i > 0; --i_i) {    // first_round_size - 2 ��
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** ��δ���û����




        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // Ϊblock��ÿ��lineд�� first_round_size - 2 ��
            if (tid_in_block < 30) {      // 0~3
                // ��ǰ����µ�һ��д�� first_round_size - 2 ��
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (30 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
                a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + (30 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ

            }

            __syncthreads();
        }

        flag = 1;
        i_round = 1;
//        first_round_size = 0; // ��Ҫ���û� 0

    } else { // first_round_size Ϊ 3 4 5 6...

//        DEBUG_PRINT("A-3", 0,0);
//        DEBUG_PRINT("A-3", 29,29);
//        DEBUG_PRINT("A-3", 30,30);
//        DEBUG_PRINT("A-3", 31,31);

        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]

        // first_round_size - 2

        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
        // Ϊÿ��line��ȡ first_round_size - 2 ��
//        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // Ϊblock��ÿ��line��ȡ first_round_size - 2 ��
//            if (tid_in_block >= first_round_size - 2)
//                break;
            if (tid_in_block < first_round_size - 2) {
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;
//
//                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 2];
                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 2    ];

            }

            __syncthreads();
        }                                                   //*** ��δ���û����



        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
        // ֻ���� first_round_size - 2 ��
        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 ��
            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
            // ����ֲ����
            tmp = (d_al + d_ar) * 0.5;
            // д��shared memory
            sh_data[tid_in_block * 33 + i_i] = tmp;
            s_eps = MAX(s_eps, fabs(d_ac - tmp));

            // ���� d_al �� d_ac
            d_al = tmp;
            d_ac = d_ar;
        }                                                   //*** ��δ���û����




        // ���е�����line_per_block����д���ڴ�
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
            // Ϊblock��ÿ��lineд�� first_round_size - 2 ��
            if (tid_in_block < first_round_size - 2) {      // 0~3
                // ��ǰ����µ�һ��д�� first_round_size - 2 ��
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
                a[    (block_Idx_mul_Dim + i_tid) * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ

            }

            __syncthreads();
        }

        flag = 1;
        i_round = 0;// ���������������ִ���Ҫһ�֣���˲���Ҫ�ı� i_round ֵ
    }                                                       //*** ��δ���û����

    __syncthreads();



    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // ������ʼ
    // Ȼ����� round_times �ֵ���
    // ��ʱ d_al  d_ac �Ѿ�׼������
    // ��ÿһ���ж�ȡ 32 �д�� 32 ��
    // 1. ����shared memory; 2. ���շ������; 3. ����д��ȫ���ڴ�
    for ( ; i_round < round_times; ++i_round) {
        if (flag == 1) {        /// flag = 1 ==> ����
            // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
            // Ϊÿ��line��ȡ32��
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;
//
//                sh_data[i_tid * 33 + tid_in_block + 1] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block];
                sh_data[i_tid * 33 + tid_in_block + 1] = a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block    ];

                __syncthreads();
            }

            // ÿ���߳���shared memory�ж��������Լ��Ĵ��м���
            for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32�����
                d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
                // ����ֲ����
                tmp = (d_al + d_ar) * 0.5;
                // д��shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // ���� d_al �� d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // ���е�����line_per_block����д���ڴ�
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
                // ��ǰ����µڶ���д�� 32 ��
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block - 1] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
                a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ

                __syncthreads();
            }                                               //*** ��δ���û����


            flag = -1;
        } else {         /// flag == -1   ==> ����
            // �Գ�����

            // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
            // Ϊÿ��line��ȡ32��
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {  // line �Ķ�ȡ˳����Ҫ������Ҫ�ı�
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block]; // �������Ҳ�a��idx���ֲ���
                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block    ]; // �������Ҳ�a��idx���ֲ���

                __syncthreads();
            }

            // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
            // ֻ���� first_round_size - 2 ��
            for (int i_i = BLOCK_SIZE; i_i > 0; --i_i) {    // 32 �����ڴ����Ϊ��׼
                d_ar = sh_data[tid_in_block * 33 + i_i - 1]; // ��ʱ flag ���� 1
                // ����ֲ����
                tmp = (d_al + d_ar) * 0.5;
                // д��shared memory
                sh_data[tid_in_block * 33 + i_i] = tmp;
                s_eps = MAX(s_eps, fabs(d_ac - tmp));

                // ���� d_al �� d_ac
                d_al = tmp;
                d_ac = d_ar;
            }

            // ���е�����line_per_block����д���ڴ�
            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д�أ�line �Ķ�ȡ˳����Ҫ������Ҫ�ı�
                // д�� 32 ��
                // ������i_tid��i��j
//                int i_thread_global_id = block_Idx_mul_Dim + i_tid;
//                int i_i = (block_Idx_mul_Dim + i_tid) / ny;
//                int i_j = (block_Idx_mul_Dim + i_tid) % ny;

//                a[    ((block_Idx_mul_Dim + i_tid) / ny) * ny * nz + ((block_Idx_mul_Dim + i_tid) % ny) * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // ����a��idx���ֲ���
                a[    (block_Idx_mul_Dim + i_tid) * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // ����a��idx���ֲ���

                __syncthreads();
            }                                               //*** ��δ���û����


            flag = 1;
        }
//        flag *= -1;   // ��Ȼ��ֱ�Ӹ�ֵ�������
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    /// ����Ϊֹ�������ڴ���Ϊ������м�ý��������Ѿ�������
    //���������Ѿ�������ļ������洢���� a[] ȫ���ڴ��У���˹����ڴ����ڿ����˻���һ����֮ǰ��ܶ౶(32*33 vs 32)��������ִ��֮ǰ�Ĺ��ܵĹ����ڴ�
    // ���Ƚ������̵߳� s_eps д�빲���ڴ�

    if (thread_global_id >= ij_total)  // ����̴߳���ά�������������Ϊ 0.
        sh_data[tid_in_block] = 0.;
    else
        sh_data[tid_in_block] = s_eps; // �� 32 ��: 0~32

//    sh_data[tid_in_block] = s_eps;    // ������ǰ��Ĵ��м�������У� >- ij_total ���߳�Ҳִ���˼��㣬��� s_eps ������
    __syncthreads();

    // ʹ�� ���� ���ڹ�Լ
    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {     // blockDim = 32; // blockDim ������ 2 �ı���������
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
    if (tid_in_block == 0) // д��block_reduce���
        d_eps[blockIdx.x] = sh_data[0];
}







// ���� ���м����ڴ��㷨
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[(BLOCK_SIZE + 1) * BLOCK_SIZE];  // 33 * 32 �������ڴ浥λ
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//
//    // �����ڿ��߳�, ���߳�ͬ�����䵱worker
////    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
////        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//
//    // ��ȡ��ǰblock�е�line����
//    int line_per_block;
//    if (blockIdx.x == ij_total / 32) {
//        line_per_block = ij_total % 32; // ���һ��block��line������Ϊ total % 32
//    } else {
//        line_per_block = 32; //  ����block��line����Ϊ 32
//    }
//
//    // ͬһ��block��ȫ���߳�Эͬ��block�����е�line�е�32�����ݵ����shared memory��
//    // ���������ڵ�һ���ߵڶ����Ҫ��������
//    // ���ȣ����Ǽ����һ�����ϵ��ܳ���nx����32�����������ǵ�һ�ֶ���һ��line���԰��˵���������֮������ݽ��ܱ�����
//    int first_round_size = nx % 32;
//    int round_times = nx / 32;  // �� i_round ����
//    int flag = 1;
//
//
//
//    __syncthreads();
//
//    int i_round = 0;    // ��ǰ�������� idx
//
//    // ����һ�ְ���1��2������ʱ���������Ƕ��صļ���ṹ�������Ҫ������Ĵ���
//    // ���ǵļ���ṹ�� d_al d_ac �������洢��Ȼ���ȡa[i+1]��д��a[i]
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
//        // ��һ��ֻ��һ��Ԫ��
//        // ÿ���̼߳�¼�Լ��� d_al d_ac
//        // ����������£�(ny,nz) �������̵߳ı�ʶ��
//        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
//        // Ϊÿ��line��ȡ32��
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // ������i_tid��i��j
//            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ];
//
//            __syncthreads();
//        }
//        // sh_data[tid_in_block + 1] = a[i * ny * nz + j * nz + tid_in_block];
//
//        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
//        for (int i_i = 1; i_i < BLOCK_SIZE; ++i_i) {    // 31���������32����һ��
//            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
//            // ����ֲ����
//            tmp = (d_al + d_ar) * 0.5;
//            // д��shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // ���� d_al �� d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }
//
//        // ���е�����line_per_block����д���ڴ�
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
//            // ��ǰ����µڶ���ֻ��д�� 31 ������� 32 ��
//            // �������ǲ���Ҫд�ص���sh_data[0]����� tid_in_block ��Ϣ
////            if (tid_in_block == 0)
////                break;
//            if (tid_in_block > 0) {
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + tid_in_block] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
//            }
//
//            __syncthreads();
//        }
//
//        flag = -1;
//        ++i_round; // ���ȣ����������������ִ���Ҫ����
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
//        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
//        // Ϊÿ��line��ȡ32��
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // ������i_tid��i��j
//            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 2    ];
//
//            __syncthreads();
//        }
//
//        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
//        for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32�����
//            d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
//            // ����ֲ����
//            tmp = (d_al + d_ar) * 0.5;
//            // д��shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // ���� d_al �� d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }
//
//        // ���е�����line_per_block����д���ڴ�
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
//            // ��ǰ����µڶ���д�� 32 ��
//            // ������i_tid��i��j
//            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//            int i_i = i_thread_global_id / ny;
//            int i_j = i_thread_global_id % ny;
//            a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
//
//            __syncthreads();
//        }
//
//        flag = -1;
//        ++i_round; // ���ȣ����������������ִ���Ҫ����
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
//        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // first_round_size - 2
//
//        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
//        // Ϊÿ��line��ȡ first_round_size - 2 ��
////        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // Ϊblock��ÿ��line��ȡ first_round_size - 2 ��
////            if (tid_in_block >= first_round_size - 2)
////                break;
//            if (tid_in_block < first_round_size - 2) {
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
//            }
//
//            __syncthreads();
//        }                                                   //*** ��δ���û����
//
//
//
//        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
//        // ֻ���� first_round_size - 2 ��
//        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 ��
//            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
//            // ����ֲ����
//            tmp = (d_al + d_ar) * 0.5;
//            // д��shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // ���� d_al �� d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }                                                   //*** ��δ���û����
//
//
//
//
//        // ���е�����line_per_block����д���ڴ�
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
//            // Ϊblock��ÿ��lineд�� first_round_size - 2 ��
//            if (tid_in_block < first_round_size - 2) {      // 0~3
//                // ��ǰ����µ�һ��д�� first_round_size - 2 ��
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
//            }
//
//            __syncthreads();
//        }
//
//        flag = 1;
//        i_round = 1;
//        first_round_size = 0; // ��Ҫ���û� 0
//
//    } else { // first_round_size Ϊ 3 4 5 6...
//
////        DEBUG_PRINT("A-3", 0,0);
////        DEBUG_PRINT("A-3", 29,29);
////        DEBUG_PRINT("A-3", 30,30);
////        DEBUG_PRINT("A-3", 31,31);
//
//        d_al = a[i * ny * nz + j * nz + 0]; // �����Ԫ��a[0]
//        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]
//
//        // first_round_size - 2
//
//        // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
//        // Ϊÿ��line��ȡ first_round_size - 2 ��
////        for (int i_tid = first_round_size - 2; i_tid > 0; --i_tid) {
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//            // Ϊblock��ÿ��line��ȡ first_round_size - 2 ��
////            if (tid_in_block >= first_round_size - 2)
////                break;
//            if (tid_in_block < first_round_size - 2) {
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + tid_in_block + 2];
//            }
//
//            __syncthreads();
//        }                                                   //*** ��δ���û����
//
//
//
//        // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
//        // ֻ���� first_round_size - 2 ��
//        for (int i_i = first_round_size - 2; i_i > 0; --i_i) {    // first_round_size - 2 ��
//            d_ar = sh_data[tid_in_block * 33 + i_i - 1];
//            // ����ֲ����
//            tmp = (d_al + d_ar) * 0.5;
//            // д��shared memory
//            sh_data[tid_in_block * 33 + i_i] = tmp;
//            s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//            // ���� d_al �� d_ac
//            d_al = tmp;
//            d_ac = d_ar;
//        }                                                   //*** ��δ���û����
//
//
//
//
//        // ���е�����line_per_block����д���ڴ�
//        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
//            // Ϊblock��ÿ��lineд�� first_round_size - 2 ��
//            if (tid_in_block < first_round_size - 2) {      // 0~3
//                // ��ǰ����µ�һ��д�� first_round_size - 2 ��
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + tid_in_block + 1] = sh_data[i_tid * 33 + (first_round_size - 2 - tid_in_block)]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
//            }
//
//            __syncthreads();
//        }
//
//        flag = 1;
//        i_round = 0;// ���������������ִ���Ҫһ�֣���˲���Ҫ�ı� i_round ֵ
//    }                                                       //*** ��δ���û����
//
//    __syncthreads();
//
//
//
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    // ������ʼ
//    // Ȼ����� round_times �ֵ���
//    // ��ʱ d_al  d_ac �Ѿ�׼������
//    // ��ÿһ���ж�ȡ 32 �д�� 32 ��
//    // 1. ����shared memory; 2. ���շ������; 3. ����д��ȫ���ڴ�
//    for ( ; i_round < round_times; ++i_round) {
//        if (flag == 1) {        /// flag = 1 ==> ����
//            // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
//            // Ϊÿ��line��ȡ32��
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + tid_in_block + 1] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block];
//
//                __syncthreads();
//            }
//
//            // ÿ���߳���shared memory�ж��������Լ��Ĵ��м���
//            for (int i_i = 0; i_i < BLOCK_SIZE; ++i_i) {    // 32�����
//                d_ar = sh_data[tid_in_block * 33 + i_i + 1]; // ��ʱ flag ���� 1
//                // ����ֲ����
//                tmp = (d_al + d_ar) * 0.5;
//                // д��shared memory
//                sh_data[tid_in_block * 33 + i_i] = tmp;
//                s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//                // ���� d_al �� d_ac
//                d_al = tmp;
//                d_ac = d_ar;
//            }
//
//            // ���е�����line_per_block����д���ڴ�
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д��
//                // ��ǰ����µڶ���д�� 32 ��
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1] = sh_data[i_tid * 33 + tid_in_block]; // ��0���Ѿ������̷߳��ض����ᱻ��ȡ
//
//                __syncthreads();
//            }                                               //*** ��δ���û����
//
//        } else {         /// flag == -1   ==> ����
//            // �Գ�����
//
//            // ���е�����line_per_block���Ա���Ϊblock��ÿ���߶�����ø��ԵĹ����ڴ�
//            // Ϊÿ��line��ȡ32��
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {  // line �Ķ�ȡ˳����Ҫ������Ҫ�ı�
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block - 1)] = a[i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block]; // �������Ҳ�a��idx���ֲ���
//
//                __syncthreads();
//            }
//
//            // ÿ���߳̿�ʼ��shared memory�ж��������Լ��Ĵ��м���
//            // ֻ���� first_round_size - 2 ��
//            for (int i_i = BLOCK_SIZE; i_i > 0; --i_i) {    // 32 �����ڴ����Ϊ��׼
//                d_ar = sh_data[tid_in_block * 33 + i_i - 1]; // ��ʱ flag ���� 1
//                // ����ֲ����
//                tmp = (d_al + d_ar) * 0.5;
//                // д��shared memory
//                sh_data[tid_in_block * 33 + i_i] = tmp;
//                s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//                // ���� d_al �� d_ac
//                d_al = tmp;
//                d_ac = d_ar;
//            }
//
//            // ���е�����line_per_block����д���ڴ�
//            for (int i_tid = 0; i_tid < line_per_block; ++i_tid) { // ��block������lineִ��д�أ�line �Ķ�ȡ˳����Ҫ������Ҫ�ı�
//                // д�� 32 ��
//                // ������i_tid��i��j
//                int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
//                int i_i = i_thread_global_id / ny;
//                int i_j = i_thread_global_id % ny;
//                a[    i_i * ny * nz + i_j * nz + i_round * 32 + first_round_size + tid_in_block - 1    ] = sh_data[i_tid * 33 + (BLOCK_SIZE - tid_in_block )]; // ����a��idx���ֲ���
//
//                __syncthreads();
//            }                                               //*** ��δ���û����
//
//        }
//        flag *= -1;
//    }
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//    /// ����Ϊֹ�������ڴ���Ϊ������м�ý��������Ѿ�������
//    //���������Ѿ�������ļ������洢���� a[] ȫ���ڴ��У���˹����ڴ����ڿ����˻���һ����֮ǰ��ܶ౶(32*33 vs 32)��������ִ��֮ǰ�Ĺ��ܵĹ����ڴ�
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//
//    if (thread_global_id >= ij_total)  // ����̴߳���ά�������������Ϊ 0.
//        sh_data[tid_in_block] = 0.;
//    else
//        sh_data[tid_in_block] = s_eps; // �� 32 ��: 0~32
//
////    sh_data[tid_in_block] = s_eps;    // ������ǰ��Ĵ��м�������У� >- ij_total ���߳�Ҳִ���˼��㣬��� s_eps ������
//    __syncthreads();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {     // blockDim = 32; // blockDim ������ 2 �ı���������
//        if (tid_in_block < s)
//            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
//        __syncthreads();
//    }
//
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
//        d_eps[blockIdx.x] = sh_data[0];
//}



//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#else






// ���ֹ�Լ
__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
    //// Ϊ block_reduce ׼���� shared memory
    __shared__ double sh_data[BLOCK_SIZE];
    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����

    double tmp;
    double s_eps = 0.;  // ��ʼ���ֲ����
    double d_al, d_ac, d_ar;  // left  center  right

    // ����� i, j
    int ij_total = nx * ny;
    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
        return;

    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����


    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1

    // ÿ���̴߳��д���һ���� z ����������
    for (int k = 1; k < nz - 1; ++k) {
        d_ar = a[i * ny * nz + j * nz + (k + 1)];
        // ����ֲ����ֵ
        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
        a[i * ny * nz + j * nz + k] = tmp;
        s_eps = MAX(s_eps, fabs(d_ac - tmp));

        // ���� d_al �� d_ac
        d_al = tmp;
        d_ac = d_ar;
    }

    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
    sh_data[tid_in_block] = s_eps;
    __syncthreads();

    // ʹ�� ���� ���ڹ�Լ
    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
    if (tid_in_block == 0) // д��block_reduce���
        d_eps[blockIdx.x] = sh_data[0];
}
#endif


//// ���ֹ�Լ   clock debug ��
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    //// Ϊ block_reduce ׼���� shared memory
//    __shared__ double sh_data[BLOCK_SIZE];
//
//    unsigned long long time_checker_1 = clock64();
//
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
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
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    unsigned long long time_checker_3 = clock64();
//    __syncthreads();
//
//
//    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
//    sh_data[tid_in_block] = s_eps;
//    __syncthreads();
//
//
//
//    unsigned long long time_checker_4 = clock64();
//
//    // ʹ�� ���� ���ڹ�Լ
//    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
//    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
//    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
//    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
//    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
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
//    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
//    if (tid_in_block == 0) // д��block_reduce���
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


//// warp �ڹ�Լ�ĸ�������
//__inline__ __device__
//double warp_reduce_max(double val) {
//    // ʹ�� warp shuffle ��������Լ
//    for (int offset = 16; offset > 0; offset /= 2)
//        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
//    return val;
//}
//// warp �ڹ�Լ
//__global__ void compute_k_direction_kernel_32_thread(double *a, double *d_eps) {
//    int tid_in_block = threadIdx.x;     // ��������ʹ�� 1 ά block, ���ֻ�� x ����
//    int thread_global_id = blockIdx.x * blockDim.x + tid_in_block;  // ͬ��, ʹ�� 1 ά grid, ֻ�� x ����
//
//    double tmp;
//    double s_eps = 0.;  // ��ʼ���ֲ����
//    double d_al, d_ac, d_ar;  // left  center  right
//
//    // ����� i, j
//    int ij_total = nx * ny;
//    if (thread_global_id >= ij_total)  // ����п��߳�, ֱ�ӷ��ؼ��� (���ҵĵ�����ȥ���������, ��������ʱ��û���κ�����)
//        return;
//
//    // ��Ϊ���� ij_total = nx * ny, ���Ӧ���� /ny; %ny
//    int i = thread_global_id / ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//    int j = thread_global_id % ny;  // ����ȷ�� i j ˳�򲻻�Ӱ����, ���ǳ�Ӱ������ٶȣ�����
//
//
//    d_al = a[i * ny * nz + j * nz + 0];   // k = 0
//    d_ac = a[i * ny * nz + j * nz + 1];   // k = 1
//
//    // ÿ���̴߳��д���һ���� z ����������
//    for (int k = 1; k < nz - 1; ++k) {
//        d_ar = a[i * ny * nz + j * nz + (k + 1)];
//        // ����ֲ����ֵ
////        tmp = (d_al + d_ar) / 2.0;   // �Է����ϱ������������Ż�
//        tmp = (d_al + d_ar) * 0.5;  // �ֶ��Ż�
//        a[i * ny * nz + j * nz + k] = tmp;
//        s_eps = MAX(s_eps, fabs(d_ac - tmp));
//
//        // ���� d_al �� d_ac
//        d_al = tmp;
//        d_ac = d_ar;
//    }
//
//    // ʹ�� warp �ڹ�Լ
//    double warp_max = warp_reduce_max(s_eps);
//    if (threadIdx.x == 0)
//        d_eps[blockIdx.x] = warp_max;
//
////    // ���Ƚ������̵߳� s_eps д�빲���ڴ�
////    sh_data[tid_in_block] = s_eps;
////    __syncthreads();
////
////    // ʹ�� ���� ���ڹ�Լ
////    // ���ֲ����洢��ȫ���ڴ���ȥ   // ԭ�Ӳ������ܽϵ�
////    // ̫��, ���ǽ����Ϊ 1. block �ڹ�Լ(block_reduce); 2. ȫ���ڴ��ڹ�Լ(global_reduce)
////    //atomicMax((unsigned long long int *) d_eps, (unsigned long long)__double_as_longlong(s_eps)); // һ�����⼼��, �������ڷǸ�������
////    // (��������ʹ�� �ݧ֧ܧ�ڧ� �Ͻ��� �ӧѧ�ڧѧߧ� 4)
////    // ��������֪���� block_reduce ��Ҫ�õ�shared memory
////    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {      // blockDime ������ 2 �ı���������
////        if (tid_in_block < s)
////            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
////        __syncthreads();
////    }
////    // ����������Ҫ�����ڴ�, ���һ���Ҫblock���������Ķ���ȫ���ڴ�, �������ÿһ��block�ڵľֲ���Լֵ(global_reduce)
////    if (tid_in_block == 0) // д��block_reduce���
////        d_eps[blockIdx.x] = sh_data[0];
//}

//__global__ void global_reduce_MAX(double* A, double* B, int num_remaining) {
//    int i = blockIdx.x; // �� i ����
//    int tid = threadIdx.x; // �� idx ������
//    int idx_thread_global = tid + i * blockDim.x; // ����� A ���飨�� block_max���ĵ�ַλ��
//    // ����������ֵ����shared mem��
//    extern __shared__ double sdata[];
//    if (idx_thread_global < num_remaining) {   // ��ÿ���̲߳�����ȫ���߳�
//        sdata[tid] = A[idx_thread_global];
//    }
//    else {  // �������һ�� block �ĳ���������������һ����Сֵ
//        sdata[tid] = 0.;
//    }
//    __syncthreads();
//    // ���ַ����ڹ�Լ
//    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//        if (tid < s) {
//            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);  // ����Ҫ����MAX������max
//        }
//        __syncthreads();
//    }
//    // �� 0 ���߳̽���ǰ���ֵд��ȫ���ڴ�
//    if (tid == 0) {
//        B[i] = sdata[0];
//    }
//}


void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared Mem per Block: %d bytes\n", prop.sharedMemPerBlock);  // ÿ�� block ����
    printf("Shared Mem per SM   : %d bytes\n", prop.sharedMemPerMultiprocessor); // ÿ�� SM ����
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

    // ��ʼ������
    maxeps = 0.01;
    itmax = IT_MAX;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());

    // ���� host �ڴ�   ��ά����ͨ������һ�������洢
    a = (double *) malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // ��ʼ������
    printf("Initializing data...\n");
    init(a);

    //**********************************************************
    //**********************************************************
    int threads_per_block = BLOCK_SIZE;     // ÿ�����ڵ��߳�����
    int total_lines_x = ny * nz;    // x �����������
    int total_lines_y = nx * nz;    // y �����������
    int total_lines_z = nx * ny;    // z �����������
    int blocks_x = (total_lines_x + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_y = (total_lines_y + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    int blocks_z = (total_lines_z + threads_per_block - 1) / threads_per_block; // �����ʱ����Ŀ���
    //**********************************************************
    //**********************************************************

    // �����豸�ڴ�
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_A, nx * ny * nz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_eps, blocks_z * sizeof(double)));  // ���ÿ��block�����ֵ

    // �Ż���1: ���ڿ�ʼʱ�������ݵ��豸
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // Ϊ CUB Ԥ�����ڴ�
    // d_eps: �����������豸�ϵ�ָ��
    int num_items = blocks_z;   // ����block��������

    printf("Starting computation...\n");
    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++) {
        check_time_a = omp_get_wtime();

//        // ��ʼ�����ֵΪ0  (ʹ��block_reduce + global_reduceʱ����Ҫ��ʼ�����)
//        double h_eps = 0.0;
//        // ***** ��ʱ���� *****
//        CHECK_CUDA_ERROR(cudaMemcpy(d_eps, &h_eps, sizeof(double), cudaMemcpyHostToDevice));


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // ÿ������ 32 ���߳�


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();

        //// ���ֹ�Լ��Ҫ shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * sizeof(double) >>>(d_A, d_eps); // ÿ������ 32 ���߳�  ÿ��block 32��double
//        //// warp �ڹ�Լ����Ҫ shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // ÿ������ 32 ���߳�
        //// ���м������ݽ� shared memory
//        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block, threads_per_block * (threads_per_block + 1) * sizeof(double) >>>(d_A, d_eps); // ÿ������ 32 ���߳�  ÿ���߳� 33��double
        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // ÿ������ 32 ���߳�  ÿ���߳� 33��double

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // ��ʱd_eps�д洢ÿ��block�ڵ����ֵ
        // global_reduce 2  thrust  // �ҳ����㣡����
        // ����һ�� Thrust device_ptr ��װ����ָ�� d_eps��device �ϵ� block �ֲ����������飩
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
        // ʹ�� thrust::reduce �ҳ����ֵ����ʼֵ������ -�ޣ�
        eps = thrust::reduce(dev_ptr, dev_ptr + num_items, 0., thrust::maximum<double>());

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);

        // �����������
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

    // �Ż���4: ֻ�ڼ�����ɺ󿽱����������
    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));



    // �ͷ��ڴ�
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
