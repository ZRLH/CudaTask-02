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
    double d_al, d_ac, d_ar;  // left  center  right

    // get the i, j
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
            int i_thread_global_id = blockIdx.x * blockDim.x + i_tid;
            int i_i = i_thread_global_id / ny;
            int i_j = i_thread_global_id % ny;
            sh_data[i_tid * 33 + tid_in_block + 1] = a[    i_i * ny * nz + i_j * nz + tid_in_block + 1    ];

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

    }
    if (first_round_size == 2) {


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


        first_round_size = 32;

        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]


        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取 first_round_size - 2 项
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
            // 为block内每个line读取 first_round_size - 2 项
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

        d_al = a[i * ny * nz + j * nz + 0]; // 最左侧元素a[0]
        d_ac = a[i * ny * nz + j * nz + 1]; // a[1]


        // 并行地运行line_per_block轮以便于为block内每条线都分配好各自的共享内存
        // 为每条line读取 first_round_size - 2 项
        for (int i_tid = 0; i_tid < line_per_block; ++i_tid) {
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

    __syncthreads();

    // 使用 二分 块内规约
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {     // blockDim = 32; // blockDim 必须是 2 的倍数！！！
        if (tid_in_block < s)
            sh_data[tid_in_block] = MAX(sh_data[tid_in_block], sh_data[tid_in_block + s]);
        __syncthreads();
    }

    if (tid_in_block == 0) // 写回block_reduce结果
        d_eps[blockIdx.x] = sh_data[0];
}
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



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


        compute_i_direction_kernel_32_thread<<<blocks_x, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        check_time_b = omp_get_wtime();


        compute_j_direction_kernel_32_thread<<<blocks_y, threads_per_block>>>(d_A); // 每个块内 32 个线程


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_c = omp_get_wtime();


        compute_k_direction_kernel_32_thread<<<blocks_z, threads_per_block>>>(d_A, d_eps); // 每个块内 32 个线程  每个线程 33个double


        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        check_time_d = omp_get_wtime();

        // 此时d_eps中存储每个block内的最大值
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_eps);
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


    printf("Computation complete, copying results back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(a, d_A, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));


    // 释放内存 Freeing up memory
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
