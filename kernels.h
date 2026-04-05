/*
这是一个“函数目录页”
它不做计算，只负责告诉别的文件 有哪些函数可用 每个函数需要什么输入 输出会放到哪里
*/

#ifndef KERNELS_H // 如果 KERNELS_H 这个标记还没有被定义过，那下面的内容就继续保留 它是一个 header guard，头文件保护机制
#define KERNELS_H // 定义 KERNELS_H 这个标记，防止头文件被多次包含

// 
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vec, double* result);

void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vec, double* result);

// baseline基准版本 后面所有优化版都拿它做比较                           
void multiply_mm_naive(const double* matrixA, int rowsA, int colsA,
                       const double* matrixB, int rowsB, int colsB,
                       double* result);

// 预处理矩阵B的转置版本 优化内存访问模式 
// naive matrix multiplication 里 访问 B 往往不连续 
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA,
                              const double* matrixB_transposed, int rowsB, int colsB,
                              double* result);
// 块状矩阵乘法 优化缓存命中率
/*
因为 cache 很小 如果你一次直接处理整个大矩阵，很多数据刚读进 cache 就被挤掉了
如果你把矩阵切成小 block：当前 block 的 A 当前 block 的 B 当前 block 的 C
更可能一起留在 cache 里，反复复用
*/
void multiply_mm_blocked(const double* matrixA, int rowsA, int colsA,
                         const double* matrixB, int rowsB, int colsB,
                         double* result, int blockSize);                              

// 和上面的 #ifndef 配对，结束头文件保护机制                         
#endif


