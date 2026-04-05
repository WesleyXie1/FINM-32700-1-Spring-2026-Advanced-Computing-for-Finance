#include "kernels.h"
#include <iostream>  // std::cerr
#include <algorithm> // std::min

// 计算
// multiply_mv_row_major 按 row-major 索引实现矩阵向量乘法 
// multiply_mv_col_major 按 column-major 索引实现矩阵向量乘法
// multiply_mm_naive     最基础三重循环矩阵乘法
// multiply_mm_transposed_b  利用转置 B 改善访存模式的矩阵乘法
// multiply_mm_blocked利用  blocking/tiling提升 cache reuse 的矩阵乘法

// basic error handling
// row-major matrix-vector multiplication
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vec, double* result) {
    if (!matrix || !vec || !result || rows <= 0 || cols <= 0) {
        std::cerr << "Error: invalid input to multiply_mv_row_major\n";
        return;
    }

    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vec[j];
        }
        result[i] = sum;
    }
}



// column-major 存储的矩阵乘向量 
void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vec, double* result) {
    if (!matrix || !vec || !result || rows <= 0 || cols <= 0) {
        std::cerr << "Error: invalid input to multiply_mv_col_major\n";
        return;
    }

    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;      // 先清零结果
    }

    for (int j = 0; j < cols; j++) {
        double v = vec[j];
        for (int i = 0; i < rows; i++) {
            result[i] += matrix[j * rows + i] * v;
        }
    }
}




void multiply_mm_naive(const double* matrixA, int rowsA, int colsA,
                       const double* matrixB, int rowsB, int colsB,
                       double* result) {
    if (!matrixA || !matrixB || !result ||
        rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0 ||
        colsA != rowsB) {
        std::cerr << "Error: invalid input to multiply_mm_naive\n";
        return;
    }
// dot product 的求和索引
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                //因为 A 是 row-major，取第 i 行第 k 列的元素是 matrixA[i * colsA + k]
                //因为 B 是 row-major，取第 k 行第 j 列的元素是 matrixB[k * colsB + j]
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            result[i * colsB + j] = sum;
        }
    }
}



// 不是直接传入 B 而是传入 B^T 也就是 B 的转置矩阵
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA,
                              const double* matrixB_transposed, int rowsB, int colsB,
                              double* result) {
    if (!matrixA || !matrixB_transposed || !result ||
        rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0 ||
        colsA != rowsB) {
        std::cerr << "Error: invalid input to multiply_mm_transposed_b\n";
        return;
    }

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += matrixA[i * colsA + k] * matrixB_transposed[j * rowsB + k];
            }
            result[i * colsB + j] = sum;
        }
    }
}

// 普通 naive 是直接算整个大矩阵 blocked 的想法是把矩阵切成一块一块的小方块 再逐块计算
void multiply_mm_blocked(const double* matrixA, int rowsA, int colsA,
                         const double* matrixB, int rowsB, int colsB,
                         double* result, int blockSize) {
    if (!matrixA || !matrixB || !result ||
        rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0 ||
        colsA != rowsB || blockSize <= 0) {
        std::cerr << "Error: invalid input to multiply_mm_blocked\n";
        return;
    }

    for (int i = 0; i < rowsA * colsB; i++) {
        result[i] = 0.0;
    }

    for (int ii = 0; ii < rowsA; ii += blockSize) {
        for (int kk = 0; kk < colsA; kk += blockSize) {
            for (int jj = 0; jj < colsB; jj += blockSize) {

                int i_max = std::min(ii + blockSize, rowsA);
                int k_max = std::min(kk + blockSize, colsA);
                int j_max = std::min(jj + blockSize, colsB);

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_val = matrixA[i * colsA + k];
                        for (int j = jj; j < j_max; j++) {
                            result[i * colsB + j] += a_val * matrixB[k * colsB + j];
                        }
                    }
                }
            }
        }
    }
}