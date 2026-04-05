#include "kernels.h"
#include <iostream>
#include <iomanip> // for std::setw 打印矩阵的时候更整齐 不会乱成一团

// 打印很多次向量结果
void print_vector(const double* v, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
}

// 把一个 row major 存储的矩阵打印出来
void print_matrix(const double* m, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << m[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    {    // 在 heap 上分配 rows * cols = 6 个 double
            // 这个矩阵是 row major 存储的
            // 1 2 3
            // 4 5 6
            // vec 是 [1, 1, 1]
            // result 应该是 [6, 15]
        int rows = 2, cols = 3;
        double* matrix = new double[rows * cols]{1, 2, 3, 4, 5, 6};
        double* vec = new double[cols]{1, 1, 1};
        double* result = new double[rows];

        multiply_mv_row_major(matrix, rows, cols, vec, result);
        std::cout << "Row-major MV result:\n";
        print_vector(result, rows);

        delete[] matrix;
        delete[] vec;
        delete[] result;
    }

    {   // column-major 存储的矩阵
            // 1 4
            // 2 5
            // 3 6
            // vec 是 [1, 1, 1]
            // result 应该是 [6, 15]
        int rows = 2, cols = 3;
        double* matrix = new double[rows * cols]{1, 4, 2, 5, 3, 6};
        double* vec = new double[cols]{1, 1, 1};
        double* result = new double[rows];

        multiply_mv_col_major(matrix, rows, cols, vec, result);
        std::cout << "Column-major MV result:\n";
        print_vector(result, rows);

        delete[] matrix;
        delete[] vec;
        delete[] result;
    }

    {   // 测 multiply_mm_naive
                // A 是 2x3 的矩阵
                // 1 2 3
                // 4 5 6
                // B 是 3x2 的矩阵
                // 7 8
                // 9 10
                // 11 12
                // C 应该是 2x2 的矩阵
                // 58 64
                // 139 154
        int rowsA = 2, colsA = 3;
        int rowsB = 3, colsB = 2;

        double* A = new double[rowsA * colsA]{1, 2, 3, 4, 5, 6};
        double* B = new double[rowsB * colsB]{7, 8, 9, 10, 11, 12};
        double* C = new double[rowsA * colsB];

        multiply_mm_naive(A, rowsA, colsA, B, rowsB, colsB, C);
        std::cout << "Naive MM result:\n";
        print_matrix(C, rowsA, colsB);

        delete[] A;
        delete[] B;
        delete[] C;
    }

    {
        int rowsA = 2, colsA = 3;
        int rowsB = 3, colsB = 2;

        double* A = new double[rowsA * colsA]{1, 2, 3, 4, 5, 6};
        double* B_transposed = new double[colsB * rowsB]{7, 9, 11, 8, 10, 12};
        double* C = new double[rowsA * colsB];

        // multiply_mm_transposed_b 版本需要预先把 B 转置成 B_transposed
        multiply_mm_transposed_b(A, rowsA, colsA, B_transposed, rowsB, colsB, C);
        std::cout << "Transposed-B MM result:\n";
        print_matrix(C, rowsA, colsB);

        delete[] A;
        delete[] B_transposed;
        delete[] C;
    }

    return 0;
}