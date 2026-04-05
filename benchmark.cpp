#include "kernels.h"
#include <iostream>
#include <vector>  // std::vector 用来存储多次运行的时间结果
#include <chrono>  // std::chrono 用来测量时间
#include <cmath>   // std::sqrt 计算标准差
#include <iomanip> // std::setw 打印表格的时候对齐
#include <random>  // std::mt19937 和 std::uniform_real_distribution 用来生成随机矩阵数据
#include <string>  // std::string 用来构造输出标签

// 这个 benchmark.cpp 文件的作用是对 kernels.h 中实现的各种矩阵乘法函数进行性能测试
// 主要测试以下函数：
// - multiply_mv_row_major
// - multiply_mv_col_major
// - multiply_mm_naive
// - multiply_mm_transposed_b           


// 给复杂类型起短名字
using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::duration<double, std::micro>;

// 给一串运行时间算平均值
double compute_mean(const std::vector<double>& times) {
    double sum = 0.0;
    for (double t : times) sum += t;
    return sum / times.size();
}


// 计算标准差 standard deviation
/*
如果 stddev 很小    每次跑出来都差不多，结果很稳定

如果 stddev 很大    实验波动比较明显
*/
double compute_stddev(const std::vector<double>& times, double mean) {
    double sum_sq = 0.0;
    for (double t : times) {
        double diff = t - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / times.size());
}

// 把数组填上随机 double
void fill_random(double* data, int size) {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < size; i++) {
        data[i] = dist(rng);
    }
}


double benchmark_mv_row_major(int rows, int cols, int runs) {
    double* matrix = new double[rows * cols];
    double* vec = new double[cols];
    double* result = new double[rows];

    fill_random(matrix, rows * cols);
    fill_random(vec, cols);

    std::vector<double> times;

    for (int r = 0; r < runs; r++) {
        auto start = Clock::now();
        multiply_mv_row_major(matrix, rows, cols, vec, result);
        auto end = Clock::now();
        
        // 把时间差转成微秒并取数值
        double elapsed = std::chrono::duration_cast<Microseconds>(end - start).count();
        times.push_back(elapsed);
    }

    double mean = compute_mean(times);
    double stddev = compute_stddev(times, mean);

    std::cout << std::setw(20) << "MV Row-Major"
              << std::setw(10) << rows
              << std::setw(10) << cols
              << std::setw(15) << mean
              << std::setw(15) << stddev << "\n";
    // 为了打印成整齐的表格
    delete[] matrix;
    delete[] vec;
    delete[] result;

    return mean;
}




double benchmark_mv_col_major(int rows, int cols, int runs) {
    double* matrix = new double[rows * cols];
    double* vec = new double[cols];
    double* result = new double[rows];

    fill_random(matrix, rows * cols);
    fill_random(vec, cols);

    std::vector<double> times;

    for (int r = 0; r < runs; r++) {
        auto start = Clock::now();
        multiply_mv_col_major(matrix, rows, cols, vec, result);
        auto end = Clock::now();

        double elapsed = std::chrono::duration_cast<Microseconds>(end - start).count();
        times.push_back(elapsed);
    }

    double mean = compute_mean(times);
    double stddev = compute_stddev(times, mean);

    std::cout << std::setw(20) << "MV Col-Major"
              << std::setw(10) << rows
              << std::setw(10) << cols
              << std::setw(15) << mean
              << std::setw(15) << stddev << "\n";

    delete[] matrix;
    delete[] vec;
    delete[] result;

    return mean;
}

double benchmark_mm_naive(int rowsA, int colsA, int colsB, int runs) {
    int rowsB = colsA;

    double* A = new double[rowsA * colsA];
    double* B = new double[rowsB * colsB];
    double* C = new double[rowsA * colsB];

    fill_random(A, rowsA * colsA);
    fill_random(B, rowsB * colsB);

    std::vector<double> times;

    for (int r = 0; r < runs; r++) {
        auto start = Clock::now();
        multiply_mm_naive(A, rowsA, colsA, B, rowsB, colsB, C);
        auto end = Clock::now();

        double elapsed = std::chrono::duration_cast<Microseconds>(end - start).count();
        times.push_back(elapsed);
    }

    double mean = compute_mean(times);
    double stddev = compute_stddev(times, mean);

    std::cout << std::setw(20) << "MM Naive"
              << std::setw(10) << rowsA
              << std::setw(10) << colsA
              << std::setw(10) << colsB
              << std::setw(15) << mean
              << std::setw(15) << stddev << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return mean;
}

double benchmark_mm_transposed_b(int rowsA, int colsA, int colsB, int runs) {
    int rowsB = colsA;

    double* A = new double[rowsA * colsA];
    double* B = new double[rowsB * colsB];
    double* B_transposed = new double[colsB * rowsB];
    double* C = new double[rowsA * colsB];

    fill_random(A, rowsA * colsA);
    fill_random(B, rowsB * colsB);

    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            B_transposed[j * rowsB + i] = B[i * colsB + j];
        }
    }

    std::vector<double> times;

    for (int r = 0; r < runs; r++) {
        auto start = Clock::now();
        multiply_mm_transposed_b(A, rowsA, colsA, B_transposed, rowsB, colsB, C);
        auto end = Clock::now();

        double elapsed = std::chrono::duration_cast<Microseconds>(end - start).count();
        times.push_back(elapsed);
    }

    double mean = compute_mean(times);
    double stddev = compute_stddev(times, mean);

    std::cout << std::setw(20) << "MM Transposed-B"
              << std::setw(10) << rowsA
              << std::setw(10) << colsA
              << std::setw(10) << colsB
              << std::setw(15) << mean
              << std::setw(15) << stddev << "\n";

    delete[] A;
    delete[] B;
    delete[] B_transposed;
    delete[] C;

    return mean;
}

double benchmark_mm_blocked(int rowsA, int colsA, int colsB, int runs, int blockSize) {
    int rowsB = colsA;

    double* A = new double[rowsA * colsA];
    double* B = new double[rowsB * colsB];
    double* C = new double[rowsA * colsB];

    fill_random(A, rowsA * colsA);
    fill_random(B, rowsB * colsB);

    std::vector<double> times;

    for (int r = 0; r < runs; r++) {
        auto start = Clock::now();
        multiply_mm_blocked(A, rowsA, colsA, B, rowsB, colsB, C, blockSize);
        auto end = Clock::now();

        double elapsed = std::chrono::duration_cast<Microseconds>(end - start).count();
        times.push_back(elapsed);
    }

    double mean = compute_mean(times);
    double stddev = compute_stddev(times, mean);

    std::string label = "MM Blocked(" + std::to_string(blockSize) + ")";
    
    std::cout << std::setw(20) << label
              << std::setw(10) << rowsA
              << std::setw(10) << colsA
              << std::setw(10) << colsB
              << std::setw(15) << mean
              << std::setw(15) << stddev << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return mean;
}




int main() {
    int runs = 10;

    std::cout << "\n===== Matrix-Vector Benchmark =====\n";
    std::cout << std::setw(20) << "Function"
              << std::setw(10) << "Rows"
              << std::setw(10) << "Cols"
              << std::setw(15) << "Avg (us)"
              << std::setw(15) << "StdDev\n";
   
    benchmark_mv_row_major(256, 256, runs);
    benchmark_mv_col_major(256, 256, runs);

    benchmark_mv_row_major(1024, 1024, runs);
    benchmark_mv_col_major(1024, 1024, runs);

    benchmark_mv_row_major(2048, 2048, runs);
    benchmark_mv_col_major(2048, 2048, runs);

    std::cout << "\n===== Matrix-Matrix Benchmark =====\n";
    std::cout << std::setw(20) << "Function"
              << std::setw(10) << "RowsA"
              << std::setw(10) << "ColsA"
              << std::setw(10) << "ColsB"
              << std::setw(15) << "Avg (us)"
              << std::setw(15) << "StdDev\n";

    benchmark_mm_naive(128, 128, 128, runs);
    benchmark_mm_transposed_b(128, 128, 128, runs);
    benchmark_mm_blocked(128, 128, 128, runs, 32);

    benchmark_mm_naive(256, 256, 256, runs);
    benchmark_mm_transposed_b(256, 256, 256, runs);
    benchmark_mm_blocked(256, 256, 256, runs, 16);
    benchmark_mm_blocked(256, 256, 256, runs, 32);
    benchmark_mm_blocked(256, 256, 256, runs, 64);



    benchmark_mm_naive(384, 384, 384, runs);
    benchmark_mm_transposed_b(384, 384, 384, runs);
    benchmark_mm_blocked(384, 384, 384, runs, 32);

    return 0;
}