## Discussion Questions

### 1. Explain the key differences between pointers and references in C++. When would you choose to use a pointer over a 
reference, and vice versa, in the context of implementing numerical algorithms?

## Answer.  
Pointers store addresses and can be reassigned, incremented, or set to nullptr.
References are aliases for existing variables, must be initialized immediately, and cannot later refer to another object.

In numerical algorithms, pointers are better for raw arrays, dynamic memory, and contiguous matrix/vector storage, 
because they give direct access to memory and make traversal efficient.
References are better when an object must always exist and you just want a cleaner function interface.

In this project, pointers were the better choice because the matrices and vectors were stored in dynamically allocated 
arrays and accessed through raw memory.


### 2. How does the row-major and column-major storage order of matrices affect memory access patterns and cache 
locality during matrix-vector and matrix-matrix multiplication? Provide specific examples from your implementations 
and benchmarking results.

## Answer.

Storage order determines which elements are contiguous in memory. In row-major format, elements in the same row are 
adjacent, while in column-major format, elements in the same column are adjacent. Since CPUs benefit from sequential 
memory access, storage order directly affects cache locality.

In matrix vector kernels, each implementation followed its natural layout. The row major version accessed 
`matrix[i * cols + j]`, which reads across a row contiguously, while the column major version accessed 
`matrix[j * rows + i]`, which reads down a column contiguously. As a result, both had reasonably good spatial locality 
and similar benchmark performance overall.

For matrix matrix multiplication, the effect was much stronger. In the naive version, the innermost loop accessed 
`matrixB[k * colsB + j]`, which reads a row major matrix with a strided, column wise pattern and hurts cache efficiency. 
In the transposed-B version, the access became `matrixB_transposed[j * rowsB + k]`, which turns that pattern into 
sequential reads. This improved cache locality and gave noticeably better benchmark results.



### 3. Describe how CPU caches work (L1, L2, L3) and explain the concepts of temporal and spatial locality. 
How did you try to exploit these concepts in your optimizations?

## Answer.

CPU caches are small, fast memory layers between the processor and main memory. L1 is the smallest and fastest, 
L2 is larger but slower, and L3 is larger again and usually shared across cores. Their purpose is to reduce the cost of 
accessing main memory.

Spatial locality means nearby memory locations are likely to be used soon, while temporal locality means recently used 
data is likely to be reused soon.

By exploit the spatial locality by accessing data sequentially whenever possible. For example, the transposed-B 
implementation improved performance by turning strided accesses into more contiguous reads. We exploited temporal 
locality with blocked matrix multiplication, where small submatrices stay in cache longer and can be reused multiple 
times.


### 4. What is memory alignment, and why is it important for performance? Did you observe a significant performance 
difference between aligned and unaligned memory in your experiments?

## Answer.

Memory alignment means placing data at addresses that are multiples of a fixed boundary, such as 16, 32, or 64 bytes. 
It matters because modern CPUs and SIMD instructions usually access aligned data more efficiently, while misaligned data 
may require extra work or reduce vectorization efficiency.

In theory, alignment is important for high performance numerical code. However, in experiments, the main performance 
gains came from better cache locality and blocking rather than explicit alignment tuning. So while alignment is 
beneficial, it did not appear to be the main factor in benchmark results.

### 5. Discuss the role of compiler optimizations (like inlining) in achieving high performance. How did the optimization
level affect the performance of your baseline and optimized implementations? What are the potential drawbacks of 
aggressive optimization?

## Answer.

Compiler optimizations play a major role in numerical performance. At higher optimization levels, the compiler may 
inline small functions, remove redundant work, unroll loops, improve register usage, and generate more efficient machine 
code. Inlining is especially helpful for very small and frequently called functions because it removes function call 
overhead and can enable further optimizations.

For matrix kernels, optimized builds usually perform much better than unoptimized builds because most of the work happens 
inside tight loops. Higher optimization levels can also improve blocked and cache-friendly implementations by making 
better use of their loop structure.

The main drawbacks of aggressive optimization are longer compile times, larger binaries, harder debugging, and less 
readable generated assembly. 



### 6. Based on your profiling experience, what were the main performance bottlenecks in your initial implementations? 
How did your profiling results guide your optimization efforts?

## Answer.

The main bottleneck in initial implementations was inefficient memory access rather than arithmetic itself. 
In the naive matrix-matrix kernel, matrix B was accessed with a strided pattern in the innermost loop, which hurt cache locality and increased memory traffic.

The benchmark and profiling results suggested that memory access pattern was the main limiting factor, since both the 
transposed-B and blocked versions performed much better than the naive baseline without changing the underlying 
computation. This guided us to focus on improving locality, so implemented blocked matrix multiplication and compared 
it with the transposed-B approach.


### 7. Reflect on the teamwork aspect of this assignment. How did dividing the initial implementation tasks and then 
collaborating on analysis and optimization work? What were the challenges and benefits of this approach?

## Answer.

All I did.