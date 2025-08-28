/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "client_utility.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

//#define DIM1 1023
//#define DIM2 1024
//#define DIM3 1025
// Using 30k pushes it to 96% memory occupancy with 6 ranks (one per XCD?)
#define DIM1 8192
#define DIM2 8192
#define DIM3 8192
#define NITERS 1

template <typename T>
void mat_mat_mult(T        alpha,
                  T        beta,
                  int      M,
                  int      N,
                  int      K,
                  const T* A,
                  int      As1,
                  int      As2,
                  const T* B,
                  int      Bs1,
                  int      Bs2,
                  T*       C,
                  int      Cs1,
                  int      Cs2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            }
            C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

int main()
{
    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    // double             alpha = 1.1, beta = 0.9;
  
	rocblas_double_complex alpha = {1.1, 0.7};
    rocblas_double_complex beta = {0.3, 0.5};
  
    rocblas_int m = DIM1, n = DIM2, k = DIM3;
    rocblas_int lda, ldb, ldc, size_a, size_b, size_c;
    int         a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    rocblas_cout << "patki zgemm example" << std::endl;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        size_a     = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
        rocblas_cout << "N";
    }
    else
    {
        lda        = k;
        size_a     = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
        rocblas_cout << "T";
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        size_b     = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        rocblas_cout << "N: ";
    }
    else
    {
        ldb        = n;
        size_b     = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        rocblas_cout << "T: ";
    }
    ldc    = m;
    size_c = n * ldc;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<rocblas_double_complex> ha(size_a);
    std::vector<rocblas_double_complex> hb(size_b);
    std::vector<rocblas_double_complex> hc(size_c);
    std::vector<rocblas_double_complex> hc_gold(size_c);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 17;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 17;
    }
    hc_gold = hc;

    // allocate memory on device
    rocblas_double_complex *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(rocblas_double_complex)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(rocblas_double_complex)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(rocblas_double_complex)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(rocblas_double_complex) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(rocblas_double_complex) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(rocblas_double_complex) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    // Patki Warmup
    CHECK_ROCBLAS_ERROR(
        rocblas_zgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));

	// Continuous high-power loop by Patki
    rocblas_cout << "Starting 100 iterations of ZGEMM for max power consumption test" << std::endl; 
    for(int i = 0; i < 100; i++) {
        rocblas_zgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc);

        // Optional: Rotate matrices to prevent optimization, suggestion from Claude
        std::swap(da, db);
    }
    
    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(rocblas_double_complex) * size_c, hipMemcpyDeviceToHost));

    rocblas_cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
                 << ", " << ldb << ", " << ldc << std::endl;

    rocblas_double_complex max_relative_error = std::numeric_limits<rocblas_double_complex>::min();

// Skip verification, need to handle diff of complex numbers.
// Also this just runs on the CPU (??) and is low, don't really need it for our tests.
/*
    for(int i = 0; i < NITERS; i++)
    {
        // calculate golden or correct result
        mat_mat_mult<rocblas_double_complex>(alpha,
                            beta,
                            m,
                            n,
                            k,
                            ha.data(),
                            a_stride_1,
                            a_stride_2,
                            hb.data(),
                            b_stride_1,
                            b_stride_2,
                            hc_gold.data(),
                            1,
                            ldc);
    }

    for(int i = 0; i < size_c; i++)
    {
        rocblas_double_complex relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
        relative_error       = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    rocblas_double_complex eps       = std::numeric_limits<rocblas_double_complex>::epsilon();
    rocblas_double_complex tolerance = 10;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        rocblas_cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
    }
    else
    {
        rocblas_cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
    }
*/
    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
