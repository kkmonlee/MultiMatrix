#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <assert.h>
#include <cmath>
// #include a timer module

cublasHandle_t handler;


// Multiplication instances
void GPU_multi(double *A, double *B, double *C, int lda, int ldb, int ldc, int xa, int xb, int xc, int ya, int yb, int yc, double aleph, double bet) {
	cublasDgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, xb, ya, xa, &aleph, B, ldb, A, lda, &bet, C, ldc);
}

