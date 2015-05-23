#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <assert.h>
#include <cmath>
#include "cudaTimer.h"

cublasHandle_t handler;

/*
	lda, ldb, ldc: memory width
	xa, xb, xc: computational width
	A = xa * ya
	B = xb * yb
	C = xc * yc
*/

// Multiplication variables
void GPU_multi(double *A, double *B, double *C, int lda, int ldb, int ldc, int xa, int xb, int xc, int ya, int yb, int yc, double aleph, double bet) {
	cublasDgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, xb, ya, xa, &aleph, B, ldb, A, lda, &bet, C, ldc);
}

// Addition variables
void GPU_add(double *A, double *B, double *C, int lda, int ldb, int ldc, int xa, int ya, double aleph, double bet) {
	cublasDgeam(handler, CUBLAS_OP_N, CUBLAS_OP_N, xa, ya, &aleph, A, lda, &bet, B, ldb, C, ldc);
}

void cublasVerification(double *dA, double *dB, double *dC, int D, int E, int F) {
	double one = 1.0;
	double zero = 0.0;
#if CMAJOR
	cublasDgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, D, E, F, &one, dA, D, db, F, *zero, dC, D);
#else
	cublasDgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, D, E, F, &one, dB, E, dA, F, &zero, dC, E);
#endif
}

void Strassen(double *A, double *B, double *C, int lda, int ldb, int ldc, int xa, int xb, int xc, int ya, int yb, int yc, int depth) {
	int xa2 = xa / 2; int xb2 = xb / 2; int xc2 = xc / 2;
	int ya2 = ya / 2; int yb2 = yb / 2; int yc2 = yc / 2;

	// Cuts off if
	bool stop = false;
#if 0
	int cutoff = 2048;
	float DD = cutoff / xb2;
	float EE = cutoff / ya2;
	float FF = cutoff / xa2;
	if ((DD + EE + FF) >= 3) stop = true;
#endif

	// Width
	double *Wid1, *Wid2;
	int lw1 = (xa2 > xc2 ? xa2 : xc2); int lw2 = xb2;
	cudaMalloc((void **)&Wid1, lw1 * ya2 * sizeof(double));
	cudaMalloc((void**)&Wid2, lw2 * yb2 * sizeof(double));

	int dXA = xa2; int dYA = ya2 * lda;
	int dXB = xb2; int dYB = yb2 * ldb;
	int dXC = xc2; int dYC = yc2 * ldc;

	double *A11, *A12, *A21, *A22;
	double *B11, *B12, *B21, *B22;
	double *C11, *C12, *C21, *C22;
	A11 = A; A12 = A + dXA; A21 = A + dYA; A22 = A + dXA + dYA;
	B11 = B; B12 = B + dXB; B21 = B + dYB; B22 = B + dXB + dYB;
	C11 = C; C12 = C + dXC; C21 = C + dYC; C22 = C + dXC + dYC;

	if (depth <= 1 || stop) {
		GPU_add(A11, A21, Wid1, lda, lda, lw1, xa2, ya2, 1.0, -1.0); // Wid1 = A11 - A21
		GPU_add(B22, B12, Wid2, ldb, ldb, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = B22 - B12
		GPU_multi(Wid1, Wid2, C21, lw1, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, 1.0, 0.0); // C21 = Wid1 * Wid2
		
		GPU_add(A21, A22, Wid1, lda, lda, lw1, xa2, ya2, 1.0, 1.0); // Wid1 = A21 + A22
		GPU_add(B12, B11, Wid2, ldb, ldb, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = B12 - B11
		GPU_multi(Wid1, Wid2, C22, lw1, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, 1.0, 0.0); // C22 = Wid1 * Wid2
		
		GPU_add(Wid1, A11, Wid1, lw1, lda, lw1, xa2, ya2, 1.0, -1.0); // Wid1 = Wid1- A11
		GPU_add(B22, Wid2, Wid2, ldb, lw2, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = B22 - Wid2
		GPU_multi(Wid1, Wid2, C11, lw1, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, 1.0, 0.0); // C11 = Wid1 * Wid2
		
		GPU_add(A12, Wid1, Wid1, lda, lw1, lw1, xa2, ya2, 1.0, -1.0); // Wid1 = A12 - Wid1
		GPU_multi(Wid1, B22, C12, lw1, ldb, ldc, xa2, xb2, xc2, ya2, yb2, yc2, 1.0, 0.0); // C12 = Wid1 * B22
		GPU_add(C22, C12, C12, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C12 = C22 + C12
		
		GPU_multi(A11, B11, Wid1, lda, ldb, lw1, xa2, xb2, xc2, ya2, yb2, yc2, 1.0, 0.0); // Wid1= A11 * B11
		GPU_add(Wid1, C11, C11, lw1, ldc, ldc, xc2, yc2, 1.0, 1.0); // C11 = Wid1 + C11
		GPU_add(C11, C12, C12, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C12 = C11 + C12
		
		GPU_add(C11, C21, C11, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C11 = C11 + C21
		GPU_add(Wid2, B21, Wid2, lw2, ldb, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = Wid2- B21
		GPU_multi(A22, Wid2, C21, lda, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, 1.0, 0.0); // C21 = A22 * Wid2
		
		GPU_add(C11, C21, C21, ldc, ldc, ldc, xc2, yc2, 1.0, -1.0); // C11 = C11 - C21
		GPU_add(C11, C22, C22, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C22 = C11 + C22
		GPU_multi(A12, B21, C11, lda, ldb, ldc, xa2, xb2, xc2, ya2, yb2, yc2, 1.0, 0.0); // C11 = A12 * B21
		
		GPU_add(Wid1, C11, C11, lw1, ldc, ldc, xc2, yc2, 1.0, 1.0); // C11 = Wid1+ C11

	}
	else {
		GPU_add(A11, A21, Wid1, lda, lda, lw1, xa2, ya2, 1.0, -1.0); // Wid1 = A11 - A21
		GPU_add(B22, B12, Wid2, ldb, ldb, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = B22 - B12
		Strassen(Wid1, Wid2, C21, lw1, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, depth - 1);
		
		GPU_add(A21, A22, Wid1, lda, lda, lw1, xa2, ya2, 1.0, 1.0); // Wid1 = A21 + A22
		GPU_add(B12, B11, Wid2, ldb, ldb, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = B12 - B11
		Strassen(Wid1, Wid2, C22, lw1, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, depth - 1);
		
		GPU_add(Wid1, A11, Wid1, lw1, lda, lw1, xa2, ya2, 1.0, -1.0); // Wid1 = Wid1- A11
		GPU_add(B22, Wid2, Wid2, ldb, lw2, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = B22 - Wid2
		Strassen(Wid1, Wid2, C11, lw1, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, depth - 1);
		
		GPU_add(A12, Wid1, Wid1, lda, lw1, lw1, xa2, ya2, 1.0, -1.0); // Wid1 = A12 - Wid1
		Strassen(Wid1, B22, C12, lw1, ldb, ldc, xa2, xb2, xc2, ya2, yb2, yc2, depth - 1);
		GPU_add(C22, C12, C12, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C12 = C22 + C12
		
		Strassen(A11, B11, Wid1, lda, ldb, lw1, xa2, xb2, xc2, ya2, yb2, yc2, depth - 1);
		GPU_add(Wid1, C11, C11, lw1, ldc, ldc, xc2, yc2, 1.0, 1.0); // C11 = Wid1 + C11
		GPU_add(C11, C12, C12, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C12 = C11 + C12
		
		GPU_add(C11, C21, C11, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C11 = C11 + C21
		GPU_add(Wid2, B21, Wid2, lw2, ldb, lw2, xb2, yb2, 1.0, -1.0); // Wid2 = Wid2- B21
		Strassen(A22, Wid2, C21, lda, lw2, ldc, xa2, xb2, xc2, ya2, yb2, yc2, depth - 1);
		
		GPU_add(C11, C21, C21, ldc, ldc, ldc, xc2, yc2, 1.0, -1.0); // C11 = C11 - C21
		GPU_add(C11, C22, C22, ldc, ldc, ldc, xc2, yc2, 1.0, 1.0); // C22 = C11 + C22
		Strassen(A12, B21, C11, lda, ldb, ldc, xa2, xb2, xc2, ya2, yb2, yc2, depth - 1);
		
		GPU_add(Wid1, C11, C11, lw1, ldc, ldc, xc2, yc2, 1.0, 1.0); // C11 = Wid1+ C11
	}
	cudaFree(Wid1);
	cudaFree(Wid2);

	/*
		Using dynamic peeling instead of padding.
		-> removes row/column after multiplication
	*/

	int pxa = xa % 2;
	int pya = ya % 2;
	int pxb = xb % 2;
	int pyb = yb % 2;
	int pxc = xc % 2;
	int pyc = yc % 2;
	int nxa = xa - pxa;
	int nya = ya - pya;
	int nxb = xb - pxb;
	int nyb = yb - pyb;
	int nxc = xc - pxc;
	int nyc = yc - pyc;

	double *a12, *a21;
	double *b12, *b21;
	double *c12, *c21;

	int dxa = nxa; int dya = nya * lda;
	int dxb = nxb; int dyb = nyb * ldb;
	int dxc = nxc; int dyc = nyc * ldc;

	a12 = A + dxa; a21 = A + dya;
	b12 = B + dxb; b21 = B + dyb;
	c12 = C + dxc; c21 = C + dyc;

	GPU_multi(a21, B11, c21, lda, ldb, ldc, nxa, xb, xc, pya, nyb, pyc, 1.0, 0.0);
	GPU_multi(A11, b12, c12, lda, ldb, ldc, nxa, pxb, pxc, ya, nyb, yc, 1.0, 0.0);
	GPU_multi(a12, b21, C11, lda, ldb, ldc, pxa, xb, xc, ya, pyb, yc, 1.0, 1.0);

}

int main(int argc, char **argv) {
	if (argc != 8) {
		printf("Using: Strasses <D> <E> <F> <iter> <check> <depth>\n");
		return -1;
	}

	int D = atoi(argv[1]);
	int E = atoi(argv[2]);
	int F = atoi(argv[3]);
	int iter = atoi(argv[4]);
	int check = atoi(argv[5]);
	int depth = atoi(argv[6]);

	int sizeA = D * F;
	int sizeB = F * E;
	int sizeC = D * E;
	
	// Memory values
	int memsizeA = sizeA * sizeof(double);
	int memsizeB = sizeB * sizeof(double);
	int memsizeC = sizeC * sizeof(double);

	// Memory allocation
	double *hA = (double *)malloc(memsizeA);
	double *hB = (double *)malloc(memsizeB);
	double *hC = (double *)malloc(memsizeC);
	double *vC = (double *)malloc(memsizeC);
	for (int i = 0; i < sizeA; i++) hA[i] = (i % 3);
	for (int i = 0; i < sizeB; i++) hB[i] = (i % 3);
	for (int i = 0; i < sizeC; i++) hC[i] = 0.0f;
	for (int i = 0; i < sizeC; i++) vC[i] = 0.0f;

	double *dA, *dB, *dC;
	cudaMalloc((void**)&dA, memsizeA);
	cudaMalloc((void**)&dB, memsizeB);
	cudaMalloc((void**)&dC, memsizeC);
	cudaMemcpy(dA, hA, memsizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, memsizeB, cudaMemcpyHostToDevice);

	if (cublasCreate(&handler) != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS failed to initialise\n"); fflush(NULL);
		return EXIT_FAILURE;
	}

	// Timerino

	CudaTimer ct;
	ct.start();
	for (int i = 0; i < iter; i++) {
		Strassen(dA, dB, dC, F, E, E, F, E, E, D, F, D, depth);
	
	}
	ct.stop();
	double Strassen_Time = ct.value() / iter;
	cudaMemcpy(hC, dC, memsizeC, cudaMemcpyDeviceToHost);

#if 1
	ct.start();
	for (int i = 0; i < iter; i++)
		cublasVerification(dA, dB, dC, D, E, F);
	ct.stop();
	double cublasTime = ct.value() / iter;
	cudaMemcpy(vC, dC, memsizeC, cudaMemcpyDeviceToHost);
	double speed = cublasTime / Strassen_Time;
	printf("%d %d %d %.2f %.2f %.2f\n", D, E, F, Strassen_Time, cublasTime, speed);
#endif

	if (check) {
		double absoluteErr = 0.0;
		for (int i = 0; i < sizeC; ++i)
			absoluteErr += abs(hC[i] - vC[i]);
		if (absoluteErr > 1) printf("Check absolute error: %lf\n", absoluteErr);
	}

	// Relieve memory
	free(hA);
	free(hB);
	free(hC);
	free(vC);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	if (cublasDestroy(handler) != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error occurred during CUBLAS shutdown.\n"); fflush(NULL);
		return EXIT_FAILURE;
	}
}