/*
 * matrix.h
 *
 *  Created on: Jul 1, 2013
 *      Author: USER
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define CUTOFF 192
#define TOL 1e-12
#define BLOCKSIZE 64
#define TBLOCK 64
#define SVDMAXITER 50


#ifdef __cplusplus
extern "C" {
#endif

float macheps();

float pmax(float a, float b);

float pmin(float a, float b);

int imax(int a, int b);

int imin(int a, int b);

float signx(float x);

float l2norm(float *vec, int N);

int compare (const void* ind1, const void* ind2);

void sort1d(float* v,int N, int* pos);

//Array Parallel Implementation may have a lot of overhead

float array_max_abs(float *array,int N);

float array_max(float *array,int N);

float array_min(float *array,int N);

//void mmult(float* A, float *B, float *C,int ra,int ca, int rb, int cb);

void dtranspose(float *sig, int rows, int cols,float *col);

void stranspose(float *sig, int rows, int cols,float *col);

void rtranspose(float *m, int rows, int cols,float *n, int r, int c);

void ctranspose(float *sig, int rows, int cols,float *col);

void mtranspose(float *sig, int rows, int cols,float *col);

void itranspose(float *A, int M, int N);

//int minverse(float *xxt, int p);

void mdisplay(float *A, int row, int col);

void madd(float* A, float* B, float* C,int rows,int cols);

void msub(float* A, float* B, float* C,int rows,int cols);

void scale(float *A, int rows, int cols, float alpha);

void nmult(float* A, float* B, float* C,int m,int n, int p);

void tmult(float* A, float* B, float* C,int m,int n, int p);

void recmult(float* A, float* B, float* C,int m,int n, int p,int sA,int sB, int sC);

void rmult(float* A, float* B, float* C,int m,int n, int p);

int findrec(int *a, int *b, int *c);

float house_2(float*x,int N,float *v);

void add_zero_pad(float *X, int rows, int cols, int zrow, int zcol,float *Y);

void remove_zero_pad(float *X, int rows, int cols, int zrow, int zcol,float *Y);

void madd_stride(float* A, float* B, float* C,int rows,int cols,int sA,int sB,int sC);

void msub_stride(float* A, float* B, float* C,int rows,int cols,int sA,int sB,int sC);

void rmadd_stride(float* A, float* B, float* C,int rows,int cols,int p,int sA,int sB,int sC);

void rmsub_stride(float* A, float* B, float* C,int rows,int cols,int p,int sA,int sB,int sC);

void srecmult(float* A, float* B, float* C,int m,int n, int p,int sA,int sB,int sC);

void smult(float* A, float* B, float* C,int m,int n, int p);

void mmult(float* A, float* B, float* C,int m,int n, int p);

void ludecomp(float *A,int N,int *ipiv);

void linsolve(float *A,int N,float *b,int *ipiv,float *x);

void minverse(float *A,int M,int *ipiv,float *inv);

void eye(float *mat,int N);

float house(float*x,int N,float *v);

void housemat(float *v, int N,float beta,float *mat);

void qrdecomp(float *A, int M, int N,float *bvec);

void getQR(float *A,int M,int N,float *bvec,float *Q, float *R);

void hessenberg(float *A,int N);

void francisQR(float *A,int N);

void eig22(float *A, int stride,float *eigre,float *eigim);

int francis_iter(float *A, int N, float *H);

void eig(float *A,int N,float *eigre,float *eigim);

int cholu(float *A, int N);

int bcholu(float *A, int N);

int chol(float *A, int N);

void chold(float *A, int N);

void svd_sort(float *U,int M,int N,float *V,float *q);

int svd(float *A,int M,int N,float *U,float *V,float *q);

int svd_transpose(float *A, int M, int N, float *U, float *V, float *q);

int rank(float *A, int M,int N);

int lls_svd_multi(float *A, float *b, int M, int N, float *x);// Ax =b where x is a matrix A - M * N, x - N * p , B - M * p

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_H_ */
