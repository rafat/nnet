
#include "matrix.h"

/*
 * matrix.c

 *
 *  Copyright (c) 2014, Rafat Hussain
 *	License : BSD 3-Clause
 *	See COPYRIGHT for more details
 */

typedef struct {
	float* a;
	int b;
} vipair;

float macheps() {
	float macheps;
	macheps = 1.0;

	while ((macheps + 1.0) > 1.0) {
		macheps = macheps / 2.0f;
	}

	macheps = macheps * 2;

	return macheps;
}

float pmax(float a, float b) {
	if (a > b) {
		return a;
	}
	else {
		return b;
	}
}

float pmin(float a, float b) {
	if (a < b) {
		return a;
	}
	else {
		return b;
	}
}

int imax(int a, int b) {
	if (a > b) {
		return a;
	}
	else {
		return b;
	}
}

int imin(int a, int b) {
	if (a < b) {
		return a;
	}
	else {
		return b;
	}
}

float signx(float x) {
	float sgn;
	if (x >= 0.) {
		sgn = 1.0;
	}
	else {
		sgn = -1.0;
	}
	return sgn;
}

float l2norm(float *vec, int N) {
	float l2, sum;
	int i;
	sum = 0.;
	for (i = 0; i < N; ++i) {
		sum += vec[i] * vec[i];
	}
	l2 = sqrtf(sum);
	return l2;
}

int compare (const void* ind1, const void* ind2)
{
	if (*((vipair *)ind1)->a > *((vipair *)ind2)->a)
		return -1;
	else if (*((vipair *)ind1)->a < *((vipair *)ind2)->a)
		return 1;
	else
		return 0;
}

void sort1d(float* v,int N, int* pos)
{
    vipair* val = NULL;
    int i;

    if (N <= 0)
        return;

    val = malloc(sizeof(vipair) * N);

    for (i = 0; i < N; ++i) {
        val[i].a = &v[i];
        val[i].b = i;
    }

    qsort(val, N, sizeof(vipair), compare);

    for (i = 0; i < N; ++i)
        pos[i] = val[i].b;

    free(val);
}

float array_max_abs(float *array,int N) {
	int i;
	float m = 0.0;

	for (i = 0; i < N;++i) {
		if (fabs(array[i]) > m ) {
			m = fabsf(array[i]);
		}
	}

	return m;
}

float array_max(float *array,int N) {
	int i;
	float m = array[0];

	for (i = 1; i < N;++i) {
		if (array[i] > m ) {
			m = array[i];
		}
	}

	return m;
}

float array_min(float *array,int N) {
	int i;
	float m = array[0];
	for (i = 1; i < N;++i) {
		if (array[i] < m ) {
			m = array[i];
			}
	}

	return m;
}

void dtranspose(float *sig, int rows, int cols,float *col) {
    int max,ud,i,k;
    if (rows >= cols) {
    	max = cols;
    } else {
    	max = rows;
    }
    ud = 0;
	for (i= -rows + 1; i < cols; i++) {
		if (i <= 0) {
			ud++;
			if (ud >= max)
				ud = max;
			for (k = 0; k < ud; k++) {
				col[k*rows+k-i] = sig[(k-i)*cols+k];
			}
		} else {
			if (i - cols + rows > 0) {
				ud--;
				if (ud >= max)
					ud = max;
			}
			for (k = 0; k < ud; k++) {
				col[(k+i)*rows+k] = sig[k*cols+k+i];
			}
		}

	}

}

void stranspose(float *sig, int rows, int cols,float *col) {
	int t,u;
	register int i,j;
//	#pragma omp parallel for private(i,j,t,u)
	for (i=0; i < rows; i++) {
		t = i * cols;
		u = 0;
		for (j=0; j < cols; j++) {
			col[u+i] = sig[j+t];
			u+=rows;
		}
	}
	
}

void rtranspose(float *m, int rows, int cols,float *n, int r, int c) {
	register int i,j;
	int rm,cm;
	int rm1,cm1,rm2,cm2;
	int block;

	block = (int) BLOCKSIZE;

	if (rows <= block && cols <= block) {
		for (i = 0; i < rows; ++i) {
			for (j = 0; j < cols; ++j) {
				n[i+j*r] = m[j+i*c];
				//cout << *(n+i+j*r) << " ";
			}
		}
		//cout << endl;

	} else if (cols >= rows) {
		rm = rows;
		cm1 = (int) ceil((float) cols/2.0);
		cm2 = cols - cm1;

		rtranspose(m,rm,cm1,n,r,c);
		rtranspose(m+cm1,rm,cm2,n+cm1*r,r,c);
	} else if (rows > cols) {
		rm1 = (int) ceil((float) rows/2.0);
		rm2 = rows - rm1;
		cm = cols;
		rtranspose(m,rm1,cm,n,r,c);
		rtranspose(m+rm1*c,rm2,cm,n+rm1,r,c);
	}

}

void ctranspose(float *sig, int rows, int cols,float *col) {
	int r,c;
	int block;

	block = (int) TBLOCK;
	r= rows;
	c = cols;
	if (rows >= block || cols >= block)  {
		rtranspose(sig,rows,cols,col,r,c);
	} else {
		stranspose(sig,rows,cols,col);
	}
}

void mtranspose(float *sig, int rows, int cols,float *col) {
	int block;
	
	block = (int) TBLOCK;

	if (rows >= block && cols >= block)  {
		ctranspose(sig,rows,cols,col);
	} else {
		stranspose(sig,rows,cols,col);
	}
}

void itranspose(float *A, int M, int N) {
	int i, j, p, iter;
	float *buf;
	float temp;

	if (M == N) {
		for (i = 0; i < N; ++i) {
			for (j = i + 1; j < N; ++j) {
				temp = A[i + j*N];
				A[i + j*N] = A[j + i*N];
				A[j + i*N] = temp;
			}
		}
	} else if (M > N) {

		p = M - N;
		buf = (float*)malloc(sizeof(float)* p * N);

		memcpy(buf, A + N * N, sizeof(*A)*p*N);

		for (i = 0; i < N; ++i) {
			for (j = i + 1; j < N; ++j) {
				temp = A[i + j*N];
				A[i + j*N] = A[j + i*N];
				A[j + i*N] = temp;
			}
		}

		for (i = N - 1; i >= 1; --i) {
			memmove(A + i*M, A + i*N, sizeof(*A)*M);
		}


		for (i = 0; i < N; ++i) {
			iter = N + i * M;
			for (j = 0; j < p; ++j) {
				A[iter + j] = buf[j*N + i];
			}
		}

		free(buf);
	}
	else if (M < N) {
		p = N - M;
		buf = (float*)malloc(sizeof(float)* p * M);

		for (i = 0; i < M; ++i) {
			iter = M + i*N;
			for (j = 0; j < p; ++j) {
				buf[j*M + i] = A[iter + j];
			}
		}

		for (i = 1; i < M; ++i) {
			memmove(A + i*M, A + i * N, sizeof(*A)*M);
		}

		for (i = 0; i < M; ++i) {
			for (j = i + 1; j < M; ++j) {
				temp = A[i + j*M];
				A[i + j*M] = A[j + i*M];
				A[j + i*M] = temp;
			}
		}
		memcpy(A + M*M, buf, sizeof(*A)*p*M);
		free(buf);
	}
}


void mdisplay(float *A, int row, int col) {
	int i,j;
	printf("\n MATRIX Order : %d X %d \n \n",row,col);
	
	for (i = 0; i < row; i++) {
		printf("R%d: ",i);
		for ( j = 0; j < col;j++) {
			printf("%f ",A[i*col + j]);
		}
		printf(":R%d \n",i);
	}
}

void madd(float* A, float* B, float* C,int rows,int cols) {
	int N,i;
	/*
	 * C = A + B . All matrices have identical dimensions rows X cols
	 */ 
	 
	N = rows * cols;
	
	#pragma omp parallel for
	for (i = 0; i < N; ++i) {
		C[i] = A[i] + B[i];
	}
}

void msub(float* A, float* B, float* C,int rows,int cols) {
	int N,i;
	/*
	 * C = A - B . All matrices have identical dimensions rows X cols
	 */ 
	 
	N = rows * cols;
	
	#pragma omp parallel for
	for (i = 0; i < N; ++i) {
		C[i] = A[i] - B[i];
	}
}

void scale(float *A, int rows, int cols, float alpha) {
	int N,i;
	/*
	 * A = alpha * A
	 * Matrix A is overwritten.
	 */ 
	 
	N = rows * cols;
	
	#pragma omp parallel for
	for (i = 0; i < N;++i) {
		A[i] = alpha * A[i];
	}
}

void nmult(float* A, float* B, float* C,int ra,int ca, int cb) {
	register int i,j,k;
	int u,v,t,rb;
	
	/*
	 * C = A * B , where A is a ra*ca matric while B is a rb*cb
	 * with ca = rb
	 * Matrix C is a ra*cb matrix
	 */ 
	 
	rb = ca;
	#pragma omp parallel for private(i,j,k,v,u,t)
	for (i = 0; i < ra; ++i) {
		for (j = 0; j < cb; ++j) {
			v = i * rb;
			u = i *cb;
			t = j + u;
			C[t] = 0.;
			for (k = 0; k < rb;++k) {
				C[t] += A[k + v] * B[j + k * cb];
			}
		}
	}


}

void tmult(float* A, float* B, float* C,int ra,int ca, int cb) {
	register int i,j,k;
	int u,v,t,rb;
	float *BT;
	BT = (float*) malloc(sizeof(float) * ca * cb);
	/*
	 * C = A * B , where A is a ra*ca matric while B is a rb*cb
	 * with ca = rb
	 * Matrix C is a ra*cb matrix
	 */ 
	 
	mtranspose(B,ca,cb,BT);
	rb = ca;
	#pragma omp parallel for private(i,j,k,v,u,t)
	for (i = 0; i < ra; ++i) {
		for (j = 0; j < cb; ++j) {
			v = i * rb;
			u = i *cb;
			t = j + u;
			C[t] = 0.;
			for (k = 0; k < rb;++k) {
				C[t] += A[k + v] * BT[k + j * rb];
			}
		}
	}
	
	free(BT);

}


void recmult(float* A, float* B, float* C,int m,int n, int p,int sA,int sB, int sC) {
	int m2,n2,p2;
	register int i,j,k;
	int u,v,t;
	if (m + n + p <= CUTOFF) {
		//#pragma omp parallel for private(i,j,k,v,u,t)
		for (i = 0; i < m; ++i) {
			for (j = 0; j < p; ++j) {
				v = i * sB;
				u = i * sC;
				t = j + u;
				for (k = 0; k < n;++k) {
					C[t] += A[k + v] * B[j + k * sC];
				}
			}
		}

		
	} else if (m >= n && m >= p) {
		m2 = (int) ceil((float) m / 2.0);
		recmult(A,B,C,m2,n,p,sA,sB,sC);
		recmult(A + m2*sB,B,C + m2*sC,m-m2,n,p,sA,sB,sC);
		
	} else if (n >= m && n >= p) {
		n2 = (int) ceil((float) n / 2.0);
		recmult(A,B,C,m,n2,p,sA,sB,sC);
		recmult(A+n2,B+n2*sC,C,m,n-n2,p,sA,sB,sC);
		
	} else if (p >= m && p >= n) {
		p2 = (int) ceil((float) p / 2.0);
		recmult(A,B,C,m,n,p2,sA,sB,sC);
		recmult(A,B+p2,C+p2,m,n,p-p2,sA,sB,sC);
	}
}

void rmult(float* A, float* B, float* C,int m,int n, int p) {
	int strA,strB,strC;
	int N;
	register int i;
	strA = m;
	strB = n;
	strC = p;
	N = m * p;
	for(i = 0; i < N; ++i) {
		C[i] = 0.;
	}
	
	recmult(A,B,C,m,n,p,strA,strB,strC);
	
}

int findrec(int *a, int *b, int *c) {
	int rec;
	float da,db,dc,mul;
	da = (float) *a;
	db = (float) *b;
	dc = (float) *c;
	rec = 0;
	mul = 1.;
	
	while (da + db + dc > (float) CUTOFF) {
		rec++;
		mul *= 2;
		da = ceilf(da/2.0f);
		db = ceilf(db/2.0f);
		dc = ceilf(dc/2.0f);
	}
	*a = (int) da * mul;
	*b = (int) db * mul;
	*c = (int) dc * mul;
	
	return rec;
}

void add_zero_pad(float *X, int rows, int cols, int zrow, int zcol,float *Y) {
	int r,c,i,j,u,v;
	r = rows + zrow;
	c = cols + zcol;
	
	for (i = 0; i < rows;++i) {
		u = i*c;
		v = i * cols;
		for (j = 0; j < cols;++j) {
			Y[u + j] = X[v + j];
		}
		for (j = cols; j < c;++j) {
			Y[u + j] = 0.;
		}
	}
	
	for (i = rows; i < r;++i) {
		u = i*c;
		for(j = 0; j < c;++j) {
			Y[u + j] = 0.;
		}
	}
	
}

void remove_zero_pad(float *Y, int rows, int cols, int zrow, int zcol,float *Z) {
	int r,c,i,j,u,v;
	r = rows - zrow;
	c = cols - zcol;
	
	for (i = 0; i < r; ++i) {
		u = i * c;
		v = i * cols;
		for (j = 0; j < c; ++j) {
			Z[j + u] = Y[j + v];
		}
	}
}

void madd_stride(float* A, float* B, float* C,int rows,int cols,int sA,int sB,int sC) {
	int i,j,u,v,w;
	 
	for (i = 0; i < rows; ++i) {
		u = i * sC;
		v = i * sA;
		w = i * sB;
		for(j = 0; j < cols;j++) {
			C[j + u] = A[j + v] + B[j + w];
		}
	}
}

void msub_stride(float* A, float* B, float* C,int rows,int cols,int sA,int sB,int sC) {
	int i,j,u,v,w;
	 
	for (i = 0; i < rows; ++i) {
		u = i * sC;
		v = i * sA;
		w = i * sB;
		for(j = 0; j < cols;j++) {
			C[j + u] = A[j + v] - B[j + w];
		}
	}
}

void rmadd_stride(float* A, float* B, float* C,int rows,int cols,int p,int sA,int sB,int sC) {
	int i,j,u,v,w;
	if (rows + cols + p <= CUTOFF) {
		for (i = 0; i < rows; ++i) {
			u = i * sC;
			v = i * sA;
			w = i * sB;
			for(j = 0; j < cols;j++) {
				C[j + u] = A[j + v] + B[j + w];
			}
		}
		
	 } else {
		 rows/=2;cols/=2;p/=2;
		 rmadd_stride(A,B,C,rows,cols,p,sA,sB,sC);
		 rmadd_stride(A + cols,B + cols,C + cols,rows,cols,p,sA,sB,sC);
		 rmadd_stride(A + rows *sB,B + rows *sC,C + rows *sC,rows,cols,p,sA,sB,sC);
		 rmadd_stride(A + rows *sB + cols,B + rows *sC + cols,C + rows *sC + cols,rows,cols,p,sA,sB,sC);
	 }
}

void rmsub_stride(float* A, float* B, float* C,int rows,int cols,int p,int sA,int sB,int sC) {
	int i,j,u,v,w;
	if (rows + cols + p <= CUTOFF) {
		for (i = 0; i < rows; ++i) {
			u = i * sC;
			v = i * sA;
			w = i * sB;
			for(j = 0; j < cols;j++) {
				C[j + u] = A[j + v] - B[j + w];
			}
		}
		
	 } else {
		 rows/=2;cols/=2;p/=2;
		 rmsub_stride(A,B,C,rows,cols,p,sA,sB,sC);
		 rmsub_stride(A + cols,B + cols,C + cols,rows,cols,p,sA,sB,sC);
		 rmsub_stride(A + rows *sB,B + rows *sC,C + rows *sC,rows,cols,p,sA,sB,sC);
		 rmsub_stride(A + rows *sB + cols,B + rows *sC + cols,C + rows *sC + cols,rows,cols,p,sA,sB,sC);
	 }
}

void srecmult(float* A, float* B, float* C,int m,int n, int p,int sA,int sB, int sC) {
	register int i,j,k;
	int u,v,t;
	float sum;
	float *A1,*B1;
	float *a11,*a12,*a21,*a22;
	float *b11,*b12,*b21,*b22;
	float *c11,*c12,*c21,*c22;
	float *m1,*m2,*m3,*m4,*m5,*m6,*m7;
	int sm1,sm2,sm3,sm4,sm5,sm6,sm7;
	int sA1,sB1;
	if (m + n + p <= CUTOFF) {
		for (i = 0; i < m; ++i) {
			for (j = 0; j < p; ++j) {
				v = i * sA;
				u = i * sC;
				t = j + u;
				sum = 0.;
				for (k = 0; k < n;++k) {
					sum += A[k + v] * B[j + k * sB];
				}
				C[t] = sum;
			}
		}

		
	} else {
		m/=2;n/=2;p/=2;
		// A size mXn, C size mXp
		a11 = A;
		a12 = A + n;
		a21 = A + m * sA;
		a22 = A + n + m * sA;
		
		//B size nXp
		
		b11 = B;
		b12 = B + p;
		b21 = B + n * sB;
		b22 = B + p + n * sB;
		
		//C size mXp
		
		c11 = C;
		c12 = C + p;
		c21 = C + m * sC;
		c22 = C + p + m * sC;
		
		// m matrices have dimension m X p each. See http://en.wikipedia.org/wiki/Strassen_algorithm
		
		m1 = (float*) malloc(sizeof(float) *m * p);
		sm1 = p;
		
		m3 = (float*) malloc(sizeof(float) *m * p);
		sm3 = p;
		
		m4 = (float*) malloc(sizeof(float) *m * p);
		sm4 = p;
		
		m2 = c21;
		sm2 = sC;
		
		m5 = c12;
		sm5 = sC;
		
		m6 = c22;
		sm6 = sC;
		
		
		m7 = c11;
		sm7 = sC;
		
		//m1
		
		sA1 = n;
		sB1 = p;
		
		A1 = (float*) malloc(sizeof(float) * m * n);
		B1 = (float*) malloc(sizeof(float) * n * p);
		
		madd_stride(a11,a22,A1,m,n,sA,sA,sA1);
		
		madd_stride(b11,b22,B1,n,p,sB,sB,sB1);
		
		srecmult(A1,B1,m1,m,n,p,sA1,sB1,sm1);
		
		free(A1);
		free(B1);
		
		
		//m2
		
		A1 = (float*) malloc(sizeof(float) * m * n);
		
		madd_stride(a21,a22,A1,m,n,sA,sA,sA1);
				
		srecmult(A1,b11,m2,m,n,p,sA1,sB,sm2);
		
		free(A1);
		
		
		//m3
		
		B1 = (float*) malloc(sizeof(float) * n * p);
		//rmsub_stride(B + p,B + p + n * sC,B1,n,p,m,sC,sC,sC/2);
		msub_stride(b12,b22,B1,n,p,sB,sB,sB1);
		srecmult(a11,B1,m3,m,n,p,sA,sB1,sm3);
		
		free(B1);
		
		//m4
		
		B1 = (float*) malloc(sizeof(float) * n * p);
		//rmsub_stride(B + p,B + p + n * sC,B1,n,p,m,sC,sC,sC/2);
		msub_stride(b21,b11,B1,n,p,sB,sB,sB1);
		srecmult(a22,B1,m4,m,n,p,sA,sB1,sm4);
		
		free(B1);
		
		//m5
		
		A1 = (float*) malloc(sizeof(float) * m * n);
		
		madd_stride(a11,a12,A1,m,n,sA,sA,sA1);
				
		srecmult(A1,b22,m5,m,n,p,sA1,sB,sm5);
		
		free(A1);
		
		
		//m6
		
		A1 = (float*) malloc(sizeof(float) * m * n);
		B1 = (float*) malloc(sizeof(float) * n * p);
		
		msub_stride(a21,a11,A1,m,n,sA,sA,sA1);
		madd_stride(b11,b12,B1,n,p,sB,sB,sB1);
		srecmult(A1,B1,m6,m,n,p,sA1,sB1,sm6);
	
		free(A1);
		free(B1);
		
		//m7
		
		A1 = (float*) malloc(sizeof(float) * m * n);
		B1 = (float*) malloc(sizeof(float) * n * p);
		
		msub_stride(a12,a22,A1,m,n,sA,sA,sA1);
		madd_stride(b21,b22,B1,n,p,sB,sB,sB1);
		srecmult(A1,B1,m7,m,n,p,sA1,sB1,sm7);
	
		free(A1);
		free(B1);
		
		
		// c11
		
		A1 = (float*) malloc(sizeof(float) * m * p);
		sA1 = p;
		madd_stride(m1,m7,m7,m,p,sm1,sm7,sm7);
		msub_stride(m4,m5,A1,m,p,sm4,sm5,sA1);
		madd_stride(m7,A1,m7,m,p,sm7,sA1,sm7);
		
		free(A1);
		
		
		// c22
		
		A1 = (float*) malloc(sizeof(float) * m * p);
		sA1 = p;
		madd_stride(m1,m6,m6,m,p,sm1,sm6,sm6);
		msub_stride(m3,m2,A1,m,p,sm3,sm2,sA1);
		madd_stride(m6,A1,m6,m,p,sm6,sA1,sm6);
		
		free(A1);
		
		//c12 
		
		madd_stride(m3,m5,m5,m,p,sm3,sm5,sm5);
		
		//c21
		
		madd_stride(m4,m2,m2,m,p,sm4,sm2,sm2);
		
		free(m1);
		free(m3);
		free(m4);
	}
}

void smult(float* A, float* B, float* C,int m,int n, int p) {
	int a,b,c,nrec;
	float *X,*Y,*Z,*P;
	a = m;
	b = n;
	c = p;
	nrec = findrec(&a,&b,&c);
	X = (float*) malloc(sizeof(float) * a * b);
	Y = (float*) malloc(sizeof(float) * b * c);
	Z = (float*) malloc(sizeof(float) * a * c);
	P = (float*) malloc(sizeof(float) * (a/2) * (c/2));

	
	add_zero_pad(A,m,n,a-m,b-n,X);
	add_zero_pad(B,n,p,b-n,c-p,Y);

	srecmult(X,Y,Z,a,b,c,b,c,c);
	// Memory allocation needs work
	
	remove_zero_pad(Z,a,c,a-m,c-p,C);
	
	// free X,Y,Z
	free(X);
	free(Y);
	free(Z);
	free(P);
	
}

void mmult(float* A, float* B, float* C,int m,int n, int p) {
	if (m+n+p <= CUTOFF/2) {
		nmult(A,B,C,m,n,p);
	} else {
		smult(A,B,C,m,n,p);
	}
}

static int pludecomp(float *A,int N,int *ipiv) {
	int k,j,l,c1,c2,mind,tempi;
	float ld,mult,mval,temp;
	for(k=0;k < N;++k)
		ipiv[k] = k;
	
	for(k = 0; k < N-1; ++k) {
		//c2 = k*N;
		mval = fabsf(A[k*N + k]);
		mind = k;
		for (j=k+1; j < N;++j) {
			if (mval < fabs(A[j*N + k])) {
				mval = A[j*N + k];
				mind = j;
			}
		}
		
		if ( mind != k) {
			c1 = k *N;
			c2 = mind * N;
			tempi = ipiv[mind];
			ipiv[mind] = ipiv[k];
			ipiv[k] = tempi;
			for (j = 0; j < N;j++) {
				temp = A[c1 + j];
				*(A + c1 + j) = *(A + c2 + j);
				*(A + c2 + j) = temp;
			}
		}
		c2 = k*N;
		ld = A[c2 + k];
		if (ld != 0.) {
			for (j = k+1; j < N; ++j) {
				c1 = j*N;
				mult = A[c1+k] /= ld;
				//printf("\n k %d j %d mult %f \n",k,j,mult);
				for(l = k+1; l < N; ++l) {
					A[c1+l] -= mult * A[c2 + l];
				}
			}
		}
		
	}
	return 0;
	
}

void ludecomp(float *A,int N,int *ipiv) {
	pludecomp(A,N,ipiv);
}

void linsolve(float *A,int N,float *b,int *ipiv,float *x) {
	int i,j,c1,l;
	float *y;
	float sum;
	
	y = (float*) malloc(sizeof(float) *N);
	/*
	 * Two step Solution L * U * x = b
	 * Let U*x = y
	 * Solve L * y = b for y (Forward Substitution
	 * Solve U * x = b for x (Back Substitution)
	 */ 
	for(i = 0; i < N;++i) {
		y[i] = 0.;
		x[i] = 0.;
		if ( A[i*N + i] == 0.) {
			printf("The Matrix system does not have a unique solution");
			exit(1);
		}
		//printf("\n B %d",ipiv[i]);
	}
	
	// Forward Substitution
	
	y[0] = b[ipiv[0]];
	for(i = 1; i < N; ++i) {
		sum = 0.;
		c1 = i*N;
		for(j = 0; j < i; ++j) {
			sum += y[j] * A[c1 + j];
		}
		y[i] = b[ipiv[i]] - sum;
	}
	
	// Back Substitution
	
	x[N - 1] = y[N - 1]/A[N * N - 1];
	
	for (i = N - 2; i >= 0; i--) {
		sum = 0.;
		c1 = i*(N+1);
		l=0;
		for(j = i+1; j < N;j++) {
			l++;
			sum += A[c1 + l] * x[j];
		}
		x[i] = (y[i] - sum) / A[c1];
	}
	
	free(y);
}

void minverse(float *A,int N,int *ipiv,float *inv) {
	int i,j,stride;
	float *col,*x;
	
	col = (float*) malloc(sizeof(float) * N);
	x = (float*) malloc(sizeof(float) * N);
	
	for (i = 0; i < N; ++i) {
		col[i] = 0.;
		x[i] = 0.;
	}
	
	for (i = 0; i < N; ++i) {
		col[i] = 1.;
		linsolve(A,N,col,ipiv,x);
		stride = i;
		for(j = 0; j < N;++j) {
			inv[stride] = x[j];
			stride+= N;
		}
		col[i] = 0.;
	}
		
	free(x);
	free(col);
}

void eye(float *mat,int N) {
	int i,j,t;
	for(i = 0;i < N;++i) {
		for(j =0; j < N;++j) {
			t = i*N;
			if (i == j) {
				mat[t+j] = 1.;
			} else {
				mat[t+j] = 0.;
			}
		}
		
	}
}

static float house_1(float*x,int N,float *v) {
	float beta,mu,temp;
	float *sigma;
	int i;
	
	sigma = (float*) malloc(sizeof(float) * 1);
	
	if (N > 1) {
		mmult(x+1,x+1,sigma,1,N-1,1);
	} else {
		sigma[0] = 0.0;
	}
	
	v[0] =1.;
	for (i = 1; i < N;++i) {
		v[i] = x[i];
	}
	
	if (sigma[0] == 0. && x[0] >= 0.) {
		beta = 0.;
	} else if (sigma[0] == 0. && x[0] < 0.) {
		beta = -2.;
	}else {
		mu = sqrtf(sigma[0] + x[0] * x[0]);
		
		if (x[0] <= 0.) {
			v[0] = x[0] - mu;
		} else {
			v[0] = - sigma[0] / (x[0] + mu);
		}
		temp = v[0];
		
		beta = (2.0f * v[0] * v[0]) /(sigma[0] + v[0] * v[0]);
		
		for (i = 0; i < N;++i) {
			v[i] /= temp;
		}
		
	}
	
	free(sigma);
	return beta;
}

float house_2(float*x,int N,float *v) {
	float sgn,beta,sc;
	float *sigma,*e;
	int i;
	
	sigma = (float*) malloc(sizeof(float) * 1);
	e = (float*) malloc(sizeof(float) * N);
	
	beta = 2.0;
	sgn = 1.0;
	mmult(x,x,sigma,1,N,1);
	sigma[0] = sqrtf(sigma[0]);
	
	e[0] =1.;
	for (i = 1; i < N;++i) {
		e[i] = 0.;
	}
	
	if (x[0] > 0.) {
		sgn = 1.0;
	} else if (x[0] < 0.) {
		sgn = -1.0;
	} else if (x[0] == 0.) {
		sgn = 0.;
	}
	
	sc = sigma[0] * sgn;
	
	//scale(e,N,1,sc);
	
	e[0] *= sc;
	
	for(i = 0; i < N;++i) {
		v[i] = e[i] + x[i];
	}
	
	mmult(v,v,sigma,1,N,1);
	sigma[0] = sqrtf(sigma[0]);
	
	for(i = 0; i < N;++i) {
		v[i] = v[i] / sigma[0];
	}
	
	free(sigma);
	free(e);
	return beta;
}

float house(float*x,int N,float *v) {
	float beta;
	beta = house_1(x,N,v);
	return beta;
}


void housemat(float *v, int N,float beta,float *mat) {
	float *temp;
	
	temp = (float*) malloc(sizeof(float) * N * N);
	eye(mat,N);
	mmult(v,v,temp,N,1,N);
	scale(temp,N,N,beta);
	msub(mat,temp,mat,N,N);
	
	free(temp);
}

void qrdecomp(float *A, int M, int N,float *bvec) {
	int j,i,k,u,t;
	float *x,*v,*AT,*w;
	float beta;
	
	if (M < N) {
			printf("M should be greater than or equal to N");
			exit(1);
	}
	x = (float*) malloc(sizeof(float) * M);
	v = (float*) malloc(sizeof(float) * M);
	AT = (float*) malloc(sizeof(float) * M * N);
	w = (float*) malloc(sizeof(float) * M * M);
	
	for(j = 0; j < N;++j) {
		for(i=j;i < M;++i) {
			x[i-j] = A[i*N+j];
			
		}
		
		beta = house(x,M-j,v);
		bvec[j] = beta;
	
		for (i=j; i < M; i++) {
			t = i * N;
			u = 0;
			for (k=j; k < N; k++) {
				AT[u+i-j] = A[k+t];
				u+=(M-j);
				
			}
			
		}
		
		
		mmult(AT,v,w,N-j,M-j,1);
		scale(w,N-j,1,beta);
		mmult(v,w,AT,M-j,1,N-j);
		for (i=j; i < M; i++) {
			t = i *N;
			for (k=j; k < N; k++) {
				A[t+k] -= AT[(i-j)*(N-j) + k - j];
			}
		}
		if (j < M) {
			for(i=j+1;i < M;++i) {
				A[i*N+j] = v[i-j];
			}
		}
		 
	}
	
	free(x);
	free(v);
	free(AT);
	free(w);
	
}

void getQR(float *A,int M,int N,float *bvec,float *Q, float *R) {
	int i,j,k,t,u;
	float *x,*v,*AT,*w;
	
	x = (float*) malloc(sizeof(float) * M);
	v = (float*) malloc(sizeof(float) * M);
	AT = (float*) malloc(sizeof(float) * M * N);
	w = (float*) malloc(sizeof(float) * M * M);
	
	for(i = 0; i < N;++i) {
		t = i *N;
		for(j = 0; j < N;++j) {
			if (i > j) {
				R[t+j] = 0.;
			} else {
				R[t+j] = A[t+j];
			}
		}
	}
	
	for(i = 0; i < M;++i) {
		t = i *N;
		for(j = 0; j < N;++j) {
			if (i == j) {
				Q[t+j] = 1.;
			} else {
				Q[t+j] = 0.;
			}
		}
	}
	
	for(j = N-1; j >= 0;--j) {
		v[0] = 1.;
		for(i=j+1;i < M;++i) {
			v[i-j] = A[i*N+j];
			
		}
		
		for (i=j; i < M; i++) {
			t = i * N;
			u = 0;
			for (k=j; k < N; k++) {
				AT[u+i-j] = Q[k+t];
				u+=(M-j);
			}
			
		}
	
		mmult(AT,v,w,N-j,M-j,1);
		scale(w,N-j,1,bvec[j]);
		mmult(v,w,AT,M-j,1,N-j);
		
		for (i=j; i < M; i++) {
			t = i *N;
			for (k=j; k < N; k++) {
				Q[t+k] -= AT[(i-j)*(N-j) + k - j];
			}
		}
	 
	}
	
	free(x);
	free(v);
	free(AT);
	free(w);
}

void hessenberg(float *A,int N) {
	int k,i,j,t,u;
	float *x,*v,*AT,*w;
	float beta;
	x = (float*) malloc(sizeof(float) * N);
	v = (float*) malloc(sizeof(float) * N);
	AT = (float*) malloc(sizeof(float) * N * N);
	w = (float*) malloc(sizeof(float) * N);
	
	for (k = 0; k < N-2;++k) {
		for(i=k + 1;i < N;++i) {
			x[i-k-1] = A[i*N+k];
			//printf("x %f \n",x[i-k-1]);
			
		}
		
		beta = house(x,N-k-1,v);
		
		
		for (i=k+1; i < N; i++) {
			t = i * N;
			u = 0;
			for (j=k; j < N; j++) {
				AT[u+i-k-1] = A[j+t];
				u+=(N-k-1);				
			}
		}
		//mdisplay(AT,N-k,N-k-1);
		
		
		mmult(AT,v,w,N-k,N-k-1,1);
		scale(w,N-k,1,beta);
		mmult(v,w,AT,N-k-1,1,N-k);
		//mdisplay(AT,N-k-1,N-k);
		
		for (i=k+1; i < N; i++) {
			t = i * N;
			for (j=k; j < N; j++) {
				A[t+j] -= AT[(i-k-1)*(N-k) + j - k];
			}
		}
		//mdisplay(A,N,N);
		
		for (i=0; i < N; i++) {
			t = i * N;
			u = i * (N-k-1);
			for (j=k+1; j < N; j++) {
				AT[u+j-k-1] = A[t+j];
			}
		}
		//mdisplay(AT,N,N-k-1);
		
		mmult(AT,v,w,N,N-k-1,1);
		scale(w,N,1,beta);
		mmult(w,v,AT,N,1,N-k-1);
		//mdisplay(AT,N,N-k-1);
		
		for (i=0; i < N; i++) {
			t = i * N;
			u = i * (N-k-1);
			for (j=k+1; j < N; j++) {
				A[t+j] -= AT[u+j-k-1];
			}
		}
		
	}
	
	free(x);
	free(v);
	free(AT);
	free(w);
}

void francisQR(float *A,int N) {
	int m,n,k,q,r,t,u,i,j;
	float s,t2,beta;
	float *x,*v,*AT,*w;
	int NN;
	/*
	 * Reference - Algorithm 7.5.1 Golub,van Loan Matrix Computations 3rd Edition
	 */ 
	x = (float*) malloc(sizeof(float) * 3);
	v = (float*) malloc(sizeof(float) * 3);
	AT = (float*) malloc(sizeof(float) * 3 * N);
	w = (float*) malloc(sizeof(float) * N);
	n = N-1;
	m = n-1;
	NN = N*N;
	
	s = A[NN-1] + A[NN-N-2];
	t2 = A[NN-1] * A[NN-N-2] - A[NN-2] * A[NN-N-1];
	
	x[0] = A[0]*A[0] + A[1]*A[N] - s*A[0] + t2;
	x[1] = A[N]*(A[0] + A[N+1] - s);
	x[2] = A[N] * A[N+N+1];
	if (N <= 2) {
		return;
	}
	
	for (k = -1; k < N - 3;++k) {
		
		beta = house(x,3,v);
		//mdisplay(x,3,1);
		if (k > 0) {
			q = k;
		} else {
			q = 0;
		}
		
		//printf("q %d \n",q);
		for (i=k+1; i < k+4; i++) {
			t = i * N;
			u = 0;
			for (j=q; j < N; j++) {
				AT[u+i-k-1] = A[j+t];
				u+=3;		
			}
		}
		
		mmult(AT,v,w,N-q,3,1);
		scale(w,N-q,1,beta);
		mmult(v,w,AT,3,1,N-q);
		
		for (i=k+1; i < k+4; i++) {
			t = i * N;
			for (j=q; j < N; j++) {
				A[t+j] -= AT[(i-k-1)*(N-q) + j - q];
			}
		}
		//mdisplay(A,N,N);
		if (k+4 >= n) {
			r = N;
		} else {
			r = k+4+1;
		}
		//printf("r %d \n",r);
		for (i=0; i < r; i++) {
			t = i * N;
			u = i * 3;
			for (j=k+1; j < k+4; j++) {
				AT[u+j-k-1] = A[t+j];
			}
		}
		
		mmult(AT,v,w,r,3,1);
		scale(w,r,1,beta);
		mmult(w,v,AT,r,1,3);
		//mdisplay(AT,N,N-k-1);
		
		for (i=0; i < r; i++) {
			t = i * N;
			u = i * 3;
			for (j=k+1; j < k+4; j++) {
				A[t+j] -= AT[u+j-k-1];
			}
		}
		//mdisplay(A,N,N);
		x[0] = A[N*(k+2) + k+1];
		x[1] = A[N*(k+3) + k+1];
		
		if (k < n-3) {
			x[2] = A[N*(k+4) + k+1];
		} 
		//mdisplay(x,3,1);
		
	}
	//mdisplay(x,2,1);
	beta = house(x,2,v);
	
	for (i=n-1; i < N; i++) {
		t = i * N;
		u = 0;
		for (j=n-2; j < N; j++) {
			AT[u+i-n+1] = A[j+t];
			u+=2;		
		}
	}
	
	mmult(AT,v,w,3,2,1);
	scale(w,3,1,beta);
	mmult(v,w,AT,2,1,3);
	for (i=n-1; i < N; i++) {
		t = i * N;
		for (j=n-2; j < N; j++) {
			A[t+j] -= AT[(i-n+1)*3 + j - n + 2];
		}
	}
	
	for (i=0; i < N; i++) {
		t = i * N;
		u = i * 2;
		for (j=n-1; j < N; j++) {
			AT[u+j-n+1] = A[t+j];
		}
	}
	
	mmult(AT,v,w,N,2,1);
	scale(w,N,1,beta);
	mmult(w,v,AT,N,1,2);
		//mdisplay(AT,N,N-k-1);
		
	for (i=0; i < N; i++) {
		t = i * N;
		u = i * 2;
		for (j=n-1; j < N; j++) {
			A[t+j] -= AT[u+j-n+1];
		}
	}
	
	
	free(x);
	free(v);
	free(AT);
	free(w);
	
}

void eig22(float *A, int stride,float *eigre,float *eigim) {
	int N;
	float a11,a12,a21,a22,c,s,c2,s2,cs,t1,t,t2,at11,at12,at21,at22;
	N = stride;
	
	a11 = A[0];
	a12 = A[1];
	a21 = A[N];
	a22 = A[N+1];
	
	if ( (a12 + a21) == 0) {
		c = 1.0f/sqrtf(2.0f);
		s = c;
	} else {
		t1 = (a11 - a22) / (a12 + a21);
		t = t1 /(1.0f + sqrtf(1+t1*t1));
		c = 1.0f/sqrtf(1 + t*t);
		s = c*t;
	}
	
	c2 = c*c;
	s2 = s*s;
	cs = c*s;

	at11 = c2 * a11 + s2 * a22 - cs * (a12 + a21);
	at12 = c2 * a12 - s2 * a21 + cs * (a11 - a22);
	at21 = c2 * a21 - s2 * a12 + cs * (a11 - a22);
	at22 = c2 * a22 + s2 * a11 + cs * (a12 + a21);
	
	eigre[0] = eigre[1] = at11;
	eigim[0] = sqrtf(-at12 * at21);
	eigim[1] = -sqrtf(-at12 * at21);
	
	if ( at12*at21 >= 0) {
		if (at12 == 0) {
			c = 0;
			s = 1;
			c2 = 0;
			s2 = 1;
			cs = 0;
		} else {
			t = sqrtf(at21/at12);
			t2 = t * t;
			cs = t/(1+t2);
			c2 = (1+t2);
			s2 = t2 /(1+t2);
		}
		eigim[0] = eigim[1] = 0.0;
		eigre[0] = at11 - cs * (at12 + at21);
		eigre[1] = at11 + cs * (at12 + at21);
		
	}
	
}

int francis_iter(float *A, int N, float *H) {
	int success,brkpoint;
	int i,j,it,p,q,t,u;
	float *temp;
	success = 0;
	brkpoint = 30 * N;
	it = 0;
	p = N - 1;
	temp = (float*) malloc(sizeof(float) * N * N);
	for(i = 0; i < N*N;++i) {
		H[i] = A[i];
	}
	
	hessenberg(H,N);
	
	while (p > 1 && it < brkpoint) {
		
		while (p > 1 && (H[N*p + p-1] == 0 || H[N*(p-1) + p-2] == 0)) {
			if (H[N*p + p-1] == 0) {
				p--;
			} else if (H[N*(p-1) + p-2] == 0) {
				p=p-2;
			}
		}
		
		if (p > 0) {
			q = p-1;
			while (q > 0 && fabs(H[N*q + q-1]) != 0) {
				q--;
			}
			//printf("%d %d \n",q,p);
			for (i=q; i <= p; i++) {
				t = i * N;
				u = (i-q) * (p-q+1);
				for (j=q; j <= p; j++) {
					temp[u+j-q] = H[t+j];
				}
			}
			francisQR(temp,p-q+1);
			for (i=q; i <= p; i++) {
				t = i * N;
				u = (i-q) * (p-q+1);
				for (j=q; j <= p; j++) {
					H[t+j] = temp[u+j-q];
				}
			}
			//mdisplay(H,N,N);
			for(i = q; i <= p-1;++i) {
				if ( fabs(H[(i+1)*N+i]) <= TOL * (fabs(H[i*N+i]) + fabs(H[(i+1)*N+i+1]) ) ) {
					H[(i+1)*N+i] = 0.;
				}
			}
			it++;
			//printf("iter %d \n",it);
		}
	}
	
	if (it == brkpoint) {
		success = 0;
	} else {
		success = 1;
	}
	
	free(temp);
	return success;
}

static void eig2t(float *A, int stride) {
	int N;
	float a11,a12,a21,a22,c,s,c2,s2,cs,t1,t,at11,at12,at21,at22;
	N = stride;
	
	a11 = A[0];
	a12 = A[1];
	a21 = A[N];
	a22 = A[N+1];
	
	if ( (a12 + a21) == 0) {
		c = 1.0f/sqrtf(2.0f);
		s = c;
	} else {
		t1 = (a11 - a22) / (a12 + a21);
		t = t1 /(1.0f + sqrtf(1+t1*t1));
		c = 1.0f/sqrtf(1 + t*t);
		s = c*t;
	}
	
	c2 = c*c;
	s2 = s*s;
	cs = c*s;

	at11 = c2 * a11 + s2 * a22 - cs * (a12 + a21);
	at12 = c2 * a12 - s2 * a21 + cs * (a11 - a22);
	at21 = c2 * a21 - s2 * a12 + cs * (a11 - a22);
	at22 = c2 * a22 + s2 * a11 + cs * (a12 + a21);
	A[0] = at11;
	A[1] = at12;
	A[N] = at21;
	A[N+1] = at22;

}

void eig(float *A,int N,float *eigre,float *eigim) {
	int i,t,u,n;
	float *H;
	float t1,t2,cs;
	H = (float*) malloc(sizeof(float) * N * N);
	n = N - 1;
	francis_iter(A,N,H);
	//mdisplay(H,N,N);
	i = 0;
	while (i < n) {
		u = i * N;
		t = (i+1)*N;
		if (H[t+i] != 0.) {
			eig2t(H+u+i,N);
			i = i +2;
		} else {
			i++;
		}
		
	}
	//mdisplay(H,N,N);
	i = 0;
	while (i < n) {
		u = i * N;
		t = (i+1)*N;
		
		if (H[t+i] != 0.) {
			if (H[u+i+1] * H[t+i] < 0.) {
				eigre[i] = H[u+i];
				eigre[i+1] = H[t+i+1];
				eigim[i] = sqrtf(-H[u+i+1] * H[t+i]);
				eigim[i+1] = -sqrtf(-H[u+i+1] * H[t+i]);
			} else {
				if (H[u+i+1] == 0.) {
					cs = 0.;
				} else {
					t1 = sqrtf(H[t+i]/H[u+i+1]);
					t2 = t1 * t1;
					cs = t1/(1+t2);
				}
				eigre[i] = H[u+i] - cs * (H[u+i+1] + H[t+i]);
				eigre[i+1] = H[u+i] + cs * (H[u+i+1] + H[t+i]);
				eigim[i] = 0.;
				eigim[i+1] = 0.;
				
			}
			
			i= i + 2;
			
		} else {
			eigre[i] = H[u+i];
			eigim[i] = 0.;
			i++;
		}
		
	}
	
	if (i == n) {
		eigre[i] = H[N*N - 1];
		eigim[i] = 0.;
	}
	
	free(H);
}

static int rcholu(float *A,int N, int stride, float *U22) {
	int sc;
	int j,i,u,w;
	float u11;
	
	if (N == 1) {
		if (A[0] > 0) {
			A[0] = sqrtf(A[0]);
			return 0;
		} else {
			return -1;
		}
	} else {
		if (A[0] < 0) {
			return -1;
		}
		u11 = sqrtf(A[0]);
		A[0] = u11;
		for (j = 1; j < N;++j) {
			A[j] /= u11;
		}
		mmult(A+1,A+1,U22,N-1,1,N-1);
		for (i = 0; i < N-1; ++i) {
			u = stride + 1+ i * stride;
			w = i * (N-1);
			for(j = i; j < N-1;j++) {
				A[j + u] -= U22[j + w];
			}
		}
		
		sc = rcholu(A+stride+1,N-1,stride,U22);
		if (sc == -1) {
			return -1;
		}
		
	}
	
	return sc;
	
}

static int rbcholu(float *A,int N, int stride, float *UB, float *UT) {
	int bs,bb,i,j,Nb,t,k,u,v,w,sc;
	float *b,*x,*U12,*U12T;
	float sum;
	
	bs = (int) BLOCKSIZE;
	bb = bs*bs;
	
	if (N <= BLOCKSIZE) {
		sc = rcholu(A,N,stride,UB);
		if (sc == -1) {
			return -1;
		}
	} else {
		Nb = N - bs;
		x = (float*) malloc(sizeof(float) * bs);
		b = (float*) malloc(sizeof(float) * bs);
		U12T = (float*) malloc(sizeof(float) * Nb * bs);
		U12 = (float*) malloc(sizeof(float) * Nb * bs);
		rcholu(A,bs,stride,UB); // U11
		
		for (i =0; i < bs;++i) {
			t = i *stride;
			u = 0;
			for(j = 0; j < N;++j) {
				UT[u+i] = A[j+t];
				u += bs;
			}
		}
		
		for(k = 0; k < Nb;++k) {
			u = k * bs;
			for(i = 0; i < bs;++i) {
				b[i] = UT[bb+u+i];
				x[i] = 0.;
			}
			for (i = 0; i < bs;++i) {
				t = i*bs;
				sum = 0;
				for (j = 0; j < i;++j) {
					sum += UT[t+j] * x[j];
				}
				x[i] = (b[i] - sum) / UT[t+i];
			}
			v = bs + k;
			for(i = 0; i < bs;++i) {
				A[v] = x[i];
				U12T[u+i] = x[i];
				v += stride;
			}
		}
		
		mtranspose(U12T,Nb,bs,U12);
		mmult(U12T,U12,UT,Nb,bs,Nb);
		free(U12T);
		free(U12);
		free(b);
		free(x);
		for (i = 0; i < Nb; ++i) {
			u = bs * stride + bs + i * stride;
			w = i * Nb;
			for(j = i; j < Nb;j++) {
				A[j + u] -= UT[j + w];
			}
		}
		
		sc = rbcholu(A + bs * stride + bs,Nb,stride,UB,UT);
		if (sc == -1) {
			return -1;
		}
	}
	
	return sc;
}

int cholu(float *A, int N) {
	int stride,i,j,t,sc;
	float *U22;
	U22 = (float*) malloc(sizeof(float) * N * N);
	stride = N; 
	
	sc = rcholu(A,N,stride,U22);
	
	for(i=0; i < N;++i) {
		t = i *N;
		for(j=0;j < i;++j) {
			A[t+j] = 0.;
		}
	}

	free(U22);
	return sc;
	
}

int bcholu(float *A, int N) {
	int stride,i,j,t,b,sc;
	float *UB,*UT;
	b = (int) BLOCKSIZE;
	UT = (float*) malloc(sizeof(float) * N * N);
	UB = (float*) malloc(sizeof(float) * b * b);
	stride = N; 
	
	sc = rbcholu(A,N,stride,UB,UT);
	
	for(i=0; i < N;++i) {
		t = i *N;
		for(j=0;j < i;++j) {
			A[t+j] = 0.;
		}
	}

	free(UB);
	free(UT);
	
	return sc;
	
}

int chol(float *A, int N) {
	int sc;
	if ( N <= (int) BLOCKSIZE) {
		sc = cholu(A,N);
	} else {
		sc = bcholu(A,N);
	}
	return sc;
}

static void rchold(float *A,int N, int stride, float *U22) {
	int j,i,u,w;
	float d1;
	
	if (N == 1) {
		return;
	} else {
		d1 = A[0];
		for (j = 1; j < N;++j) {
			A[j] /= d1;
		}
		mmult(A+1,A+1,U22,N-1,1,N-1);
		scale(U22,N-1,N-1,d1);
		for (i = 0; i < N-1; ++i) {
			u = stride + 1+ i * stride;
			w = i * (N-1);
			for(j = i; j < N-1;j++) {
				A[j + u] -= U22[j + w];
			}
		}
		
		rchold(A+stride+1,N-1,stride,U22);
	
	}
		
}

void chold(float *A, int N) {
	int stride,i,j,t;
	float *U22;
	U22 = (float*) malloc(sizeof(float) * N * N);
	stride = N; 
	
	rchold(A,N,stride,U22);
	
	for(i=0; i < N;++i) {
		t = i *N;
		for(j=0;j < i;++j) {
			A[t+j] = 0.;
		}
	}

	free(U22);
	
}

void svd_sort(float *U,int M,int N,float *V,float *q) {
	/*
	 * Pavel Sakov's CSA SVD sort routine is used with some minor
	 * modifications. See The License below
	 */
	/*
	 * Copyright (C) 2000-2008 Pavel Sakov and CSIRO

Redistribution and use of material from the package `csa', with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of material must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. The names of the authors may not be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
	 */
	int i,j;
	float *UT,*VT,*qq;
	int *pos;

	UT = (float*) malloc(sizeof(float) * N * M);
	VT = (float*) malloc(sizeof(float) * N * N);
	qq = (float*) malloc(sizeof(float) * N);
	pos = (int*) malloc(sizeof(int) * N);

	for(i = 0;i < N;++i) {
		qq[i] = q[i];
	}

	for(i = 0;i < M*N;++i) {
		UT[i] = U[i];
	}

	for(i = 0;i < N*N;++i) {
		VT[i] = V[i];
	}

	//mtranspose(U,M,N,UT);
	//mtranspose(V,N,N,VT);

	sort1d(q,N,pos);

	for(i = 0; i < N;++i) {
		q[i] = qq[pos[i]];

		for (j = 0; j < M;++j) {
			U[j*N+i] = UT[j*N+pos[i]];
		}
		for (j = 0; j < N;++j) {
			V[j*N+i] = VT[j*N+pos[i]];
		}
	}

	free(UT);
	free(VT);
	free(qq);
	free(pos);

}

int svd(float *A,int M,int N,float *U,float *V,float *q) {
	int i,j,k,l,t,t2,ierr,cancel,iter,l1;
	float eps,g,x,s,temp,f,h,c,y,z,scale;
	float *e;
	/*
     THIS SUBROUTINE IS THE MODIFIED C TRANSLATION OF THE
     EISPACK FORTRAN TRANSLATION OF THE ALGOL PROCEDURE SVD,
     NUM. MATH. 14, 403-420(1970) BY GOLUB AND REINSCH.
     HANDBOOK FOR AUTO. COMP., VOL II-LINEAR ALGEBRA, 134-151(1971).
	 */
	/*
	 * U = MXN
	 * V - NXN
	 * Q - NX1
	 */

	/*
	 * The program return error codes
	 *
	 *  Code 0 if the computation is successful
	 *  Code -1 If  M < N . Transpose the matrix such that rows > columns and trye again
	 *  Code 15 if maximum iterations are reached without achieving convergence. Increase SVDMAXITER value
	 *  in matrix.h header file. Default Value is 50
	 *
	 */
	if (M < N) {
		printf("Rows (M) should be greater than Columns (B) \n");
		printf("Retry By Transposing the Input Matrix");
		return -1;
	}
	e = (float*) malloc(sizeof(float) * N);
	ierr = 0;
	eps = macheps();
	g = scale = x = 0.0;

	for(i = 0; i < M*N;++i) {
		U[i] = A[i];
	}

	for(i = 0; i < N;++i) {
		l = i+1;
		e[i] = scale * g;
		g = 0.0;
		s = 0.0;
		scale = 0.0;

		if (i < M) {
			for(k = i; k < M;++k) {
				scale += fabsf(U[k*N+i]);
			}

			if (scale != 0.0) {
				for(k = i; k < M;++k) {
					t = k * N;
					U[t+i] /= scale;
					temp = U[t+i];
					s += temp*temp;
				}
				f = U[i*N+i];
				g = (f < 0) ? sqrtf(s) : -sqrtf(s);
				h = f * g - s;
				U[i*N+i] = f - g;

				if (i < N - 1) {
					for(j = l; j < N;++j) {
						s = 0.0;
						for(k = i; k < M;++k) {
							t = k * N;
							s += U[t+i]*U[t+j];
						}
						f = s / h;
						for(k = i; k < M;++k) {
							t = k * N;
							U[t+j] += f * U[t+i];
						}
					}
				}
				for(k = i; k < M;++k) {
					t = k * N;
					U[t+i] *= scale;
				}
			}
		}
        q[i] = scale * g;
        g = 0.0;
        s = 0.0;
        scale = 0.0;

        if (i < M && i != N - 1) {
        	t = i *N;
        	for(k = l; k < M;++k) {
        		scale += fabsf(U[t+k]);
        	}
        	if (scale != 0.0) {
        		for(k = l; k < N;++k) {
        			U[t+k] /= scale;
        			temp = U[t+k];
        			s = s + temp*temp;
        		}
        		f = U[t+l];
        		g = (f < 0) ? sqrtf(s) : -sqrtf(s);
                h = f * g - s;
                U[t+l] = f - g;
                for(k = l;k < N;++k) {
                	e[k] = U[t+k] / h;
                }

				for (j = l; j < M; j++) {
					s = 0.0;
					t2 = j * N;
					for (k = l; k < N; k++) {
						s += U[t2+k] * U[t+k];
					}
					for (k = l; k < N; k++) {
						U[t2+k] += s * e[k];
					}
				}
                for (k = l; k < N; k++)
                    U[t+k] *= scale;
        	}

        }

        temp = fabsf(q[i]) + fabsf(e[i]);

        if (x < temp) {
        	x = temp;
        }
	}

	/*
	ierr = 0;
	eps = macheps();
	tol = eps;
	g = x = 0.0;

	for(i = 0; i < M*N;++i) {
		U[i] = A[i];
	}

	for(i = 0; i < N;++i) {
		l = i+1;
		e[i] = g;
		s = 0.0;

		for(k = i; k < M;++k) {
			t = k * N;
			temp = U[t+i];
			s += temp*temp;
		}
		if (s < tol) {
			g = 0.0;
		} else {
			f = U[i*N+i];
			g = (f < 0) ? sqrtf(s) : -sqrtf(s);
			h = f * g - s;
			U[i*N+i] = f - g;

			for(j = l; j < N;++j) {
				s = 0.0;
				for(k = i; k < M;++k) {
					t = k * N;
					s += (U[t+i]*U[t+j]);
				}
				f = s / h;
				for(k = i; k < M;++k) {
					t = k * N;
					U[t+j] += (f * U[t+i]);
				}
			}

		}

        q[i] = g;
        s = 0.0;
        t = i * N;
    	for(k = l; k < N;++k) {
    		temp = U[t+k];
    		s = s + temp*temp;
    	}
        if (s < tol) {
        	g = 0.0;
        } else {
        	f = U[t+l];
			g = (f < 0) ? sqrtf(s) : -sqrtf(s);
			h = f * g - s;
            U[t+l] = f - g;
            for(k = l;k < N;++k) {
              	e[k] = U[t+k] / h;
            }

            for (j = l; j < M; j++) {
                s = 0.0;
                t2 = j * N;
                for (k = l; k < N; k++) {
                     s += U[t2+k] * U[t+k];
                }
                for (k = l; k < N; k++) {
                     U[t2+k] += s * e[k];
                }
            }

        }

        temp = fabs(q[i]) + fabs(e[i]);

        if (x < temp) {
        	x = temp;
        }
	}
*/


//Accumulating Right Hand Transformations

	for(i = N - 1;i >= 0;--i) {
		t = i * N;
		if (i < N - 1) {
			if (g != 0.0) {
				h = U[t+i+1] * g;
				for(j = l;j < N;++j) {
					V[j*N+i] = U[t+j] / h;
				}
				for(j = l;j < N;++j) {
					s = 0.0;
					for(k = l; k < N;++k) {
						s += U[t+k] * V[k*N+j];
					}
					for(k = l; k < N;++k) {
						V[k*N+j] += (s * V[k*N+i]);
					}
				}
			}
			for(j = l; j < N;++j) {
				V[t+j] = V[j*N+i] = 0.0;
			}
		}
	    V[t+i] = 1.0;
		g = e[i];
		l = i;
	}



//Accumulating Left Hand Transformations

	for(i = N - 1;i >= 0;--i) {
		t = i * N;
		l = i+1;
		g = q[i];

		if (i < N - 1) {
			for(j = l;j < N;++j) {
				U[t+j] = 0.0;
			}
		}

		if (g != 0.0) {
			if (i != N - 1) {
				//h = U[t+i] * g;
				for(j = l;j < N;++j) {
					s = 0.0;
					for(k = l; k < M;++k) {
						s += (U[k*N+i] * U[k*N+j]);
					}
					f = (s / U[t+i]) / g;
					for(k = i; k < M;++k) {
						U[k*N+j] += (f * U[k*N+i]);
					}
				}
			}
			for(j = i; j < M;++j) {
				U[j*N+i] = U[j*N+i] / g;
			}
		} else {
			for(j = i; j < M;++j) {
				U[j*N+i] = 0.0;
			}
		}

		U[t+i] += 1.0;
	}
//	mdisplay(U,M,N);

	eps = eps * x;

	for(k = N - 1; k >= 0; --k) {
		iter = 0;

		while(1) {
			iter++;
			if (iter > SVDMAXITER) {
				printf("Convergence Not Achieved \n");
				return 15;
			}

			cancel = 1;
			for(l = k; l >= 0; --l) {
				if (fabs(e[l]) <= eps) {
					cancel = 0; //test f convergence
					break;
				}
				if (fabs(q[l-1]) <= eps) {
					//Cancel
					break;
				}
			}
			if (cancel) {
				c = 0.0;
				s = 1.0;
				l1 = l - 1;
				for(i = l; i <= k;++i) {
					f = s*e[i];
					e[i] *= c;
					if (fabs(f) <= eps) {
						break;
					}
					g = q[i];
					h = q[i] = hypotf(f,g);
					c = g/h;
					s = -f/h;
					for(j = 0; j < M;++j) {
						t = j * N;
						y = U[t+l1];
						z = U[t+i];

						U[t+l1] = y * c + z * s;
						U[t+i] = z * c - y * s;
					}
				}
			}
			z = q[k];
			if (l != k) {
				x = q[l];
				y = q[k-1];
				g = e[k-1];
				h = e[k];
				f = 0.5f * (((g + z) / h) * ((g - z) / y) + y / h - h / y);
				g = hypotf(f,1.0);
				if (f < 0.0) {
					temp = f - g;
				} else {
					temp = f+g;
				}
				f = x - (z / x) * z + (h / x) * (y / temp - h);

				//Next QR Transformation

				c = s = 1.0;
				for(i = l+1; i <= k;++i) {
					g = e[i];
					y = q[i];
					h = s * g;
					g = c * g;
					e[i-1] = z = hypotf(f,h);
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;
                    for(j = 0; j < N;++j) {
                    	t = j * N;
                        x = V[t+i-1];
                        z = V[t+i];
                        V[t+i-1] = x * c + z * s;
                        V[t+i] = z * c - x * s;
                    }
                    q[i-1] = z = hypotf(f,h);
                    if (z != 0.0) {
                        c = f / z;
                        s = h / z;
                    }
                    f = c * g + s * y;
                    x = c * y - s * g;
                    for(j = 0; j < M;++j) {
                    	t = j * N;
                        y = U[t+i-1];
                        z = U[t+i];
                        U[t+i-1] = y * c + z * s;
                        U[t+i] = z * c - y * s;
                    }
				}
                    e[l] = 0.0;
                    e[k] = f;
                    q[k] = x;

			} else {
				//convergence
                if (z < 0.0) {
                    q[k] = -z;
                    for (j = 0; j < N; j++) {
                    	t = j *N;
                        V[t+k] = -V[t+k];
                    }
                }
                break;
			}
		}
	}

	svd_sort(U,M,N,V,q);

	free(e);
	return ierr;
}

int svd_transpose(float *A, int M, int N, float *U, float *V, float *q) {
	int ret;
	/* Call this routine if M < N
	* U = MXM
	* V - NXM
	* Q - MX1
	*/

	if (M >= N) {
		printf("M>=N. Use svd routine.\n");
		exit(-1);
	}

	mtranspose(A, M, N, V);

	ret = svd(V, N, M, V, U, q);
	return ret;
}

static int rank_c(float *A, int M,int N) {
	int i,rnk,ret;
	float eps,tol,szmax,qmax;
	float *U,*V,*q;

	U = (float*) malloc(sizeof(float) * M*N);
	V = (float*) malloc(sizeof(float) * N*N);
	q = (float*) malloc(sizeof(float) * N);

	eps = macheps();
	rnk = 0;
	if (M < N) {
		//mtranspose(A,M,N,U);
		szmax = (float) N;
	} else {
		szmax = (float) M;
	}
	ret = svd(A,M,N,U,V,q);
	qmax = q[0];
	if ( ret != 0) {
		printf("Failed to Compute SVD");
		free(U);
		free(V);
		free(q);
		return -1;
	}

	tol = qmax*szmax *eps;

	for(i = 0; i < N;++i) {
		if (q[i] > tol) {
			rnk++;
		}
	}



	free(U);
	free(V);
	free(q);

	return rnk;
}

int rank(float *A, int M,int N) {
	int rnk;
	float *AT;

	AT = (float*) malloc(sizeof(float) * M*N);

	if (M < N) {
		mtranspose(A,M,N,AT);
		rnk = rank_c(AT,N,M);
	} else {
		rnk = rank_c(A,M,N);
	}

	free(AT);
	return rnk;

}

int lls_svd_multi(float *A, float *b, int M,int N, float *x) {
	int rnk, ret, i;
	float *U, *V, *q, *UT, *d;
	float eps, tol, szmax, qmax;

	if (M < N) {
		printf("Rows (M) should be greater than Columns (B) \n");\
		return -1;
	}

	U = (float*)malloc(sizeof(float)* M*N);
	V = (float*)malloc(sizeof(float)* N*N);
	q = (float*)malloc(sizeof(float)* N);
	UT = (float*)malloc(sizeof(float)* M*N);
	d = (float*)malloc(sizeof(float)* N);
	/*
	The code returns -1 if SVD computation fails else it returns the rank of the matrix A (and the real size of vector x)
	*/
	ret = svd(A, M, N, U, V, q);

	if (ret != 0) {
		printf("Failed to Compute SVD");
		free(U);
		free(V);
		free(q);
		free(UT);
		free(d);
		return -1;
	}

	szmax = (float)M;

	eps = macheps();
	rnk = 0;

	qmax = q[0];

	tol = qmax*szmax *eps;

	for (i = 0; i < N; ++i) {
		if (q[i] > tol) {
			rnk++;
		}
	}

	mtranspose(U, M, N, UT);

	d = (float*)malloc(sizeof(float)* N);

	mmult(UT, b, d, N, M, 1);

	for (i = 0; i < rnk; ++i) {
		d[i] /= q[i];
	}

	for (i = rnk; i < N; ++i) {
		d[i] = 0.0;
	}

	mmult(V, d, x, N, N, 1);

	free(U);
	free(V);
	free(q);
	free(UT);
	free(d);

	return(rnk);
}

