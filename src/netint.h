#ifndef NETINT_H_
#define NETINT_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <math.h>

#include "optimc.h"

#ifdef __cplusplus
extern "C" {
#endif

void logsig(double *x, double N, double *y);

void tansig(double *x, double N, double *y);

void hardlim(double *x, double N, double *y);

void purelin(double *x, double N, double *y);

double clip_value(double x, double lo, double hi);

double logsig_der(double value);

double tansig_der(double value);

int intmax(int* x, int N);

double mean(double* vec, int N);

double std(double* vec, int N);

double dmax(double* vec, int N);

double dmin(double* vec, int N);

double neuron_oup(double *inp, int N, double *weights, double bias);

void neuronlayer_logsig_oup(double *inp, int N, int S, double *weights, double *oup);

void neuronlayer_tansig_oup(double *inp, int N, int S, double *weights, double *oup);

void neuronlayer_purelin_oup(double *inp, int N, int S, double *weights, double *oup);


#ifdef __cplusplus
}
#endif

#endif /* NETINT_H_ */