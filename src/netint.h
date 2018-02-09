#ifndef NETINT_H_
#define NETINT_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <math.h>

#include "matrix.h"

#pragma warning(disable:4996)

#define NNET_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define NNET_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define NNET_SIGN(a) (((a) >= (0.0)) ? (1) : (-1))

#ifdef __cplusplus
extern "C" {
#endif

void logsig(float *x, float N, float *y);

void tansig(float *x, float N, float *y);

void hardlim(float *x, float N, float *y);

void purelin(float *x, float N, float *y);

float clip_value(float x, float lo, float hi);

float logsig_der(float value);

float tansig_der(float value);

int intmax(int* x, int N);

float mean(float* vec, int N);

float std(float* vec, int N);

float dmax(float* vec, int N);

float dmin(float* vec, int N);

float neuron_oup(float *inp, int N, float *weights, float bias);

void neuronlayer_logsig_oup(float *inp, int N, int S, float *weights, float *oup);

void neuronlayer_tansig_oup(float *inp, int N, int S, float *weights, float *oup);

void neuronlayer_purelin_oup(float *inp, int N, int S, float *weights, float *oup);


#ifdef __cplusplus
}
#endif

#endif /* NETINT_H_ */