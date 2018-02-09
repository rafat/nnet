#ifndef NETDATA_H_
#define NETDATA_H_

#include <ctype.h>
#include "netint.h"

#define BUFFER 4096

#ifdef __cplusplus
extern "C" {
#endif


typedef struct ndata_set* ndata_object;

ndata_object ndata_init(int inputs, int outputs, int patterns);

struct ndata_set {
	int I;
	int O;
	int P;
	int tsize;
	int gsize;
	int vsize;
	float *data;
	float *target;
	float params[1];
};

void interleave(float *inp, int size, int M, float *oup);

void data_enter(ndata_object obj, float *data, float *target);

void data_interleave_enter(ndata_object obj, float *data, float *target);

void csvreader(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void file_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void file_rev_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void file_sep_line_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void ndata_check(ndata_object obj);

void ndata_free(ndata_object obj);


#ifdef __cplusplus
}
#endif

#endif /* NETDATA_H_ */