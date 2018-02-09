#ifndef NNET_H_
#define NNET_H_

#ifdef _OPENMP
#include <omp.h>
#endif
#include "netdata.h"

#define ETA_MAX 1.0
#define ETA_MIN 1e-05

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nnet_set* nnet_object;

nnet_object nnet_init(int layers, int *arch, int *actfcn);

/* layers = Total Number of Layers = Input Layer + Output Layer + Hidden Layers <= 5. 1 input layer and 1 output layers + Upto 100 Hidden Layers
arch is the multilayer architecture. Number of neurons in each layer. arch[0] - Number of Inputs. arch[layers-1] - Number of Outputs.
eg. arch = {4,4,2} is a neural network with 4 inputs ,one hidden layer with 4 neurons and 2 outputs
arch = {3,4,5,2} is a neural network with 3 inputs , two hidden layer with 4 and 5 neurons respectively and 2 outputs
*/

struct nnet_set {
	int layers;
	int arch[5];
	int lweight[5];
	int actfcn[5];// 0 - NULL, 1 - purelin, 2 - logsig, 3 - tansig. By default set first element to 0 as input layer doesn't have any activation function
	int normmethod;// 0 - NULL, 1 - Minmax {-1,1}, 2 - Std (Mean = 0, Variance = 1}. Default = 0
	char trainfcn[50];
	char trainmethod[20];// Options "online" or "batch"
	float *weight;// Weights vector including biases
	float *gradient; // Gradient Vector
	float *tout; // Output Vector contains outputs at all the nodes including hidden ones during the current iteration.
	float *input;// current input
	float *dmin;
	float *dmax;
	float *tmin;
	float *tmax;
	float *dmean;
	float *dstd;
	float *tmean;
	float *tstd;
	int datasize;
	int lw;
	int ld;
	int lm1;
	int nmax;
	int emax;
	int generalize;
	int validate;
	int verbose;
	int batchsize;// batchsize . Used only if the batch method selected.
	float qp_threshold;
	float qp_shrink_factor;
	float qp_max_factor;
	float qp_decay;
	float rp_eta_p;
	float rp_eta_n;
	float rp_delta_min;
	float rp_init_upd;
	float rp_max_step;
	float rp_zero_tol;
	float mse;
	float tmse; // Training Set MSE
	float gmse; // Generalization Set MSE
	float imse;
	float eta;
	float alpha;
	float steepness;
	float eta_inc;
	float eta_dec;
	float perf_inc;
	float tratio;
	float gratio;
	float vratio;
	float inpnmin;
	float inpnmax;
	float inpnmean;
	float inpnstd;
	float oupnmin;
	float oupnmax;
	float oupnmean;
	float oupnstd;
	float params[1];
};

void set_learning_rate(nnet_object obj, float eta);

void set_momentum(nnet_object obj, float alpha);

void set_target_mse(nnet_object obj, float mse);

void set_generalization_mse(nnet_object obj, float gmse);

void set_max_epoch(nnet_object obj, int max_epoch);

void set_verbose(nnet_object obj, int verb);

void set_training_ratios(nnet_object obj, float tratio, float gratio, float vratio);

void set_trainfcn(nnet_object obj, char *trainfcn);

void set_trainmethod(nnet_object obj,char *method, int batchsize);// batchsize is only used if method is set to "batch". online training is done incrementally using one set of data at a time

void set_norm_method(nnet_object obj, int nmethod);

void set_mnmx(nnet_object obj, float inpmin, float inpmax, float oupmin, float oupmax);

void set_mstd(nnet_object obj, float inpmean, float inpstd, float oupmean, float oupstd);

void initweights(nnet_object obj);

void initweightsnw(nnet_object obj);

void shuffle(int N, int *index);

void initweights_seed(nnet_object obj, int seed);

void feedforward(nnet_object obj, float *inp, int leninp, int lenoup, float *oup,float *tempi, float *tempo);

void backpropagate(nnet_object obj, float *output, float *desired, int lenoup, float *delta, float *tinp);

void mapminmax(float *x, int N, float ymin, float ymax, float *y);

void mapstd(float *x, int N, float ymean, float ystd, float *y);

void premnmx(int size, float *p, int leninp, float *t, int lenoup, float *pn, float *tn, float ymin, float ymax, float omin, float omax,float *pmin, float *pmax, float *tmin, float *tmax);

void applymnmx(nnet_object obj, int size, float *p, int leninp, float *pn);

void prestd(int size, float *p, int leninp, float *t, int lenoup, float *pn, float *tn, float ymean, float ystd, float omean, float ostd, float *dmean, float *dstd, float *tmean, float *tstd);

void postmnmx(nnet_object obj, int size, float *oupn, int lenoup, float *oup);

void poststd(nnet_object obj, int size, float *oupn, int lenoup, float *oup);

void train_null(nnet_object obj, int tsize, float *data, float *target);

void train_mnmx(nnet_object obj, int size, float *inp, float *out);

void train_mstd(nnet_object obj, int size, float *inp, float *out);

void train(nnet_object obj, int tsize, float *data, float *target);

void sim(nnet_object obj, int size, float *data, float *output);

float nnet_test(nnet_object obj, int tsize, float *data, float *target);

void nnet_save(nnet_object obj, const char *fileName);

nnet_object nnet_load(const char *fileName);

void nnet_free(nnet_object obj);


#ifdef __cplusplus
}
#endif

#endif /* NNET_H_ */