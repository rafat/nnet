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
	double *weight;// Weights vector including biases
	double *gradient; // Gradient Vector
	double *tout; // Output Vector contains outputs at all the nodes including hidden ones during the current iteration.
	double *input;// current input
	double *dmin;
	double *dmax;
	double *tmin;
	double *tmax;
	double *dmean;
	double *dstd;
	double *tmean;
	double *tstd;
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
	double qp_threshold;
	double qp_shrink_factor;
	double qp_max_factor;
	double qp_decay;
	double rp_eta_p;
	double rp_eta_n;
	double rp_delta_min;
	double rp_init_upd;
	double rp_max_step;
	double rp_zero_tol;
	double mse;
	double tmse; // Training Set MSE
	double gmse; // Generalization Set MSE
	double imse;
	double eta;
	double alpha;
	double steepness;
	double eta_inc;
	double eta_dec;
	double perf_inc;
	double tratio;
	double gratio;
	double vratio;
	double inpnmin;
	double inpnmax;
	double inpnmean;
	double inpnstd;
	double oupnmin;
	double oupnmax;
	double oupnmean;
	double oupnstd;
	double params[1];
};

void set_learning_rate(nnet_object obj, double eta);

void set_momentum(nnet_object obj, double alpha);

void set_target_mse(nnet_object obj, double mse);

void set_generalization_mse(nnet_object obj, double gmse);

void set_max_epoch(nnet_object obj, int max_epoch);

void set_verbose(nnet_object obj, int verb);

void set_training_ratios(nnet_object obj, double tratio, double gratio, double vratio);

void set_trainfcn(nnet_object obj, char *trainfcn);

void set_trainmethod(nnet_object obj,char *method, int batchsize);// batchsize is only used if method is set to "batch". online training is done incrementally using one set of data at a time

void set_norm_method(nnet_object obj, int nmethod);

void set_mnmx(nnet_object obj, double inpmin, double inpmax, double oupmin, double oupmax);

void set_mstd(nnet_object obj, double inpmean, double inpstd, double oupmean, double oupstd);

void initweights(nnet_object obj);

void initweightsnw(nnet_object obj);

void shuffle(int N, int *index);

void initweights_seed(nnet_object obj, int seed);

void feedforward(nnet_object obj, double *inp, int leninp, int lenoup, double *oup,double *tempi, double *tempo);

void backpropagate(nnet_object obj, double *output, double *desired, int lenoup, double *delta, double *tinp);

void mapminmax(double *x, int N, double ymin, double ymax, double *y);

void mapstd(double *x, int N, double ymean, double ystd, double *y);

void premnmx(int size, double *p, int leninp, double *t, int lenoup, double *pn, double *tn, double ymin, double ymax, double omin, double omax,double *pmin, double *pmax, double *tmin, double *tmax);

void applymnmx(nnet_object obj, int size, double *p, int leninp, double *pn);

void prestd(int size, double *p, int leninp, double *t, int lenoup, double *pn, double *tn, double ymean, double ystd, double omean, double ostd, double *dmean, double *dstd, double *tmean, double *tstd);

void postmnmx(nnet_object obj, int size, double *oupn, int lenoup, double *oup);

void poststd(nnet_object obj, int size, double *oupn, int lenoup, double *oup);

void train_null(nnet_object obj, int tsize, double *data, double *target);

void train_mnmx(nnet_object obj, int size, double *inp, double *out);

void train_mstd(nnet_object obj, int size, double *inp, double *out);

void train(nnet_object obj, int tsize, double *data, double *target);

void sim(nnet_object obj, int size, double *data, double *output);

double nnet_test(nnet_object obj, int tsize, double *data, double *target);

void nnet_save(nnet_object obj, const char *fileName);

nnet_object nnet_load(const char *fileName);

void nnet_free(nnet_object obj);


#ifdef __cplusplus
}
#endif

#endif /* NNET_H_ */