#include <stdio.h>
#include "../src/nnet.h"

int main() {
	int N, i, leninp, lenoup;
	int tsize;
	nnet_object obj;

	N = 3;
	tsize = 4;

	double p[8] = { 1, 1, 0, 1, 1, 0, 0, 0 };// Interleaved input x1,x2,x1,x2,....
	double gp[4] = { 0, 1, 1, 0 };

	int arch[3] = { 2, 2, 1 }; // architecture 2-2-1
	int actfcn[3] = { 0, 3, 1 };// {Null,'tansig','purelin'}

	leninp = arch[0];
	lenoup = arch[2];

	obj = nnet_init(N, arch, actfcn);
	initweights_seed(obj, 10);// set seed to 10
	set_trainfcn(obj, "traingdx");
	set_training_ratios(obj, 1.0, 0.0, 0.0);
	set_max_epoch(obj, 500);
	set_target_mse(obj, 1e-04);// Target MSE error
	set_learning_rate(obj, 0.1);// learning rate
	set_momentum(obj, 0.9);// No momentum term
	set_norm_method(obj, 2);// Input/Output Normalization Method (0,1,2)

	/*
	obj->weight[0] = 0.8;//bias
	obj->weight[1] = 0.5;
	obj->weight[2] = 0.4;
	obj->weight[3] = -0.1;//bias
	obj->weight[4] = 0.9;
	obj->weight[5] = 1.0;
	obj->weight[6] = 0.3;//bias
	obj->weight[7] = -1.2;
	obj->weight[8] = 1.1;
	*/

	train(obj, tsize, p, gp);
	/*
	for (i = 0; i < obj->lw; ++i) {
	printf("W %g ", obj->weight[i]);
	}
	*/
	double out[4] = { 0, 0, 0, 0 };

	sim(obj, tsize, p, out);


	for (i = 0; i < tsize; ++i) {
		//feedforward(obj, p + i*leninp, leninp, lenoup, out + i*lenoup);
		printf("\n%g %g ", gp[i], out[i]);
	}

	//backpropagate2(obj, out, gp, lenoup);

	for (i = 0; i < obj->lw; ++i) {
		printf("\nW %g ", obj->weight[i]);
	}

	nnet_free(obj);
	return 0;
}