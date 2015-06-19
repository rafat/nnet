#include "nnet.h"

nnet_object nnet_init(int layers, int *arch, int *actfcn) {
	nnet_object obj = NULL;
	int i,ld,lw,li;
	int add_vector,add_vector2, t_vector;

	// Error Checks
	if (layers < 2) {
		printf("\nThis Neural Network cannot have fewer than 2 layers (Input and Output\n");
		printf("Returning NULL Object\n");
		return obj;
	}
	else if (layers > 102) {
		printf("\nThis Neural Network cannot have more than 102 layers (Input+Output+ 100 Hidden Layers\n");
		printf("Returning NULL Object\n");
		return obj;
	}

	for (i = 0; i < layers - 1; ++i) {
		if (arch[i] <= 0) {
			printf("\nThere should be at least one neuron in each layer.\n");
			printf("Returning NULL Object\n");
			return obj;
		}
	}

	if (actfcn[0] != 0) {
		printf("\n Error : The first value of activation function array corresponding to input layer must be zero \n");
		printf("Eg., A three layer network may have actfcn[3] = {0,3,1} coresponding to input layer (Null), the\n");
		printf(" hidden layer (tansig) and the output layer (purelin) \n");
		printf("Returning NULL Object\n");
		return obj;
	}

	for (i = 1; i < layers; ++i) {
		if (actfcn[i] <= 0 || actfcn[i] > 3) {
			printf("Error : Allowed Activatiuon Functions 1 - purelin, 2 - logsig, 3 - tansig\n");
			return obj;
		}
	}

	// End Error Checks

	ld = lw = 0;

	for (i = 0; i < layers - 1; ++i) {
		ld += arch[i+1];
		lw += (arch[i] + 1) * arch[i + 1];
	}
	li = arch[0];

	add_vector2 = 2 * (arch[0] + arch[layers - 1]);
	add_vector = 2 * add_vector2;
	t_vector = 2 * ld + 2 * lw + li;

	obj = (nnet_object)malloc(sizeof(struct nnet_set) + sizeof(double)* (t_vector + add_vector) );

	obj->layers = layers;
	obj->lm1 = layers - 1;
	obj->ld = ld;
	obj->lw = lw;

	obj->alpha = 0.9; // momentum
	obj->eta = 0.01;// learning rate
	obj->eta_inc = 1.05;// increase learning rate
	obj->eta_dec = 0.7;// decrease learning rate
	obj->perf_inc = 1.04; // maximum performance increase


	obj->emax = 1000; // Maximum Epoch

	for (i = 0; i < layers; ++i) {
		obj->arch[i] = arch[i];
	}

	obj->lweight[0] = 0;

	for (i = 1; i < layers; ++i) {
		obj->lweight[i]= (arch[i-1] + 1) * arch[i];
	}
	for (i = 0; i < t_vector + add_vector; ++i) {
		obj->params[i] = 0.0;// Can't initialize all values to zero
	}
	obj->weight = &obj->params[0];
	obj->delta = &obj->params[lw];
	obj->gradient = &obj->params[2*lw];
	obj->tout = &obj->params[2 * lw + ld];
	obj->input = &obj->params[2 * lw + 2*ld];

	obj->dmin = &obj->params[t_vector];
	obj->dmax = &obj->params[t_vector + arch[0]];
	obj->tmin = &obj->params[t_vector + 2 * arch[0]];
	obj->tmax = &obj->params[t_vector + 2 * arch[0] + arch[layers-1]];

	obj->dmean = &obj->params[t_vector + add_vector2];
	obj->dstd = &obj->params[t_vector + add_vector2 + arch[0]];
	obj->tmean = &obj->params[t_vector + add_vector2 + 2 * arch[0]];
	obj->tstd = &obj->params[t_vector + add_vector2 + 2 * arch[0] + arch[layers - 1]];

	obj->nmax = imax(arch, layers);
	obj->mse = 1.0;
	obj->tmse = 1.0e-04;

	obj->tratio = 0.70;
	obj->gratio = 0.15;
	obj->vratio = 0.15;

	obj->actfcn[0] = 0;
	for (i = 1; i < layers; ++i) {
		obj->actfcn[i] = actfcn[i];
	}

	obj->normmethod = 0; // Input Normalization Method
	obj->inpnmin = obj->oupnmin = - 1;
	obj->inpnmax = obj->oupnmax = 1;
	obj->inpnmean = obj->oupnmean = 0;
	obj->inpnstd = obj->oupnstd = 1;

	initweights(obj);
	//initweightsnw(obj);
	strcpy(obj->trainfcn, "traingd");

	return obj;
}

void set_learning_rate(nnet_object obj,double eta) {
	if (eta > 0 && eta < 1.0) {
		obj->eta = eta;
	}
	else {
		printf("Learning Rate only takes values between 0.0 and 1.0\n");
		exit(1);
	}
}

void set_momentum(nnet_object obj, double alpha) {
	if (alpha >= 0.0 && alpha < 1.0) {
		obj->alpha = alpha;
	}
	else {
		printf("Momentum only takes values between 0.0 and 1.0\n");
		exit(1);
	}
}

void set_target_mse(nnet_object obj, double tmse) {
	if (tmse < 0.0) {
		printf("MSE only takes values over 0.0\n");
		exit(1);
	}
	else {
		obj->tmse = tmse;
	}
}

void set_max_epoch(nnet_object obj, int max_epoch) {
	if (max_epoch <= 0) {
		printf("Epoch only takes values >= 1\n");
		exit(1);
	}
	else {
		obj->emax = max_epoch;
	}
}

void set_training_ratios(nnet_object obj, double tratio, double gratio, double vratio) {
	if (tratio + gratio + vratio != 1.0) {
		printf("Ratios must sum to 1.0\n");
		exit(1);
	}

	obj->tratio = tratio;
	obj->gratio = gratio;
	obj->vratio = vratio;
}

void set_trainfcn(nnet_object obj, char *trainfcn) {
	if (!strcmp(trainfcn, "traingd")) {
		strcpy(obj->trainfcn, "traingd");
	}
	else if (!strcmp(trainfcn, "traingdm")) {
		strcpy(obj->trainfcn, "traingdm");
	}
	else if (!strcmp(trainfcn, "traingda")) {
		strcpy(obj->trainfcn, "traingda");
	}
	else if (!strcmp(trainfcn, "traingdx")) {
		strcpy(obj->trainfcn, "traingdx");
	}
	else {
		printf("Error : Available Training Functions - traingd, traingdm, traingda, traingdx");
		exit(1);
	}
}

void set_norm_method(nnet_object obj, int nmethod) {
	if (nmethod >= 0 && nmethod <= 2) {
		obj->normmethod = nmethod;
	}
	else {
		printf("\n Available Normalization Methods : 0 - NULL,1 - Minmax ,2 - Mean/Std method\n");
		exit(1);
	}
}

void set_mnmx(nnet_object obj, double inpmin, double inpmax, double oupmin, double oupmax) {
	if (obj->normmethod != 1) {
		obj->normmethod = 1;
	}
	obj->inpnmin = inpmin;
	obj->inpnmax = inpmax;
	obj->oupnmin = oupmin;
	obj->oupnmax = oupmax;
}

void set_mstd(nnet_object obj, double inpmean, double inpstd, double oupmean, double oupstd) {
	if (obj->normmethod != 2) {
		obj->normmethod = 2;
	}
	obj->inpnmean = inpmean;
	obj->inpnstd = inpstd;
	obj->oupnmean = oupmean;
	obj->oupnstd = oupstd;
}

static double mean_stride(double* vec, int N, int stride) {
	int i;
	double m;
	m = 0.0;

	for (i = 0; i < N; ++i) {
		m += vec[i*stride];
	}
	m = m / N;
	return m;
}

static double std_stride(double* vec, int N, int stride) {
	double v, temp, m;
	int i;
	v = 0.0;
	m = mean_stride(vec, N, stride);

	for (i = 0; i < N; ++i) {
		temp = vec[i*stride] - m;
		v += temp*temp;
	}

	v = v / N;
	v = sqrt(v);

	return v;

}

static double dmax_stride(double* x, int N, int stride) {
	int i;
	double m;

	m = -DBL_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i*stride] > m) {
			m = x[i*stride];
		}
	}

	return m;
}

static double dmin_stride(double* x, int N, int stride) {
	int i;
	double m;

	m = DBL_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i*stride] < m) {
			m = x[i*stride];
		}
	}

	return m;
}

static void norm_nw(int S, double *weight,double beta) {
	int i;
	double temp;

	temp = 0.0;

	for (i = 0; i < S; ++i) {
		temp += (weight[i] * weight[i]);
	}

	temp = sqrt(temp);

	for (i = 0; i < S; ++i) {
		weight[i] *= beta / temp;
	}
}

static double dsign(double val) {
	double sign;

	if (val >= 0.0) {
		sign = 1.0;
	}
	else {
		sign = -1.0;
	}

	return sign;
}

static void calc_nw_hidden(int N, int S, double *weight) {
	// N - Number of Inputs
	// S - Number of Neurons in Hidden layer
	// Total Number of Weights - N*S + S biases
	double beta;
	int j, k, N1, itr;
	double spc;

	beta = 0.7 * pow((double)S, 1.0 / (double)N);
	N1 = N + 1;
	srand(time(NULL));
	//srand(100);

	if (S == 1) {
		spc = 0.0;
	}
	else {
		spc = 2.0 / (S - 1);
	}

	for (j = 0; j < S; ++j) {
		itr = j * N1;
		weight[itr] = -1.0 + j*spc;
		for (k = 1; k < N1; ++k) {
			weight[itr + k] = ((((double)(rand() % 100) + 1) / 100 * 2) - 1.0) * beta;
		}
	}

	if (S == 1) {
		weight[0] = 0.0;
	}

	for (j = 0; j < S; ++j) {
		itr = j * N1 + 1;
		norm_nw(N, weight+itr, beta);
		weight[itr - 1] *= beta * dsign(weight[itr]);
	}

}

void initweightsnw(nnet_object obj) {
	int i, hid, itr,N,S,N1;

	hid = obj->layers - 2;
	itr = 0;
	for (i = 0; i < hid+1; ++i) {
		N = obj->arch[i];
		S = obj->arch[i+1];
		N1 = N + 1;
		calc_nw_hidden(N, S, obj->weight + itr);
		itr += (N1 * S);
	}
	/* Output layer
	lm1 = obj->lm1;
	lm2 = lm1 - 1;
	*/

}

void initweights(nnet_object obj) {
	int i, lm1, j, k, S, N, itr, itr3;
	double nrm;

	lm1 = obj->lm1;
	itr3 = 0;
	srand(time(NULL));
	//srand(100);
	for (i = 0; i < lm1; ++i) {
		nrm = 1.0 / sqrt((double)obj->arch[i]);
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			//obj->bias[itr2 + j] = (((double)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
			//printf("\n %d bias %g", itr2 + j, obj->bias[itr2 + j]);
			itr = j * N;
			for (k = 0; k < N; ++k) {
				obj->weight[itr3 + itr + k] = (((double)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
				//obj->weight[itr3 + itr + k] = (((double)(rand() % 100) + 1) / 100) * (0.001 - 0.0001) + 0.0001;
			}
		}
		itr3 += S * N;
	}

}

void initweights_seed(nnet_object obj, int seed) {
	int i, lm1, j, k, S, N, itr, itr3;
	double nrm;

	lm1 = obj->lm1;
	itr3 = 0;
	srand(seed);
	//srand(100);
	for (i = 0; i < lm1; ++i) {
		nrm = 1.0 / sqrt((double)obj->arch[i]);
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			//obj->bias[itr2 + j] = (((double)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
			//printf("\n %d bias %g", itr2 + j, obj->bias[itr2 + j]);
			itr = j * N;
			for (k = 0; k < N; ++k) {
				obj->weight[itr3 + itr + k] = (((double)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
				//obj->weight[itr3 + itr + k] = (((double)(rand() % 100) + 1) / 100) * (0.001 - 0.0001) + 0.0001;
			}
		}
		itr3 += S * N;
	}

}

static double gvmse(nnet_object obj, int tsize, double *data, double *target, int *index, double *output) {
	double gmse, temp;
	int i, itrd, itrt, leninp, lenoup, j;

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	gmse = gmse / (2.0 * lenoup * tsize);

	return gmse;
}

void feedforward(nnet_object obj, double *inp, int leninp, int lenoup, double *oup) {
	int lm1,i,N,S,itr,itr2,j,N1;
	lm1 = obj->lm1;
	double *tempi, *tempo;// To-DO Add a temp vector of length 2*nmax to the object obj

	tempi = (double*)malloc(sizeof(double)* obj->nmax);
	tempo = (double*)malloc(sizeof(double)* obj->nmax);

	if (leninp != obj->arch[0] || lenoup != obj->arch[lm1]) {
		printf("\nError The Neural network is designed for %d Inputs and %d Outputs", obj->arch[0], obj->arch[lm1]);
		exit(0);
	}

	N = obj->arch[0];
	for (i = 0; i < N; ++i) {
		tempi[i] = inp[i];
	}
	itr = 0;
	itr2 = 0;
	for (i = 0; i < lm1; ++i) {
		S = obj->arch[i+1];
		N1 = N + 1;
		if (obj->actfcn[i+1] == 1) {
			neuronlayer_purelin_oup(tempi, N, S, &obj->weight[itr], tempo);
		}
		else if (obj->actfcn[i+1] == 2) {
			neuronlayer_logsig_oup(tempi, N, S, &obj->weight[itr], tempo);
		}
		else if (obj->actfcn[i+1] == 3) {
			neuronlayer_tansig_oup(tempi, N, S, &obj->weight[itr], tempo);
		}
		itr += S*N1;
		for (j = 0; j < S; ++j) {
			obj->tout[j + itr2] = tempo[j];
			tempi[j] = tempo[j];
		}
		N = S;
		itr2 += S;
	}

	for (i = 0; i < lenoup; ++i) {
		oup[i] = tempo[i];
	}

	for (i = 0; i < leninp; ++i) {
		obj->input[i] = inp[i];
	}

	free(tempi);
	free(tempo);

}



void backpropagate(nnet_object obj, double *output, double *desired, int lenoup) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2,itr4;
	int S, itr3, index, in0;
	double temp,lr,mc;
	double *tinp;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}

	tinp = (double*)malloc(sizeof(double)* (ld + obj->arch[0]));

	// Local Gradients Calculation
	itr = ld - loup;
	obj->mse = 0.0;

	if (obj->actfcn[lm1] == 1) {
		for (i = 0; i < loup; ++i) {
			//printf("Wcv %g ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = -2.0 * temp;
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);
	
		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = -2.0 * temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = -2.0 * temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
		}
	}

	obj->mse /= 2.0;

	for (i = lm1 - 1; i > 0; --i) {
		if (obj->actfcn[i] == 1) {
			N = obj->arch[i];
			jinit = itr - N;
			kfin = obj->arch[i + 1];
			lw = lw - obj->lweight[i + 1];
			itr2 = 1;
			for (j = jinit; j < itr; ++j) {
				temp = 0.0;
				for (k = 0; k < kfin; ++k) {
					temp += obj->gradient[itr + k] * obj->weight[lw + itr2];// add weights
					//printf("W %d ", lw + itr2);
					itr2 += (N + 1);
				}
				itr2 = j - jinit + 2;
				obj->gradient[j] = temp;
			}
			itr -= N;
		}
		else if (obj->actfcn[i] == 2) {
			N = obj->arch[i];
			jinit = itr - N;
			//logsig(obj->tout + jinit, N, beta + jinit);
			kfin = obj->arch[i + 1];
			lw = lw - obj->lweight[i + 1];
			itr2 = 1;
			for (j = jinit; j < itr; ++j) {
				temp = 0.0;
				for (k = 0; k < kfin; ++k) {
					temp += obj->gradient[itr + k] * obj->weight[lw + itr2];// add weights
					//printf("W %d ", lw + itr2);
					itr2 += (N + 1);
				}
				itr2 = j - jinit + 2;
				obj->gradient[j] = temp * obj->tout[j] * (1.0 - obj->tout[j]);
				//printf("temp %g %g ", temp, obj->tout[j]);
			}
			itr -= N;
		}
		else if (obj->actfcn[i] == 3) {
			N = obj->arch[i];
			jinit = itr - N;
			//logsig(obj->tout + jinit, N, beta + jinit);
			kfin = obj->arch[i + 1];
			lw = lw - obj->lweight[i + 1];
			itr2 = 1;
			for (j = jinit; j < itr; ++j) {
				temp = 0.0;
				for (k = 0; k < kfin; ++k) {
					temp += obj->gradient[itr + k] * obj->weight[lw + itr2];// add weights
					//printf("W %d ", lw + itr2);
					itr2 += (N + 1);
				}
				itr2 = j - jinit + 2;
				obj->gradient[j] = temp * (1.0 + obj->tout[j]) * (1.0 - obj->tout[j]);
			}
			itr -= N;
		}
	}


	// Calculate weights and deltas

	lw = obj->lw;
	in0 = obj->arch[0];

	for (i = 0; i < in0; ++i) {
		tinp[i] = obj->input[i];
	}

	for (i = in0; i < in0 + ld; ++i) {
		tinp[i] = obj->tout[i - in0];
	}


	itr3 = 0;
	itr2 = itr4 = 0;
	if (!strcmp(obj->trainfcn, "traingd")) {
		for (i = 0; i < lm1; ++i) {
			N = obj->arch[i] + 1;
			S = obj->arch[i + 1];
			for (j = 0; j < S; ++j) {
				itr = j * N;// iterates over one row of weights
				index = itr3 + itr;
				obj->delta[index] = - lr * obj->gradient[itr2 + j];
				obj->weight[index] += obj->delta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					obj->delta[index] = - lr * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					obj->weight[index] += obj->delta[index];
					//printf(" ind %d", itr + k - 1);
				}

			}
			itr3 += S * N;// iterates over all the weights going into a layer
			itr2 += S;// iterates over each output layer
			itr4 += (N - 1);// iterates over each input layer
			//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		}
	}
	else if (!strcmp(obj->trainfcn, "traingdm")) {
		for (i = 0; i < lm1; ++i) {
			N = obj->arch[i] + 1;
			S = obj->arch[i + 1];
			for (j = 0; j < S; ++j) {
				itr = j * N;// iterates over one row of weights
				index = itr3 + itr;
				temp = obj->delta[index];
				obj->delta[index] = mc * temp - lr * (1.0 - mc) * obj->gradient[itr2 + j];
				obj->weight[index] += obj->delta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					temp = obj->delta[index];
					obj->delta[index] = mc * temp - lr * (1.0 - mc) * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					obj->weight[index] += obj->delta[index];
					//printf(" ind %d", itr + k - 1);
				}

			}
			itr3 += S * N;// iterates over all the weights going into a layer
			itr2 += S;// iterates over each output layer
			itr4 += (N - 1);// iterates over each input layer
			//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		}
	}
	//printf("WT %g \n", obj->weight[0]);

	free(tinp);

}

static void backpropagate_alr(nnet_object obj, double *output, double *desired, int lenoup,double *tvec) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	double temp, lr, mc;
	double *tinp;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}

	tinp = (double*)malloc(sizeof(double)* (ld + obj->arch[0]));

	// Local Gradients Calculation
	itr = ld - loup;
	//obj->mse = 0.0;

	if (obj->actfcn[lm1] == 1) {
		for (i = 0; i < loup; ++i) {
			//printf("Wcv %g ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = -2.0 * temp;
			//obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = -2.0 * temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			//obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = -2.0 * temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
			//obj->mse += temp*temp;
		}
	}

	//obj->mse /= 2.0;

	for (i = lm1 - 1; i > 0; --i) {
		if (obj->actfcn[i] == 1) {
			N = obj->arch[i];
			jinit = itr - N;
			kfin = obj->arch[i + 1];
			lw = lw - obj->lweight[i + 1];
			itr2 = 1;
			for (j = jinit; j < itr; ++j) {
				temp = 0.0;
				for (k = 0; k < kfin; ++k) {
					temp += obj->gradient[itr + k] * obj->weight[lw + itr2];// add weights
					//printf("W %d ", lw + itr2);
					itr2 += (N + 1);
				}
				itr2 = j - jinit + 2;
				obj->gradient[j] = temp;
			}
			itr -= N;
		}
		else if (obj->actfcn[i] == 2) {
			N = obj->arch[i];
			jinit = itr - N;
			//logsig(obj->tout + jinit, N, beta + jinit);
			kfin = obj->arch[i + 1];
			lw = lw - obj->lweight[i + 1];
			itr2 = 1;
			for (j = jinit; j < itr; ++j) {
				temp = 0.0;
				for (k = 0; k < kfin; ++k) {
					temp += obj->gradient[itr + k] * obj->weight[lw + itr2];// add weights
					//printf("W %d ", lw + itr2);
					itr2 += (N + 1);
				}
				itr2 = j - jinit + 2;
				obj->gradient[j] = temp * obj->tout[j] * (1.0 - obj->tout[j]);
				//printf("temp %g %g ", temp, obj->tout[j]);
			}
			itr -= N;
		}
		else if (obj->actfcn[i] == 3) {
			N = obj->arch[i];
			jinit = itr - N;
			//logsig(obj->tout + jinit, N, beta + jinit);
			kfin = obj->arch[i + 1];
			lw = lw - obj->lweight[i + 1];
			itr2 = 1;
			for (j = jinit; j < itr; ++j) {
				temp = 0.0;
				for (k = 0; k < kfin; ++k) {
					temp += obj->gradient[itr + k] * obj->weight[lw + itr2];// add weights
					//printf("W %d ", lw + itr2);
					itr2 += (N + 1);
				}
				itr2 = j - jinit + 2;
				obj->gradient[j] = temp * (1.0 + obj->tout[j]) * (1.0 - obj->tout[j]);
			}
			itr -= N;
		}
	}


	// Calculate weights and deltas

	lw = obj->lw;
	in0 = obj->arch[0];

	for (i = 0; i < in0; ++i) {
		tinp[i] = obj->input[i];
	}

	for (i = in0; i < in0 + ld; ++i) {
		tinp[i] = obj->tout[i - in0];
	}


	itr3 = 0;
	itr2 = itr4 = 0;
	if (!strcmp(obj->trainfcn, "traingda")) {
		for (i = 0; i < lm1; ++i) {
			N = obj->arch[i] + 1;
			S = obj->arch[i + 1];
			for (j = 0; j < S; ++j) {
				itr = j * N;// iterates over one row of weights
				index = itr3 + itr;
				tvec[index] += -lr * obj->gradient[itr2 + j];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					tvec[index] += -lr * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					//printf(" ind %d", itr + k - 1);
				}

			}
			itr3 += S * N;// iterates over all the weights going into a layer
			itr2 += S;// iterates over each output layer
			itr4 += (N - 1);// iterates over each input layer
			//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		}
	}
	else if (!strcmp(obj->trainfcn, "traingdx")) {
		for (i = 0; i < lm1; ++i) {
			N = obj->arch[i] + 1;
			S = obj->arch[i + 1];
			for (j = 0; j < S; ++j) {
				itr = j * N;// iterates over one row of weights
				index = itr3 + itr;
				temp = obj->delta[index];
				tvec[index] += mc * temp - lr * (1.0 - mc) * obj->gradient[itr2 + j];
				//obj->weight[index] += obj->delta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					temp = obj->delta[index];
					tvec[index] += mc * temp - lr * (1.0 - mc) * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					//obj->weight[index] += obj->delta[index];
					//printf(" ind %d", itr + k - 1);
				}

			}
			itr3 += S * N;// iterates over all the weights going into a layer
			itr2 += S;// iterates over each output layer
			itr4 += (N - 1);// iterates over each input layer
			//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		}
	}


	free(tinp);

}

void mapminmax(double *x, int N, double ymin, double ymax, double *y) {
	double xmin, xmax, t;
	int i;

	xmin = dmin(x, N);
	xmax = dmax(x, N);

	t = (ymax - ymin) / (xmax - xmin);

	for (i = 0; i < N; ++i) {
		y[i] = (x[i] - xmin) * t + ymin;
	}
}

void mapstd(double *x, int N, double ymean, double ystd, double *y) {
	double xmean, xstd, t;
	int i;

	xmean = mean(x, N);
	xstd = std(x, N);

	t = ystd / xstd;

	for (i = 0; i < N; ++i) {
		y[i] = (x[i] - xmean) * t + ymean;
	}

}

static void mapminmax_stride(double *x, int N, int stride,double ymin, double ymax, double *y) {
	double xmin, xmax, t;
	int i;

	xmin = dmin_stride(x, N,stride);
	xmax = dmax_stride(x, N,stride);

	t = (ymax - ymin) / (xmax - xmin);

	for (i = 0; i < N; ++i) {
		y[i*stride] = (x[i*stride] - xmin) * t + ymin;
	}
}

static void mapminmax_stride_apply(double *x, int N, int stride, double ymin, double ymax, double xmin, double xmax, double *y) {
	double t;
	int i;

	t = (ymax - ymin) / (xmax - xmin);

	for (i = 0; i < N; ++i) {
		y[i*stride] = (x[i*stride] - xmin) * t + ymin;
	}
}

static void mapstd_stride(double *x, int N, int stride,double ymean, double ystd, double *y) {
	double xmean, xstd, t;
	int i;

	xmean = mean_stride(x, N,stride);
	xstd = std_stride(x, N,stride);

	t = ystd / xstd;

	for (i = 0; i < N; ++i) {
		y[i*stride] = (x[i*stride] - xmean) * t + ymean;
	}
}

static void mapstd_stride_apply(double *x, int N, int stride, double ymean, double ystd, double xmean, double xstd, double *y) {
	double t;
	int i;

	t = ystd/xstd;

	for (i = 0; i < N; ++i) {
		y[i*stride] = (x[i*stride] - xmean) * t + ymean;
	}
}

static void shuffle(int N, int *index) {
	int i,j,temp;

	for (i = 0; i < N; ++i) {
		index[i] = i;
	}

	for (i = 0; i < N; ++i) {
		j = i + rand() % (N - i);
		temp = index[i];
		index[i] = index[j];
		index[j] = temp;
	}
}

static void shuffleinput(int N, int leninp, double *input, double *shuffled, int lenoup, double *output, double *target) {
	int i,j;
	int *index;

	index = (int*)malloc(sizeof(int)*N);
	shuffle(N, index);

	for (i = 0; i < N; ++i) {
		for (j = 0; j < leninp; ++j) {
			shuffled[index[i] * leninp + j] = input[i* leninp + j];
		}
		for (j = 0; j < lenoup; ++j) {
			target[index[i] * lenoup + j] = output[i* lenoup + j];
		}
	}

	free(index);
}

void premnmx(int size,double *p, int leninp, double *t, int lenoup, double *pn, double *tn, double ymin, double ymax, double omin, double omax, double *pmin, double *pmax, double *tmin, double *tmax) {
	// pmax and pmin have the length leninp each
	// tmax and tmin have the length lenoup each
	// pn is of size leninp*size
	// tn is of size lenoup*size

	int i;
	double temp;

	for (i = 0; i < leninp; ++i) {
		mapminmax_stride(p + i, size, leninp, ymin,ymax, pn + i);
		temp = dmin_stride(p+i, size, leninp);
		pmin[i] = temp;
		temp = dmax_stride(p + i, size, leninp);
		pmax[i] = temp;
	}

	for (i = 0; i < lenoup; ++i) {
		mapminmax_stride(t + i, size, lenoup, omin,omax, tn + i);
		temp = dmin_stride(t + i, size, lenoup);
		tmin[i] = temp;
		temp = dmax_stride(t + i, size, lenoup);
		tmax[i] = temp;
	}
}

void applymnmx(nnet_object obj,int size, double *p,int leninp, double *pn) {
	int i;
	double temp1,temp2;

	for (i = 0; i < leninp; ++i) {
		temp1 = obj->dmin[i];
		temp2 = obj->dmax[i];
		mapminmax_stride_apply(p+i, size, leninp, obj->inpnmin,obj->inpnmax,temp1, temp2, pn+i); 
	}
}

void applystd(nnet_object obj, int size, double *p, int leninp, double *pn) {
	int i;
	double temp1, temp2;

	for (i = 0; i < leninp; ++i) {
		temp1 = obj->dmean[i];
		temp2 = obj->dstd[i];
		mapstd_stride_apply(p + i, size, leninp, obj->inpnmean, obj->inpnstd, temp1, temp2, pn + i);
	}
}

void prestd(int size, double *p, int leninp, double *t, int lenoup, double *pn, double *tn,double ymean,double ystd,double omean,double ostd, double *dmean, double *dstd, double *tmean, double *tstd) {
	// dmean and dstd have the length leninp each
	// tmean and tstd have the length lenoup each
	// pn is of size leninp*size
	// tn is of size lenoup*size

	int i;
	double temp;

	for (i = 0; i < leninp; ++i) {
		mapstd_stride(p + i, size, leninp, ymean, ystd, pn + i);
		temp = mean_stride(p + i, size, leninp);
		dmean[i] = temp;
		temp = std_stride(p + i, size, leninp);
		dstd[i] = temp;
	}

	for (i = 0; i < lenoup; ++i) {
		mapstd_stride(t + i, size, lenoup, omean, ostd, tn + i);
		temp = mean_stride(t + i, size, lenoup);
		tmean[i] = temp;
		temp = std_stride(t + i, size, lenoup);
		tstd[i] = temp;
	}
}

void postmnmx(nnet_object obj,int size,double *oupn,int lenoup,double *oup) {
	int i;
	double temp1, temp2;

	for (i = 0; i < lenoup; ++i) {
		temp1 = obj->tmin[i];
		temp2 = obj->tmax[i];
		mapminmax_stride_apply(oupn + i, size, lenoup, temp1, temp2, obj->oupnmin, obj->oupnmax, oup + i);
	}
}

void poststd(nnet_object obj, int size, double *oupn, int lenoup, double *oup) {
	int i;
	double temp1, temp2;

	for (i = 0; i < lenoup; ++i) {
		temp1 = obj->tmean[i];
		temp2 = obj->tstd[i];
		mapstd_stride_apply(oupn + i, size, lenoup, temp1, temp2,obj->oupnmean,obj->oupnstd, oup + i);
	}
}

static void epoch_gdm_alr(nnet_object obj, int tsize, double *data, double *target, int *index, double *tvec, double *output) {
	int lendata, lentarget, i, j,itrd, itrt, leninp, lenoup;
	double mse,gmse,temp;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;

	shuffle(tsize, index);

	for (i = 0; i < obj->lw; ++i) {
		tvec[i] = 0.0;
	}

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output);
		backpropagate_alr(obj, output, target + itrt, lenoup,tvec);
		
	}
	//printf("\n");

	for (i = 0; i < obj->lw; ++i) {
		//obj->delta[i] = tvec[i];
		obj->weight[i] += tvec[i];
	}

	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (2.0 * lenoup * tsize);

}

static void epoch_gdm(nnet_object obj, int tsize, double *data, double *target,int *index,double *output) {
	int lendata, lentarget, i,j,itrd,itrt,leninp,lenoup;
	double mse,gmse,temp;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;

	shuffle(tsize, index);

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output);
		backpropagate(obj, output, target + itrt, lenoup);
	}
	//printf("\n");
	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (2.0 * lenoup * tsize);
}

void train_null(nnet_object obj, int size, double *inp, double *out) {
	int epoch,i;
	int tsize, gsize, vsize;
	int itrd,itrt,leninp,lenoup;
	double mse,gmse,vmse,omse,mcval;
	double mpe, lr_inc, lr_dec;
	double *output,*data,*target;
	double *tweight;
	int *index, *indexg,*indexv;
	int gen, val;

	gen = val = 0;
	obj->normmethod = 0;

	tsize = (int) (obj->tratio * size); // training size
	gsize = (int)(obj->gratio * size); // generalization size
	vsize = size - tsize - gsize; // validation size

	output = (double*)malloc(sizeof(double)* obj->arch[obj->lm1]);
	index = (int*)malloc(sizeof(int)*tsize);
	indexg = (int*)malloc(sizeof(int)*gsize);
	indexv = (int*)malloc(sizeof(int)*vsize);

	data = (double*)malloc(sizeof(double)* size * obj->arch[0]);
	target = (double*)malloc(sizeof(double)* size * obj->arch[obj->lm1]);
	tweight = (double*)malloc(sizeof(double)*obj->lw);

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];

	shuffleinput(size, obj->arch[0], inp, data, obj->arch[obj->lm1],out, target);

	//printf("size %d %d %d \n", tsize,gsize,vsize);


	mcval = obj->alpha;
	mpe = obj->perf_inc;
	lr_inc = obj->eta_inc;
	lr_dec = obj->eta_dec;

	itrd = tsize * obj->arch[0];
	itrt = tsize * obj->arch[obj->lm1];

	for (i = 0; i < tsize; ++i) {
		index[i] = i;
	}

	for (i = 0; i < gsize; ++i) {
		indexg[i] = i;
	}

	for (i = 0; i < vsize; ++i) {
		indexv[i] = i;
	}

	if (gsize > 0) {
		gen = 1;
	}

	if (vsize > 0) {
		val = 1;
	}


	for (i = 0; i < obj->lw; ++i) {
		tweight[i] = obj->weight[i];
	}

	if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingdm")) {
		epoch_gdm(obj, tsize, data, target, index, output);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm(obj, tsize, data, target, index, output);
			mse = obj->mse;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g \n", epoch, mse);
			}

			epoch++;
		}
	}
	else if (!strcmp(obj->trainfcn, "traingda")) {
		epoch_gdm_alr(obj, tsize, data, target, index,tweight, output);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm_alr(obj, tsize, data, target, index, tweight,output);
			mse = obj->mse;
			if (mse > mpe*omse) {
				obj->eta *= lr_dec;	
				for (i = 0; i < obj->lw; ++i) {
					obj->weight[i] -= tweight[i];
				}
				mse = omse;
			}
			else {
				if (mse < omse) {
					obj->eta *= lr_inc;
				}
				omse = mse;
			}

			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g omse %g eta %g \n", epoch, mse,omse,obj->eta);
			}

			epoch++;
		}
	} else if (!strcmp(obj->trainfcn, "traingdx")) {
		epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
		for (i = 0; i < obj->lw; ++i) {
			obj->delta[i] = tweight[i];
		}
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
			mse = obj->mse;
			if (mse > mpe*omse) {
				obj->eta *= lr_dec;
				obj->alpha = 0.0;
				for (i = 0; i < obj->lw; ++i) {
					obj->weight[i] -= tweight[i];
				}
				//mse = omse;
			}
			else {
				if (mse < omse) {
					obj->eta *= lr_inc;
					obj->alpha = mcval;
				}
				for (i = 0; i < obj->lw; ++i) {
					obj->delta[i] = tweight[i];
				}
				omse = mse;
			}

			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g omse %g lr %g mc %g \n", epoch, mse, omse, obj->eta,obj->alpha);
			}

			epoch++;
		}
	}

	// Validate

	itrd += gsize * obj->arch[0];
	itrt += gsize * obj->arch[obj->lm1];

	if (val == 1) {
		vmse = gvmse(obj, vsize, data + itrd, target + itrt, indexv, output);

		printf("\n Validation MSE %g \n", vmse);
	}

	free(output);
	free(index);
	free(indexg);
	free(indexv);
	free(data);
	free(target);
	free(tweight);
}

void train_mnmx(nnet_object obj, int size, double *inp, double *out) {
	int epoch, i;
	int tsize, gsize, vsize;
	int itrd, itrt, leninp, lenoup;
	double mse, gmse, vmse, omse, mcval;
	double mpe, lr_inc, lr_dec;
	double *output, *data, *target,*tweight;
	double *pn, *tn;
	int *index, *indexg, *indexv;
	int gen, val;

	gen = val = 0;

	obj->normmethod = 1;

	tsize = (int)(obj->tratio * size); // training size
	gsize = (int)(obj->gratio * size); // generalization size
	vsize = size - tsize - gsize; // validation size

	output = (double*)malloc(sizeof(double)* obj->arch[obj->lm1]);
	index = (int*)malloc(sizeof(int)*tsize);
	indexg = (int*)malloc(sizeof(int)*gsize);
	indexv = (int*)malloc(sizeof(int)*vsize);

	data = (double*)malloc(sizeof(double)* size * obj->arch[0]);
	target = (double*)malloc(sizeof(double)* size * obj->arch[obj->lm1]);

	pn = (double*)malloc(sizeof(double)* size * obj->arch[0]);
	tn = (double*)malloc(sizeof(double)* size * obj->arch[obj->lm1]);
	tweight = (double*)malloc(sizeof(double)*obj->lw);

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];

	mcval = obj->alpha;
	mpe = obj->perf_inc;
	lr_inc = obj->eta_inc;
	lr_dec = obj->eta_dec;

	premnmx(size, inp, leninp, out, lenoup, pn, tn, obj->inpnmin, obj->inpnmax, obj->oupnmin, obj->oupnmax, obj->dmin, obj->dmax, obj->tmin, obj->tmax);
	/*
	for (i = 0; i < size; ++i) {
		printf("%g %g %g %g \n", pn[i*leninp], pn[i*leninp+1], pn[i*leninp+2], pn[i*leninp+3]);
	}
	*/

	shuffleinput(size, obj->arch[0], pn, data, obj->arch[obj->lm1], tn, target);

	//printf("size %d %d %d \n", tsize,gsize,vsize);



	itrd = tsize * obj->arch[0];
	itrt = tsize * obj->arch[obj->lm1];

	for (i = 0; i < tsize; ++i) {
		index[i] = i;
	}

	for (i = 0; i < gsize; ++i) {
		indexg[i] = i;
	}

	for (i = 0; i < vsize; ++i) {
		indexv[i] = i;
	}

	if (gsize > 0) {
		gen = 1;
	}

	if (vsize > 0) {
		val = 1;
	}

	for (i = 0; i < obj->lw; ++i) {
		tweight[i] = obj->weight[i];
	}

	if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingdm")) {
		epoch_gdm(obj, tsize, data, target, index, output);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm(obj, tsize, data, target, index, output);
			mse = obj->mse;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g \n", epoch, mse);
			}

			epoch++;
		}
	}
	else if (!strcmp(obj->trainfcn, "traingda")) {
		epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
			mse = obj->mse;
			if (mse > mpe*omse) {
				obj->eta *= lr_dec;
				for (i = 0; i < obj->lw; ++i) {
					obj->weight[i] -= tweight[i];
				}
				mse = omse;
			}
			else {
				if (mse < omse) {
					obj->eta *= lr_inc;
				}
				omse = mse;
			}

			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g omse %g eta %g \n", epoch, mse, omse, obj->eta);
			}

			epoch++;
		}
	}
	else if (!strcmp(obj->trainfcn, "traingdx")) {
		epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
		for (i = 0; i < obj->lw; ++i) {
			obj->delta[i] = tweight[i];
		}
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
			mse = obj->mse;
			if (mse > mpe*omse) {
				obj->eta *= lr_dec;
				obj->alpha = 0.0;
				for (i = 0; i < obj->lw; ++i) {
					obj->weight[i] -= tweight[i];
				}
				//mse = omse;
			}
			else {
				if (mse < omse) {
					obj->eta *= lr_inc;
					obj->alpha = mcval;
				}
				for (i = 0; i < obj->lw; ++i) {
					obj->delta[i] = tweight[i];
				}
				omse = mse;
			}

			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g omse %g lr %g mc %g \n", epoch, mse, omse, obj->eta, obj->alpha);
			}

			epoch++;
		}
	}

	// Validate

	itrd += gsize * obj->arch[0];
	itrt += gsize * obj->arch[obj->lm1];

	if (val == 1) {
		vmse = gvmse(obj, vsize, data + itrd, target + itrt, indexv, output);

		printf("\n Validation MSE %g \n", vmse);
	}

	free(output);
	free(index);
	free(indexg);
	free(indexv);
	free(data);
	free(target);
	free(pn);
	free(tn);
	free(tweight);
}

void train_mstd(nnet_object obj, int size, double *inp, double *out) {
	int epoch, i;
	int tsize, gsize, vsize;
	int itrd, itrt, leninp, lenoup;
	double mse, gmse, vmse, omse, mcval;
	double mpe, lr_inc, lr_dec;
	double *output, *data, *target, *tweight;
	double *pn, *tn;
	int *index, *indexg, *indexv;
	int gen, val;

	gen = val = 0;

	obj->normmethod = 2;

	tsize = (int)(obj->tratio * size); // training size
	gsize = (int)(obj->gratio * size); // generalization size
	vsize = size - tsize - gsize; // validation size

	output = (double*)malloc(sizeof(double)* obj->arch[obj->lm1]);
	index = (int*)malloc(sizeof(int)*tsize);
	indexg = (int*)malloc(sizeof(int)*gsize);
	indexv = (int*)malloc(sizeof(int)*vsize);

	data = (double*)malloc(sizeof(double)* size * obj->arch[0]);
	target = (double*)malloc(sizeof(double)* size * obj->arch[obj->lm1]);

	pn = (double*)malloc(sizeof(double)* size * obj->arch[0]);
	tn = (double*)malloc(sizeof(double)* size * obj->arch[obj->lm1]);
	tweight = (double*)malloc(sizeof(double)*obj->lw);

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];

	mcval = obj->alpha;
	mpe = obj->perf_inc;
	lr_inc = obj->eta_inc;
	lr_dec = obj->eta_dec;

	//premnmx(size, inp, leninp, out, lenoup, pn, tn, obj->inpnmin, obj->inpnmax, obj->oupnmin, obj->oupnmax, obj->dmin, obj->dmax, obj->tmin, obj->tmax);
	prestd(size, inp, leninp, out, lenoup, pn, tn, obj->inpnmean, obj->inpnstd, obj->oupnmean, obj->oupnstd, obj->dmean, obj->dstd, obj->tmean, obj->tstd);

	

	shuffleinput(size, obj->arch[0], pn, data, obj->arch[obj->lm1], tn, target);

	//printf("size %d %d %d \n", tsize,gsize,vsize);



	itrd = tsize * obj->arch[0];
	itrt = tsize * obj->arch[obj->lm1];

	for (i = 0; i < tsize; ++i) {
		index[i] = i;
	}

	for (i = 0; i < gsize; ++i) {
		indexg[i] = i;
	}

	for (i = 0; i < vsize; ++i) {
		indexv[i] = i;
	}

	if (gsize > 0) {
		gen = 1;
	}

	if (vsize > 0) {
		val = 1;
	}

	for (i = 0; i < obj->lw; ++i) {
		tweight[i] = obj->weight[i];
	}

	if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingdm")) {
		epoch_gdm(obj, tsize, data, target, index, output);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm(obj, tsize, data, target, index, output);
			mse = obj->mse;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g \n", epoch, mse);
			}

			epoch++;
		}
	}
	else if (!strcmp(obj->trainfcn, "traingda")) {
		epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
			mse = obj->mse;
			if (mse > mpe*omse) {
				obj->eta *= lr_dec;
				for (i = 0; i < obj->lw; ++i) {
					obj->weight[i] -= tweight[i];
				}
				mse = omse;
			}
			else {
				if (mse < omse) {
					obj->eta *= lr_inc;
				}
				omse = mse;
			}

			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g omse %g eta %g \n", epoch, mse, omse, obj->eta);
			}

			epoch++;
		}
	}
	else if (!strcmp(obj->trainfcn, "traingdx")) {
		epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
		for (i = 0; i < obj->lw; ++i) {
			obj->delta[i] = tweight[i];
		}
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_gdm_alr(obj, tsize, data, target, index, tweight, output);
			mse = obj->mse;
			if (mse > mpe*omse) {
				obj->eta *= lr_dec;
				obj->alpha = 0.0;
				for (i = 0; i < obj->lw; ++i) {
					obj->weight[i] -= tweight[i];
				}
				//mse = omse;
			}
			else {
				if (mse < omse) {
					obj->eta *= lr_inc;
					obj->alpha = mcval;
				}
				for (i = 0; i < obj->lw; ++i) {
					obj->delta[i] = tweight[i];
				}
				omse = mse;
			}

			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output);
				printf("EPOCH %d MSE %g %g \n", epoch, mse, gmse);
			}
			else {
				printf("EPOCH %d MSE %g omse %g lr %g mc %g \n", epoch, mse, omse, obj->eta, obj->alpha);
			}

			epoch++;
		}
	}

	// Validate

	itrd += gsize * obj->arch[0];
	itrt += gsize * obj->arch[obj->lm1];

	if (val == 1) {
		vmse = gvmse(obj, vsize, data + itrd, target + itrt, indexv, output);

		printf("\n Validation MSE %g \n", vmse);
	}

	free(output);
	free(index);
	free(indexg);
	free(indexv);
	free(data);
	free(target);
	free(pn);
	free(tn);
	free(tweight);
}

void train(nnet_object obj, int tsize, double *data, double *target) {

	if (obj->normmethod == 0) {
		train_null(obj, tsize, data, target);
	}
	else if (obj->normmethod == 1) {
		train_mnmx(obj, tsize, data, target);
	}
	else if (obj->normmethod == 2) {
		train_mstd(obj, tsize, data, target);
	}
	else {
		printf("\n Available Normalization Methods : 0 - NULL,1 - Minmax ,2 - Mean/Std method\n");
		exit(1);
	}
}

void sim(nnet_object obj, int size, double *data, double *output) {
	int leninp, lenoup,i;
	double *pn,*tn;

	pn = (double*)malloc(sizeof(double)* size * obj->arch[0]);
	tn = (double*)malloc(sizeof(double)* size * obj->arch[obj->lm1]);

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];

	if (obj->normmethod == 0) {

		for (i = 0; i < size; ++i) {
			feedforward(obj, data + i*leninp, leninp, lenoup, output + i *lenoup);
		}

	} else if (obj->normmethod == 1) {
		for (i = 0; i < lenoup*size; ++i) {
			tn[i] = 0.0;
		}

		applymnmx(obj, size, data, leninp, pn);

		for (i = 0; i < size; ++i) {
			feedforward(obj, pn + i*leninp, leninp, lenoup, tn + i *lenoup);
		}

		postmnmx(obj, size, tn, lenoup, output);
	}
	else if (obj->normmethod == 2) {
		for (i = 0; i < lenoup*size; ++i) {
			tn[i] = 0.0;
		}

		applystd(obj, size, data, leninp, pn);
		
		for (i = 0; i < size; ++i) {
			feedforward(obj, pn + i*leninp, leninp, lenoup, tn + i *lenoup);
		}
		

		poststd(obj, size, tn, lenoup, output);
	}
	else {
		printf("\n Available Normalization Methods : 0 - NULL,1 - Minmax ,2 - Mean/Std method\n");
		exit(1);
	}

	free(pn);
	free(tn);
}

void nnet_free(nnet_object obj) {
	free(obj);
}
