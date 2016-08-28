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
	t_vector = 2 * ld + lw + li;

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

	obj->generalize = 0;// Generalize and Validation initialized to 0.
	obj->validate = 0; //  They will automatically change to 1 if a generalization and/or
	// a validation set is used. see void set_training_ratios(nnet_object obj, double tratio, double gratio, double vratio);

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
	obj->gradient = &obj->params[lw];
	obj->tout = &obj->params[lw + ld];
	obj->input = &obj->params[lw + 2*ld];

	obj->dmin = &obj->params[t_vector];
	obj->dmax = &obj->params[t_vector + arch[0]];
	obj->tmin = &obj->params[t_vector + 2 * arch[0]];
	obj->tmax = &obj->params[t_vector + 2 * arch[0] + arch[layers-1]];

	obj->dmean = &obj->params[t_vector + add_vector2];
	obj->dstd = &obj->params[t_vector + add_vector2 + arch[0]];
	obj->tmean = &obj->params[t_vector + add_vector2 + 2 * arch[0]];
	obj->tstd = &obj->params[t_vector + add_vector2 + 2 * arch[0] + arch[layers - 1]];

	obj->nmax = intmax(arch, layers);
	obj->mse = 1.0;
	obj->tmse = 1.0e-04;
	obj->gmse = 1.0e-04;
	obj->imse = DBL_MAX;

	obj->steepness = 0.5;

	obj->tratio = 1.00;
	obj->gratio = 0.00;
	obj->vratio = 0.00;

	obj->actfcn[0] = 0;
	for (i = 1; i < layers; ++i) {
		obj->actfcn[i] = actfcn[i];
	}

	obj->normmethod = 0; // Input Normalization Method
	obj->inpnmin = obj->oupnmin = - 1;
	obj->inpnmax = obj->oupnmax = 1;
	obj->inpnmean = obj->oupnmean = 0;
	obj->inpnstd = obj->oupnstd = 1;

	//Quickpropagation Parameter;
	obj->qp_threshold = 0.001;
	obj->qp_max_factor = 1.75;
	obj->qp_shrink_factor = obj->qp_max_factor / (1.0 + obj->qp_max_factor);
	obj->qp_decay = -0.0001;

	// RPROP parameters

	obj->rp_eta_p = 1.2;
	obj->rp_eta_n = 0.5;
	obj->rp_delta_min = 1e-06;
	obj->rp_init_upd = 0.1;
	obj->rp_max_step = 50;
	obj->rp_zero_tol = 0.00000000000000001;

	initweights(obj);
	//initweightsnw(obj);
	strcpy(obj->trainfcn, "traingd");
	strcpy(obj->trainmethod, "online");
	obj->batchsize = 1;// This value is ignored by default and will be used only if "batch" method is selected by the user using set_trainmethod()

	return obj;
}

lm_object lm_init(nnet_object nnet, ndata_object ndata) {
	lm_object lm = NULL;

	lm = (lm_object)malloc(sizeof(struct lm_set));

	lm->net = nnet;
	lm->data = ndata;

	return lm;
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

void set_generalization_mse(nnet_object obj, double gmse) {
	if (gmse < 0.0) {
		printf("MSE only takes values over 0.0\n");
		exit(1);
	}
	else {
		obj->gmse = gmse;
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
	else if (!strcmp(trainfcn, "trainqp")) {
		strcpy(obj->trainfcn, "trainqp");
	}
	else if (!strcmp(trainfcn, "trainrp")) {
		strcpy(obj->trainfcn, "trainrp");
	}
	else if (!strcmp(trainfcn, "trainirp")) {
		strcpy(obj->trainfcn, "trainirp");
	}
	else {
		printf("Error : Available Training Functions - traingd, traingdm, traingda, traingdx,trainqp,trainrp,trainirp");
		exit(1);
	}
}

void set_trainmethod(nnet_object obj, char *method, int batchsize) {
	if (!strcmp(method, "online")) {
		strcpy(obj->trainmethod, "online");
	}
	else if (!strcmp(method, "batch")) {
		strcpy(obj->trainmethod, "batch");
		obj->batchsize = batchsize;
	}
	else {
		printf("Error : Available Training Methods - online and batch");
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
	srand((unsigned int) time(NULL));
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
	srand((unsigned int) time(NULL));
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

static double gvmse(nnet_object obj, int tsize, double *data, double *target, int *index, double *output, double *tempi,double *tempo) {
	double gmse, temp;
	int i, itrd, itrt, leninp, lenoup, j;

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output,tempi,tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	gmse = gmse / (lenoup * tsize);

	return gmse;
}

void feedforward(nnet_object obj, double *inp, int leninp, int lenoup, double *oup,double *tempi, double *tempo) {
	int lm1,i,N,S,itr,itr2,j,N1;
	lm1 = obj->lm1;


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


}



void backpropagate(nnet_object obj, double *output, double *desired, int lenoup,double *delta,double *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2,itr4;
	int S, itr3, index, in0;
	double temp,lr,mc;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}

	// Local Gradients Calculation
	itr = ld - loup;

	if (obj->actfcn[lm1] == 1) {
		for (i = 0; i < loup; ++i) {
			//printf("Wcv %g ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);
	
		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
		}
	}


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
				obj->weight[index] += lr * obj->gradient[itr2 + j];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					obj->weight[index] += lr * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
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
				temp = delta[index];
				delta[index] = mc * temp + lr * (1.0 - mc) * obj->gradient[itr2 + j];
				obj->weight[index] += delta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					temp = delta[index];
					delta[index] = mc * temp + lr * (1.0 - mc) * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					obj->weight[index] += delta[index];
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


}

static void backpropagate_alr(nnet_object obj, double *output, double *desired, int lenoup,double *delta,double *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	double temp, lr, mc;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}

	// Local Gradients Calculation
	itr = ld - loup;

	if (obj->actfcn[lm1] == 1) {
		for (i = 0; i < loup; ++i) {
			//printf("Wcv %g ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
		}
	}

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
				obj->weight[index] += lr * obj->gradient[itr2 + j];
				//tvec[index] += -lr * obj->gradient[itr2 + j];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					obj->weight[index] += lr * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					//tvec[index] += -lr * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					//printf(" index %d", index);
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
				temp = delta[index];
				delta[index] = mc * temp + lr * (1.0 - mc) * obj->gradient[itr2 + j];
				obj->weight[index] += delta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					temp = delta[index];
					delta[index] = mc * temp + lr * (1.0 - mc) * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					obj->weight[index] += delta[index];
					//printf(" ind %d", itr + k - 1);
				}

			}
			itr3 += S * N;// iterates over all the weights going into a layer
			itr2 += S;// iterates over each output layer
			itr4 += (N - 1);// iterates over each input layer
			//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		}
	}

}

static void backpropagate_mb_1(nnet_object obj, double *output, double *desired, int lenoup, double *delta,double *tdelta,double *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	double temp, lr, mc;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}


	// Local Gradients Calculation
	itr = ld - loup;

	if (obj->actfcn[lm1] == 1) {
		for (i = 0; i < loup; ++i) {
			//printf("Wcv %g ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
			//printf("%g %g \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
			obj->mse += temp*temp;
		}
	}


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

	for (i = 0; i < lm1; ++i) {
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			itr = j * N;// iterates over one row of weights
			index = itr3 + itr;
			tdelta[index] += obj->gradient[itr2 + j];
			//printf(" ind %d", itr2+j);

			for (k = 1; k < N; ++k) {
				index = itr3 + itr + k;
				tdelta[index] += tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
				//printf(" ind %d", itr + k - 1);
			}

		}
		itr3 += S * N;// iterates over all the weights going into a layer
		itr2 += S;// iterates over each output layer
		itr4 += (N - 1);// iterates over each input layer
		//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
	}

	//printf("WT %g \n", obj->weight[0]);


}

static void backpropagate_mb_2(nnet_object obj, double *delta, double *tdelta) {
	int itr, itr2, itr3, itr4;
	int lm1,i,N,S,j,k,index;

	double lr, mc,temp;
	lr = obj->eta;
	mc = obj->alpha;

	lm1 = obj->lm1;
	itr3 = 0;
	itr2 = itr4 = 0;

	if (!strcmp(obj->trainfcn, "traingd")) {
		for (i = 0; i < lm1; ++i) {
			N = obj->arch[i] + 1;
			S = obj->arch[i + 1];
			for (j = 0; j < S; ++j) {
				itr = j * N;// iterates over one row of weights
				index = itr3 + itr;
				obj->weight[index] += lr * tdelta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					obj->weight[index] += lr * tdelta[index];
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
				temp = delta[index];
				delta[index] = mc * temp + lr * (1.0 - mc) * tdelta[index];
				obj->weight[index] += delta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					temp = delta[index];
					delta[index] = mc * temp + lr * (1.0 - mc) * tdelta[index];
					obj->weight[index] += delta[index];
					//printf(" ind %d", itr + k - 1);
				}

			}
			itr3 += S * N;// iterates over all the weights going into a layer
			itr2 += S;// iterates over each output layer
			itr4 += (N - 1);// iterates over each input layer
			//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		}
	}
}

static void backpropagate_mb_3(nnet_object obj, double *delta, double *tdelta) {
	int itr, itr2, itr3, itr4;
	int lm1, i, N, S, j, k, index;

	double lr, mc, temp;
	lr = obj->eta;
	mc = obj->alpha;

	lm1 = obj->lm1;
	itr3 = 0;
	itr2 = itr4 = 0;

	if (!strcmp(obj->trainfcn, "traingda")) {
		for (i = 0; i < lm1; ++i) {
			N = obj->arch[i] + 1;
			S = obj->arch[i + 1];
			for (j = 0; j < S; ++j) {
				itr = j * N;// iterates over one row of weights
				index = itr3 + itr;
				obj->weight[index] += lr * tdelta[index];
				//tvec[index] += -lr * obj->gradient[itr2 + j];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					obj->weight[index] += lr * tdelta[index];
					//tvec[index] += -lr * tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
					//printf(" index %d", index);
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
				temp = delta[index];
				delta[index] = mc * temp + lr * (1.0 - mc) * tdelta[index];
				obj->weight[index] += delta[index];
				//printf(" ind %d", itr2+j);

				for (k = 1; k < N; ++k) {
					index = itr3 + itr + k;
					temp = delta[index];
					delta[index] = mc * temp + lr * (1.0 - mc) * tdelta[index];
					obj->weight[index] += delta[index];
					//printf(" ind %d", itr + k - 1);
				}

			}
			itr3 += S * N;// iterates over all the weights going into a layer
			itr2 += S;// iterates over each output layer
			itr4 += (N - 1);// iterates over each input layer
			//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		}
	}

}

void backpropagate_qp_1(nnet_object obj, double *output, double *desired, int lenoup, double *delta,double *slope,double *tslope,double *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	double temp, lr, mc,del,epsilon,dslope,temp2;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];

	//epsilon = lr / obj->datasize;

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}


	// Local Gradients Calculation
	itr = ld - loup;

	if (obj->actfcn[lm1] == 1) {
		for (i = 0; i < loup; ++i) {
			//printf("Wcv %g ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			obj->mse += (temp*temp);
			//printf("%g %g \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * logsig_der(obj->tout[itr + i]);
			obj->mse += (temp*temp);
			//printf("%g %g \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * tansig_der(obj->tout[itr + i]);
			obj->mse += (temp*temp);
		}
	}


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
				obj->gradient[j] = temp * logsig_der(obj->tout[j]);;
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
				obj->gradient[j] = temp * tansig_der(obj->tout[j]);
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

	itr3 = itr = 0;
	itr2 = itr4 = 0;
	
	for (i = 0; i < lm1; ++i) {
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			itr = j * N;// iterates over one row of weights
			index = itr3 + itr;
			slope[index] += obj->gradient[itr2 + j];

			for (k = 1; k < N; ++k) {
				index = itr3 + itr + k;
				slope[index] += tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
				
			}

		}
		itr3 += S * N;// iterates over all the weights going into a layer
		itr2 += S;// iterates over each output layer
		itr4 += (N - 1);// iterates over each input layer
		//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
	}


}

void backpropagate_qp_2(nnet_object obj, double *delta, double *slope, double *tslope) {
	int i;
	double del, dslope, epsilon;
	double th_p, th_n, shrink_factor, max_factor,decay;

	//epsilon = 0.55 / (double)obj->lw;
	epsilon = obj->eta / obj->datasize;
	th_p = obj->qp_threshold;
	th_n = -1.0 * th_p;
	shrink_factor = obj->qp_shrink_factor;
	max_factor = obj->qp_max_factor;
	decay = obj->qp_decay;

	for (i = 0; i < obj->lw; ++i) {
		del = 0.0;
		dslope = slope[i] + decay * obj->weight[i];
		if (delta[i] > th_p) {
			if (dslope > 0.0) {
				del += epsilon * dslope;
			}
			if (dslope > (shrink_factor * tslope[i]))  {
				del += max_factor *delta[i];
			}
			else {
				del += delta[i] * dslope / (tslope[i] - dslope);
			}
		}
		else if (delta[i] < th_n) {
			if (dslope < 0.0) {
				del += epsilon * dslope;
			}
			if (dslope < (shrink_factor * tslope[i]))  {
				del += max_factor *delta[i];
			}
			else {
				del += delta[i] * dslope / (tslope[i] - dslope);
			}
		}
		else {
			del += epsilon * dslope;// +mc * delta[i];
		}

		delta[i] = del;
		obj->weight[i] += del;
		if (obj->weight[i] > 1500) {
			obj->weight[i] = 1500;
		}
		else if (obj->weight[i] < -1500) {
			obj->weight[i] = -1500;
		}
		tslope[i] = dslope;
		slope[i] = 0.0;
		//printf("%g ", obj->weight[i]);
	}
}

void backpropagate_rp_1(nnet_object obj, double *output, double *desired, int lenoup, double *delta, double *slope, double *tslope, double *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	double temp, lr, mc, del, epsilon, dslope;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];

	//epsilon = lr / obj->datasize;

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}


	// Local Gradients Calculation
	itr = ld - loup;

	if (obj->actfcn[lm1] == 1) {
		for (i = 0; i < loup; ++i) {
			//printf("Wcv %g ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			obj->mse += (temp*temp);
			//printf("%g %g \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * logsig_der(obj->tout[itr + i]);
			obj->mse += (temp*temp);
			//printf("%g %g \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * tansig_der(obj->tout[itr + i]);
			obj->mse += (temp*temp);
		}
	}


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
				obj->gradient[j] = temp * logsig_der(obj->tout[j]);
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
				obj->gradient[j] = temp * tansig_der(obj->tout[j]);
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

	for (i = 0; i < lm1; ++i) {
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			itr = j * N;// iterates over one row of weights
			index = itr3 + itr;
			slope[index] += obj->gradient[itr2 + j];

			for (k = 1; k < N; ++k) {
				index = itr3 + itr + k;
				slope[index] += tinp[itr4 + k - 1] * obj->gradient[itr2 + j];

			}

		}
		itr3 += S * N;// iterates over all the weights going into a layer
		itr2 += S;// iterates over each output layer
		itr4 += (N - 1);// iterates over each input layer
		//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
	}

}

void backpropagate_rp_2(nnet_object obj, double *delta, double *slope, double *tslope, double *updatevalue) {
	int i;
	double value, max_step,delta_min,ndelta,del;

	max_step = obj->rp_max_step;
	delta_min = obj->rp_delta_min;
	del = 0.0;
	
	for (i = 0; i < obj->lw; ++i) {
		value = signx(slope[i] * tslope[i]);
		if (value > 0) {
			ndelta = updatevalue[i] * obj->rp_eta_p;
			ndelta = pmin(ndelta, max_step);
			del = signx(slope[i]) * ndelta;
			updatevalue[i] = ndelta;
			tslope[i] = slope[i];
		}
		else if (value < 0) {
			ndelta = updatevalue[i] * obj->rp_eta_n;
			ndelta = pmax(ndelta, delta_min);
			updatevalue[i] = ndelta;
			tslope[i] = 0.0;
			del = -delta[i];
		}
		else {
			del = signx(slope[i]) * updatevalue[i];
			tslope[i] = slope[i];
		}

		obj->weight[i] += del;
		delta[i] = del;
		slope[i] = 0.0;
	}

}

void backpropagate_irp_2(nnet_object obj, double *delta, double *slope, double *tslope, double *updatevalue) {
	int i;
	double value, max_step, delta_min, ndelta, del;

	max_step = obj->rp_max_step;
	delta_min = obj->rp_delta_min;
	del = 0.0;

	for (i = 0; i < obj->lw; ++i) {
		value = signx(slope[i] * tslope[i]);
		if (value > 0) {
			ndelta = updatevalue[i] * obj->rp_eta_p;
			ndelta = pmin(ndelta, max_step);
			del = signx(slope[i]) * ndelta;
			updatevalue[i] = ndelta;
			tslope[i] = slope[i];
		}
		else if (value < 0) {
			ndelta = updatevalue[i] * obj->rp_eta_n;
			ndelta = pmax(ndelta, delta_min);
			updatevalue[i] = ndelta;
			tslope[i] = 0.0;
			if (obj->mse > obj->imse) {
				del = -delta[i];
			}
		}
		else {
			del = signx(slope[i]) * updatevalue[i];
			tslope[i] = slope[i];
		}

		obj->weight[i] += del;
		delta[i] = del;
		slope[i] = 0.0;
	}

}

void backpropagate_rqp_1(nnet_object obj, double *output, double *desired, int lenoup, double *slope, double *tinp,double *gradient2) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4,itr5;
	int S, itr3, index, in0, linp;
	double temp, lr, mc, del, epsilon, dslope, temp2;

	lw = obj->lw;
	ld = obj->ld;
	lm1 = obj->lm1;

	lr = obj->eta;
	mc = obj->alpha;

	loup = obj->arch[lm1];
	linp = obj->arch[0];

	//epsilon = lr / obj->datasize;

	if (lenoup != loup) {
		printf("Outputs of this Network are of length %d \n", loup);
	}


	// Local Gradients Calculation
	itr = linp + ld - loup;

	for (i = 0; i < itr; ++i)
	{
		gradient2[i] = 0.0;
	}
	
	for (i = 0; i < loup; ++i) {
		//printf("Wcv %g ", output[i]);
		temp = (desired[i] - output[i]);
		gradient2[itr + i] = temp;
		obj->mse += (temp*temp);
		//printf("%g %g \n", desired[i], output[i]);

	}

	lw = obj->lw;
	in0 = obj->arch[0];

	for (i = 0; i < in0; ++i) {
		tinp[i] = obj->input[i];
	}

	for (i = in0; i < in0 + ld; ++i) {
		tinp[i] = obj->tout[i - in0];
	}

	itr4 = in0 + ld;
	itr3 = lw;
	itr2 = 0;
	itr5 = itr;

	for (i = lm1; i > 0; --i) {
		N = obj->arch[i];
		S = obj->arch[i - 1] + 1;
		itr4 -= N;
		itr3 -= N*S;
		itr5 -= (S - 1);
		for (k = 0; k < N; ++k) {
			if (obj->actfcn[i] == 1) {
				obj->gradient[itr - linp + k] = gradient2[itr + k];
			}
			else if (obj->actfcn[i] == 2) {
				obj->gradient[itr - linp + k] = gradient2[itr + k] * logsig_der(tinp[itr4 + k]);
			}
			else if (obj->actfcn[i] == 3) {
				obj->gradient[itr - linp + k] = gradient2[itr + k] * tansig_der(tinp[itr4 + k]);
			}
			itr2 = k * S;
			index = itr3 + itr2;
			slope[index] += obj->gradient[itr - linp + k];
			gradient2[itr5] += obj->gradient[itr - linp + k];
			for (j = 1; j < S; ++j) {
				index = itr3 + itr2 + j;
				gradient2[itr5 + j - 1] += obj->gradient[itr - linp + k] * obj->weight[index];
				slope[index] += tinp[itr5 + j - 1] * obj->gradient[itr -linp + k];
			}
		}
		itr -= (S - 1);
	}

	/*
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
				obj->gradient[j] = temp * logsig_der(obj->tout[j]);;
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
				obj->gradient[j] = temp * tansig_der(obj->tout[j]);
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

	itr3 = itr = 0;
	itr2 = itr4 = 0;

	for (i = 0; i < lm1; ++i) {
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			itr = j * N;// iterates over one row of weights
			index = itr3 + itr;
			slope[index] += obj->gradient[itr2 + j];

			for (k = 1; k < N; ++k) {
				index = itr3 + itr + k;
				slope[index] += tinp[itr4 + k - 1] * obj->gradient[itr2 + j];

			}

		}
		itr3 += S * N;// iterates over all the weights going into a layer
		itr2 += S;// iterates over each output layer
		itr4 += (N - 1);// iterates over each input layer
		//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
	}
	*/

}
void mapminmax(double *x, int N, double ymin, double ymax, double *y) {
	double xmin, xmax, t;
	int i;

	xmin = dmin(x, N);
	xmax = dmax(x, N);
	if (xmax != xmin) {
		t = (ymax - ymin) / (xmax - xmin);

		for (i = 0; i < N; ++i) {
			y[i] = (x[i] - xmin) * t + ymin;
		}

	}
	else {
		for (i = 0; i < N; ++i) {
			y[i] = x[i];
		}
	}
}

void mapstd(double *x, int N, double ymean, double ystd, double *y) {
	double xmean, xstd, t;
	int i;

	xmean = mean(x, N);
	xstd = std(x, N);

	if (xstd != 0) {

		t = ystd / xstd;

		for (i = 0; i < N; ++i) {
			y[i] = (x[i] - xmean) * t + ymean;
		}
	}
	else {
		for (i = 0; i < N; ++i) {
			y[i] = x[i];
		}
	}
}

static void mapminmax_stride(double *x, int N, int stride,double ymin, double ymax, double *y) {
	double xmin, xmax, t;
	int i;

	xmin = dmin_stride(x, N,stride);
	xmax = dmax_stride(x, N,stride);

	if (xmax != xmin) {

		t = (ymax - ymin) / (xmax - xmin);

		for (i = 0; i < N; ++i) {
			y[i*stride] = (x[i*stride] - xmin) * t + ymin;
		}
	}
	else {
		for (i = 0; i < N; ++i) {
			y[i*stride] = x[i*stride];
		}
	}
}

static void mapminmax_stride_apply(double *x, int N, int stride, double ymin, double ymax, double xmin, double xmax, double *y) {
	double t;
	int i;

	if (xmax != xmin) {
		t = (ymax - ymin) / (xmax - xmin);

		for (i = 0; i < N; ++i) {
			y[i*stride] = (x[i*stride] - xmin) * t + ymin;
		}
	}
	else {
		for (i = 0; i < N; ++i) {
			y[i*stride] = x[i*stride];
		}
	}
}

static void mapstd_stride(double *x, int N, int stride,double ymean, double ystd, double *y) {
	double xmean, xstd, t;
	int i;

	xmean = mean_stride(x, N,stride);
	xstd = std_stride(x, N,stride);

	if (xstd != 0) {
		t = ystd / xstd;

		for (i = 0; i < N; ++i) {
			y[i*stride] = (x[i*stride] - xmean) * t + ymean;
		}
	}
	else {
		for (i = 0; i < N; ++i) {
			y[i*stride] = x[i*stride];
		}
	}
}

static void mapstd_stride_apply(double *x, int N, int stride, double ymean, double ystd, double xmean, double xstd, double *y) {
	double t;
	int i;
	if (xstd != 0) {
		t = ystd / xstd;

		for (i = 0; i < N; ++i) {
			y[i*stride] = (x[i*stride] - xmean) * t + ymean;
		}
	}
	else {
		for (i = 0; i < N; ++i) {
			y[i*stride] = x[i*stride] ;
		}
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

static void epoch_gdm_alr2(nnet_object obj, int tsize, double *data, double *target, int *index, double *delta,double *output,double *tinp,double *tempi,double *tempo) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	double mse, gmse, temp;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	obj->mse = 0.0;
	shuffle(tsize, index);


	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		//printf("output %g \n", data[i]);
		backpropagate_alr(obj, output, target + itrt, lenoup,delta,tinp);

	}

	obj->mse /= (lenoup * tsize);

}

static void epoch_gdm(nnet_object obj, int tsize, double *data, double *target,int *index,double *delta,double *output,double *tinp,double *tempi,double *tempo) {
	int lendata, lentarget, i,j,itrd,itrt,leninp,lenoup;
	double mse,gmse,temp;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	obj->mse = 0.0;
	shuffle(tsize, index);

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate(obj, output, target + itrt, lenoup,delta,tinp);
	}

	obj->mse /= (lenoup * tsize);
}

static void epoch_qp(nnet_object obj, int tsize, double *data, double *target, int *index, double *delta, double *slope, double *tslope,double *output,double *tinp,double *tempi,double *tempo,
	double *gradient2) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	double mse, gmse, temp;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	obj->mse = 0.0;

	shuffle(tsize, index);

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		//backpropagate_qp_1(obj, output, target + itrt, lenoup, delta,slope,tslope,tinp);
		backpropagate_rqp_1(obj, output, target + itrt, lenoup, slope, tinp,gradient2);
	}
	//printf("\n");
	gmse = 0.0;
	obj->mse /= (lenoup * tsize);

	backpropagate_qp_2(obj, delta, slope, tslope);

}

static void epoch_rp(nnet_object obj, int tsize, double *data, double *target, int *index, double *delta, double *slope, double *tslope, double *output, double *tinp, double *tempi, double *tempo,double *updatevalue) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	double mse, gmse, temp;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	obj->mse = 0.0;

	shuffle(tsize, index);

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate_rp_1(obj, output, target + itrt, lenoup, delta, slope, tslope, tinp);
	}
	//printf("\n");
	
	gmse = 0.0;
	obj->mse /= (lenoup * tsize);

	backpropagate_rp_2(obj, delta, slope, tslope,updatevalue);

}

static void epoch_irp(nnet_object obj, int tsize, double *data, double *target, int *index, double *delta, double *slope, double *tslope, double *output, double *tinp, double *tempi, double *tempo, double *updatevalue) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	double mse, gmse, temp;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	//obj->imse = obj->mse;
	obj->mse = 0.0;

	shuffle(tsize, index);

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate_rp_1(obj, output, target + itrt, lenoup, delta, slope, tslope, tinp);
	}
	//printf("\n");

	gmse = 0.0;
	obj->mse /= (lenoup * tsize);

	backpropagate_irp_2(obj, delta, slope, tslope, updatevalue);

}

static void epoch_mb(nnet_object obj, int tsize, double *data, double *target, int *index, double *delta, double *tdelta, double *output,double *tinp,double *tempi,double *tempo) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	double mse, gmse, temp;
	int batchsize,mbsize;

	batchsize = obj->batchsize;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	mbsize = 0;

	obj->mse = 0.0;

	shuffle(tsize, index);

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate_mb_1(obj, output, target + itrt, lenoup, delta,tdelta,tinp);
		mbsize++;
		if (mbsize == batchsize || (i == tsize-1 && mbsize >= 1) || mbsize == tsize-1) {
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] /= mbsize;
			}
			backpropagate_mb_2(obj, delta, tdelta);
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] = 0.0;
			}
			mbsize = 0;
		}
	}
	obj->mse /= (lenoup * tsize);
}

static void epoch_mbp(nnet_object obj, int tsize, double *data, double *target, int *index, double *delta, double *tdelta, double *output,double *tinp,double *tempi, double *tempo) {
	int lendata, lentarget, i, j, k, itrd, itrt, leninp, lenoup;
	double mse, gmse, temp;
	int batchsize, iters,maxsize,litr;

	batchsize = obj->batchsize;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	litr = 0;

	shuffle(tsize, index);

	iters = ceil(tsize / batchsize);

	for (k = 0; k < iters; ++k) {
		maxsize = (k + 1)*batchsize;
		if ((k + 1)*batchsize > tsize) {
			maxsize = tsize;
			litr = 1;
		}
//#pragma omp parallel for shared(tdelta)
		for (i = k*batchsize; i < maxsize; ++i) {
			itrd = index[i] * leninp;
			itrt = index[i] * lenoup;
			feedforward(obj, data + itrd, leninp, lenoup, output,tempi,tempo);
			backpropagate_mb_1(obj, output, target + itrt, lenoup, delta, tdelta,tinp);
		}
//#pragma omp barrier
		
		if (litr == 1) {
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] /= ((k+1)*batchsize- tsize);
			}
			litr = 0;
		}
		else {
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] /= batchsize;
			}
		}
		backpropagate_mb_2(obj, delta, tdelta);
		for (j = 0; j < obj->lw; ++j) {
			tdelta[j] = 0.0;
		}
		
	}
	//printf("\n");
	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (lenoup * tsize);
}
void train_null(nnet_object obj, int size, double *inp, double *out) {
	int epoch,i;
	int tsize, gsize, vsize;
	int itrd,itrt,leninp,lenoup;
	double mse,gmse,vmse,omse,mcval;
	double mpe, lr_inc, lr_dec;
	double *output,*data,*target;
	double *tweight,*delta,*tdelta,*slope,*tslope,*tinp,*gradient2;
	double *tempi, *tempo;
	int *index, *indexg,*indexv;
	int gen, val;
	double tstart, tend;

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
	tinp = (double*)malloc(sizeof(double)* (obj->ld + obj->arch[0]));

	tempi = (double*)malloc(sizeof(double)* obj->nmax);
	tempo = (double*)malloc(sizeof(double)* obj->nmax);

	gradient2 = (double*)malloc(sizeof(double)* (obj->ld + obj->arch[0]));

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];

	shuffleinput(size, obj->arch[0], inp, data, obj->arch[obj->lm1],out, target);

	//printf("size %d %d %d \n", tsize,gsize,vsize);

	if (gsize > 0) {
		gen = obj->generalize = 1;
	}

	if (vsize > 0) {
		val = obj->validate = 1;
	}


	mcval = obj->alpha;
	mpe = obj->perf_inc;
	lr_inc = obj->eta_inc;
	lr_dec = obj->eta_dec;

	itrd = tsize * obj->arch[0];
	itrt = tsize * obj->arch[obj->lm1];

	for (i = 0; i < tsize; ++i) {
		index[i] = i;
	}

	for (i = 0; i < obj->nmax; ++i) {
		tempi[i] = tempo[i] = 0;
	}

	for (i = 0; i < gsize; ++i) {
		indexg[i] = i;
	}

	for (i = 0; i < vsize; ++i) {
		indexv[i] = i;
	}
	for (i = 0; i < obj->ld + obj->arch[0]; ++i) {
		tinp[i] = 0;
	}

	if (gsize > 0) {
		gen = 1;
	}

	if (vsize > 0) {
		val = 1;
	}

	if (!strcmp(obj->trainfcn, "traingdm") || !strcmp(obj->trainfcn, "traingdx") || !strcmp(obj->trainfcn, "trainqp") ||
		!strcmp(obj->trainfcn, "trainrp") || !strcmp(obj->trainfcn, "trainirp")) {
		delta = (double*)malloc(sizeof(double)*obj->lw);
		for (i = 0; i < obj->lw; ++i) {
			delta[i] = 0;
		}
	}
	else {
		delta = (double*)malloc(sizeof(double)* 1);
		delta[0] = 0;
	}

	if (!strcmp(obj->trainfcn, "trainqp") || !strcmp(obj->trainfcn, "trainrp") || !strcmp(obj->trainfcn, "trainirp")) {
		slope = (double*)malloc(sizeof(double)*obj->lw);
		tslope = (double*)malloc(sizeof(double)*obj->lw);
		for (i = 0; i < obj->lw; ++i) {
			slope[i] = tslope[i] = 0;
		}
	}
	else {
		slope = (double*)malloc(sizeof(double)* 1);
		tslope = (double*)malloc(sizeof(double)* 1);
		slope[0] = tslope[0] = 0;
	}

	for (i = 0; i < obj->lw; ++i) {
		tweight[i] = obj->weight[i];
	}

	if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingdm")) {
		if (!strcmp(obj->trainmethod, "online")) {
			tdelta = (double*)malloc(sizeof(double)* 1);
			epoch_gdm(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gdm(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
				mse = obj->mse;
				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g \n", epoch, mse);
				}

				epoch++;
			}
		}
		else if (!strcmp(obj->trainmethod, "batch")) {
			tdelta = (double*)malloc(sizeof(double)*obj->lw);
			for (i = 0; i < obj->lw; ++i) {
				tdelta[i] = 0.0;
			}
			epoch_mbp(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				tstart = omp_get_wtime();
				epoch_mbp(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
				tend = omp_get_wtime();
				mse = obj->mse;
				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g Time %.16g \n", epoch, mse, tend - tstart);
				}

				epoch++;
			}
		}
	}

	if (!strcmp(obj->trainfcn, "traingda")) {
		if (!strcmp(obj->trainmethod, "online")) {
			tdelta = (double*)malloc(sizeof(double)* 1);
			epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			//printf("EPOCH %d MSE %g omse %g eta %g \n", epoch, mse, omse, obj->eta);
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				if (mse > mpe*omse) {
					obj->eta *= lr_dec;
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
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
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g omse %g eta %g \n", epoch, mse, omse, obj->eta);
				}

				epoch++;
			}
		}
	}

	if (!strcmp(obj->trainfcn, "traingdx")) {
		if (!strcmp(obj->trainmethod, "online")) {
			tdelta = (double*)malloc(sizeof(double)* 1);
			epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				if (mse > mpe*omse) {
					obj->eta *= lr_dec;
					obj->alpha = 0.0;
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
					}
					//mse = omse;
				}
				else {
					if (mse < omse) {
						obj->eta *= lr_inc;
						obj->alpha = mcval;
					}
					omse = mse;
				}

				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g omse %g lr %g mc %g \n", epoch, mse, omse, obj->eta, obj->alpha);
				}

				epoch++;
			}
		}
	}

	if (!strcmp(obj->trainfcn, "trainqp")) {
		tdelta = (double*)malloc(sizeof(double)* 1);
		epoch_qp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo,gradient2);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_qp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo,gradient2);
			mse = obj->mse;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
				printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
				if (gmse <= obj->gmse) {
					printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
					break;
				}
			}
			else {
				printf("EPOCH %d MSE %g \n", epoch, mse);
			}

			epoch++;
		}
	}
	if (!strcmp(obj->trainfcn, "trainrp")) {
		tdelta = (double*)malloc(sizeof(double)*obj->lw);
		for (i = 0; i < obj->lw; ++i) {
			tdelta[i] = obj->rp_init_upd;
		}
		epoch_rp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo,tdelta);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_rp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo,tdelta);
			mse = obj->mse;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
				printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
				if (gmse <= obj->gmse) {
					printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
					break;
				}
			}
			else {
				printf("EPOCH %d MSE %g \n", epoch, mse);
			}

			epoch++;
		}
	}

	if (!strcmp(obj->trainfcn, "trainirp")) {
		tdelta = (double*)malloc(sizeof(double)*obj->lw);
		for (i = 0; i < obj->lw; ++i) {
			tdelta[i] = obj->rp_init_upd;
		}
		epoch_irp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo, tdelta);
		obj->imse = obj->mse;
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_irp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo, tdelta);
			mse = obj->imse = obj->mse;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
				printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
				if (gmse <= obj->gmse) {
					printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
					break;
				}
			}
			else {
				printf("EPOCH %d MSE %g \n", epoch, mse);
			}

			epoch++;
		}
	}
	/*
	if (!strcmp(obj->trainmethod, "online")) {
		tdelta = (double*)malloc(sizeof(double)*1);
		if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingdm")) {
			epoch_gdm(obj, tsize, data, target, index, delta, output,tinp,tempi,tempo);
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gdm(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
				mse = obj->mse;
				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g \n", epoch, mse);
				}

				epoch++;
			}
		}
		else if (!strcmp(obj->trainfcn, "traingda")) {
			epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				if (mse > mpe*omse) {
					obj->eta *= lr_dec;
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
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
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g omse %g eta %g \n", epoch, mse, omse, obj->eta);
				}

				epoch++;
			}
		}
		else if (!strcmp(obj->trainfcn, "traingdx")) {
			epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gdm_alr2(obj, tsize, data, target, index, delta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				if (mse > mpe*omse) {
					obj->eta *= lr_dec;
					obj->alpha = 0.0;
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
					}
					//mse = omse;
				}
				else {
					if (mse < omse) {
						obj->eta *= lr_inc;
						obj->alpha = mcval;
					}
					omse = mse;
				}

				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g omse %g lr %g mc %g \n", epoch, mse, omse, obj->eta, obj->alpha);
				}

				epoch++;
			}
		}
		
	}
	else if (!strcmp(obj->trainmethod, "batch")) {
		tdelta = (double*)malloc(sizeof(double)*obj->lw);
		for (i = 0; i < obj->lw; ++i) {
			tdelta[i] = 0.0;
		}
		if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingdm")) {
			epoch_mbp(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				tstart = omp_get_wtime();
				epoch_mbp(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
				tend = omp_get_wtime();
				mse = obj->mse;
				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g Time %.16g \n", epoch, mse,tend-tstart);
				}

				epoch++;
			}
		}
		else if (!strcmp(obj->trainfcn, "trainqp")) {
			epoch_qp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo);
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_qp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo);
				mse = obj->mse;
				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					printf("EPOCH %d MSE %g GMSE %g \n", epoch, mse, gmse);
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %g \n", obj->gmse);
						break;
					}
				}
				else {
					printf("EPOCH %d MSE %g \n", epoch, mse);
				}

				epoch++;
			}
		}
	}
	*/
	// Validate

	itrd += gsize * obj->arch[0];
	itrt += gsize * obj->arch[obj->lm1];

	if (val == 1) {
		vmse = gvmse(obj, vsize, data + itrd, target + itrt, indexv, output, tempi, tempo);

		printf("\n Validation MSE %g \n", vmse);
	}

	free(output);
	free(index);
	free(indexg);
	free(indexv);
	free(data);
	free(target);
	free(tweight);
	free(delta);
	free(slope);
	free(tslope);
	free(tinp);
	free(tempi);
	free(tempo);
	free(gradient2);
}

void func_lm(double *x,int MP,int N,void *params) {
	int M, P,i,j;
	nnet_object obj = (nnet_object)params;
	M = obj->arch[obj->lm1];
	P = MP / M;
	//printf("\n%d \n", M);
}
/*
static void epoch_lm(nnet_object obj, int tsize, double *data, double *target, int *index, double *output) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	double mse, gmse, temp;

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
		//backpropagate(obj, output, target + itrt, lenoup,delta);
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

	obj->mse = gmse / (lenoup * tsize);
}

*/
void train_lm(nnet_object netobj,ndata_object dataobj, double *out) {
	lm_object obj;

	obj = lm_init(netobj, dataobj);
	
	obj->data->tsize = (int)(obj->net->tratio * obj->data->P);
	obj->data->gsize = (int)(obj->net->gratio * obj->data->P);
	obj->data->vsize = obj->data->P - obj->data->tsize - obj->data->gsize;
	
	lm_free(obj);
}

void train(nnet_object obj, int tsize, double *data, double *target) {

	double *pn, *tn;
	int leninp, lenoup;

	obj->datasize = tsize;

	if (obj->normmethod == 0) {
		train_null(obj, tsize, data, target);
	}
	else if (obj->normmethod == 1) {
		pn = (double*)malloc(sizeof(double)* tsize * obj->arch[0]);
		tn = (double*)malloc(sizeof(double)* tsize * obj->arch[obj->lm1]);
		leninp = obj->arch[0];
		lenoup = obj->arch[obj->lm1];

		premnmx(tsize, data, leninp, target, lenoup, pn, tn, obj->inpnmin, obj->inpnmax, obj->oupnmin, obj->oupnmax, obj->dmin, obj->dmax, obj->tmin, obj->tmax);
		train_null(obj, tsize, pn, tn);

		free(pn);
		free(tn);
	}
	else if (obj->normmethod == 2) {
		pn = (double*)malloc(sizeof(double)* tsize * obj->arch[0]);
		tn = (double*)malloc(sizeof(double)* tsize * obj->arch[obj->lm1]);
		leninp = obj->arch[0];
		lenoup = obj->arch[obj->lm1];

		prestd(tsize, data, leninp, target, lenoup, pn, tn, obj->inpnmean, obj->inpnstd, obj->oupnmean, obj->oupnstd, obj->dmean, obj->dstd, obj->tmean, obj->tstd);
		train_null(obj, tsize, pn, tn);
		//train_mstd(obj, tsize, data, target);

		free(pn);
		free(tn);
	}
	else {
		printf("\n Available Normalization Methods : 0 - NULL,1 - Minmax ,2 - Mean/Std method\n");
		exit(1);
	}
}

void train_mnmx(nnet_object obj, int size, double *inp, double *out) {

	obj->normmethod = 1;

	train(obj, size, inp, out);
}

void train_mstd(nnet_object obj, int size, double *inp, double *out) {

	obj->normmethod = 2;
	train(obj, size, inp, out);

}

void sim(nnet_object obj, int size, double *data, double *output) {
	int leninp, lenoup,i;
	double *pn,*tn, *tempi,*tempo;

	pn = (double*)malloc(sizeof(double)* size * obj->arch[0]);
	tn = (double*)malloc(sizeof(double)* size * obj->arch[obj->lm1]);
	tempi = (double*)malloc(sizeof(double)* obj->nmax);
	tempo = (double*)malloc(sizeof(double)* obj->nmax);

	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];

	if (obj->normmethod == 0) {

		for (i = 0; i < size; ++i) {
			feedforward(obj, data + i*leninp, leninp, lenoup, output + i *lenoup,tempi,tempo);
		}

	} else if (obj->normmethod == 1) {
		for (i = 0; i < lenoup*size; ++i) {
			tn[i] = 0.0;
		}

		applymnmx(obj, size, data, leninp, pn);

		for (i = 0; i < size; ++i) {
			feedforward(obj, pn + i*leninp, leninp, lenoup, tn + i *lenoup,tempi,tempo);
		}

		postmnmx(obj, size, tn, lenoup, output);
	}
	else if (obj->normmethod == 2) {
		for (i = 0; i < lenoup*size; ++i) {
			tn[i] = 0.0;
		}

		applystd(obj, size, data, leninp, pn);
		
		for (i = 0; i < size; ++i) {
			feedforward(obj, pn + i*leninp, leninp, lenoup, tn + i *lenoup,tempi,tempo);
		}
		

		poststd(obj, size, tn, lenoup, output);
	}
	else {
		printf("\n Available Normalization Methods : 0 - NULL,1 - Minmax ,2 - Mean/Std method\n");
		exit(1);
	}

	free(tempi);
	free(tempo);
	free(pn);
	free(tn);
}

void nnet_free(nnet_object obj) {
	free(obj);
}

void lm_free(lm_object obj) {
	free(obj);
}