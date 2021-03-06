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
	else if (layers > 5) {
		printf("\nThis Neural Network cannot have more than 5 layers (Input+Output+ 100 Hidden Layers\n");
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

	obj = (nnet_object)malloc(sizeof(struct nnet_set) + sizeof(float)* (t_vector + add_vector) );

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
	obj->verbose = 10; // Display update every "verbose" iteration. Set it to 0 to suppress any outputs.

	obj->generalize = 0;// Generalize and Validation initialized to 0.
	obj->validate = 0; //  They will automatically change to 1 if a generalization and/or
	// a validation set is used. see void set_training_ratios(nnet_object obj, float tratio, float gratio, float vratio);

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
	obj->imse = FLT_MAX;

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

void set_learning_rate(nnet_object obj,float eta) {
	if (eta > 0.0f && eta < 1.0f) {
		obj->eta = eta;
	}
	else {
		printf("Learning Rate only takes values between 0.0 and 1.0\n");
		exit(1);
	}
}

void set_momentum(nnet_object obj, float alpha) {
	if (alpha >= 0.0f && alpha < 1.0f) {
		obj->alpha = alpha;
	}
	else {
		printf("Momentum only takes values between 0.0 and 1.0\n");
		exit(1);
	}
}

void set_target_mse(nnet_object obj, float tmse) {
	if (tmse < 0.0f) {
		printf("MSE only takes values over 0.0\n");
		exit(1);
	}
	else {
		obj->tmse = tmse;
	}
}

void set_generalization_mse(nnet_object obj, float gmse) {
	if (gmse < 0.0f) {
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

void set_verbose(nnet_object obj, int verb) {
	obj->verbose = verb;
}

void set_training_ratios(nnet_object obj, float tratio, float gratio, float vratio) {

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

void set_mnmx(nnet_object obj, float inpmin, float inpmax, float oupmin, float oupmax) {
	if (obj->normmethod != 1) {
		obj->normmethod = 1;
	}
	obj->inpnmin = inpmin;
	obj->inpnmax = inpmax;
	obj->oupnmin = oupmin;
	obj->oupnmax = oupmax;
}

void set_mstd(nnet_object obj, float inpmean, float inpstd, float oupmean, float oupstd) {
	if (obj->normmethod != 2) {
		obj->normmethod = 2;
	}
	obj->inpnmean = inpmean;
	obj->inpnstd = inpstd;
	obj->oupnmean = oupmean;
	obj->oupnstd = oupstd;
}

static float mean_stride(float* vec, int N, int stride) {
	int i;
	float m;
	m = 0.0;

	for (i = 0; i < N; ++i) {
		m += vec[i*stride];
	}
	m = m / N;
	return m;
}

static float std_stride(float* vec, int N, int stride) {
	float v, temp, m;
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

static float dmax_stride(float* x, int N, int stride) {
	int i;
	float m;

	m = -FLT_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i*stride] > m) {
			m = x[i*stride];
		}
	}

	return m;
}

static float dmin_stride(float* x, int N, int stride) {
	int i;
	float m;

	m = FLT_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i*stride] < m) {
			m = x[i*stride];
		}
	}

	return m;
}

static void norm_nw(int S, float *weight,float beta) {
	int i;
	float temp;

	temp = 0.0f;

	for (i = 0; i < S; ++i) {
		temp += (weight[i] * weight[i]);
	}

	temp = sqrt(temp);

	for (i = 0; i < S; ++i) {
		weight[i] *= beta / temp;
	}
}

static float dsign(float val) {
	float sign;

	if (val >= 0.0f) {
		sign = 1.0f;
	}
	else {
		sign = -1.0f;
	}

	return sign;
}

static void calc_nw_hidden(int N, int S, float *weight) {
	// N - Number of Inputs
	// S - Number of Neurons in Hidden layer
	// Total Number of Weights - N*S + S biases
	float beta;
	int j, k, N1, itr;
	float spc;

	beta = 0.7f * pow((float)S, 1.0f / (float)N);
	N1 = N + 1;
	srand((unsigned int) time(NULL));
	//srand(100);

	if (S == 1) {
		spc = 0.0f;
	}
	else {
		spc = 2.0f / (S - 1);
	}

	for (j = 0; j < S; ++j) {
		itr = j * N1;
		weight[itr] = -1.0 + j*spc;
		for (k = 1; k < N1; ++k) {
			weight[itr + k] = ((((float)(rand() % 100) + 1) / 100 * 2) - 1) * beta;
		}
	}

	if (S == 1) {
		weight[0] = 0.0f;
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
	float nrm;

	lm1 = obj->lm1;
	itr3 = 0;
	srand((unsigned int) time(NULL));
	//srand(100);
	for (i = 0; i < lm1; ++i) {
		nrm = 1.0f / sqrt((float)obj->arch[i]);
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			//obj->bias[itr2 + j] = (((float)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
			//printf("\n %d bias %f", itr2 + j, obj->bias[itr2 + j]);
			itr = j * N;
			for (k = 0; k < N; ++k) {
				obj->weight[itr3 + itr + k] = (((float)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
				//obj->weight[itr3 + itr + k] = (((float)(rand() % 100) + 1) / 100) * (0.001 - 0.0001) + 0.0001;
			}
		}
		itr3 += S * N;
	}

}

void initweights_seed(nnet_object obj, int seed) {
	int i, lm1, j, k, S, N, itr, itr3;
	float nrm;

	lm1 = obj->lm1;
	itr3 = 0;
	srand(seed);
	//srand(100);
	for (i = 0; i < lm1; ++i) {
		nrm = 1.0 / sqrt((float)obj->arch[i]);
		N = obj->arch[i] + 1;
		S = obj->arch[i + 1];
		for (j = 0; j < S; ++j) {
			//obj->bias[itr2 + j] = (((float)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
			//printf("\n %d bias %f", itr2 + j, obj->bias[itr2 + j]);
			itr = j * N;
			for (k = 0; k < N; ++k) {
				obj->weight[itr3 + itr + k] = (((float)(rand() % 100) + 1) / 100 * 2 * nrm) - nrm;
				//obj->weight[itr3 + itr + k] = (((float)(rand() % 100) + 1) / 100) * (0.001 - 0.0001) + 0.0001;
			}
		}
		itr3 += S * N;
	}

}

static float gvmse(nnet_object obj, int tsize, float *data, float *target, int *index, float *output, float *tempi,float *tempo) {
	float gmse, temp;
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

void feedforward(nnet_object obj, float *inp, int leninp, int lenoup, float *oup,float *tempi, float *tempo) {
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



void backpropagate(nnet_object obj, float *output, float *desired, int lenoup,float *delta,float *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2,itr4;
	int S, itr3, index, in0;
	float temp,lr,mc;

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
			//printf("Wcv %f ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			//printf("%f %f \n", desired[i], output[i]);
	
		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			//printf("%f %f \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
		}
	}
	//printf("Wcv %f ", obj->mse);

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
				//printf("temp %f %f ", temp, obj->tout[j]);
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
	//printf("WT %f \n", obj->weight[0]);


}

static void backpropagate_gd(nnet_object obj, float *output, float *desired, int lenoup, float *delta, float *tdelta, float *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4, itr5, itr6, N2;
	int S, itr3, index, in0;
	float temp, lr, mc;

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
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
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

	N = obj->arch[lm1 - 1] + 1;
	S = obj->arch[lm1];
	itr3 = lw - N * S;
	itr2 = ld - S;
	itr4 = ld + in0 - S - obj->arch[lm1 - 1];
	itr5 = ld - loup;

	for (i = lm1 - 1; i >= 0; --i) {
		for (j = 0; j < S; ++j) {
			itr = j * N;// iterates over one row of weights
			index = itr3 + itr;
			tdelta[index] = obj->gradient[itr2 + j];
			//printf(" ind %d", itr2+j);

			for (k = 1; k < N; ++k) {
				index = itr3 + itr + k;
				tdelta[index] = tinp[itr4 + k - 1] * obj->gradient[itr2 + j];
				//printf(" ind %d", itr + k - 1);
			}

		}
		N = obj->arch[i - 1] + 1;
		S = obj->arch[i];
		itr3 -= S * N;// iterates over all the weights going into a layer
		itr2 -= S;// iterates over each output layer
		itr4 -= (N - 1);// iterates over each input layer
		//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		if (i > 0) {
			if (obj->actfcn[i] == 1) {
				N2 = obj->arch[i];
				jinit = itr5 - N2;
				kfin = obj->arch[i + 1];
				lw = lw - obj->lweight[i + 1];
				itr6 = 1;
				for (j = jinit; j < itr5; ++j) {
					temp = 0.0;
					for (k = 0; k < kfin; ++k) {
						temp += obj->gradient[itr5 + k] * obj->weight[lw + itr6];// add weights
						//printf("W %d ", lw + itr2);
						itr6 += (N2 + 1);
					}
					itr6 = j - jinit + 2;
					obj->gradient[j] = temp;
				}
				itr5 -= N2;
			}
			else if (obj->actfcn[i] == 2) {
				N2 = obj->arch[i];
				jinit = itr5 - N2;
				kfin = obj->arch[i + 1];
				lw = lw - obj->lweight[i + 1];
				itr6 = 1;
				for (j = jinit; j < itr5; ++j) {
					temp = 0.0;
					for (k = 0; k < kfin; ++k) {
						temp += obj->gradient[itr5 + k] * obj->weight[lw + itr6];// add weights
						//printf("W %d ", lw + itr2);
						itr6 += (N2 + 1);
					}
					itr6 = j - jinit + 2;
					obj->gradient[j] = temp * obj->tout[j] * (1.0 - obj->tout[j]);
					//printf("temp %f %f ", temp, obj->tout[j]);
				}
				itr5 -= N2;
			}
			else if (obj->actfcn[i] == 3) {
				N2 = obj->arch[i];
				jinit = itr5 - N2;
				kfin = obj->arch[i + 1];
				lw = lw - obj->lweight[i + 1];
				itr6 = 1;
				for (j = jinit; j < itr5; ++j) {
					temp = 0.0;
					for (k = 0; k < kfin; ++k) {
						temp += obj->gradient[itr5 + k] * obj->weight[lw + itr6];// add weights
						//printf("W %d ", lw + itr2);
						itr6 += (N2 + 1);
					}
					itr6 = j - jinit + 2;
					obj->gradient[j] = temp * (1.0 + obj->tout[j]) * (1.0 - obj->tout[j]);
				}
				itr5 -= N2;
			}
		}
	}
	itr3 = 0;
	itr2 = itr4 = 0;

	if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingda")) {
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
	else if (!strcmp(obj->trainfcn, "traingdm") || !strcmp(obj->trainfcn, "traingdx")) {
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

static void backpropagate_alr(nnet_object obj, float *output, float *desired, int lenoup,float *delta,float *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	float temp, lr, mc;

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
			//printf("Wcv %f ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			//printf("%f %f \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			//printf("%f %f \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
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
				//printf("temp %f %f ", temp, obj->tout[j]);
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

static void backpropagate_mb_1(nnet_object obj, float *output, float *desired, int lenoup, float *delta,float *tdelta,float *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	float temp, lr, mc;

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
			//printf("Wcv %f ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			//printf("%f %f \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
			//printf("%f %f \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
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
				//printf("temp %f %f ", temp, obj->tout[j]);
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


	//printf("WT %f \n", obj->weight[0]);


}

static void backpropagate_mb(nnet_object obj, float *output, float *desired, int lenoup, float *delta, float *tdelta, float *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4,itr5,itr6,N2;
	int S, itr3, index, in0;
	float temp, lr, mc;

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
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * obj->tout[itr + i] * (1.0 - obj->tout[itr + i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * (1.0 + obj->tout[itr + i]) * (1.0 - obj->tout[itr + i]);
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

	N = obj->arch[lm1-1] + 1;
	S = obj->arch[lm1];
	itr3 = lw - N * S;
	itr2 = ld - S; 
	itr4 = ld + in0 - S - obj->arch[lm1 - 1];
	itr5 = ld - loup;

	for (i = lm1 - 1; i >= 0; --i) {
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
		N = obj->arch[i - 1] + 1;
		S = obj->arch[i];
		itr3 -= S * N;// iterates over all the weights going into a layer
		itr2 -= S;// iterates over each output layer
		itr4 -= (N - 1);// iterates over each input layer
		//printf("\n itr %d itr2 %d itr3 %d \n", itr, itr2, itr3);
		if (i > 0) {
			if (obj->actfcn[i] == 1) {
				N2 = obj->arch[i];
				jinit = itr5 - N2;
				kfin = obj->arch[i + 1];
				lw = lw - obj->lweight[i + 1];
				itr6 = 1;
				for (j = jinit; j < itr5; ++j) {
					temp = 0.0;
					for (k = 0; k < kfin; ++k) {
						temp += obj->gradient[itr5 + k] * obj->weight[lw + itr6];// add weights
						//printf("W %d ", lw + itr2);
						itr6 += (N2 + 1);
					}
					itr6 = j - jinit + 2;
					obj->gradient[j] = temp;
				}
				itr5 -= N2;
			}
			else if (obj->actfcn[i] == 2) {
				N2 = obj->arch[i];
				jinit = itr5 - N2;
				kfin = obj->arch[i + 1];
				lw = lw - obj->lweight[i + 1];
				itr6 = 1;
				for (j = jinit; j < itr5; ++j) {
					temp = 0.0;
					for (k = 0; k < kfin; ++k) {
						temp += obj->gradient[itr5 + k] * obj->weight[lw + itr6];// add weights
						//printf("W %d ", lw + itr2);
						itr6 += (N2 + 1);
					}
					itr6 = j - jinit + 2;
					obj->gradient[j] = temp * obj->tout[j] * (1.0 - obj->tout[j]);
					//printf("temp %f %f ", temp, obj->tout[j]);
				}
				itr5 -= N2;
			}
			else if (obj->actfcn[i] == 3) {
				N2 = obj->arch[i];
				jinit = itr5 - N2;
				kfin = obj->arch[i + 1];
				lw = lw - obj->lweight[i + 1];
				itr6 = 1;
				for (j = jinit; j < itr5; ++j) {
					temp = 0.0;
					for (k = 0; k < kfin; ++k) {
						temp += obj->gradient[itr5 + k] * obj->weight[lw + itr6];// add weights
						//printf("W %d ", lw + itr2);
						itr6 += (N2 + 1);
					}
					itr6 = j - jinit + 2;
					obj->gradient[j] = temp * (1.0 + obj->tout[j]) * (1.0 - obj->tout[j]);
				}
				itr5 -= N2;
			}
		}
	}

}

static void backpropagate_mb_2(nnet_object obj, float *delta, float *tdelta) {
	int itr, itr2, itr3, itr4;
	int lm1,i,N,S,j,k,index;

	float lr, mc,temp;
	lr = obj->eta;
	mc = obj->alpha;

	lm1 = obj->lm1;
	itr3 = 0;
	itr2 = itr4 = 0;

	if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingda")) {
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
	else if (!strcmp(obj->trainfcn, "traingdm") || !strcmp(obj->trainfcn, "traingdx")) {
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

static void backpropagate_mb_3(nnet_object obj, float *delta, float *tdelta) {
	int itr, itr2, itr3, itr4;
	int lm1, i, N, S, j, k, index;

	float lr, mc, temp;
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

void backpropagate_qp_1(nnet_object obj, float *output, float *desired, int lenoup, float *delta,float *slope,float *tslope,float *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	float temp, lr, mc,del,epsilon,dslope,temp2;

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
			//printf("Wcv %f ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			//printf("%f %f \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * logsig_der(obj->tout[itr + i]);
			//printf("%f %f \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * tansig_der(obj->tout[itr + i]);
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
				//printf("temp %f %f ", temp, obj->tout[j]);
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

void backpropagate_qp_2(nnet_object obj, float *delta, float *slope, float *tslope) {
	int i;
	float del, dslope, epsilon;
	float th_p, th_n, shrink_factor, max_factor,decay;

	//epsilon = 0.55 / (float)obj->lw;
	epsilon = obj->eta / obj->datasize;
	th_p = obj->qp_threshold * obj->qp_threshold;
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
		//printf("%f ", obj->weight[i]);
	}
}

void backpropagate_rp_1(nnet_object obj, float *output, float *desired, int lenoup, float *delta, float *slope, float *tslope, float *tinp) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4;
	int S, itr3, index, in0;
	float temp, lr, mc, del, epsilon, dslope;

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
			//printf("Wcv %f ", output[i]);
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp;
			//printf("%f %f \n", desired[i], output[i]);

		}
	}
	else if (obj->actfcn[lm1] == 2) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * logsig_der(obj->tout[itr + i]);
			//printf("%f %f \n", desired[i], output[i]);
		}
	}
	else if (obj->actfcn[lm1] == 3) {
		for (i = 0; i < loup; ++i) {
			temp = (desired[i] - output[i]);
			obj->gradient[itr + i] = temp * tansig_der(obj->tout[itr + i]);
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
				//printf("temp %f %f ", temp, obj->tout[j]);
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

void backpropagate_rp_2(nnet_object obj, float *delta, float *slope, float *tslope, float *updatevalue) {
	int i;
	float value, max_step,delta_min,ndelta,del;
	float rp_eta_n, rp_eta_p;

	max_step = obj->rp_max_step;
	delta_min = obj->rp_delta_min;
	rp_eta_n = obj->rp_eta_n;
	rp_eta_p = obj->rp_eta_p;
	del = 0.0;
	
	for (i = 0; i < obj->lw; ++i) {
		value = NNET_SIGN(slope[i] * tslope[i]);
		if (value > 0) {
			ndelta = updatevalue[i] * rp_eta_p;
			ndelta = NNET_MIN(ndelta, max_step);
			del = NNET_SIGN(slope[i]) * ndelta;
			updatevalue[i] = ndelta;
			tslope[i] = slope[i];
		}
		else if (value < 0) {
			ndelta = updatevalue[i] * rp_eta_n;
			ndelta = NNET_MAX(ndelta, delta_min);
			updatevalue[i] = ndelta;
			tslope[i] = 0.0;
			del = -delta[i];
		}
		else {
			del = NNET_SIGN(slope[i]) * updatevalue[i];
			tslope[i] = slope[i];
		}

		obj->weight[i] += del;
		delta[i] = del;
		slope[i] = 0.0;
	}

}

void backpropagate_irp_2(nnet_object obj, float *slope, float *tslope, float *updatevalue) {
	int i;
	float value, max_step, delta_min, ndelta,del;
	float rp_eta_n,rp_eta_p;

	max_step = obj->rp_max_step;
	delta_min = obj->rp_delta_min;
	rp_eta_n = obj->rp_eta_n;
	rp_eta_p = obj->rp_eta_p;
	del = 0.0;

	for (i = 0; i < obj->lw; ++i) {
		value = NNET_SIGN(slope[i] * tslope[i]);
		if (value >= 0) {
			ndelta = updatevalue[i] * rp_eta_p;
			ndelta = NNET_MIN(ndelta, max_step);
			updatevalue[i] = ndelta;
			tslope[i] = slope[i];
		}
		else if (value < 0) {
			ndelta = updatevalue[i] * rp_eta_n;
			ndelta = NNET_MAX(ndelta, delta_min);
			updatevalue[i] = ndelta;
			tslope[i] = 0.0;
			
		}

		del = NNET_SIGN(slope[i]) * ndelta;
		obj->weight[i] += del;
		slope[i] = 0.0;
	}

}

void backpropagate_rqp_1(nnet_object obj, float *output, float *desired, int lenoup, float *slope, float *tinp,float *gradient2) {
	int lm1, i, lw, ld, loup, itr, jinit, j, k, kfin, N, itr2, itr4,itr5;
	int S, itr3, index, in0, linp;
	float temp, lr, mc, del, epsilon, dslope, temp2;

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
		//printf("Wcv %f ", output[i]);
		temp = (desired[i] - output[i]);
		gradient2[itr + i] = temp;
		obj->mse += (temp*temp);
		//printf("%f %f \n", desired[i], output[i]);

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

}
void mapminmax(float *x, int N, float ymin, float ymax, float *y) {
	float xmin, xmax, t;
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

void mapstd(float *x, int N, float ymean, float ystd, float *y) {
	float xmean, xstd, t;
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

static void mapminmax_stride(float *x, int N, int stride,float ymin, float ymax, float *y) {
	float xmin, xmax, t;
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

static void mapminmax_stride_apply(float *x, int N, int stride, float ymin, float ymax, float xmin, float xmax, float *y) {
	float t;
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

static void mapstd_stride(float *x, int N, int stride,float ymean, float ystd, float *y) {
	float xmean, xstd, t;
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

static void mapstd_stride_apply(float *x, int N, int stride, float ymean, float ystd, float xmean, float xstd, float *y) {
	float t;
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

void shuffle(int N, int *index) {
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

static void shuffleinput(int N, int leninp, float *input, float *shuffled, int lenoup, float *output, float *target) {
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

void premnmx(int size,float *p, int leninp, float *t, int lenoup, float *pn, float *tn, float ymin, float ymax, float omin, float omax, float *pmin, float *pmax, float *tmin, float *tmax) {
	// pmax and pmin have the length leninp each
	// tmax and tmin have the length lenoup each
	// pn is of size leninp*size
	// tn is of size lenoup*size

	int i;
	float temp;

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

void applymnmx(nnet_object obj,int size, float *p,int leninp, float *pn) {
	int i;
	float temp1,temp2;

	for (i = 0; i < leninp; ++i) {
		temp1 = obj->dmin[i];
		temp2 = obj->dmax[i];
		mapminmax_stride_apply(p+i, size, leninp, obj->inpnmin,obj->inpnmax,temp1, temp2, pn+i); 
	}
}

void applystd(nnet_object obj, int size, float *p, int leninp, float *pn) {
	int i;
	float temp1, temp2;

	for (i = 0; i < leninp; ++i) {
		temp1 = obj->dmean[i];
		temp2 = obj->dstd[i];
		mapstd_stride_apply(p + i, size, leninp, obj->inpnmean, obj->inpnstd, temp1, temp2, pn + i);
	}
}

void prestd(int size, float *p, int leninp, float *t, int lenoup, float *pn, float *tn,float ymean,float ystd,float omean,float ostd, float *dmean, float *dstd, float *tmean, float *tstd) {
	// dmean and dstd have the length leninp each
	// tmean and tstd have the length lenoup each
	// pn is of size leninp*size
	// tn is of size lenoup*size

	int i;
	float temp;

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

void postmnmx(nnet_object obj,int size,float *oupn,int lenoup,float *oup) {
	int i;
	float temp1, temp2;

	for (i = 0; i < lenoup; ++i) {
		temp1 = obj->tmin[i];
		temp2 = obj->tmax[i];
		mapminmax_stride_apply(oupn + i, size, lenoup, temp1, temp2, obj->oupnmin, obj->oupnmax, oup + i);
	}
}

void poststd(nnet_object obj, int size, float *oupn, int lenoup, float *oup) {
	int i;
	float temp1, temp2;

	for (i = 0; i < lenoup; ++i) {
		temp1 = obj->tmean[i];
		temp2 = obj->tstd[i];
		mapstd_stride_apply(oupn + i, size, lenoup, temp1, temp2,obj->oupnmean,obj->oupnstd, oup + i);
	}
}

static void epoch_gdm_alr2(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta,float *output,float *tinp,float *tempi,float *tempo) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;

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
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		//printf("output %f \n", data[i]);
		backpropagate_alr(obj, output, target + itrt, lenoup,delta,tinp);

	}

	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = i * leninp;
		itrt = i * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (lenoup * tsize);
}

static void epoch_gdm(nnet_object obj, int tsize, float *data, float *target,int *index,float *delta,float *output,float *tinp,float *tempi,float *tempo) {
	int lendata, lentarget, i,j,itrd,itrt,leninp,lenoup;
	float mse,gmse,temp;

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
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate(obj, output, target + itrt, lenoup,delta,tinp);
	}

	//printf("GMSE %f \n", obj->mse);
	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = i * leninp;
		itrt = i * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (lenoup * tsize);
}

static void epoch_qp(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta, float *slope, float *tslope,float *output,float *tinp,float *tempi,float *tempo,
	float *gradient2) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;

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
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		//backpropagate_qp_1(obj, output, target + itrt, lenoup, delta,slope,tslope,tinp);
		//backpropagate_rqp_1(obj, output, target + itrt, lenoup, slope, tinp,gradient2);
		backpropagate_mb(obj, output, target + itrt, lenoup, delta, slope, tinp);
	}
	//printf("\n");

	backpropagate_qp_2(obj, delta, slope, tslope);

	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = i * leninp;
		itrt = i * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (lenoup * tsize);

}

static void epoch_rp(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta, float *slope, float *tslope, float *output, float *tinp, float *tempi, float *tempo,float *updatevalue) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;

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
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate_rp_1(obj, output, target + itrt, lenoup, delta, slope, tslope, tinp);
		//backpropagate_mb(obj, output, target + itrt, lenoup, delta, slope, tinp);
	}
	//printf("\n");

	backpropagate_rp_2(obj, delta, slope, tslope,updatevalue);

	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = i * leninp;
		itrt = i * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (lenoup * tsize);

}

static void epoch_irp(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta, float *slope, float *tslope, float *output, float *tinp, float *tempi, float *tempo, float *updatevalue) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;

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
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate_rp_1(obj, output, target + itrt, lenoup, delta, slope, tslope, tinp);
		//backpropagate_mb(obj, output, target + itrt, lenoup, delta, slope, tinp);
	}
	//printf("\n");

	backpropagate_irp_2(obj, slope, tslope, updatevalue);

	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = i * leninp;
		itrt = i * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->imse = obj->mse;

	obj->mse = gmse / (lenoup * tsize);

}

static void epoch_mb(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta, float *tdelta, float *output,float *tinp,float *tempi,float *tempo) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;
	int batchsize,mbsize;

	batchsize = obj->batchsize;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	mbsize = 0;

	shuffle(tsize, index);

	for (i = 0; i < tsize; ++i) {
		itrd = index[i] * leninp;
		itrt = index[i] * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate_mb(obj, output, target + itrt, lenoup, delta,tdelta,tinp);
		mbsize++;
		if (mbsize == batchsize || i == tsize-1) {
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] /= mbsize;
			}
			//printf("%d \n", mbsize);
			backpropagate_mb_2(obj, delta, tdelta);
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] = 0.0;
			}
			mbsize = 0;
		}
	}
	gmse = 0.0;

	for (i = 0; i < tsize; ++i) {
		itrd = i * leninp;
		itrt = i * lenoup;
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	obj->mse = gmse / (lenoup * tsize);
}

static void epoch_mbp(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta, float *tdelta, float *output,float *tinp,float *tempi, float *tempo) {
	int lendata, lentarget, i, j, k, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;
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

	iters = (int) ceil((double) tsize /(double) batchsize);

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
			backpropagate_mb(obj, output, target + itrt, lenoup, delta, tdelta,tinp);
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

static void epoch_mbp_alr2(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta, float *tdelta, float *output, float *tinp, float *tempi, float *tempo) {
	int lendata, lentarget, i, j, k, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;
	int batchsize, iters, maxsize, litr;

	batchsize = obj->batchsize;

	lendata = obj->arch[0] * tsize;
	lentarget = obj->arch[obj->lm1] * tsize;
	itrt = itrd = 0;
	leninp = obj->arch[0];
	lenoup = obj->arch[obj->lm1];
	mse = 0.0;
	litr = 0;

	shuffle(tsize, index);

	iters = (int) ceil((double)tsize / (double)batchsize);

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
			feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
			backpropagate_mb(obj, output, target + itrt, lenoup, delta, tdelta, tinp);
		}
		//#pragma omp barrier

		if (litr == 1) {
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] /= ((k + 1)*batchsize - tsize);
			}
			litr = 0;
		}
		else {
			for (j = 0; j < obj->lw; ++j) {
				tdelta[j] /= batchsize;
			}
		}
		backpropagate_mb_3(obj, delta, tdelta);
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

static void epoch_gd(nnet_object obj, int tsize, float *data, float *target, int *index, float *delta,float *tdelta, float *output, float *tinp, float *tempi, float *tempo) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;

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
		feedforward(obj, data + itrd, leninp, lenoup, output, tempi, tempo);
		backpropagate_gd(obj, output, target + itrt, lenoup, delta,tdelta, tinp);
	}

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

void train_null(nnet_object obj, int size, float *inp, float *out) {
	int epoch,i;
	int tsize, gsize, vsize,vitr;
	int itrd,itrt,leninp,lenoup;
	float mse,gmse,vmse,omse,mcval;
	float mpe, lr_inc, lr_dec,eta_tmp;
	float *output,*data,*target;
	float *tweight,*delta,*tdelta,*slope,*tslope,*tinp,*gradient2;
	float *tempi, *tempo;
	int *index, *indexg,*indexv;
	int gen, val;

	gen = val = 0;

	tsize = (int) (obj->tratio * size); // training size
	gsize = 0;
	vsize = 0;
	if (tsize >= size) {
		tsize = size;
	}
	else {
		gsize = (int)(obj->gratio * size); // generalization size
		if (tsize + gsize >= size) {
			gsize = size - tsize;
		}
		else {
			vsize = size - tsize - gsize; // validation size
		}
	}

	output = (float*)malloc(sizeof(float)* obj->arch[obj->lm1]);
	index = (int*)malloc(sizeof(int)*tsize);
	indexg = (int*)malloc(sizeof(int)*gsize);
	indexv = (int*)malloc(sizeof(int)*vsize);

	data = (float*)malloc(sizeof(float)* size * obj->arch[0]);
	target = (float*)malloc(sizeof(float)* size * obj->arch[obj->lm1]);
	tweight = (float*)malloc(sizeof(float)*obj->lw);
	tinp = (float*)malloc(sizeof(float)* (obj->ld + obj->arch[0]));

	tempi = (float*)malloc(sizeof(float)* obj->nmax);
	tempo = (float*)malloc(sizeof(float)* obj->nmax);

	gradient2 = (float*)malloc(sizeof(float)* (obj->ld + obj->arch[0]));

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
		delta = (float*)malloc(sizeof(float)*obj->lw);
		for (i = 0; i < obj->lw; ++i) {
			delta[i] = 0;
		}
	}
	else {
		delta = (float*)malloc(sizeof(float)* 1);
		delta[0] = 0;
	}

	if (!strcmp(obj->trainfcn, "trainqp") || !strcmp(obj->trainfcn, "trainrp") || !strcmp(obj->trainfcn, "trainirp")) {
		slope = (float*)malloc(sizeof(float)*obj->lw);
		tslope = (float*)malloc(sizeof(float)*obj->lw);
		for (i = 0; i < obj->lw; ++i) {
			slope[i] = tslope[i] = 0;
		}
	}
	else {
		slope = (float*)malloc(sizeof(float)* 1);
		tslope = (float*)malloc(sizeof(float)* 1);
		slope[0] = tslope[0] = 0;
	}

	for (i = 0; i < obj->lw; ++i) {
		tweight[i] = obj->weight[i];
	}
	vitr = 0;
	if (!strcmp(obj->trainfcn, "traingd") || !strcmp(obj->trainfcn, "traingdm")) {
		if (!strcmp(obj->trainmethod, "online")) {
			tdelta = (float*)malloc(sizeof(float)*obj->lw);
			for (i = 0; i < obj->lw; ++i) {
				tdelta[i] = 0.0;
			}
			epoch_gd(obj, tsize, data, target, index, delta,tdelta, output, tinp, tempi, tempo);
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			//printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, obj->tmse);
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gd(obj, tsize, data, target, index, delta,tdelta, output, tinp, tempi, tempo);
				mse = obj->mse;
				vitr++;
				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
						vitr = 0;
					}
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
						break;
					}
				}
				else {
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f \n", epoch, mse);
						vitr = 0;
					}
				}

				epoch++;
			}
		}
		else if (!strcmp(obj->trainmethod, "batch")) {
			tdelta = (float*)malloc(sizeof(float)*obj->lw);
			for (i = 0; i < obj->lw; ++i) {
				tdelta[i] = 0.0;
			}
			epoch_mb(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_mb(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
				mse = obj->mse;
				vitr++;
				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
						vitr = 0;
					}
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
						break;
					}
				}
				else {
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f \n", epoch, mse);
						vitr = 0;
					}
				}

				epoch++;
			}
		}
	}

	if (!strcmp(obj->trainfcn, "traingda")) {
		if (!strcmp(obj->trainmethod, "online")) {
			tdelta = (float*)malloc(sizeof(float)*obj->lw);
			for (i = 0; i < obj->lw; ++i) {
				tdelta[i] = 0.0;
			}
			epoch_gd(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			//printf("EPOCH %d MSE %f omse %f eta %f \n", epoch, mse, omse, obj->eta);
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gd(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				vitr++;
				if (mse > mpe*omse) {
					eta_tmp = obj->eta * lr_dec;
					obj->eta = pmax(eta_tmp,(float)ETA_MIN);
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
					}
					mse = omse;
				}
				else {
					if (mse < omse) {
						eta_tmp = obj->eta * lr_inc;
						obj->eta = pmin(eta_tmp, (float)ETA_MAX);
					}
					omse = mse;
				}

				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
						vitr = 0;
					}
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
						break;
					}
				}
				else {
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f \n", epoch, mse);
						vitr = 0;
					}
				}

				epoch++;
			}
		}
		else if (!strcmp(obj->trainmethod, "batch")) {
			tdelta = (float*)malloc(sizeof(float) * obj->lw);
			for (i = 0; i < obj->lw; ++i) {
				tdelta[i] = 0.0;
			}
			epoch_mb(obj, tsize, data, target, index, delta,tdelta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			//printf("EPOCH %d MSE %f omse %f eta %f \n", epoch, mse, omse, obj->eta);
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_mb(obj, tsize, data, target, index, delta,tdelta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				vitr++;
				if (mse > mpe*omse) {
					eta_tmp = obj->eta * lr_dec;
					obj->eta = pmax(eta_tmp, (float)ETA_MIN);
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
					}
					mse = omse;
				}
				else {
					if (mse < omse) {
						eta_tmp = obj->eta * lr_inc;
						obj->eta = pmin(eta_tmp, (float)ETA_MAX);
					}
					omse = mse;
				}

				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
						vitr = 0;
					}
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
						break;
					}
				}
				else {
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f \n", epoch, mse);
						vitr = 0;
					}
				}

				epoch++;
			}
		}
	}

	if (!strcmp(obj->trainfcn, "traingdx")) {
		if (!strcmp(obj->trainmethod, "online")) {
			tdelta = (float*)malloc(sizeof(float) * obj->lw);
			for (i = 0; i < obj->lw; ++i) {
				tdelta[i] = 0.0;
			}
			epoch_gd(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_gd(obj, tsize, data, target, index, delta, tdelta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				vitr++;
				if (mse > mpe*omse) {
					eta_tmp = obj->eta * lr_dec;
					obj->eta = pmax(eta_tmp, (float)ETA_MIN);
					obj->alpha = 0.0;
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
					}
					//mse = omse;
				}
				else {
					if (mse < omse) {
						eta_tmp = obj->eta * lr_inc;
						obj->eta = pmin(eta_tmp, (float)ETA_MAX);
						obj->alpha = mcval;
					}
					omse = mse;
				}

				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
						vitr = 0;
					}
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
						break;
					}
				}
				else {
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f \n", epoch, mse);
						vitr = 0;
					}
				}

				epoch++;
			}
		}
		else if (!strcmp(obj->trainmethod, "batch")) {
			tdelta = (float*)malloc(sizeof(float) * obj->lw);
			for (i = 0; i < obj->lw; ++i) {
				tdelta[i] = 0.0;
			}
			epoch_mb(obj, tsize, data, target, index, delta,tdelta, output, tinp, tempi, tempo);
			for (i = 0; i < obj->lw; ++i) {
				tweight[i] = obj->weight[i];
			}
			mse = obj->mse;
			omse = mse;
			epoch = 1;
			while (mse > obj->tmse && epoch < obj->emax) {
				epoch_mb(obj, tsize, data, target, index, delta,tdelta, output, tinp, tempi, tempo);
				for (i = 0; i < obj->lw; ++i) {
					tweight[i] = obj->weight[i];
				}
				mse = obj->mse;
				vitr++;
				if (mse > mpe*omse) {
					eta_tmp = obj->eta * lr_dec;
					obj->eta = pmax(eta_tmp, (float)ETA_MIN);
					obj->alpha = 0.0;
					for (i = 0; i < obj->lw; ++i) {
						obj->weight[i] = tweight[i];
					}
					//mse = omse;
				}
				else {
					if (mse < omse) {
						eta_tmp = obj->eta * lr_inc;
						obj->eta = pmin(eta_tmp, (float)ETA_MAX);
						obj->alpha = mcval;
					}
					omse = mse;
				}

				if (gen == 1) {
					gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
						vitr = 0;
					}
					if (gmse <= obj->gmse) {
						printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
						break;
					}
				}
				else {
					if (vitr == obj->verbose) {
						printf("EPOCH %d MSE %f \n", epoch, mse);
						vitr = 0;
					}
				}

				epoch++;
			}
		}
	}

	if (!strcmp(obj->trainfcn, "trainqp")) {
		tdelta = (float*)malloc(sizeof(float)* 1);
		epoch_qp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo,gradient2);
		mse = obj->mse;
		omse = mse;
		epoch = 1;
		while (mse > obj->tmse && epoch < obj->emax) {
			epoch_qp(obj, tsize, data, target, index, delta, slope, tslope, output, tinp, tempi, tempo,gradient2);
			mse = obj->mse;
			vitr++;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
				if (vitr == obj->verbose) {
					printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
					vitr = 0;
				}
				if (gmse <= obj->gmse) {
					printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
					break;
				}
			}
			else {
				if (vitr == obj->verbose) {
					printf("EPOCH %d MSE %f \n", epoch, mse);
					vitr = 0;
				}
			}

			epoch++;
		}
	}
	if (!strcmp(obj->trainfcn, "trainrp")) {
		tdelta = (float*)malloc(sizeof(float)*obj->lw);
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
			vitr++;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
				if (vitr == obj->verbose) {
					printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
					vitr = 0;
				}
				if (gmse <= obj->gmse) {
					printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
					break;
				}
			}
			else {
				if (vitr == obj->verbose) {
					printf("EPOCH %d MSE %f \n", epoch, mse);
					vitr = 0;
				}
			}

			epoch++;
		}
	}

	if (!strcmp(obj->trainfcn, "trainirp")) {
		tdelta = (float*)malloc(sizeof(float)*obj->lw);
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
			mse = obj->mse;
			vitr++;
			if (gen == 1) {
				gmse = gvmse(obj, gsize, data + itrd, target + itrt, indexg, output, tempi, tempo);
				if (vitr == obj->verbose) {
					printf("EPOCH %d MSE %f GMSE %f \n", epoch, mse, gmse);
					vitr = 0;
				}
				if (gmse <= obj->gmse) {
					printf("Convergence based on Generalization MSE dropping under %f \n", obj->gmse);
					break;
				}
			}
			else {
				if (vitr == obj->verbose) {
					printf("EPOCH %d MSE %f \n", epoch, mse);
					vitr = 0;
				}
			}

			epoch++;
		}
	}


	// Check for failure
	if (obj->mse != obj->mse) {
		printf("\n Failure. Re-try with different parameter values.  \n");
	}
	else {
		printf("EPOCH %d MSE %f \n", epoch-1, obj->mse);
	}
	// Validate

	itrd += gsize * obj->arch[0];
	itrt += gsize * obj->arch[obj->lm1];

	if (val == 1) {
		vmse = gvmse(obj, vsize, data + itrd, target + itrt, indexv, output, tempi, tempo);

		printf("\n Validation MSE %f \n", vmse);
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

void func_lm(float *x,int MP,int N,void *params) {
	int M, P,i,j;
	nnet_object obj = (nnet_object)params;
	M = obj->arch[obj->lm1];
	P = MP / M;
	//printf("\n%d \n", M);
}
/*
static void epoch_lm(nnet_object obj, int tsize, float *data, float *target, int *index, float *output) {
	int lendata, lentarget, i, j, itrd, itrt, leninp, lenoup;
	float mse, gmse, temp;

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

void train(nnet_object obj, int tsize, float *data, float *target) {

	float *pn, *tn;
	int leninp, lenoup;

	obj->datasize = tsize;

	if (obj->normmethod == 0) {
		train_null(obj, tsize, data, target);
	}
	else if (obj->normmethod == 1) {
		pn = (float*)malloc(sizeof(float)* tsize * obj->arch[0]);
		tn = (float*)malloc(sizeof(float)* tsize * obj->arch[obj->lm1]);
		leninp = obj->arch[0];
		lenoup = obj->arch[obj->lm1];

		premnmx(tsize, data, leninp, target, lenoup, pn, tn, obj->inpnmin, obj->inpnmax, obj->oupnmin, obj->oupnmax, obj->dmin, obj->dmax, obj->tmin, obj->tmax);
		train_null(obj, tsize, pn, tn);

		free(pn);
		free(tn);
	}
	else if (obj->normmethod == 2) {
		pn = (float*)malloc(sizeof(float)* tsize * obj->arch[0]);
		tn = (float*)malloc(sizeof(float)* tsize * obj->arch[obj->lm1]);
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

void train_mnmx(nnet_object obj, int size, float *inp, float *out) {

	obj->normmethod = 1;

	train(obj, size, inp, out);
}

void train_mstd(nnet_object obj, int size, float *inp, float *out) {

	obj->normmethod = 2;
	train(obj, size, inp, out);

}

void sim(nnet_object obj, int size, float *data, float *output) {
	int leninp, lenoup,i;
	float *pn,*tn, *tempi,*tempo;

	pn = (float*)malloc(sizeof(float)* size * obj->arch[0]);
	tn = (float*)malloc(sizeof(float)* size * obj->arch[obj->lm1]);
	tempi = (float*)malloc(sizeof(float)* obj->nmax);
	tempo = (float*)malloc(sizeof(float)* obj->nmax);

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

float nnet_test(nnet_object obj, int tsize, float *data, float *target) {
	float gmse, temp;
	int i, itrt, lenoup, j;
	float  *output;

	lenoup = obj->arch[obj->lm1];
	gmse = 0.0;

	output = (float*)malloc(sizeof(float)* obj->arch[obj->lm1] * tsize);

	sim(obj, tsize, data, output);

	for (i = 0; i < tsize; ++i) {
		itrt = i * lenoup;
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[itrt+j];
			gmse += (temp*temp);
		}
	}

	gmse = gmse / (lenoup * tsize);
	free(output);
	return gmse;
}

void nnet_save(nnet_object obj, const char *fileName) {
	int i;
	FILE *file;
	file = fopen(fileName, "w");

	fprintf(file, "layers : %d\n", obj->layers);

	fprintf(file, "arch : {");
	for (i = 0; i < obj->layers - 1; ++i) {
		fprintf(file, "%d,", obj->arch[i]);
	}
	fprintf(file, "%d}\n", obj->arch[obj->layers - 1]);

	fprintf(file, "actfcn : {");
	for (i = 0; i < obj->layers - 1; ++i) {
		fprintf(file, "%d,", obj->actfcn[i]);
	}
	fprintf(file, "%d}\n", obj->actfcn[obj->layers - 1]);

	fprintf(file, "trainfcn : %s\n", obj->trainfcn);
	fprintf(file, "trainmethod : %s\n", obj->trainmethod);
	fprintf(file, "batch size : %d\n", obj->batchsize);
	fprintf(file, "learning rate : %f\n", obj->eta);
	fprintf(file, "momentum : %f\n", obj->alpha);
	fprintf(file, "eta_inc : %f\n", obj->eta_inc);
	fprintf(file, "eta_dec : %f\n", obj->eta_dec);
	fprintf(file, "qp_threshold : %f\n", obj->qp_threshold);
	fprintf(file, "qp_max_factor : %f\n", obj->qp_max_factor);
	fprintf(file, "qp_shrink_factor : %f\n", obj->qp_shrink_factor);
	fprintf(file, "qp_decay : %f\n", obj->qp_decay);
	fprintf(file, "rp_eta_p : %f\n", obj->rp_eta_p);
	fprintf(file, "rp_eta_n : %f\n", obj->rp_eta_n);
	fprintf(file, "rp_delta_min : %f\n", obj->rp_delta_min);
	fprintf(file, "rp_init_upd : %f\n", obj->rp_init_upd);
	fprintf(file, "rp_max_step : %f\n", obj->rp_max_step);
	fprintf(file, "rp_zero_tol : %f\n", obj->rp_zero_tol);

	fprintf(file, "perf_inc : %f\n", obj->perf_inc);
	fprintf(file, "emax : %d\n", obj->emax);
	fprintf(file, "nmax : %d\n", obj->nmax);
	fprintf(file, "mse : %f\n", obj->mse);
	fprintf(file, "tmse : %f\n", obj->tmse);
	fprintf(file, "gmse : %f\n", obj->gmse);
	fprintf(file, "imse : %f\n", obj->imse);

	fprintf(file, "normmethod : %d\n", obj->normmethod);
	fprintf(file, "inpnmin : %f\n", obj->inpnmin);
	fprintf(file, "inpnmax : %f\n", obj->inpnmax);
	fprintf(file, "oupnmin : %f\n", obj->oupnmin);
	fprintf(file, "oupnmax : %f\n", obj->oupnmax);
	fprintf(file, "inpnmean : %f\n", obj->inpnmean);
	fprintf(file, "inpnstd : %f\n", obj->inpnstd);
	fprintf(file, "oupnmean : %f\n", obj->oupnmean);
	fprintf(file, "oupnstd : %f\n", obj->oupnstd);

	fprintf(file, "steepness : %f\n", obj->steepness);
	fprintf(file, "tratio : %f\n", obj->tratio);
	fprintf(file, "gratio : %f\n", obj->gratio);
	fprintf(file, "vratio : %f\n", obj->vratio);
	fprintf(file, "lm1 : %d\n", obj->lm1);
	fprintf(file, "ld : %d\n", obj->ld);
	fprintf(file, "lw : %d\n", obj->lw);
	fprintf(file, "generalize : %d\n", obj->generalize);
	fprintf(file, "validate : %d\n", obj->validate);

	fprintf(file, "lweight : {");
	for (i = 0; i < obj->layers - 1; ++i) {
		fprintf(file, "%d,", obj->lweight[i]);
	}
	fprintf(file, "%d}\n", obj->lweight[obj->layers - 1]);

	fprintf(file, "weight : {");
	for (i = 0; i < obj->lw - 1; ++i) {
		fprintf(file, "%f,", obj->weight[i]);
	}
	fprintf(file, "%f}\n", obj->weight[obj->lw - 1]);

	fprintf(file, "gradient : {");
	for (i = 0; i < obj->ld - 1; ++i) {
		fprintf(file, "%f,", obj->gradient[i]);
	}
	fprintf(file, "%f}\n", obj->gradient[obj->ld - 1]);

	fprintf(file, "tout : {");
	for (i = 0; i < obj->ld - 1; ++i) {
		fprintf(file, "%f,", obj->tout[i]);
	}
	fprintf(file, "%f}\n", obj->tout[obj->ld - 1]);

	fprintf(file, "input : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fprintf(file, "%f,", obj->input[i]);
	}
	fprintf(file, "%f}\n", obj->input[obj->arch[0] - 1]);

	fprintf(file, "dmin : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fprintf(file, "%f,", obj->dmin[i]);
	}
	fprintf(file, "%f}\n", obj->dmin[obj->arch[0] - 1]);

	fprintf(file, "dmax : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fprintf(file, "%f,", obj->dmax[i]);
	}
	fprintf(file, "%f}\n", obj->dmax[obj->arch[0] - 1]);

	fprintf(file, "tmin : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fprintf(file, "%f,", obj->tmin[i]);
	}
	fprintf(file, "%f}\n", obj->tmin[obj->arch[obj->layers - 1] - 1]);

	fprintf(file, "tmax : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fprintf(file, "%f,", obj->tmax[i]);
	}
	fprintf(file, "%f}\n", obj->tmax[obj->arch[obj->layers - 1] - 1]);

	fprintf(file, "dmean : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fprintf(file, "%f,", obj->dmean[i]);
	}
	fprintf(file, "%f}\n", obj->dmean[obj->arch[0] - 1]);

	fprintf(file, "dstd : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fprintf(file, "%f,", obj->dstd[i]);
	}
	fprintf(file, "%f}\n", obj->dstd[obj->arch[0] - 1]);

	fprintf(file, "tmean : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fprintf(file, "%f,", obj->tmean[i]);
	}
	fprintf(file, "%f}\n", obj->tmean[obj->arch[obj->layers - 1] - 1]);

	fprintf(file, "tstd : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fprintf(file, "%f,", obj->tstd[i]);
	}
	fprintf(file, "%f}\n", obj->tstd[obj->arch[obj->layers - 1] - 1]);

	fclose(file);
}

nnet_object nnet_load(const char *fileName) {
	nnet_object obj;
	int i, layers;
	int *arch, *actfcn;
	FILE *file;
	file = fopen(fileName, "r");

	fscanf(file, "layers : %d\n", &layers);

	arch = (int*)malloc(sizeof(int) * layers);
	actfcn = (int*)malloc(sizeof(int) * layers);

	fscanf(file, "arch : {");
	for (i = 0; i < layers - 1; ++i) {
		fscanf(file, "%d,", &arch[i]);
	}
	fscanf(file, "%d}\n", &arch[layers - 1]);

	fscanf(file, "actfcn : {");
	for (i = 0; i < layers - 1; ++i) {
		fscanf(file, "%d,", &actfcn[i]);
	}
	fscanf(file, "%d}\n", &actfcn[layers - 1]);

	fclose(file);

	obj = nnet_init(layers, arch, actfcn);

	file = fopen(fileName, "r");

	fscanf(file, "layers : %d\n", &obj->layers);

	arch = (int*)malloc(sizeof(int) * layers);
	actfcn = (int*)malloc(sizeof(int) * layers);

	fscanf(file, "arch : {");
	for (i = 0; i < layers - 1; ++i) {
		fscanf(file, "%d,", &obj->arch[i]);
	}
	fscanf(file, "%d}\n", &obj->arch[layers - 1]);

	fscanf(file, "actfcn : {");
	for (i = 0; i < layers - 1; ++i) {
		fscanf(file, "%d,", &obj->actfcn[i]);
	}
	fscanf(file, "%d}\n", &obj->actfcn[layers - 1]);

	fscanf(file, "trainfcn : %s\n", obj->trainfcn);
	fscanf(file, "trainmethod : %s\n", obj->trainmethod);
	fscanf(file, "batch size : %d\n", &obj->batchsize);
	fscanf(file, "learning rate : %f\n", &obj->eta);
	fscanf(file, "momentum : %f\n", &obj->alpha);
	fscanf(file, "eta_inc : %f\n", &obj->eta_inc);
	fscanf(file, "eta_dec : %f\n", &obj->eta_dec);
	fscanf(file, "qp_threshold : %f\n", &obj->qp_threshold);
	fscanf(file, "qp_max_factor : %f\n", &obj->qp_max_factor);
	fscanf(file, "qp_shrink_factor : %f\n", &obj->qp_shrink_factor);
	fscanf(file, "qp_decay : %f\n", &obj->qp_decay);
	fscanf(file, "rp_eta_p : %f\n", &obj->rp_eta_p);
	fscanf(file, "rp_eta_n : %f\n", &obj->rp_eta_n);
	fscanf(file, "rp_delta_min : %f\n", &obj->rp_delta_min);
	fscanf(file, "rp_init_upd : %f\n", &obj->rp_init_upd);
	fscanf(file, "rp_max_step : %f\n", &obj->rp_max_step);
	fscanf(file, "rp_zero_tol : %f\n", &obj->rp_zero_tol);

	fscanf(file, "perf_inc : %f\n", &obj->perf_inc);
	fscanf(file, "emax : %d\n", &obj->emax);
	fscanf(file, "nmax : %d\n", &obj->nmax);
	fscanf(file, "mse : %f\n", &obj->mse);
	fscanf(file, "tmse : %f\n", &obj->tmse);
	fscanf(file, "gmse : %f\n", &obj->gmse);
	fscanf(file, "imse : %f\n", &obj->imse);

	fscanf(file, "normmethod : %d\n", &obj->normmethod);
	fscanf(file, "inpnmin : %f\n", &obj->inpnmin);
	fscanf(file, "inpnmax : %f\n", &obj->inpnmax);
	fscanf(file, "oupnmin : %f\n", &obj->oupnmin);
	fscanf(file, "oupnmax : %f\n", &obj->oupnmax);
	fscanf(file, "inpnmean : %f\n", &obj->inpnmean);
	fscanf(file, "inpnstd : %f\n", &obj->inpnstd);
	fscanf(file, "oupnmean : %f\n", &obj->oupnmean);
	fscanf(file, "oupnstd : %f\n", &obj->oupnstd);

	fscanf(file, "steepness : %f\n", &obj->steepness);
	fscanf(file, "tratio : %f\n", &obj->tratio);
	fscanf(file, "gratio : %f\n", &obj->gratio);
	fscanf(file, "vratio : %f\n", &obj->vratio);
	fscanf(file, "lm1 : %d\n", &obj->lm1);
	fscanf(file, "ld : %d\n", &obj->ld);
	fscanf(file, "lw : %d\n", &obj->lw);
	fscanf(file, "generalize : %d\n", &obj->generalize);
	fscanf(file, "validate : %d\n", &obj->validate);

	fscanf(file, "lweight : {");
	for (i = 0; i < obj->layers - 1; ++i) {
		fscanf(file, "%d,", &obj->lweight[i]);
	}
	fscanf(file, "%d}\n", &obj->lweight[obj->layers - 1]);

	fscanf(file, "weight : {");
	for (i = 0; i < obj->lw - 1; ++i) {
		fscanf(file, "%f,", &obj->weight[i]);
	}
	fscanf(file, "%f}\n", &obj->weight[obj->lw - 1]);

	fscanf(file, "gradient : {");
	for (i = 0; i < obj->ld - 1; ++i) {
		fscanf(file, "%f,", &obj->gradient[i]);
	}
	fscanf(file, "%f}\n", &obj->gradient[obj->ld - 1]);

	fscanf(file, "tout : {");
	for (i = 0; i < obj->ld - 1; ++i) {
		fscanf(file, "%f,", &obj->tout[i]);
	}
	fscanf(file, "%f}\n", &obj->tout[obj->ld - 1]);

	fscanf(file, "input : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fscanf(file, "%f,", &obj->input[i]);
	}
	fscanf(file, "%f}\n", &obj->input[obj->arch[0] - 1]);

	fscanf(file, "dmin : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fscanf(file, "%f,", &obj->dmin[i]);
	}
	fscanf(file, "%f}\n", &obj->dmin[obj->arch[0] - 1]);

	fscanf(file, "dmax : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fscanf(file, "%f,", &obj->dmax[i]);
	}
	fscanf(file, "%f}\n", &obj->dmax[obj->arch[0] - 1]);

	fscanf(file, "tmin : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fscanf(file, "%f,", &obj->tmin[i]);
	}
	fscanf(file, "%f}\n", &obj->tmin[obj->arch[obj->layers - 1] - 1]);

	fscanf(file, "tmax : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fscanf(file, "%f,", &obj->tmax[i]);
	}
	fscanf(file, "%f}\n", &obj->tmax[obj->arch[obj->layers - 1] - 1]);

	fscanf(file, "dmean : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fscanf(file, "%f,", &obj->dmean[i]);
	}
	fscanf(file, "%f}\n", &obj->dmean[obj->arch[0] - 1]);

	fscanf(file, "dstd : {");
	for (i = 0; i < obj->arch[0] - 1; ++i) {
		fscanf(file, "%f,", &obj->dstd[i]);
	}
	fscanf(file, "%f}\n", &obj->dstd[obj->arch[0] - 1]);

	fscanf(file, "tmean : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fscanf(file, "%f,", &obj->tmean[i]);
	}
	fscanf(file, "%f}\n", &obj->tmean[obj->arch[obj->layers - 1] - 1]);

	fscanf(file, "tstd : {");
	for (i = 0; i < obj->arch[obj->layers - 1] - 1; ++i) {
		fscanf(file, "%f,", &obj->tstd[i]);
	}
	fscanf(file, "%f}\n", &obj->tstd[obj->arch[obj->layers - 1] - 1]);


	fclose(file);
	free(arch);
	free(actfcn);
	return obj;
}

void nnet_free(nnet_object obj) {
	free(obj);
}