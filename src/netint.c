#include "netint.h"

void logsig(double *x, double N, double *y) {
	int i;
	for (i = 0; i < N; ++i) {
		y[i] = (1.0 / (1.0 + exp(-x[i])));
		if (y[i] != y[i]) {
			y[i] = signx(x[i]) * 0.5 + 0.5;
		}
	}
}

void tansig(double *x, double N, double *y) {
	int i;
	double a, b;
	for (i = 0; i < N; ++i) {
		//y[i] = (1.0 - exp(-2 * x[i])) / (1.0 + exp(-2 * x[i]));
		//y[i] = -1.0 + 2.0 / (1.0 + exp(-2 * x[i]));
		a = exp(x[i]);
		b = exp(-x[i]);
		y[i] = (a-b) / (a+b);
		if (y[i] != y[i]) {
			y[i] = signx(x[i]);
		}
		// y[i] = tanh(x[i]);
		
	}
}

void hardlim(double *x, double N, double *y) {
	int i;
	for (i = 0; i < N; ++i) {
		if (x[i] <= 0.0) {
			y[i] = 0.0;
		}
		else {
			y[i] = 1.0;
		}
	}
}

void purelin(double *x, double N, double *y) {
	int i;
	for (i = 0; i < N; ++i) {
		y[i] = x[i];
	}
}

//Clip

double clip_value(double x, double lo, double hi) {
	double clip;
	clip = x;
	if (x < lo) {
		clip = lo;
	}

	if (x > hi) {
		clip = hi;
	}
	return clip;
}

// Derivatives

double logsig_der(double value) {
	double temp2,df;

	temp2 = clip_value(value,0.01, 0.99);

	df = temp2 * (1.0 - temp2);

	return df;
}

double tansig_der(double value) {
	double temp2, df;

	temp2 = clip_value(value, -0.98, 0.98);

	df = (1.0 - temp2 * temp2);

	return df;
}

int intmax(int* x, int N) {
	int m, i;

	m = -INT_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i] > m) {
			m = x[i];
		}
	}

	return m;
}

double mean(double* vec, int N) {
	int i;
	double m;
	m = 0.0;

	for (i = 0; i < N; ++i) {
		m += vec[i];
	}
	m = m / N;
	return m;
}

double std(double* vec, int N) {
	double v, temp, m;
	int i;
	v = 0.0;
	m = mean(vec, N);

	for (i = 0; i < N; ++i) {
		temp = vec[i] - m;
		v += temp*temp;
	}

	v = v / N;
	v = sqrt(v);

	return v;

}



double dmax(double* x, int N) {
	int i;
	double m;

	m = -DBL_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i] > m) {
			m = x[i];
		}
	}

	return m;
}


double dmin(double* x, int N) {
	int i;
	double m;

	m = DBL_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i] < m) {
			m = x[i];
		}
	}

	return m;
}


double neuron_oup(double *inp, int N, double *weights, double bias) {
	double a, tmp;
	int i;
	tmp = bias;

	for (i = 0; i < N; ++i) {
		tmp += inp[i] * weights[i];
	}

	logsig(&tmp, 1, &a);

	return a;
}

void neuronlayer_logsig_oup(double *inp, int N, int S, double *weights, double *oup) {
	int i, j, itr, N1;
	double* tmp;
	tmp = (double*)malloc(sizeof(double)* S);
	/*
	N - Number of Inputs going into each Neuron in the Layer
	S - Number of Neutrons in the Layer
	weights - (S X N) Each row contains N weights corresponding to N inputs going into the Neuron
	where i =0,..,S-1 and j = 0,.,.,N-1
	inp - (N X 1) Inputs going into the layer of neutron
	*/
	N1 = N + 1;
	for (j = 0; j < S; ++j) {
		itr = j * N1;
		tmp[j] = weights[itr];
		for (i = 1; i < N1; ++i) {
			tmp[j] += inp[i - 1] * weights[itr + i];
		}
	}

	logsig(tmp, S, oup);

	free(tmp);
}

void neuronlayer_tansig_oup(double *inp, int N, int S, double *weights, double *oup) {
	int i, j, itr, N1;
	double* tmp;
	tmp = (double*)malloc(sizeof(double)* S);
	/*
	N - Number of Inputs going into each Neuron in the Layer
	S - Number of Neutrons in the Layer
	weights - (S X N) Each row contains N weights corresponding to N inputs going into the Neuron
	where i =0,..,S-1 and j = 0,.,.,N-1
	inp - (N X 1) Inputs going into the layer of neutron
	*/
	N1 = N + 1;
	for (j = 0; j < S; ++j) {
		itr = j * N1;
		tmp[j] = weights[itr];
		for (i = 1; i < N1; ++i) {
			tmp[j] += inp[i - 1] * weights[itr + i];
		}
	}


	tansig(tmp, S, oup);

	free(tmp);
}

void neuronlayer_purelin_oup(double *inp, int N, int S, double *weights, double *oup) {
	int i, j, itr, N1;
	double* tmp;
	tmp = (double*)malloc(sizeof(double)* S);
	/*
	N - Number of Inputs going into each Neuron in the Layer
	S - Number of Neutrons in the Layer
	weights - (S X N) Each row contains N weights corresponding to N inputs going into the Neuron
	where i =0,..,S-1 and j = 0,.,.,N-1
	inp - (N X 1) Inputs going into the layer of neutron
	*/
	N1 = N + 1;
	for (j = 0; j < S; ++j) {
		itr = j * N1;
		tmp[j] = weights[itr];
		for (i = 1; i < N1; ++i) {
			tmp[j] += inp[i - 1] * weights[itr + i];
		}
	}


	purelin(tmp, S, oup);

	free(tmp);
}