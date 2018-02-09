#include "netint.h"

void logsig(float *x, float N, float *y) {
	int i;
	for (i = 0; i < N; ++i) {
		y[i] = (1.0 / (1.0 + exp(-x[i])));
		if (y[i] != y[i]) {
			y[i] = signx(x[i]) * 0.5 + 0.5;
		}
	}
}

void tansig(float *x, float N, float *y) {
	int i;
	float a, b;
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

void hardlim(float *x, float N, float *y) {
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

void purelin(float *x, float N, float *y) {
	int i;
	for (i = 0; i < N; ++i) {
		y[i] = x[i];
	}
}

//Clip

float clip_value(float x, float lo, float hi) {
	float clip;
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

float logsig_der(float value) {
	float temp2,df;

	temp2 = clip_value(value,0.01, 0.99);

	df = temp2 * (1.0 - temp2);

	return df;
}

float tansig_der(float value) {
	float temp2, df;

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

float mean(float* vec, int N) {
	int i;
	float m;
	m = 0.0;

	for (i = 0; i < N; ++i) {
		m += vec[i];
	}
	m = m / N;
	return m;
}

float std(float* vec, int N) {
	float v, temp, m;
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



float dmax(float* x, int N) {
	int i;
	float m;

	m = -FLT_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i] > m) {
			m = x[i];
		}
	}

	return m;
}


float dmin(float* x, int N) {
	int i;
	float m;

	m = FLT_MAX;

	for (i = 0; i < N; ++i) {
		if (x[i] < m) {
			m = x[i];
		}
	}

	return m;
}


float neuron_oup(float *inp, int N, float *weights, float bias) {
	float a, tmp;
	int i;
	tmp = bias;

	for (i = 0; i < N; ++i) {
		tmp += inp[i] * weights[i];
	}

	logsig(&tmp, 1, &a);

	return a;
}

void neuronlayer_logsig_oup(float *inp, int N, int S, float *weights, float *oup) {
	int i, j, itr, N1;
	float* tmp;
	tmp = (float*)malloc(sizeof(float)* S);
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

void neuronlayer_tansig_oup(float *inp, int N, int S, float *weights, float *oup) {
	int i, j, itr, N1;
	float* tmp;
	tmp = (float*)malloc(sizeof(float)* S);
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

void neuronlayer_purelin_oup(float *inp, int N, int S, float *weights, float *oup) {
	int i, j, itr, N1;
	float* tmp;
	tmp = (float*)malloc(sizeof(float)* S);
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