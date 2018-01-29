#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

void irisout(double *out, int N, double *clamp) {
	int i;
	double temp;

	temp = out[0];
	clamp[0] = 1.0;

	for (i = 1; i < N; ++i) {
		if (out[i] > temp) {
			clamp[i] = 1.0;
			clamp[i - 1] = 0.0;
		}
		else {
			clamp[i] = 0.0;
		}
	}
}

void building() {
	ndata_object data;
	nnet_object net;
	int inp, oup, patterns;
	int N;
	int isheader = 1;
	char* file = "datasets/building.train";
	char *delimiter = " ";

	inp = 14;
	oup = 3;
	patterns = 2104;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 3;
	int arch[3] = { inp, 16, oup }; // 
	int actfcn[3] = { 0, 3, 1 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "traingd");
	set_training_ratios(net, 1.0, 0.0, 0.0);
	set_trainmethod(net, "batch", patterns);
	set_max_epoch(net, 3000);
	set_target_mse(net, 1e-05);// Target MSE error
	set_learning_rate(net, 0.01);// learning rate
	//set_momentum(net, 0.9);// No momentum term

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
}

void gene() {
	ndata_object data;
	nnet_object net;
	int inp, oup, patterns;
	int N;
	int isheader = 1;
	char* file = "datasets/gene.train";
	char *delimiter = " ";

	inp = 120;
	oup = 3;
	patterns = 1588;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 4;
	int arch[4] = { inp, 4, 2, oup }; // 
	int actfcn[4] = { 0, 3, 3, 1 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "traingda");
	set_training_ratios(net, 1.0, 0.0, 0.0);
	set_trainmethod(net, "online", patterns);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05);// Target MSE error
	set_learning_rate(net, 0.01);// learning rate
	//set_momentum(net, 0.9);// No momentum term
	set_norm_method(net,1);
	set_mnmx(net, 0, 1, 0, 1);

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
}

void mushroom() {
	ndata_object data;
	nnet_object net;
	int inp, oup, patterns;
	int N;
	int isheader = 1;
	char* file = "datasets/mushroom.train";
	char *delimiter = " ";

	inp = 125;
	oup = 2;
	patterns = 4062;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 3;
	int arch[3] = { inp, 32, oup }; // 
	int actfcn[3] = { 0, 3, 2 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "trainqp");
	set_training_ratios(net, 1.0, 0.0, 0.0);
	//set_trainmethod(net, "batch", patterns);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05);// Target MSE error
	set_learning_rate(net, 0.7);// learning rate
	set_momentum(net, 0.95);// No momentum term

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
}

void robot() {
	ndata_object data;
	nnet_object net;
	int inp, oup, patterns;
	int N;
	int isheader = 1;
	char* file = "datasets/robot.train";
	char *delimiter = " ";

	inp = 48;
	oup = 3;
	patterns = 374;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 3;
	int arch[3] = { inp, 96 , oup }; // 
	int actfcn[3] = { 0, 3, 2 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "traingd");
	set_training_ratios(net, 1.0, 0.0, 0.0);
	//set_trainmethod(net, "batch", patterns);
	set_max_epoch(net, 3000);
	set_target_mse(net, 1e-03);// Target MSE error
	set_learning_rate(net, 0.7);// learning rate
	set_momentum(net, 0.4);// No momentum term

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
}

void soybean() {
	ndata_object data;
	nnet_object net;
	int inp, oup, patterns;
	int N;
	int isheader = 1;
	char* file = "datasets/soybean.train";
	char *delimiter = " ";

	inp = 82;
	oup = 19;
	patterns = 342;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 4;
	int arch[4] = { inp, 16, 8, oup }; // 
	int actfcn[4] = { 0, 3, 3, 1 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "trainrp");
	set_training_ratios(net, 1.0, 0.0, 0.0);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05);// Target MSE error
	set_learning_rate(net, 0.07);// learning rate
	set_momentum(net, 0.9);// No momentum term
	set_norm_method(net, 0);
	set_mnmx(net, 0, 1, 0, 1);

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
}

void thyroid() {
	ndata_object data;
	nnet_object net;
	int inp, oup, patterns;
	int N;
	int isheader = 1;
	char* file = "datasets/thyroid.train";
	char *delimiter = " ";

	inp = 21;
	oup = 3;
	patterns = 3600;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 4;
	int arch[4] = { inp, 16, 8, oup }; // 
	int actfcn[4] = { 0, 3, 3, 1 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "traingda");
	set_training_ratios(net, 1.0, 0.0, 0.0);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05);// Target MSE error
	set_learning_rate(net, 0.07);// learning rate
	//set_momentum(net, 0.9);// No momentum term

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
}

void iris() {
	ndata_object data;
	nnet_object net;
	int inp, oup, patterns;
	int N;
	int isheader = 0;
	char* file = "iris.data.txt";
	char *delimiter = " ";

	inp = 4;
	oup = 3;
	patterns = 150;

	data = ndata_init(inp, oup, patterns);
	file_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 3;
	int arch[3] = { inp, 7, oup }; // 
	int actfcn[3] = { 0, 3, 2 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "traingdm");
	set_training_ratios(net, 0.8, 0.0, 0.2);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05);// Target MSE error
	set_learning_rate(net, 0.07);// learning rate
	set_momentum(net, 0.9);// No momentum term
	set_mnmx(net, 0, 1, 0, 1);

	train_mnmx(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
}

void iris2() {
	int N, i, lendata, lentarget;
	int tsize, leninp, lenoup;
	double *data, *target, *output, *out;
	int iter;
	double accr;
	nnet_object obj;

	FILE *fp;

	N = 3;

	tsize = 150;
	leninp = 4;
	lenoup = 3;
	lendata = tsize * leninp;
	lentarget = tsize * lenoup;

	data = (double*)malloc(sizeof(double)* lendata);
	target = (double*)malloc(sizeof(double)* lentarget);
	output = (double*)malloc(sizeof(double)* lenoup);
	out = (double*)malloc(sizeof(double)* lenoup * tsize);

	fp = fopen("iris.data.txt", "r");

	for (i = 0; i < tsize; ++i) {
		fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", &(data[leninp * i]), &(data[leninp * i + 1]), &(data[leninp * i + 2]), &(data[leninp * i + 3]),
			&(target[i*lenoup]), &(target[i*lenoup + 1]), &(target[i*lenoup + 2]));
	}

	fclose(fp);
	/*
	for (i = 0; i < tsize; ++i) {
	//target[i] = 0.5 * (target[i] + 1.0);
	printf("%g %g %g %g %g %g %g \n", data[leninp * i], data[leninp * i + 1], data[leninp * i + 2], data[leninp * i + 3], target[i*lenoup], target[i*lenoup+1], target[i*lenoup+2]);
	}
	*/


	int arch[3] = { 4, 7, 3 };
	int actfcn[3] = { 0, 3, 2 };

	obj = nnet_init(N, arch, actfcn);

	set_trainfcn(obj, "traingd");
	//set_training_ratios(obj, 1.0, 0.0, 0.0);
	//obj->eta = 0.05;
	//obj->alpha = 0.01;
	set_learning_rate(obj, 0.1);// learning rate
	set_momentum(obj, 0.9);// No momentum term
	set_max_epoch(obj, 1000);
	set_norm_method(obj, 1);
	set_mnmx(obj, 0, 1, 0, 1);

	train(obj, tsize, data, target);
	double clamp[3] = { 0, 0, 0 };
	accr = 0;

	sim(obj, tsize, data, out);

	for (i = 0; i < tsize; ++i) {
		//feedforward(obj, data + i*leninp, leninp, lenoup, output);
		irisout(out + i*lenoup, lenoup, clamp);
		//printf("%g %g %g %g %g %g \n", target[i*lenoup], target[i*lenoup + 1], target[i*lenoup + 2], clamp[0], clamp[1], clamp[2]);
		if (target[i*lenoup] == clamp[0] && target[i*lenoup + 1] == clamp[1] && target[i*lenoup + 2] == clamp[2]) {
			accr++;
		}
	}
	printf("Accuracy %g \n", (double)accr / tsize);

	for (i = 0; i < obj->lw; ++i) {
		printf("%g ", obj->weight[i]);
	}

	//func_lm(data, 4, 4, obj);

	nnet_free(obj);
	free(data);
	free(target);
	free(output);
	free(out);
}


int main() {
	//building();
	//gene();
	//soybean();
	//thyroid();
	robot();
	//mushroom();
	//waveinit();
	//iris();
	return 0;
}