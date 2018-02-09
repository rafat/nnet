#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

int main() {
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
	set_training_ratios(net, 0.8f, 0.1f, 0.1f);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05f);// Target MSE error
	set_learning_rate(net, 0.07f);// learning rate
	//set_momentum(net, 0.9);// No momentum term

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
    return 0;
}