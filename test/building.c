#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

int main() {
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
	set_training_ratios(net, 0.8f, 0.1f, 0.1f);
	set_trainmethod(net, "batch", 5);
	set_max_epoch(net, 3000);
	set_target_mse(net, 1e-05f);// Target MSE error
	set_learning_rate(net, 0.01f);// learning rate
	//set_momentum(net, 0.9);// No momentum term

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
    return 0;
}

