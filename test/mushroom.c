#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

int main() {
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
	set_training_ratios(net, 1.0f, 0.0f, 0.0f);
	//set_trainmethod(net, "batch", patterns);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05f);// Target MSE error
	set_learning_rate(net, 0.7f);// learning rate
	//set_momentum(net, 0.95);// No momentum term

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
    return 0;
}