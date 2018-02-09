#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

int main() {
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

	set_trainfcn(net, "traingdm");
	set_training_ratios(net, 1.0f, 0.0f, 0.0f);
	set_max_epoch(net, 1000);
	set_target_mse(net, 1e-05f);// Target MSE error
	set_learning_rate(net, 0.07f);// learning rate
	set_momentum(net, 0.9f);// No momentum term
	set_norm_method(net, 0);
	//set_mnmx(net, 0, 1, 0, 1);

	train(net, patterns, data->data, data->target);

	ndata_free(data);
	nnet_free(net);
    return 0;
}