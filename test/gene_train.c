#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

int main() {
	ndata_object data,tdata;
	nnet_object net;
	int inp, oup, patterns;
	int N,tsize;
	int isheader = 1;
	char* file = "datasets/gene.train";
	char* tfile = "datasets/gene.test";
	char *delimiter = " ";
	float tmse;

	inp = 120;
	oup = 3;
	patterns = 1588;
	tsize = 1587;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 4;
	int arch[4] = { inp, 4, 2, oup }; // 
	int actfcn[4] = { 0, 3, 3, 1 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "traingda");
	set_training_ratios(net, 1.0f, 0.0f, 0.0f);
	set_trainmethod(net, "online", patterns);
	set_max_epoch(net, 500);
	set_target_mse(net, 1e-05f);// Target MSE error
	set_learning_rate(net, 0.01f);// learning rate
	//set_momentum(net, 0.9);// No momentum term
	set_norm_method(net,1);
	set_mnmx(net, 0, 1, 0, 1);

	train(net, patterns, data->data, data->target);

	tdata = ndata_init(inp, oup, tsize);
	file_sep_line_enter(tdata, tfile, delimiter, isheader);

	tmse = nnet_test(net, tsize, tdata->data, tdata->target);

	printf("\n Test MSE %f \n", tmse);

	nnet_save(net, "gene.nnet");

	ndata_free(data);
	ndata_free(tdata);
	nnet_free(net);
    return 0;
}