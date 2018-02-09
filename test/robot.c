#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

int main() {
	ndata_object data,tdata;
	nnet_object net;
	int inp, oup, patterns,tsize;
	int N;
	int isheader = 1;
	double tmse;
	char* file = "datasets/robot.train";
	char *tfile = "datasets/robot.test";
	char *delimiter = " ";

	inp = 48;
	oup = 3;
	patterns = 374;
	tsize = 594;

	data = ndata_init(inp, oup, patterns);
	file_sep_line_enter(data, file, delimiter, isheader);
	ndata_check(data);

	N = 3;
	int arch[3] = { inp, 96 , oup }; // 
	int actfcn[3] = { 0, 3, 2 };// {Null,'tansig','purelin'}

	net = nnet_init(N, arch, actfcn);

	set_trainfcn(net, "traingd");
	set_training_ratios(net, 1.0f, 0.0f, 0.0f);
	set_trainmethod(net, "online", 1);
	set_max_epoch(net, 3000);
	set_target_mse(net, 1e-04f);// Target MSE error
	set_learning_rate(net, 0.7f);// learning rate
	set_momentum(net, 0.4f);// No momentum term

	train(net, patterns, data->data, data->target);

	tdata = ndata_init(inp, oup, tsize);
	file_sep_line_enter(tdata, tfile, delimiter, isheader);

	tmse = nnet_test(net, tsize, tdata->data, tdata->target);

	printf("\n Test MSE %g \n", tmse);

	ndata_free(data);
	ndata_free(tdata);
	nnet_free(net);
    return 0;
}