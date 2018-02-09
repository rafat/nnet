#include <stdio.h>
#include <stdlib.h>
#include "../header/nnet.h"

int main() {
	ndata_object tdata;
	nnet_object net;
	int inp, oup;
	int N, tsize;
	int isheader = 1;
	char* tfile = "datasets/gene.test";
	char *delimiter = " ";
	float tmse;

	inp = 120;
	oup = 3;
	tsize = 1587;

	net = nnet_load("gene.nnet");

	tdata = ndata_init(inp, oup, tsize);
	file_sep_line_enter(tdata, tfile, delimiter, isheader);

	tmse = nnet_test(net, tsize, tdata->data, tdata->target);

	printf("\n Test MSE %f \n", tmse);

	ndata_free(tdata);
	nnet_free(net);

    return 0;
}