#ifndef BASIC_OP_H
#define BASIC_OP_H

#include"xtop_fun.h" // accelerator driver
#include"xil_cache.h"
#include"sd_io.h"
#include "xtime_l.h" // for measuring latency

using namespace std;

static XTop_fun topf_inst;

#define SCALE 512

void topf_init();

void top_fun(data_t* in, data_n* fin, data_t* par, data_n* fpar, data_t* weight, data_n* fweight, data_t* bias, data_n* fbias, data_t* out, data_n* fout, data_t* rescale,
			 int ch_in, int ch_out,
		 	 int input_row, int input_col, bool act, bool use_bias,
			 int choice
			);


#endif // BASIC_OP_H
