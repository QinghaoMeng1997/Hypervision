#include "deepwise_conv.h"
#include "down_sample.h"
#include "trans_linear.h"
#include "head_linear.h"
#include "normalize.h"
#include "normalize_par.h"
#include "pointwise_conv.h"
//#include "pointwise_conv_m.h"
#include "softmax.h"
#include "std_conv.h"
#include "transpose_conv.h"
#include "withbias_layernorm.h"
#include "withbias_layernorm_par.h"
#include "add_leaky.h"

void top_fun(data_t* in1, data_t* in2, data_t* in3, data_t* in4,
			 data_n* fin1, data_n* fin2, data_n* fin3, data_n* fin4,

			 data_t* par1, data_t* par2, data_t* par3, data_t* par4,
			 data_n* fpar1, data_n* fpar2, data_n* fpar3, data_n* fpar4,

			 data_t* w1, data_t* w2, data_t* w3, data_t* w4,
			 data_n* fw1,

			 data_t* bias,
			 data_n* fbias,

			 data_t* out1, data_t* out2, data_t* out3, data_t* out4,
			 data_n* fout1, data_n* fout2, data_n* fout3, data_n* fout4,

			 data_t* rescale,
			 int ch_in, int ch_out,
			 int input_row, int input_col, bool act, bool use_bias,
			 int choice);
