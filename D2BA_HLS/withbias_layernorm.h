#include "data_type.h"
#include "hls_math.h"


void layernorm(data_n* ln_in1, data_n* ln_in2, data_n* ln_in3, data_n* ln_in4,
			   data_n* ln_par1, data_n* ln_par2, data_n* ln_par3, data_n* ln_par4,
               data_n* ln_weight,
               data_n* ln_bias,
			   data_n* ln_out1, data_n* ln_out2, data_n* ln_out3, data_n* ln_out4,
               int row, int col,
               int channel);
