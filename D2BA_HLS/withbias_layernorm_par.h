#include "data_type.h"
#include "hls_math.h"

void layernorm_par(data_n* ln_in1, data_n* ln_in2, data_n* ln_in3, data_n* ln_in4,
			   	   data_n* ln_out1,
				   int row, int col,
				   int channel);
