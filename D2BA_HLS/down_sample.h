#include "data_type.h"

void down_sample(data_t* ds_in1, data_t* ds_in2, data_t* ds_in3, data_t* ds_in4,
                 data_t* ds_w1,
				 data_t* ds_bias,
				 data_t* ds_out1,
				 unsigned short ds_ch_in, unsigned short ds_ch_out,
				 unsigned short ds_input_row, unsigned short ds_input_col, bool ds_act, bool ds_use_bias
            	);
