#include "data_type.h"

void transpose_conv(data_t* tc_in1, data_t* tc_in2, data_t* tc_in3, data_t* tc_in4,
              	  	data_t* tc_w1, data_t* tc_w2, data_t* tc_w3, data_t* tc_w4,
					data_t* tc_bias,
					data_t* tc_out1, data_t* tc_out2, data_t* tc_out3, data_t* tc_out4,
					unsigned short tc_ch_in, unsigned short tc_ch_out,
					unsigned short tc_input_row, unsigned tc_input_col, bool tc_act, bool tc_use_bias
            		);

