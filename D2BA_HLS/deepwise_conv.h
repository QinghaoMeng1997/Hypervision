#include "data_type.h"

void deepwise_conv(data_t* dc_in1, data_t* dc_in2, data_t* dc_in3, data_t* dc_in4,
                   data_t* dc_w1, data_t* dc_w2, data_t* dc_w3, data_t* dc_w4,
				   data_t* dc_bias,
				   data_t* dc_out1, data_t* dc_out2, data_t* dc_out3, data_t* dc_out4,
				   unsigned short dc_ch_in,
				   unsigned short dc_input_row, unsigned dc_input_col, bool dc_act, bool dc_use_bias
            		);

