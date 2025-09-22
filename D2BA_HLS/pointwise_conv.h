#include "data_type.h"

void pointwise_conv(data_t* pc_in1, data_t* pc_in2, data_t* pc_in3, data_t* pc_in4,
            		data_t* pc_w1,
					data_t* pc_bias,
					data_t* pc_out1,
					int pc_input_row, int pc_input_col, int pc_output_col,
					bool act, bool pc_use_bias
           			);
