#include "data_type.h"

void std_conv(data_t* sc_in1, data_t* sc_in2, data_t* sc_in3, data_t* sc_in4,
              data_t* sc_w1, data_t* sc_w2, data_t* sc_w3, data_t* sc_w4,
              data_t* sc_bias,
              data_t* sc_out1, data_t* sc_out2, data_t* sc_out3, data_t* sc_out4,
              unsigned short sc_ch_in, unsigned short sc_ch_out,
			  unsigned short sc_input_row, unsigned sc_input_col, bool sc_act, bool sc_use_bias
            );
