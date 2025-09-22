#include "data_type.h"

void trans_linear(data_t* in_ptr1, data_t* in_ptr2, data_t* in_ptr3, data_t* in_ptr4,
				  data_t* weight_ptr1,
				  data_t* bias_ptr,
				  data_t* out_ptr1,
				  int head, int pre_channel, int size, int feature, bool use_bias);
