#include "top_fun.h"

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
			 int input_row, int input_col,
			 bool act, bool use_bias,
			 int choice){

//	 AXI4-Lite
#pragma HLS INTERFACE s_axilite port=ch_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ch_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=input_row bundle=CTRL
#pragma HLS INTERFACE s_axilite port=input_col bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite port=act bundle=CTRL
#pragma HLS INTERFACE s_axilite port=use_bias bundle=CTRL
#pragma HLS INTERFACE s_axilite port=choice bundle=CTRL

//	 AXI4-Master
#pragma HLS INTERFACE m_axi depth=65536 port=in1 offset=slave bundle=FM1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=in2 offset=slave bundle=FM2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=in3 offset=slave bundle=FM3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=in4 offset=slave bundle=FM4 max_read_burst_length=256 max_write_burst_length=256

#pragma HLS INTERFACE m_axi depth=65536 port=fin1 offset=slave bundle=FM1_32 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=fin2 offset=slave bundle=FM2_32 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=fin3 offset=slave bundle=FM3_32 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=fin4 offset=slave bundle=FM4_32 max_read_burst_length=256 max_write_burst_length=256

#pragma HLS INTERFACE m_axi depth=65536 port=w1 offset=slave bundle=W1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w2 offset=slave bundle=W2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w3 offset=slave bundle=W3 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w4 offset=slave bundle=W4 max_read_burst_length=256

#pragma HLS INTERFACE m_axi depth=65536 port=fw1 offset=slave bundle=W1_32 max_read_burst_length=256

#pragma HLS INTERFACE m_axi depth=65536 port=par1 offset=slave bundle=W1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=par2 offset=slave bundle=W2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=par3 offset=slave bundle=W3 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=par4 offset=slave bundle=W4 max_read_burst_length=256

#pragma HLS INTERFACE m_axi depth=65536 port=fpar1 offset=slave bundle=W1_32 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=fpar2 offset=slave bundle=W2_32 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=fpar3 offset=slave bundle=W3_32 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=fpar4 offset=slave bundle=W4_32 max_read_burst_length=256

#pragma HLS INTERFACE m_axi depth=1024 port=bias offset=slave bundle=W1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=1024 port=fbias offset=slave bundle=W1_32 max_read_burst_length=256

#pragma HLS INTERFACE m_axi depth=1024 port=rescale offset=slave bundle=W1 max_read_burst_length=256

#pragma HLS INTERFACE m_axi port=out1 offset=slave bundle=FM1
#pragma HLS INTERFACE m_axi port=out2 offset=slave bundle=FM2
#pragma HLS INTERFACE m_axi port=out3 offset=slave bundle=FM3
#pragma HLS INTERFACE m_axi port=out4 offset=slave bundle=FM4

#pragma HLS INTERFACE m_axi port=fout1 offset=slave bundle=FM1_32
#pragma HLS INTERFACE m_axi port=fout2 offset=slave bundle=FM2_32
#pragma HLS INTERFACE m_axi port=fout3 offset=slave bundle=FM3_32
#pragma HLS INTERFACE m_axi port=fout4 offset=slave bundle=FM4_32

	switch(choice){

	case 1:
		std_conv(in1, in2, in3, in4,
		         w1, w2, w3, w4,
		         bias,
				 out1, out2, out3, out4,
				 ch_in, ch_out,
				 input_row, input_col, act, use_bias
				);
		break;

	case 2:
		down_sample(in1, in2, in3, in4,
		            w1,
					bias,
					out1,
		            ch_in, ch_out,
					input_row, input_col,act, use_bias
		            );
		break;

	case 3:
		transpose_conv(in1, in2, in3, in4,
				       w1,w2,w3,w4,
					   bias,
				 	   out1,out2,out3,out4,
			  		   ch_in, ch_out,
					   input_row, input_col,act, use_bias
					   );
		break;

	case 4:
		deepwise_conv(in1, in2, in3, in4,
					  w1,w2,w3,w4,
					  bias,
					  out1,out2,out3,out4,
					  ch_in,
					  input_row, input_col,act, use_bias
					);
		break;

	case 5:
		pointwise_conv(in1, in2, in3, in4,
	   				   w1,
					   bias,
					   out1,
					   ch_in,
					   input_row, input_col, act, use_bias
					 );
		break;

//	case 6:
//		pointwise_conv_m(in1,in2,in3,in4,
//				         w1,
//						 bias,
//						 out1,
//						 ch_in,
//						 input_row, input_col,  act,use_bias
//					 );
//		break;
//
	case 7:
		head_linear(in1,
				    w1,
					bias,
				    out1, out2, out3, out4,
					rescale,
					ch_in, ch_out,
					input_row, input_col, act, use_bias
				   );
		break;

	case 8:
		trans_linear(in1, in2, in3, in4,
				     w1,
				     bias,
				     out1,
				     ch_in, ch_out,
				     input_row, input_col, use_bias
					);
		break;

	case 9:
		normalize_par(fin1,fin2,fin3,fin4,
				      fout1,
					  input_row, input_col, ch_in
					 );
		break;

	case 10:
		normalize(fin1, fin2, fin3, fin4,
				  fpar1,
				  fout1, fout2, fout3, fout4,
				  input_row, input_col, ch_in
				);
		break;

	case 11:
		softmax(fin1,fin2,fin3,fin4,
				fout1,
				ch_in, ch_out,
				input_row, input_col);
		break;

	case 12:
		layernorm_par(fin1, fin2, fin3, fin4,
				      fout1,
				      input_row, input_col, ch_in
				);
		break;

	case 13:
		layernorm(fin1, fin2, fin3, fin4,
				  fpar1, fpar2, fpar3, fpar4,
		          fw1,
		          fbias,
			      fout1, fout2, fout3, fout4,
			      input_row, input_col, ch_in
				);
		break;

	case 14:
		add_leaky(fin1,
		          fw1,
			      fout1,
			      ch_in,
				  act
				);
		break;
	}
}

