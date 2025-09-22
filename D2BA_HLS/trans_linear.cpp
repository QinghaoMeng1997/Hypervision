#include "trans_linear.h"

#define Tr 256 //height
#define Th 4 //hidden
#define Tc 8 //width
#define MAX_LEN 1024

void load_tlin(data_t fm_in_buff[Th][Tr],
               data_t* in1, data_t* in2, data_t* in3, data_t* in4,
			   int n, int basePixAddr,
			   int input_row
               ){
                    int i;
                    int base_addr = basePixAddr+n*input_row;
                    for (i = 0; i < Tr; i++)
                    {
#pragma HLS PIPELINE II=1
                    	fm_in_buff[0][i] = *(in1+base_addr+i);
                    	fm_in_buff[1][i] = *(in2+base_addr+i+input_row);
                    	fm_in_buff[2][i] = *(in3+base_addr+i+2*input_row);
                    	fm_in_buff[3][i] = *(in4+base_addr+i+3*input_row);
                    }   
                }

void load_tlw(data_t wt_buff[Tc][Th],
              data_t* w1,
			  int n, int m,
			  int input_col, int output_col
			){
                    int i,j;
                    for (i = 0; i < Tc; i++)
                    {
                    	for(j = 0; j < Th; j++)
                    	{
#pragma HLS PIPELINE II=1
                    		wt_buff[i][j] = *(w1+(m+i)*input_col+n+j);
                    	}
                    }
                }

void load_tlb(data_t fm_out_buff[Tc][Tr],
              data_t bias_buff[MAX_LEN/Tc][Tc],
              int m, bool use_bias
               ){
#pragma HLS INLINE off
                    int i, j;
                    for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    	for (j = 0; j < Tc; j++){
//                        	if(use_bias)
////                        		bias_buff[m/Tc][j]
//                        		fm_out_buff[j][i] = (data_t)0;
//                        	else
                        	fm_out_buff[j][i] = (data_t)0;
                        }
                    }
                }

void compute_tl(data_t fm_in_buff[Th][Tr],
             	data_t fm_out_buff[Tc][Tr],
				data_t wt_buff[Tc][Th]
            	){
                int i, j;
                for (i = 0; i < Tr; i++)
               {
#pragma HLS PIPELINE II=1
                	for (j = 0; j < Tc; j++){
#pragma HLS UNROLL
                        data_t mult1 = fm_in_buff[0][i]*wt_buff[j][0];
                        data_t mult2 = fm_in_buff[1][i]*wt_buff[j][1];
                        data_t mult3 = fm_in_buff[2][i]*wt_buff[j][2];
                        data_t mult4 = fm_in_buff[3][i]*wt_buff[j][3];
                        data_t sum1 = mult1+mult2;
                        data_t sum2 = mult3+mult4;
                        data_t sum = sum1+sum2;
                        data_t accu = fm_out_buff[j][i];
                        fm_out_buff[j][i] = accu+sum;
                    }
                }
            }

void compute_tlout(data_t fm_out[Tc][Tr],
                   data_t bias_buff[MAX_LEN/Tc][Tc],
				   data_t* in1, data_t* in2, data_t* in3, data_t* in4,
				   data_t* w1,
				   int m, int basePixAddr,
				   int input_col, int input_row, int output_col, bool use_bias
                   ){
		                data_t fm_in_buff1[Th][Tr];
	                    data_t fm_in_buff2[Th][Tr];

	                    data_t wt_buff1[Tc][Th];
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=1
	                    data_t wt_buff2[Tc][Th];
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=1

		                int ti = 0;
                        load_tlb(fm_out,bias_buff,m,use_bias);
		                load_tlin(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,input_row);
		                load_tlw(wt_buff1,w1,ti,m,input_col,output_col);
		                bool pingpong = true;
                        for (ti = Th; ti < input_col; ti+=Th)
                        {
#pragma HLS LOOP_TRIPCOUNT min=31 max=31 avg=31
                            if(pingpong){
				                load_tlin(fm_in_buff2,in1,in2,in3,in4,ti,basePixAddr,input_row);
				                load_tlw(wt_buff2,w1,ti,m,input_col,output_col);
				                compute_tl(fm_in_buff1,fm_out,wt_buff1);
				                pingpong = false;
			                }
			                else{
				                load_tlin(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,input_row);
				                load_tlw(wt_buff1,w1,ti,m,output_col,output_col);
				                compute_tl(fm_in_buff2,fm_out,wt_buff2);
				                pingpong = true;
			                }
                        }
                        if(pingpong)
				            compute_tl(fm_in_buff1,fm_out,wt_buff1);
		                else
				            compute_tl(fm_in_buff2,fm_out,wt_buff2);
                    }

void store_tlout(data_t fm_out_buff[Tc][Tr],
                 data_t* out1,
				 int basePixAddr, int m,
				 int input_row, int output_col
                  ){
                    int i, j;
                    for (j = 0; j < Tc; j++)
                    {
                    	for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    		 *(out1+basePixAddr+i+(j+m)*input_row) = fm_out_buff[j][i];
                        }
                    }
                }

void next_tlblock(int input_row,
                  int basePixAddr, int baseFeature,
				  int &next_basePixAddr, int &next_baseFeature){

					if(basePixAddr+Tr >= input_row)
	                {
						next_basePixAddr = 0;
						next_baseFeature = baseFeature+Tc;
	                }
					else{
						next_basePixAddr = basePixAddr+Tr;
						next_baseFeature = baseFeature;
	                }
                }

void tl_linear(data_t* in1, data_t* in2, data_t* in3, data_t* in4,
               data_t* w1,
			   data_t* bias,
			   data_t* out1,
			   int input_row, int input_col, int output_col,bool use_bias
            	){

                int basePixAddr,baseFeature;
                int next_basePixAddr,next_baseFeature;

                data_t bias_buff[MAX_LEN/Tc][Tc];
//#pragma HLS ARRAY_PARTITION variable=bias_buff complete dim=2

                data_t fm_out1[Tc][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=1
                data_t fm_out2[Tc][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=1

                bool pingpong;
//                if(use_bias)
//                	memcpy((data_t*)bias_buff,(const data_t*)bias,sizeof(data_t)*output_col);
                pingpong = true;
                basePixAddr = 0;
                baseFeature = 0;
                next_basePixAddr = Tr;
                next_baseFeature = 0;
                compute_tlout(fm_out1,bias_buff,in1,in2,in3,in4,w1,baseFeature,basePixAddr,input_col,input_row,output_col,use_bias);
                while(true){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4

    	            next_tlblock(input_row,basePixAddr,baseFeature,next_basePixAddr,next_baseFeature);
    	            if(pingpong){
    		            compute_tlout(fm_out2,bias_buff,in1,in2,in3,in4,w1,next_baseFeature,next_basePixAddr,input_col,input_row,output_col,use_bias);
    		            store_tlout(fm_out1,out1,basePixAddr,baseFeature,input_row,output_col);
    		            pingpong = false;
    	            }
    	            else{
    		            compute_tlout(fm_out1,bias_buff,in1,in2,in3,in4,w1,next_baseFeature,next_basePixAddr,input_col,input_row,output_col,use_bias);
    		            store_tlout(fm_out2,out1,basePixAddr,baseFeature,input_row,output_col);
    		            pingpong = true;
    	            }
    	            basePixAddr=next_basePixAddr;
    	            baseFeature=next_baseFeature;

                    if(basePixAddr+Tr >= input_row && baseFeature+Tc >= output_col)
    		            break;
                }
                if(pingpong){
		            store_tlout(fm_out1,out1,basePixAddr,baseFeature,input_row,output_col);
                }
                else{
		            store_tlout(fm_out2,out1,basePixAddr,baseFeature,input_row,output_col);
                }
            }

void trans_linear(data_t* in_ptr1, data_t* in_ptr2, data_t* in_ptr3, data_t* in_ptr4,
				  data_t* weight_ptr1,
				  data_t* bias_ptr,
				  data_t* out_ptr1,
				  int head, int pre_channel, int size, int feature, bool use_bias){

					int i, in_size, weight_size, bias_size, out_size;

					in_size = pre_channel*size;
					weight_size = feature*pre_channel;
					bias_size = feature;
					out_size = pre_channel*size;

					for (i = 0; i < head; i++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
						data_t* in1 = in_ptr1+i*in_size;
						data_t* in2 = in_ptr2+i*in_size;
						data_t* in3 = in_ptr3+i*in_size;
						data_t* in4 = in_ptr4+i*in_size;

						data_t* w1 = weight_ptr1+i*weight_size;

						data_t* bias = bias_ptr+i*bias_size;

						data_t* out1 = out_ptr1+i*out_size;

						tl_linear(in1, in2, in3, in4,
								  w1,
								  bias,
								  out1,
								  size, pre_channel, feature, use_bias);
						}
					}




