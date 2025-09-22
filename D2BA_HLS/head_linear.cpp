#include "head_linear.h"

#define Folder 64
#define Tr 32 //height
#define Th 32 //hidden 一次计算输入特征
#define Tc 32 //width 输出特征
#define MAX_LEN 1024

void load_hlin(data_t fm_in_buff[Th][Tr],
               data_t* in1,
               int n, int basePixAddr,
               int fold_incol
                ){
                    int i, j;
                    int base_addr = basePixAddr*Folder+n;

                    for(j = 0; j < Th; j++)
                    {
                    	for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    		fm_in_buff[j][i] = *(in1+base_addr+i*fold_incol*Folder+j);
                    	}
                    }
                }

void load_hlw(data_t wt_buff[Tc][Th],
              data_t* w1,
			  int n, int m,
			  int fold_incol, int output_col
              ){
                    int i,j;
                    for (i = 0; i < Tc; i++)
                    {
                    	for(j = 0; j < Th; j++)
                    	{
#pragma HLS PIPELINE II=1
                    		wt_buff[i][j] = *(w1+(m+i*fold_incol)*Folder+j+n);
                    	}
                    }
                }

void load_hlb(data_t fm_out_buff[Tc][Tr],
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

void compute_hl(data_t fm_in_buff[Th][Tr],
             	data_t fm_out_buff[Tc][Tr],
				data_t wt_buff[Tc][Th]
            	){
                int i, j, k;
                for (j = 0; j < Tc; j++){
                	for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                		for (k = 0; k < Th; k++){
#pragma HLS UNROLL
                			data_t mult1 = fm_in_buff[k][i]*wt_buff[j][k];
                			data_t accu = fm_out_buff[j][i];
                			fm_out_buff[j][i] = accu+mult1;
                		}
                    }
                }
            }

void compute_hlout(data_t fm_out[Tc][Tr],
                   data_t bias_buff[MAX_LEN/Tc][Tc],
				   data_t* in1,
				   data_t* w1,
				   int m, int basePixAddr,
				   int fold_incol, int output_col,bool use_bias
					){
		                data_t fm_in_buff1[Th][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff1 complete dim=1
	                    data_t fm_in_buff2[Th][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff2 complete dim=1

	                    data_t wt_buff1[Tc][Th];
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=2
	                    data_t wt_buff2[Tc][Th];
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=2

		                int ti=0;
                        load_hlb(fm_out,bias_buff,m,use_bias);
		                load_hlin(fm_in_buff1,in1,ti,basePixAddr,fold_incol);
		                load_hlw(wt_buff1,w1,ti,m,fold_incol,output_col);
		                bool pingpong = true;
                        for (ti = Th; ti < Folder; ti+=Th)
                        {
#pragma HLS LOOP_TRIPCOUNT min=31 max=31 avg=31
                            if(pingpong){
				                load_hlin(fm_in_buff2,in1,ti,basePixAddr,fold_incol);
				                load_hlw(wt_buff2,w1,ti,m,fold_incol,output_col);
				                compute_hl(fm_in_buff1,fm_out,wt_buff1);
				                pingpong = false;
			                }
			                else{
				                load_hlin(fm_in_buff1,in1,ti,basePixAddr,fold_incol);
				                load_hlw(wt_buff1,w1,ti,m,fold_incol,output_col);
				                compute_hl(fm_in_buff2,fm_out,wt_buff2);
				                pingpong = true;
			                }
                        }
                        if(pingpong)
				            compute_hl(fm_in_buff1,fm_out,wt_buff1);
		                else
				            compute_hl(fm_in_buff2,fm_out,wt_buff2);
                    }

void store_hlout(data_t fm_out_buff[Tc][Tr],
                 data_t* out1, data_t* out2, data_t* out3, data_t* out4,
				 int basePixAddr, int m,
				 int fold_incol, int output_col,
				 data_t re,
				 bool act
				){
                    int i, j;
                    for (j = 0; j < Tc; j+=4)
                    {
                    	for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                        	if(act)
                        	{
                        		*(out1+basePixAddr+i*fold_incol*Tr+j*fold_incol) = fm_out_buff[j][i]*re;
                        		*(out2+basePixAddr+i*fold_incol*Tr+(j+1)*fold_incol) = fm_out_buff[j+1][i]*re;
                        		*(out3+basePixAddr+i*fold_incol*Tr+(j+2)*fold_incol) = fm_out_buff[j+2][i]*re;
                        		*(out4+basePixAddr+i*fold_incol*Tr+(j+3)*fold_incol) = fm_out_buff[j+3][i]*re;
                        	}
                        	else
                        	{
                        		*(out1+basePixAddr+i*fold_incol*Tr+j*fold_incol) = fm_out_buff[j][i];
                        		*(out2+basePixAddr+i*fold_incol*Tr+(j+1)*fold_incol) = fm_out_buff[j+1][i];
                        		*(out3+basePixAddr+i*fold_incol*Tr+(j+2)*fold_incol) = fm_out_buff[j+2][i];
                        		*(out4+basePixAddr+i*fold_incol*Tr+(j+3)*fold_incol) = fm_out_buff[j+3][i];
                        	}
                        }
                    }
                }

void next_hlblock(int input_row,
                  int basePixAddr, int baseFeature,
                  int &next_basePixAddr, int &next_baseFeature){

					next_basePixAddr = basePixAddr+1;
					next_baseFeature = baseFeature+1;
                }


void hl_linear(data_t* in1,
               data_t* w1,
			   data_t* bias,
			   data_t* out1, data_t* out2, data_t* out3, data_t* out4,
			   data_t re,
			   int input_row, int fold_incol, int output_col,bool act, bool use_bias
            	){
                int basePixAddr, baseFeature;
                int next_basePixAddr, next_baseFeature;
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
                compute_hlout(fm_out1,bias_buff,in1,w1,baseFeature,basePixAddr,fold_incol,output_col,use_bias);
                while(true){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4

    	            next_hlblock(input_row,basePixAddr,baseFeature,next_basePixAddr,next_baseFeature);
    	            if(pingpong){
    		            compute_hlout(fm_out2,bias_buff,in1,w1,next_baseFeature,next_basePixAddr,fold_incol,output_col,use_bias);
    		            store_hlout(fm_out1,out1,out2,out3,out4,basePixAddr,baseFeature,fold_incol,output_col,re,act);
    		            pingpong = false;
    	            }
    	            else{
    		            compute_hlout(fm_out1,bias_buff,in1,w1,next_baseFeature,next_basePixAddr,fold_incol,output_col,use_bias);
    		            store_hlout(fm_out2,out1,out2,out3,out4,basePixAddr,baseFeature,fold_incol,output_col,re,act);
    		            pingpong = true;
    	            }
    	            basePixAddr = next_basePixAddr;
    	            baseFeature = next_baseFeature;

                    if(basePixAddr+1 >= fold_incol && baseFeature+1 >= fold_incol)
    		            break;
                }
                if(pingpong){
		            store_hlout(fm_out1,out1,out2,out3,out4,basePixAddr,baseFeature,fold_incol,output_col,re,act);
                }
                else{
		            store_hlout(fm_out2,out1,out2,out3,out4,basePixAddr,baseFeature,fold_incol,output_col,re,act);
                }
            }

void head_linear(data_t* in_ptr1,
				 data_t* weight_ptr1,
				 data_t* bias_ptr,
				 data_t* out_ptr1, data_t* out_ptr2, data_t* out_ptr3, data_t* out_ptr4,
				 data_t* rescale,
				 int head, int pre_channel, int size, int feature, bool act, bool use_bias){

						int i, fold_incol, in_size, weight_size, bias_size, out_size;
						data_t re;
						fold_incol = size/Folder;
						in_size = pre_channel*size;
						weight_size = feature*size;
						bias_size = feature;
						out_size = pre_channel*feature*fold_incol;

						for (i = 0; i < head; i++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
							if(act)
								re = *(rescale+i);
							data_t* in1 = in_ptr1+i*in_size;

							data_t* w1 = weight_ptr1+i*weight_size;

							data_t* bias = bias_ptr+i*bias_size;

							data_t* out1 = out_ptr1+i*out_size;
							data_t* out2 = out_ptr2+i*out_size;
							data_t* out3 = out_ptr3+i*out_size;
							data_t* out4 = out_ptr4+i*out_size;

							hl_linear(in1,
									w1,
									bias,
									out1, out2, out3, out4,
									re,
									pre_channel, fold_incol, feature, act, use_bias);
							}
						}
