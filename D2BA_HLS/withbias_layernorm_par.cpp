#include "withbias_layernorm_par.h"

#define epsilon 0.00001
#define Tr 128
#define Th 4

void load_lnp_in(data_n fm_in_buff[Th][Tr],
        		 data_n* in1, data_n* in2, data_n* in3, data_n* in4,
				 int n, int basePixAddr,
				 int size){

					int i;
					long base_addr = basePixAddr+n*size;
					for (i = 0; i < Tr; i++)
					{
#pragma HLS PIPELINE II=1
						fm_in_buff[0][i] = *(in1+base_addr+i);
						fm_in_buff[1][i] = *(in2+base_addr+i+1*size);
						fm_in_buff[2][i] = *(in3+base_addr+i+2*size);
						fm_in_buff[3][i] = *(in4+base_addr+i+3*size);
					}
				}

void load_lnp_b(data_n mean_out_buff[Tr], data_n var_out_buff[Tr]){
#pragma HLS INLINE off
				int i;
				for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
					mean_out_buff[i] = (data_n)0;
					var_out_buff[i] = (data_n)0;
                }
			}

void compute_add(data_n fm_in_buff[Th][Tr], data_n mean_out_buff[Tr], data_n var_out_buff[Tr]){

                int i;
                for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                	data_n mult1 = fm_in_buff[0][i];
                	data_n mult2 = fm_in_buff[1][i];
                	data_n mult3 = fm_in_buff[2][i];
                	data_n mult4 = fm_in_buff[3][i];

                	data_n sum1 = mult1+mult2;
                	data_n sum2 = mult3+mult4;
                	data_n sum_add = sum1+sum2;
                	data_n accu1 = mean_out_buff[i];
                	mean_out_buff[i] = accu1+sum_add;

                	data_n sum3 = mult1*mult1 + mult2*mult2;
                	data_n sum4 = mult3*mult3 + mult4*mult4;
                	data_n sum_var = sum3+sum4;
                	data_n accu2 = var_out_buff[i];
                	var_out_buff[i] = accu2+sum_var;
                }
            }

void compute_lnp_out(data_n mean_fm_out[Tr], data_n var_fm_out[Tr],
                     data_n* in1, data_n* in2, data_n* in3, data_n* in4,
					 int basePixAddr,
					 int channel,
					 int size
                     ){
						data_n fm_in_buff1[Th][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff1 complete dim=1
	                    data_n fm_in_buff2[Th][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff2 complete dim=1

		                int ti = 0;
                        load_lnp_b(mean_fm_out, var_fm_out);
		                load_lnp_in(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,size);
		                bool pingpong = true;

                        for (ti = Th; ti < channel; ti+=Th)
                        {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
                            if(pingpong){
				                load_lnp_in(fm_in_buff2,in1,in2,in3,in4,ti,basePixAddr,size);
				                compute_add(fm_in_buff1,mean_fm_out,var_fm_out);
				                pingpong = false;
			                }
			                else{
				                load_lnp_in(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,size);
				                compute_add(fm_in_buff2,mean_fm_out,var_fm_out);
				                pingpong = true;
			                }
                        }
                        if(pingpong){
				            compute_add(fm_in_buff1,mean_fm_out,var_fm_out);
                        }
		                else{
				            compute_add(fm_in_buff2,mean_fm_out,var_fm_out);
		                }
                    }

void store_lnp_out(data_n mean_out_buff[Tr], data_n var_out_buff[Tr],
                   data_n* out1,
                   int basePixAddr,
				   int channel,
                   int size
                   ){

                    	int i, j;
                    	for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    		*(out1+basePixAddr+i) = mean_out_buff[i]/channel;
                    	}

                    	for (j = 0; j < Tr; j++){
#pragma HLS PIPELINE II=1
                    		*(out1+size+basePixAddr+j) = hls::sqrt(var_out_buff[j]/channel - (mean_out_buff[j]/channel)*(mean_out_buff[j]/channel) + data_n(epsilon));
                    	}
                	}

void layernorm_par(data_n* ln_in1, data_n* ln_in2, data_n* ln_in3, data_n* ln_in4,
			   	   data_n* ln_out1,
				   int row, int col,
				   int channel){

				int basePixAddr, next_basePixAddr;
				int size = row*col;
				data_n mean_fm_out1[Tr];
				data_n mean_fm_out2[Tr];

				data_n var_fm_out1[Tr];
				data_n var_fm_out2[Tr];
				bool pingpong;
				pingpong = true;
				basePixAddr = 0;
				next_basePixAddr = basePixAddr+Tr;
				compute_lnp_out(mean_fm_out1,var_fm_out1,ln_in1,ln_in2,ln_in3,ln_in4,basePixAddr,channel,size);
				while(true){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4

					if(pingpong){
						compute_lnp_out(mean_fm_out2,var_fm_out2,ln_in1,ln_in2,ln_in3,ln_in4,next_basePixAddr,channel,size);
						store_lnp_out(mean_fm_out1,var_fm_out1,ln_out1,basePixAddr,channel,size);
						pingpong = false;
					}
					else{
						compute_lnp_out(mean_fm_out1,var_fm_out1,ln_in1,ln_in2,ln_in3,ln_in4,next_basePixAddr,channel,size);
						store_lnp_out(mean_fm_out2,var_fm_out2,ln_out1,basePixAddr,channel,size);
						pingpong = true;
					}
					basePixAddr = next_basePixAddr;
					next_basePixAddr = basePixAddr+Tr;
					if(next_basePixAddr >= size)
						break;
				}
				if(pingpong){
					store_lnp_out(mean_fm_out1,var_fm_out1,ln_out1,basePixAddr,channel,size);
				}
				else{
					store_lnp_out(mean_fm_out2,var_fm_out2,ln_out1,basePixAddr,channel,size);
				}
			}

