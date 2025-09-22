#include "normalize_par.h"

#define epsilon 0.00001

#define Tr 16
#define Th 16

void load_norp_in(data_n fm_in_buff[Th][Tr],
        		  data_n* in1, data_n* in2, data_n* in3, data_n* in4,
				  int n, int basePixAddr,
				  int size
        	){
			int i, j;
			int base_addr = basePixAddr*size+n;
			for (i = 0; i < Tr; i++)
			{
				for (j = 0; j < Th; j+=4){
#pragma HLS PIPELINE II=1
					int offset = i*size;
#pragma HLS RESOURCE variable=offset core=Mul_LUT
					fm_in_buff[j][i] = *(in1+base_addr+offset+j);
					fm_in_buff[j+1][i] = *(in2+base_addr+offset+j+1);
					fm_in_buff[j+2][i] = *(in3+base_addr+offset+j+2);
					fm_in_buff[j+3][i] = *(in4+base_addr+offset+j+3);
				}
			}
		}

void load_norp_b(data_n p2_out_buff[Tr]){
#pragma HLS INLINE off
					int i;
                    for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    	p2_out_buff[i] = (data_n)0;
                    }
                }

void compute_p2(data_n fm_in_buff[Th][Tr], data_n p2_out_buff[Tr]){

                int i, j;
                for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                	 for (j = 0; j < Th; j++){
                		 data_n mult1 = fm_in_buff[j][i];
						 data_n sum1 = mult1*mult1;
						 data_n accu = p2_out_buff[i];
						 p2_out_buff[i] = accu+sum1;
                	}
                }
			}

void compute_norp_out(data_n p2_fm_out[Tr],
                      data_n* in1, data_n* in2, data_n* in3, data_n* in4,
					  int basePixAddr,
					  int size
                      ){
		                data_n fm_in_buff1[Th][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff1 complete dim=1
	                    data_n fm_in_buff2[Th][Tr];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff2 complete dim=1

		                int ti = 0;
                        load_norp_b(p2_fm_out);
		                load_norp_in(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,size);
		                bool pingpong=true;
                        for (ti = Th; ti < size; ti+=Th)
                        {
#pragma HLS LOOP_TRIPCOUNT min=31 max=31 avg=31
                            if(pingpong){
				                load_norp_in(fm_in_buff2,in1,in2,in3,in4,ti,basePixAddr,size);
				                compute_p2(fm_in_buff1,p2_fm_out);
				                pingpong = false;
			                }
			                else{
				                load_norp_in(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,size);
				                compute_p2(fm_in_buff2,p2_fm_out);
				                pingpong = true;
			                }
                        }
                        if(pingpong){
				            compute_p2(fm_in_buff1,p2_fm_out);
                        }
		                else{
				            compute_p2(fm_in_buff2,p2_fm_out);
		                }
                    }

void store_norp_out(data_n p2_out_buff[Tr],
		            data_n* out1,
					int basePixAddr
                  	){
                    int i;
                    for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    	*(out1+basePixAddr+i) = hls::sqrt(p2_out_buff[i]+data_n(epsilon));
                    }
                }

void normalize_par(data_n* norp_in1, data_n* norp_in2, data_n* norp_in3, data_n* norp_in4,
			   	   data_n* norp_out1,
				   int norp_row, int norp_col,
				   int norp_channel){

				int basePixAddr, next_basePixAddr;
				int size = norp_row*norp_col;

				data_n p2_fm_out1[Tr];
				data_n p2_fm_out2[Tr];

				bool pingpong;

				pingpong = true;
				basePixAddr = 0;
				next_basePixAddr = Tr;
				compute_norp_out(p2_fm_out1,norp_in1,norp_in2,norp_in3,norp_in4,basePixAddr,size);
				while(true){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
					next_basePixAddr = basePixAddr+Tr;
					if(pingpong){
						compute_norp_out(p2_fm_out2,norp_in1,norp_in2,norp_in3,norp_in4,next_basePixAddr,size);
						store_norp_out(p2_fm_out1,norp_out1,basePixAddr);
						pingpong = false;
					}
					else{
						compute_norp_out(p2_fm_out1,norp_in1,norp_in2,norp_in3,norp_in4,next_basePixAddr,size);
						store_norp_out(p2_fm_out2,norp_out1,basePixAddr);
						pingpong = true;
					}
					basePixAddr = next_basePixAddr;

					if(basePixAddr+Tr >= norp_channel)
						break;
				}
				if(pingpong){
					store_norp_out(p2_fm_out1,norp_out1,basePixAddr);
				}
				else{
					store_norp_out(p2_fm_out2,norp_out1,basePixAddr);
				}
			}
