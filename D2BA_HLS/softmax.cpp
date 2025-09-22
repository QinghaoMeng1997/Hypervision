#include "softmax.h"

#define Tr 32
#define Th 4
#define Max_Num 1

data_n sof_lim_e(data_n x){
#pragma HLS INLINE off
	return hls::powf((1+x/64),64);
}

void compute_max(data_n fm_out_buff[Tr], data_n max[Max_Num], unsigned short size){

		int i;
        for (i = 0; i < size; i++){
#pragma HLS PIPELINE II=1
        	if(max[0] < fm_out_buff[i])
        		max[0] = fm_out_buff[i];
        }
	}

void compute_sum(data_n fm_out_buff[Tr], data_n max[Max_Num], data_n sum[Max_Num], unsigned short size){

		int i;
		for (i = 0; i < size; i++){
#pragma HLS PIPELINE II=1
			data_n accu = fm_out_buff[i];
			data_n tem_sum = sum[0];
			data_n re = sof_lim_e(accu-max[0]);
			fm_out_buff[i] = re;
			sum[0] = tem_sum+re;
		}
	}

void load_sofin(data_n fm_in_buff[Th][Tr],
                data_n* in1, data_n* in2, data_n* in3, data_n* in4,
				int n, int basePixAddr,
				int folder
        		){
			int i, j;
			int base_addr = basePixAddr*folder+n;
			for (i = 0; i < Tr; i++)
			{
#pragma HLS PIPELINE II=1
				int offset = i*folder;
#pragma HLS RESOURCE variable=offset core=Mul_LUT
				fm_in_buff[0][i] = *(in1+base_addr+offset);
				fm_in_buff[1][i] = *(in2+base_addr+offset+1);
				fm_in_buff[2][i] = *(in3+base_addr+offset+2);
				fm_in_buff[3][i] = *(in4+base_addr+offset+3);
			}
		}

void load_sofb(data_n fm_out_buff[Tr]){
#pragma HLS INLINE off

                    int i;
                    for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    	fm_out_buff[i] = (data_n)0;
                    }
                }

void compute_add(data_n fm_in_buff[Th][Tr], data_n fm_out_buff[Tr]){

                int i;
                for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
                	data_n mult1 = fm_in_buff[0][i];
                	data_n mult2 = fm_in_buff[1][i];
                	data_n mult3 = fm_in_buff[2][i];
                	data_n mult4 = fm_in_buff[3][i];
                	data_n sum1 = mult1+mult2;
                	data_n sum2 = mult3+mult4;
                	data_n sum = sum1+sum2;
                	data_n accu = fm_out_buff[i];
                	fm_out_buff[i] = accu+sum;
                }
            }

void compute_sofout(data_n fm_out[Tr], data_n max[Max_Num], data_n sum[Max_Num],
                    data_n* in1, data_n* in2, data_n* in3, data_n* in4,
					unsigned short basePixAddr,
					unsigned short folder,
					unsigned short size
                    ){
		                data_n fm_in_buff1[Th][Tr];
	                    data_n fm_in_buff2[Th][Tr];

		                int ti = 0;
                        load_sofb(fm_out);
                        max[0] = (data_n)0;
                        sum[0] = (data_n)0;

		                load_sofin(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,folder);
		                bool pingpong=true;
                        for (ti = Th; ti < folder; ti+=Th)
                        {
#pragma HLS LOOP_TRIPCOUNT min=31 max=31 avg=31
                            if(pingpong){
				                load_sofin(fm_in_buff2,in1,in2,in3,in4,ti,basePixAddr,folder);
				                compute_add(fm_in_buff1,fm_out);
				                pingpong = false;
			                }
			                else{
				                load_sofin(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,folder);
				                compute_add(fm_in_buff2,fm_out);
				                pingpong = true;
			                }
                        }
                        if(pingpong){
				            compute_add(fm_in_buff1,fm_out);
							compute_max(fm_out, max, size);
							compute_sum(fm_out, max, sum, size);
                        }
		                else{
				            compute_add(fm_in_buff2,fm_out);
							compute_max(fm_out, max, size);
							compute_sum(fm_out, max, sum, size);
		                }
                    }


void store_sofout(data_n fm_out_buff[Tr], data_n sum[Max_Num],
                  data_n* out1,
                  unsigned basePixAddr
                  ){
                    int i;
                    for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    	*(out1+basePixAddr+i) = fm_out_buff[i]/sum[0];
                    }
                }

void softmax(data_n* sof_in1, data_n* sof_in2, data_n* sof_in3, data_n* sof_in4,
			 data_n* sof_out1,
			 unsigned short head, unsigned short pre_channel, unsigned short size, unsigned short folder
			){
				int basePixAddr, next_basePixAddr;
				int border = head*pre_channel*size;

				data_n fm_out1[Tr];
				data_n fm_out2[Tr];

				data_n max1[Max_Num];
				data_n max2[Max_Num];

				data_n sum1[Max_Num];
				data_n sum2[Max_Num];
				bool pingpong;

				pingpong = true;
				basePixAddr = 0;
				next_basePixAddr = Tr;
				compute_sofout(fm_out1,max1,sum1,sof_in1,sof_in2,sof_in3,sof_in4,basePixAddr,folder,size);

				while(true){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
					next_basePixAddr=basePixAddr+Tr;
					if(pingpong){
						compute_sofout(fm_out2,max2,sum2,sof_in1,sof_in2,sof_in3,sof_in4,next_basePixAddr,folder,size);
						store_sofout(fm_out1,sum1,sof_out1,basePixAddr);
						pingpong = false;
					}
					else{
						compute_sofout(fm_out1,max1,sum1,sof_in1,sof_in2,sof_in3,sof_in4,next_basePixAddr,folder,size);
						store_sofout(fm_out2,sum2,sof_out1,basePixAddr);
						pingpong = true;
					}
					basePixAddr = next_basePixAddr;

					if(basePixAddr+Tr >= border)
						break;
				}
				if(pingpong){
					store_sofout(fm_out1,sum1,sof_out1,basePixAddr);
				}
				else{
					store_sofout(fm_out2,sum2,sof_out1,basePixAddr);
				}
}

