#include "withbias_layernorm.h"

#define epsilon 0.00001
#define Tr 256
#define Th 4
#define Max_Num 256

void load_lnin(data_n fm_in_buff[Th][Tr],
               data_n* in1, data_n* in2, data_n* in3, data_n* in4,
			   int n, int basePixAddr,
			   int size
        	   ){
					int i;
					int base_addr = basePixAddr+n*size;
					for (i = 0; i < Tr; i++)
					{
#pragma HLS PIPELINE II=1
						fm_in_buff[0][i] = *(in1+base_addr+i);
						fm_in_buff[1][i] = *(in2+base_addr+i+1*size);
						fm_in_buff[2][i] = *(in3+base_addr+i+2*size);
						fm_in_buff[3][i] = *(in4+base_addr+i+3*size);
					}
				}

void load_lnpar(data_n mean_buff[Tr], data_n std_buff[Tr],
				data_n* par1, data_n* par2, data_n* par3, data_n* par4,
				int basePixAddr,
				int size){

					int i;
					for (i = 0; i < Tr; i+=4)
					{
#pragma HLS PIPELINE II=1
						mean_buff[i] = *(par1+basePixAddr+i);
						mean_buff[i+1] = *(par2+basePixAddr+i+1);
						mean_buff[i+2] = *(par3+basePixAddr+i+2);
						mean_buff[i+3] = *(par4+basePixAddr+i+3);

					}
					for (i = 0; i < Tr; i+=4)
					{
#pragma HLS PIPELINE II=1
						std_buff[i] = *(par1+basePixAddr+size+i);
						std_buff[i+1] = *(par2+basePixAddr+size+i+1);
						std_buff[i+2] = *(par3+basePixAddr+size+i+2);
						std_buff[i+3] = *(par4+basePixAddr+size+i+3);
					}
				}

void load_lnb(data_n fm_out_buff[Th][Tr], data_n bias_buff[Max_Num], int n){
#pragma HLS INLINE off

                    int i;
                    for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    	fm_out_buff[0][i] = bias_buff[n];
                    	fm_out_buff[1][i] = bias_buff[n+1];
                    	fm_out_buff[2][i] = bias_buff[n+2];
                    	fm_out_buff[3][i] = bias_buff[n+3];
                    }
}

void compute_ln(data_n fm_in_buff[Th][Tr], data_n mean_buff[Tr], data_n std_buff[Tr],
				data_n weight_buff[Max_Num],
				data_n fm_out_buff[Th][Tr],
				int n){

                int i;
                for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                	data_n fm1 = fm_in_buff[0][i];
                	data_n fm2 = fm_in_buff[1][i];
                	data_n fm3 = fm_in_buff[2][i];
                	data_n fm4 = fm_in_buff[3][i];
                	data_n mean = mean_buff[i];
                	data_n std = std_buff[i];
                	data_n weight1 = weight_buff[n];
                	data_n weight2 = weight_buff[n+1];
                	data_n weight3 = weight_buff[n+2];
                	data_n weight4 = weight_buff[n+3];

                	data_n out1 = fm_out_buff[0][i]+(fm1-mean)*weight1/std;
                	data_n out2 = fm_out_buff[1][i]+(fm2-mean)*weight2/std;
                	data_n out3 = fm_out_buff[2][i]+(fm3-mean)*weight3/std;
                	data_n out4 = fm_out_buff[3][i]+(fm4-mean)*weight4/std;

                	fm_out_buff[0][i] = out1;
                	fm_out_buff[1][i] = out2;
                	fm_out_buff[2][i] = out3;
                	fm_out_buff[3][i] = out4;
                }
            }

void compute_lnout(data_n fm_out_buff[Th][Tr],
				   data_n weight_buff[Max_Num], data_n bias_buff[Max_Num],
				   data_n* in1, data_n* in2, data_n* in3, data_n* in4,
			       data_n* par1, data_n* par2, data_n* par3, data_n* par4,
			       int n, int basePixAddr,
				   int size
                    ){
		                data_n fm_in_buff[Th][Tr];

		                data_n mean_buff[Tr];
#pragma HLS ARRAY_PARTITION variable=mean_buff cyclic factor=32 dim=1

		                data_n std_buff[Tr];
#pragma HLS ARRAY_PARTITION variable=std_buff cyclic factor=32 dim=1

		                int trigger = -1;
                        load_lnb(fm_out_buff, bias_buff, n);
		                load_lnin(fm_in_buff,in1,in2,in3,in4,n,basePixAddr,size);

		                if(trigger != basePixAddr){
		                	trigger = basePixAddr;
		                	load_lnpar(mean_buff,std_buff,par1,par2,par3,par4,basePixAddr,size);
		                }
		                compute_ln(fm_in_buff,mean_buff,std_buff,weight_buff,fm_out_buff,n);
                    }

void store_lnout(data_n fm_out_buff[Th][Tr],
                  data_n* out1, data_n* out2, data_n* out3, data_n* out4,
                  int n, int basePixAddr,
                  int size
                  ){
                    	int i;
                    	int base_addr = basePixAddr+n*size;
                    	for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
                    		*(out1+base_addr+i) = fm_out_buff[0][i];
                    		*(out2+base_addr+i+1*size) = fm_out_buff[1][i];
                    		*(out3+base_addr+i+2*size) = fm_out_buff[2][i];
                    		*(out4+base_addr+i+3*size) = fm_out_buff[3][i];
                    	}
                	}

void next_lnblock(int n,int basePixAddr,
				  int &next_n, int &next_basePixAddr,
				  int channel){
    					if(n+Th >= channel){
    						next_n = 0;
    						next_basePixAddr = basePixAddr+Tr;
    					}
    					else{
    						next_n = n+Th;
    						next_basePixAddr = basePixAddr;
    					}
					}

void layernorm(data_n* ln_in1, data_n* ln_in2, data_n* ln_in3, data_n* ln_in4,
			   data_n* ln_par1, data_n* ln_par2, data_n* ln_par3, data_n* ln_par4,
               data_n* ln_weight,
               data_n* ln_bias,
			   data_n* ln_out1, data_n* ln_out2, data_n* ln_out3, data_n* ln_out4,
               int row, int col,
               int channel){

				int n, basePixAddr, next_n, next_basePixAddr;
				int size = row*col;

				data_n fm_out1[Th][Tr];
				data_n fm_out2[Th][Tr];

				data_n weight[Max_Num];
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
				data_n bias[Max_Num];
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1

				memcpy((data_n*)weight,(const data_n*)ln_weight,sizeof(data_n)*Max_Num);
				memcpy((data_n*)bias,(const data_n*)ln_bias,sizeof(data_n)*Max_Num);
				bool pingpong;
				pingpong = true;
				basePixAddr = 0;
				n = 0;

				compute_lnout(fm_out1,weight,bias,ln_in1,ln_in2,ln_in3,ln_in4,ln_par1,ln_par2,ln_par3,ln_par4,n,basePixAddr,size);
				next_lnblock(n, basePixAddr,next_n, next_basePixAddr, channel);
				while(true){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
					if(pingpong){
						compute_lnout(fm_out2,weight,bias,ln_in1,ln_in2,ln_in3,ln_in4,ln_par1,ln_par2,ln_par3,ln_par4,next_n,next_basePixAddr,size);
						store_lnout(fm_out1,ln_out1,ln_out2,ln_out3,ln_out4,n,basePixAddr,size);
						pingpong = false;
					}
					else{
						compute_lnout(fm_out1,weight,bias,ln_in1,ln_in2,ln_in3,ln_in4,ln_par1,ln_par2,ln_par3,ln_par4,next_n,next_basePixAddr,size);
						store_lnout(fm_out2,ln_out1,ln_out2,ln_out3,ln_out4,n,basePixAddr,size);
						pingpong = true;
					}
					n = next_n;
					basePixAddr = next_basePixAddr;
					next_lnblock(n,basePixAddr,next_n,next_basePixAddr,channel);

					if(next_basePixAddr >= size)
						break;
				}
				if(pingpong){
					store_lnout(fm_out1,ln_out1,ln_out2,ln_out3,ln_out4,n,basePixAddr,size);
				}
				else{
					store_lnout(fm_out2,ln_out1,ln_out2,ln_out3,ln_out4,n,basePixAddr,size);
				}
			}
