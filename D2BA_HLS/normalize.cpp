#include "normalize.h"

#define epsilon 0.00001

#define Tr 256
#define Th 4
#define Max_Num 256

void load_norin(data_n fm_in_buff[Th][Tr],
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

void compute_nor(data_n fm_in_buff[Th][Tr], data_n p2_buff[Max_Num], data_n fm_out_buff[Th][Tr],
				int n){

				int i;
				for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
					data_n fm1 = fm_in_buff[0][i];
					data_n fm2 = fm_in_buff[1][i];
					data_n fm3 = fm_in_buff[2][i];
					data_n fm4 = fm_in_buff[3][i];

					data_n p2_1 = p2_buff[n];
					data_n p2_2 = p2_buff[n+1];
					data_n p2_3 = p2_buff[n+2];
					data_n p2_4 = p2_buff[n+3];

					data_n out1 = fm1/p2_1;
					data_n out2 = fm2/p2_2;
					data_n out3 = fm3/p2_3;
					data_n out4 = fm4/p2_4;

					fm_out_buff[0][i] = out1;
					fm_out_buff[1][i] = out2;
					fm_out_buff[2][i] = out3;
					fm_out_buff[3][i] = out4;
				}
			}

void compute_norout(data_n fm_out_buff[Th][Tr], data_n p2_buff[Max_Num],
                    data_n* in1, data_n* in2, data_n* in3, data_n* in4,
					int n, int basePixAddr,
					int size
                    ){
		                data_n fm_in_buff[Th][Tr];

		                load_norin(fm_in_buff,in1,in2,in3,in4,n,basePixAddr,size);
		                compute_nor(fm_in_buff,p2_buff,fm_out_buff,n);
                    }

void store_norout(data_n fm_out_buff[Th][Tr],
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

void next_norblock(int n, int basePixAddr,
				  int &next_n, int &next_basePixAddr,
				  int size){
    		if(basePixAddr+Tr >= size){
    			next_n = n+Th;
    			next_basePixAddr = 0;
    		}
    		else{
    			next_n = n;
    			next_basePixAddr = basePixAddr+Tr;
    }
}

void normalize(data_n* nor_in1, data_n* nor_in2, data_n* nor_in3, data_n* nor_in4,
               data_n* nor_p2,
			   data_n* nor_out1, data_n* nor_out2, data_n* nor_out3, data_n* nor_out4,
               int nor_row, int nor_col,
               int nor_channel){

				int n, basePixAddr, next_n, next_basePixAddr;
				int size = nor_row*nor_col;

				data_n fm_out1[Th][Tr];
				data_n fm_out2[Th][Tr];

				data_n p2_buff[Max_Num];
#pragma HLS ARRAY_PARTITION variable=p2_buff complete dim=1

				memcpy((data_n*)p2_buff,(const data_n*)nor_p2,sizeof(data_n)*Max_Num);
				bool pingpong;
				pingpong = true;
				basePixAddr = 0;
				n = 0;
				compute_norout(fm_out1,p2_buff,nor_in1,nor_in2,nor_in3,nor_in4,n,basePixAddr,size);
				next_norblock(n,basePixAddr,next_n,next_basePixAddr,size);
				while(true){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
					if(pingpong){
						compute_norout(fm_out2,p2_buff,nor_in1,nor_in2,nor_in3,nor_in4,next_n,next_basePixAddr,size);
						store_norout(fm_out1,nor_out1,nor_out2,nor_out3,nor_out4,n,basePixAddr,size);
						pingpong = false;
					}
					else{
						compute_norout(fm_out1,p2_buff,nor_in1,nor_in2,nor_in3,nor_in4,next_n,next_basePixAddr,size);
						store_norout(fm_out2,nor_out1,nor_out2,nor_out3,nor_out4,n,basePixAddr,size);
						pingpong = true;
					}
					n = next_n;
					basePixAddr = next_basePixAddr;
					next_norblock(n, basePixAddr,next_n, next_basePixAddr,size);

					if(next_n >= nor_channel)
						break;
				}
				if(pingpong){
					store_norout(fm_out1,nor_out1,nor_out2,nor_out3,nor_out4,n,basePixAddr,size);
				}
				else{
					store_norout(fm_out2,nor_out1,nor_out2,nor_out3,nor_out4,n,basePixAddr,size);
				}
			}
