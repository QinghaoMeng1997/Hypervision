#include "pointwise_conv.h"

#define Tr 256//height
#define Th 4 //hidden
#define Tc 16 //width
#define MAX_LEN 1024

//data_t pc_leaky_relu(data_t x){
//    const data_t alpha = (data_t)0.1;
//    if (x >= (data_t)0 )
//    {
//        return x;
//    }
//    else
//    {
//        return x*alpha;
//    }
//}

void load_pcin(data_t fm_in_buff[Th][Tr],
               data_t* in1, data_t* in2, data_t* in3, data_t* in4,
			   int n, int basePixAddr,
			   int input_row
                ){
                    int i;
                    int base_addr = basePixAddr+n*input_row;
                    for (i = 0; i < Tr; i++)
                    {
#pragma HLS PIPELINE II=1
                    	fm_in_buff[0][i]=*(in1+base_addr+i);
                    	fm_in_buff[1][i]=*(in2+base_addr+i+input_row);
                    	fm_in_buff[2][i]=*(in3+base_addr+i+2*input_row);
                    	fm_in_buff[3][i]=*(in4+base_addr+i+3*input_row);
                    }   
                }

void load_pcw(data_t wt_buff[Tc][Th],
              data_t* w1,
			  int n, int m,
			  int input_col, int output_col
                 ){
                    int i, j;
                    for (i = 0; i < Tc; i++)
                    {
                    	for(j = 0; j < Th; j++){
#pragma HLS PIPELINE II=1
                    		wt_buff[i][j]=*(w1+(m+i)*input_col+n+j);
                    	}
                    }
                 }

void load_pcb(data_t fm_out_buff[Tc][Tr],
              data_t bias_buff[MAX_LEN/Tc][Tc],
               int m,bool use_bias
               ){
#pragma HLS INLINE off
                    int i, j;
                    for (i = 0; i < Tr; i++)
                    {
#pragma HLS PIPELINE II=1
                    	for (j = 0; j < Tc; j++){
//                        	if(use_bias)
////                        		bias_buff[m/Tc][j]
//                        		fm_out_buff[j][i] =	(data_t)0;
//                        	else
                        	fm_out_buff[j][i] = (data_t)0;
                        } 
                    }
                }

void compute_pc(data_t fm_in_buff[Th][Tr],
             	data_t fm_out_buff[Tc][Tr],
				data_t wt_buff[Tc][Th]
            	){
                int i, j;

                for (i = 0; i < Tr; i++)
               {
#pragma HLS PIPELINE II=1
                	for (j = 0; j < Tc; j++){
#pragma HLS UNROLL
                        data_t mult1=fm_in_buff[0][i]*wt_buff[j][0];
                        data_t mult2=fm_in_buff[1][i]*wt_buff[j][1];
                        data_t mult3=fm_in_buff[2][i]*wt_buff[j][2];
                        data_t mult4=fm_in_buff[3][i]*wt_buff[j][3];
                        data_t sum1=mult1+mult2;
                        data_t sum2=mult3+mult4;
                        data_t sum=sum1+sum2;
                        data_t accu=fm_out_buff[j][i];
                        fm_out_buff[j][i]=accu+sum;
                    }
                }
            }

void compute_pcout(data_t fm_out_buff[Tc][Tr],
                   data_t bias_buff[MAX_LEN/Tc][Tc],
				   data_t* in1, data_t* in2, data_t* in3, data_t* in4,
				   data_t* w1,
				   int m, int basePixAddr,
				   int input_col, int input_row, int output_col,
				   bool use_bias
                    ){
		                data_t fm_in_buff1[Th][Tr];
	                    data_t fm_in_buff2[Th][Tr];

	                    data_t wt_buff1[Tc][Th];
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=1
	                    data_t wt_buff2[Tc][Th];
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=1

		                int ti=0;
                        load_pcb(fm_out_buff,bias_buff,m,use_bias);
		                load_pcin(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,input_row);
		                load_pcw(wt_buff1,w1,ti,m,input_col,output_col);
		                bool pingpong=true;

                        for (ti=Th; ti < input_col; ti+=Th)
                        {
#pragma HLS LOOP_TRIPCOUNT min=31 max=31 avg=31
                            if(pingpong){          
				                load_pcin(fm_in_buff2,in1,in2,in3,in4,ti,basePixAddr,input_row);
				                load_pcw(wt_buff2,w1,ti,m,input_col,output_col);
				                compute_pc(fm_in_buff1,fm_out_buff,wt_buff1);
				                pingpong=false;
			                }
			                else{
				                load_pcin(fm_in_buff1,in1,in2,in3,in4,ti,basePixAddr,input_row);
				                load_pcw(wt_buff1,w1,ti,m,input_col,output_col);
				                compute_pc(fm_in_buff2,fm_out_buff,wt_buff2);
				                pingpong=true;
			                }    
                        }
                        if(pingpong)
				            compute_pc(fm_in_buff1,fm_out_buff,wt_buff1);
		                else
				            compute_pc(fm_in_buff2,fm_out_buff,wt_buff2);
                    }

void store_pcout(data_t fm_out_buff[Tc][Tr],
                 data_t* out1,
				 int basePixAddr, int m,
				 int input_row, int output_col,
				 bool act
                  ){
                    int i, j;
                    for (j = 0; j < Tc; j++)
                    {
                    	for (i = 0; i < Tr; i++){
#pragma HLS PIPELINE II=1
//                    		if(act)
//                    			*(out1+basePixAddr+i+(j+m)*input_row)=pc_leaky_relu(fm_out_buff[j][i]);
//                    		else
                    		*(out1+basePixAddr+i+(j+m)*input_row)=fm_out_buff[j][i];
                        }
                    }
				}

void next_pcblock(int input_row,
                  int basePixAddr, int baseFeature,
                  int &next_basePixAddr, int &next_baseFeature){
                    if(basePixAddr+Tr>=input_row)
                    {
    	                next_basePixAddr=0;
    	                next_baseFeature=baseFeature+Tc;

                    }
                    else{
    	                next_basePixAddr=basePixAddr+Tr;
    	                next_baseFeature=baseFeature;
                    } 
                }

void pointwise_conv(data_t* pc_in1, data_t* pc_in2, data_t* pc_in3, data_t* pc_in4,
            		data_t* pc_w1,
					data_t* pc_bias,
					data_t* pc_out1,
					int pc_input_row, int pc_input_col, int pc_output_col,
					bool act, bool pc_use_bias
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
//                if(pc_use_bias)
//                	memcpy((data_t*)bias_buff,(const data_t*)pc_bias,sizeof(data_t)*pc_output_col);
                pingpong=true;
                basePixAddr=0;
                baseFeature=0;
                next_basePixAddr=Tr;
                next_baseFeature=0;
                compute_pcout(fm_out1,bias_buff,pc_in1,pc_in2,pc_in3,pc_in4,pc_w1,baseFeature,basePixAddr,pc_input_col,pc_input_row,pc_output_col,pc_use_bias);
                while(true){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
    	            next_pcblock(pc_input_row,basePixAddr,baseFeature,next_basePixAddr,next_baseFeature);
    	            if(pingpong){
    		            compute_pcout(fm_out2,bias_buff,pc_in1,pc_in2,pc_in3,pc_in4,pc_w1,next_baseFeature,next_basePixAddr,pc_input_col,pc_input_row,pc_output_col,pc_use_bias);
    		            store_pcout(fm_out1,pc_out1,basePixAddr,baseFeature,pc_input_row,pc_output_col,act);
    		            pingpong=false;
    	            }
    	            else{
    		            compute_pcout(fm_out1,bias_buff,pc_in1,pc_in2,pc_in3,pc_in4,pc_w1,next_baseFeature,next_basePixAddr,pc_input_col,pc_input_row,pc_output_col,pc_use_bias);
    		            store_pcout(fm_out2,pc_out1,basePixAddr,baseFeature,pc_input_row,pc_output_col,act);
    		            pingpong=true;
    	            }
    	            basePixAddr=next_basePixAddr;
    	            baseFeature=next_baseFeature;

                    if(basePixAddr+Tr>=pc_input_row&&baseFeature+Tc>=pc_output_col)
    		            break;
                }
                if(pingpong){
		            store_pcout(fm_out1,pc_out1,basePixAddr,baseFeature,pc_input_row,pc_output_col,act);
                }
                else{
		            store_pcout(fm_out2,pc_out1,basePixAddr,baseFeature,pc_input_row,pc_output_col,act);
                }
            }
