#include "deepwise_conv.h"

#define Tr 8  //height
#define Tc 16  //width
#define Tn 4

#define K 3
#define P 1
#define S 1

#define MAX_LEN 1024

#define TRin Tr+K-1
#define TCin Tc+K-1

//data_t dc_leaky_relu(data_t x){
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

void load_dcin(data_t fm_in_buff[Tn][TRin][TCin],
               data_t* in1, data_t* in2, data_t* in3, data_t* in4,
			   unsigned short n, unsigned short fm_row, unsigned short fm_col,
			   unsigned short input_row, unsigned input_col
                ){
                    unsigned short rr, cc;
                    data_t tmp1, tmp2, tmp3, tmp4;
                    int size = input_row*input_col;
                    int base_addr = n*size+(fm_row-P)*input_col+(fm_col-P);

                   	for(rr = 0; rr < TRin; rr++){
                   		for(cc = 0; cc < TCin; cc++){
#pragma HLS PIPELINE II=1
                   			int rr_offset = rr*input_col;
#pragma HLS RESOURCE variable=rr_offset core=Mul_LUT

                   			tmp1 = *(in1+base_addr+rr_offset+cc);
                   			tmp2 = *(in2+base_addr+size+rr_offset+cc);
                   			tmp3 = *(in3+base_addr+2*size+rr_offset+cc);
                   			tmp4 = *(in4+base_addr+3*size+rr_offset+cc);
                   			if(fm_row+rr >= P && fm_row+rr < input_row+P && fm_col+cc >= P && fm_col+cc < input_col+P){
                   				fm_in_buff[0][rr][cc] = tmp1;
                   				fm_in_buff[1][rr][cc] = tmp2;
                   				fm_in_buff[2][rr][cc] = tmp3;
                   				fm_in_buff[3][rr][cc] = tmp4;
			            	}
                   			else{
                   				fm_in_buff[0][rr][cc] = (data_t)0;
                   				fm_in_buff[1][rr][cc] = (data_t)0;
                   				fm_in_buff[2][rr][cc] = (data_t)0;
                   				fm_in_buff[3][rr][cc] = (data_t)0;
			            	}
			            }
                   	}
				}

void load_dcw(data_t wt_buff[Tn][K][K],
              data_t* w1, data_t* w2, data_t* w3, data_t* w4,
              unsigned short n,
			  unsigned short ch_in
              ){
                    unsigned short k;
                    unsigned short ch_in_kk = K*K;
#pragma HLS RESOURCE variable=ch_in_kk core=Mul_LUT

                    int base_addr = n*K*K;
                    for(k = 0; k < K*K; k++)
                    {
#pragma HLS PIPELINE II=1
                    	unsigned short i = (k%(K*K))/K; // row
                    	unsigned short j = k%K;         // col
                    	wt_buff[0][i][j] = *(w1+base_addr+k);
                    	wt_buff[1][i][j] = *(w2+base_addr+ch_in_kk+k);
                    	wt_buff[2][i][j] = *(w3+base_addr+2*ch_in_kk+k);
                    	wt_buff[3][i][j] = *(w4+base_addr+3*ch_in_kk+k);
                    }
                }

void load_dcb(data_t fm_out_buff[Tn][Tr][Tc],
              data_t bias_buff[MAX_LEN/Tn][Tn],
			  unsigned short n,
			  bool use_bias
              ){
                int rr, cc, mm;
                for(cc = 0; cc < Tc; cc++){
#pragma HLS PIPELINE II=1
                	for (rr = 0; rr < Tr; rr++){
#pragma HLS UNROLL
    		            for(mm = 0; mm < Tn; mm++){
//    		            	if(use_bias)
////    		            		bias_buff[n/Tn][mm]
//    		            		fm_out_buff[mm][rr][cc]=(data_t)0;
//    		            	else
    		            	fm_out_buff[mm][rr][cc] = (data_t)0;
			            }
                    }
                }
            }

void compute_dc(data_t fm_in_buff[Tn][TRin][TCin],
			 data_t fm_out_buff[Tn][Tr][Tc],
			 data_t wt_buff[Tn][K][K]
			 ){

                unsigned short kx, ky, rr, cc, nn;
                for(ky = 0; ky < K; ky++)
                	for(cc = 0; cc < Tc; cc++){
#pragma HLS PIPELINE II=1
                		for(rr = 0; rr < Tr; rr++)
                			for(kx = 0; kx < K; kx++)
                				for(nn = 0; nn < Tn; nn++)
                				{
    						        data_t mult = fm_in_buff[nn][rr+kx][cc+ky]*wt_buff[nn][kx][ky];
    						        data_t psum = mult+fm_out_buff[nn][rr][cc];
                                    fm_out_buff[nn][rr][cc] = psum;
    					        }
    			            }
				}

void store_dcout(data_t fm_out_buff[Tn][Tr][Tc],
                 data_t* out1, data_t* out2, data_t* out3, data_t* out4,
				 unsigned short fm_row, unsigned short fm_col, unsigned short ti,
				 unsigned short input_row, unsigned input_col,
				 unsigned short ch_in, bool act
                ){
                    unsigned short rr, cc;
                    unsigned short output_row = input_row;
                    unsigned short output_col = input_col;
                    int o_size = output_col*output_row;
//                    data_t tmp1,tmp2,tmp3,tmp4;
                    for(rr = 0; rr < Tr; rr++)
                    	for(cc = 0; cc < Tc; cc++)
                    	{
#pragma HLS PIPELINE II=1
//                    		if(act){
//                    			tmp1=dc_leaky_relu(fm_out_buff[0][rr][cc]);
//                    			tmp2=dc_leaky_relu(fm_out_buff[1][rr][cc]);
//                    			tmp3=dc_leaky_relu(fm_out_buff[2][rr][cc]);
//                    			tmp4=dc_leaky_relu(fm_out_buff[3][rr][cc]);
//                    		}
//                    		else{
//                    			tmp1=fm_out_buff[0][rr][cc];
//                    			tmp2=fm_out_buff[1][rr][cc];
//                    			tmp3=fm_out_buff[2][rr][cc];
//                    			tmp4=fm_out_buff[3][rr][cc];
//                    		}
                    		int base_addr = ti*o_size+(fm_row+rr)*output_col+fm_col;
                    		*(out1+base_addr+cc) = fm_out_buff[0][rr][cc];
                    		*(out2+base_addr+o_size+cc) = fm_out_buff[1][rr][cc];
                    		*(out3+base_addr+2*o_size+cc) = fm_out_buff[2][rr][cc];
                    		*(out4+base_addr+3*o_size+cc) = fm_out_buff[3][rr][cc];

                    	}
                    }

void compute_dcout(data_t fm_out_buff[Tn][Tr][Tc], data_t bias_buff[MAX_LEN/Tn][Tn],
				   data_t* in1, data_t* in2, data_t* in3, data_t* in4,
				   data_t* w1, data_t* w2, data_t* w3, data_t* w4,
				   unsigned short n, unsigned short input_row, unsigned input_col, unsigned short ch_in,
				   unsigned short fm_row, unsigned short fm_col, bool use_bias){

                    data_t fm_in_buff[Tn][TRin][TCin];
                    #pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=1
					#pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=2

	                data_t wt_buff[Tn][K][K];
                    #pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=1
					#pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=2


	                unsigned short trigger = -1;

	                load_dcb(fm_out_buff,bias_buff,n,use_bias);
                    load_dcin(fm_in_buff,in1,in2,in3,in4,n,fm_row,fm_col,input_row,input_col);
                    if(trigger != n){
                    	trigger = n;
                    	load_dcw(wt_buff,w1,w2,w3,w4,n,ch_in);
                    }
                    compute_dc(fm_in_buff,fm_out_buff,wt_buff);
                }



void next_dcblock(unsigned short r, unsigned short c, unsigned short n,
				  unsigned short &next_r, unsigned short &next_c, unsigned short &next_n,
				  unsigned short input_row, unsigned input_col){
    				if(c+Tc >= input_col){
    					if(r+Tr >= input_row){
    						next_n = n+Tn;
    						next_r = 0;
    						next_c = 0;
    					}
    					else{
    						next_n = n;
    						next_r = r+Tr;
    						next_c = 0;
    					}
    				}
    				else{
    					next_n = n;
    					next_r = r;
    					next_c = c+Tc;
    				}
				}

//top function
void deepwise_conv(data_t* dc_in1, data_t* dc_in2, data_t* dc_in3, data_t* dc_in4,
              	   data_t* dc_w1, data_t* dc_w2, data_t* dc_w3, data_t* dc_w4,
				   data_t* dc_bias,
				   data_t* dc_out1, data_t* dc_out2, data_t* dc_out3, data_t* dc_out4,
				   unsigned short dc_ch_in,
				   unsigned short dc_input_row, unsigned dc_input_col, bool dc_act, bool dc_use_bias
            	){

                data_t bias_buff[MAX_LEN/Tn][Tn];
//                #pragma HLS ARRAY_PARTITION variable=bias_buff complete dim=2

                data_t fm_out1[Tn][Tr][Tc];
                #pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=1
				#pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=2
                data_t fm_out2[Tn][Tr][Tc];
				#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=1
				#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=2
                bool pingpong;

                unsigned short r, c, n;
                unsigned short next_r, next_c, next_n;

                r = 0;
                c = 0;
                n = 0;
                pingpong = true;
//                if(dc_use_bias)
//                	memcpy((data_t*)bias_buff,(const data_t*)dc_bias,sizeof(data_t)*dc_ch_in);
                compute_dcout(fm_out1,bias_buff,dc_in1,dc_in2,dc_in3,dc_in4,dc_w1,dc_w2,dc_w3,dc_w4,n,dc_input_row,dc_input_col,dc_ch_in,r,c,dc_use_bias);
                next_dcblock(r,c,n,next_r,next_c,next_n,dc_input_row,dc_input_col);
                while(true){
				#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
    	            if(pingpong){
    		            compute_dcout(fm_out2,bias_buff,dc_in1,dc_in2,dc_in3,dc_in4,dc_w1,dc_w2,dc_w3,dc_w4,next_n,dc_input_row,dc_input_col,dc_ch_in,next_r,next_c,dc_use_bias);
    		            store_dcout(fm_out1,dc_out1,dc_out2,dc_out3,dc_out4,r,c,n,dc_input_row,dc_input_col,dc_ch_in,dc_act);
    		            pingpong = false;
    	            }
    	            else{
    		            compute_dcout(fm_out1,bias_buff,dc_in1,dc_in2,dc_in3,dc_in4,dc_w1,dc_w2,dc_w3,dc_w4,next_n,dc_input_row,dc_input_col,dc_ch_in,next_r,next_c,dc_use_bias);
			            store_dcout(fm_out2,dc_out1,dc_out2,dc_out3,dc_out4,r,c,n,dc_input_row,dc_input_col,dc_ch_in,dc_act);
			            pingpong = true;
    	            }
    	            n = next_n;
    	            r = next_r;
    	            c = next_c;
    	            next_dcblock(r,c,n,next_r,next_c,next_n,dc_input_row,dc_input_col);
		            if(next_n >= dc_ch_in)
		                 break;
                    }

                if(pingpong){
    	            store_dcout(fm_out1,dc_out1,dc_out2,dc_out3,dc_out4,r,c,n,dc_input_row,dc_input_col,dc_ch_in,dc_act);
                }
                else{
    	            store_dcout(fm_out2,dc_out1,dc_out2,dc_out3,dc_out4,r,c,n,dc_input_row,dc_input_col,dc_ch_in,dc_act);
                }
            }

