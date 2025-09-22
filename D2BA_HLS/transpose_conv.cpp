#include "transpose_conv.h"

#define Tr 16    //height
#define Tc 16    //width 64
#define Tn 4
#define Tm 32

#define K 2
#define P 0
#define S 2

#define MAX_LEN 64

#define TRin (Tr-K)/2+1
#define TCin (Tc-K)/2+1

//data_t tc_leaky_relu(data_t x){
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

void load_tcin(data_t fm_in_buff[Tn][TRin][TCin],
               data_t* in1, data_t* in2, data_t* in3, data_t* in4,
			   unsigned short n, unsigned short fm_row, unsigned short fm_col,
			   unsigned short input_row, unsigned input_col
			 ){
                    unsigned short nn, rr, cc;
                    data_t tmp1, tmp2, tmp3, tmp4;
                    int size = input_row*input_col;
                    int base_addr = n*size+(fm_row/2-P)*input_col+(fm_col/2-P);

                    for(rr = 0; rr < TRin; rr++){
			            for(cc = 0; cc < TCin; cc++){
#pragma HLS PIPELINE II=1
				            int rr_offset = rr*input_col;
#pragma HLS RESOURCE variable=rr_offset core=Mul_LUT
			                tmp1 = *(in1+base_addr+rr_offset+cc);
				            tmp2 = *(in2+base_addr+size+rr_offset+cc);
				            tmp3 = *(in3+base_addr+2*size+rr_offset+cc);
				            tmp4 = *(in4+base_addr+3*size+rr_offset+cc);

				            if(fm_row/2+rr >= P && fm_row/2+rr < input_row+P && fm_col/2+cc >= P && fm_col/2+cc < input_col+P){
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

void load_tcw(data_t wt_buff[Tm][Tn][K][K],
              data_t* w1, data_t* w2, data_t* w3, data_t* w4,
			  unsigned short n, unsigned short m,
			  unsigned short ch_in, unsigned short ch_out
			){
                    unsigned short k;
	                int ch_out_kk = ch_out*K*K;
                    int base_addr = m*K*K + n*ch_out_kk;
#pragma HLS RESOURCE variable=ch_in_kk core=Mul_LUT

                    	for(k=0; k < Tm*K*K; k++)
                    	{
#pragma HLS PIPELINE II=1
                    		unsigned short mm = k/(K*K);    // channel
                    		unsigned short i = (k%(K*K))/K; // row
                    		unsigned short j = k%K;         // col
                    		wt_buff[mm][0][i][j] = *(w1+base_addr+k);
                    		wt_buff[mm][1][i][j] = *(w2+base_addr+ch_out_kk+k);
                    		wt_buff[mm][2][i][j] = *(w3+base_addr+2*ch_out_kk+k);
                    		wt_buff[mm][3][i][j] = *(w4+base_addr+3*ch_out_kk+k);
                    	}
                }

void load_tcb(data_t fm_out_buff[Tm][Tr][Tc],
              data_t bias_buff[MAX_LEN/Tm][Tm],
			  unsigned short m,
			  bool use_bias
              ){
                int rr, cc, mm;
                for(mm = 0; mm < Tm; mm++){
                	for(cc = 0; cc < Tc; cc++){
#pragma HLS PIPELINE II=1
                		for (rr = 0; rr < Tr; rr++){
#pragma HLS UNROLL
    		            	if(use_bias)
    		            		fm_out_buff[mm][rr][cc] = bias_buff[m/Tm][mm];
    		            	else
    		            		fm_out_buff[mm][rr][cc] = (data_t)0;
			            }
                    }
                }
            }

void compute_tc(data_t fm_in_buff[Tn][TRin][TCin],
			 	data_t fm_out_buff[Tm][Tr][Tc],
				data_t wt_buff[Tm][Tn][K][K]
			 	){

                unsigned short kx, ky, rr, cc, nn, mm;
                for(mm = 0; mm < Tm; mm++)
                	for(cc = 0; cc < Tc; cc++){
#pragma HLS PIPELINE II=1
                		for(rr = 0; rr < Tr; rr++){
    					    for(nn = 0; nn < Tn ;nn++){
    						    data_t mult = fm_in_buff[nn][int(rr*0.5)][int(cc*0.5)]*wt_buff[mm][nn][rr%2][cc%2];
    						    data_t psum = mult+fm_out_buff[mm][rr][cc];
                                fm_out_buff[mm][rr][cc] = psum;
    					    }
    				    }
    			    }
				}

void compute_tcout(data_t fm_out_buff[Tm][Tr][Tc],
				   data_t bias_buff[MAX_LEN/Tm][Tm],
				   data_t* in1, data_t* in2, data_t* in3, data_t* in4,
				   data_t* w1, data_t* w2, data_t* w3, data_t* w4,
				   unsigned short m, unsigned short input_row, unsigned input_col, unsigned short ch_in, unsigned short ch_out,
				   unsigned short fm_row, unsigned short fm_col, bool use_bias){

                    data_t fm_in_buff1[Tn][TRin][TCin];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=fm_in_buff1 complete dim=2

	                data_t fm_in_buff2[Tn][TRin][TCin];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=fm_in_buff2 complete dim=2

	                data_t wt_buff1[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=3

	                data_t wt_buff2[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=3

	                unsigned short ti = 0;
	                load_tcb(fm_out_buff,bias_buff,m,use_bias);
                    load_tcin(fm_in_buff1,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col);
                    load_tcw(wt_buff1,w1,w2,w3,w4,ti,m,ch_in,ch_out);

                    bool pingpong = true;
                    for(ti = Tn; ti < ch_in; ti+=Tn){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
                        if (pingpong)
                        {
                            load_tcin(fm_in_buff2,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col);
                            load_tcw(wt_buff2,w1,w2,w3,w4,ti,m,ch_in,ch_out);
                            compute_tc(fm_in_buff1,fm_out_buff,wt_buff1);
                            pingpong = false;
                        }
                        else{
                            load_tcin(fm_in_buff1,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col);
                            load_tcw(wt_buff1,w1,w2,w3,w4,ti,m,ch_in,ch_out);
                            compute_tc(fm_in_buff2,fm_out_buff,wt_buff2);
                            pingpong = true;
                        }
                    }
                    if(pingpong){
		                compute_tc(fm_in_buff1,fm_out_buff,wt_buff1);
	                }
	                else{
		                compute_tc(fm_in_buff2,fm_out_buff,wt_buff2);
	                }
                }

void store_tcout(data_t fm_out_buff[Tm][Tr][Tc],
                 data_t* out1, data_t* out2, data_t* out3, data_t* out4,
				 unsigned short fm_row, unsigned short fm_col, unsigned short m,
				 unsigned short input_row, unsigned input_col,
				 unsigned short ch_out, bool act
                ){
                    unsigned short mm, rr, cc;
                    unsigned short output_row = input_row*2;
                    unsigned short output_col = input_col*2;
                    int o_size = output_col*output_row;
//                    data_t tmp1,tmp2,tmp3,tmp4;
                    for(mm = 0; mm < Tm; mm++)
    	                for(rr = 0; rr < Tr; rr+=4)
    		                for(cc = 0; cc < Tc; cc++)
                            {
#pragma HLS PIPELINE II=1
//    		                	if(act){
//    		                		tmp1=tc_leaky_relu(fm_out_buff[mm][rr][cc]);
//    		                		tmp2=tc_leaky_relu(fm_out_buff[mm][rr+1][cc]);
//    		                		tmp3=tc_leaky_relu(fm_out_buff[mm][rr+2][cc]);
//    		                		tmp4=tc_leaky_relu(fm_out_buff[mm][rr+3][cc]);
//    		                	}
//    		                	else{
//    		                		tmp1=fm_out_buff[mm][rr][cc];
//    		                		tmp2=fm_out_buff[mm][rr+1][cc];
//    		                		tmp3=fm_out_buff[mm][rr+2][cc];
//    		                		tmp4=fm_out_buff[mm][rr+3][cc];
//    		                	}
    		                	int base_addr = (m+mm)*o_size+(fm_row+rr)*output_col+fm_col;
    		                	*(out1+base_addr+cc) = fm_out_buff[mm][rr][cc];
    		                	*(out2+base_addr+output_col+cc) = fm_out_buff[mm][rr+1][cc];
    		                	*(out3+base_addr+2*output_col+cc) = fm_out_buff[mm][rr+2][cc];
    		                	*(out4+base_addr+3*output_col+cc) = fm_out_buff[mm][rr+3][cc];
                            }
                		}

void next_tcblock(unsigned short r, unsigned short c, unsigned short m,
				  unsigned short &next_r, unsigned short &next_c, unsigned short &next_m,
				  unsigned short out_row, unsigned out_col){
    				if(c+Tc >= out_col){
    					if(r+Tr >= out_row){
    						next_m = m+Tm;
    						next_r = 0;
    						next_c = 0;
    					}
    					else{
    						next_m = m;
    						next_r = r+Tr;
    						next_c = 0;
    					}
    				}
    				else{
    					next_m = m;
    					next_r = r;
    					next_c = c+Tc;
    				}
				}

//top function
void transpose_conv(data_t* tc_in1, data_t* tc_in2, data_t* tc_in3, data_t* tc_in4,
              	  	data_t* tc_w1, data_t* tc_w2, data_t* tc_w3, data_t* tc_w4,
					data_t* tc_bias,
					data_t* tc_out1, data_t* tc_out2, data_t* tc_out3, data_t* tc_out4,
					unsigned short tc_ch_in, unsigned short tc_ch_out,
					unsigned short tc_input_row, unsigned tc_input_col, bool tc_act, bool tc_use_bias
            	){

                data_t bias_buff[MAX_LEN/Tm][Tm];
#pragma HLS ARRAY_PARTITION variable=bias_buff complete dim=2

                data_t fm_out1[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=2
                data_t fm_out2[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=2
                bool pingpong;

                unsigned short r, c, m;
                unsigned short next_r, next_c, next_m;
	            unsigned short out_row = tc_input_row*2;
                unsigned short out_col = tc_input_col*2;

                r = 0;
                c = 0;
                m = 0;
                pingpong = true;
                if(tc_use_bias)
                	memcpy((data_t*)bias_buff,(const data_t*)tc_bias,sizeof(data_t)*MAX_LEN);
                compute_tcout(fm_out1,bias_buff,tc_in1,tc_in2,tc_in3,tc_in4, tc_w1,tc_w2,tc_w3,tc_w4,m,tc_input_row,tc_input_col,tc_ch_in,tc_ch_out,r,c,tc_use_bias);
                next_tcblock(r,c,m,next_r,next_c,next_m,out_row,out_col);
                while(true){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
    	            if(pingpong){
    		            compute_tcout(fm_out2,bias_buff,tc_in1,tc_in2,tc_in3,tc_in4,tc_w1,tc_w2,tc_w3,tc_w4,next_m,tc_input_row,tc_input_col,tc_ch_in,tc_ch_out,next_r,next_c,tc_use_bias);
    		            store_tcout(fm_out1,tc_out1,tc_out2,tc_out3,tc_out4,r,c,m,tc_input_row,tc_input_col,tc_ch_out,tc_act);
    		            pingpong = false;
    	            }
    	            else{
    		            compute_tcout(fm_out1,bias_buff,tc_in1,tc_in2,tc_in3,tc_in4,tc_w1,tc_w2,tc_w3,tc_w4,next_m,tc_input_row,tc_input_col,tc_ch_in,tc_ch_out,next_r,next_c,tc_use_bias);
			            store_tcout(fm_out2,tc_out1,tc_out2,tc_out3,tc_out4,r,c,m,tc_input_row,tc_input_col,tc_ch_out,tc_act);
			            pingpong = true;
    	            }
    	            m = next_m;
    	            r = next_r;
    	            c = next_c;
    	            next_tcblock(r,c,m,next_r,next_c,next_m,out_row,out_col);
		            if(next_m >= tc_ch_out)
		                 break;
                    }
                if(pingpong){
    	            store_tcout(fm_out1,tc_out1,tc_out2,tc_out3,tc_out4,r,c,m,tc_input_row,tc_input_col,tc_ch_out,tc_act);
                }
                else{
    	            store_tcout(fm_out2,tc_out1,tc_out2,tc_out3,tc_out4,r,c,m,tc_input_row,tc_input_col,tc_ch_out,tc_act);
                }
            }


