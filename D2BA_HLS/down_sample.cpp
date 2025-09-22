#include "down_sample.h"

#define Tr 16	 //height
#define Tc 16   //width
#define Tn 4
#define Tm 32

#define K 4
#define P 1
#define S 2

#define MAX_LEN 512

#define TRin (Tr-1)*2+K
#define TCin (Tc-1)*2+K

//data_t ds_leaky_relu(data_t x){
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

void load_dsin(data_t fm_in_buff[Tn][TRin][TCin],
               data_t* in1, data_t* in2, data_t* in3, data_t* in4,
			   unsigned short n, unsigned short fm_row, unsigned short fm_col,
			   unsigned short input_row, unsigned input_col
               ){
                    unsigned short rr, cc;
                    data_t tmp1, tmp2, tmp3, tmp4;
                    int size = input_row*input_col;
                    int base_addr = n*size+(fm_row*2-P)*input_col+(fm_col*2-P);

                    for(rr = 0; rr < TRin; rr++){
			            for(cc = 0; cc < TCin; cc++){
#pragma HLS PIPELINE II=1
				            int rr_offset = rr*input_col;
#pragma HLS RESOURCE variable=rr_offset core=Mul_LUT

			                tmp1 = *(in1+base_addr+rr_offset+cc);
				            tmp2 = *(in2+base_addr+size+rr_offset+cc);
				            tmp3 = *(in3+base_addr+2*size+rr_offset+cc);
				            tmp4 = *(in4+base_addr+3*size+rr_offset+cc);
				            if(fm_row*2+rr >= P && fm_row*2+rr < input_row+P && fm_col*2+cc >= P && fm_col*2+cc < input_col+P){
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

void load_dsw(data_t wt_buff[Tm][Tn][K][K],
              data_t* w1,
			  unsigned short n, unsigned short m,
			  unsigned short ch_in, unsigned short ch_out
              ){
                    unsigned short mm, k;
	                int ch_in_kk = ch_in*K*K;
                    int base_addr = m*ch_in_kk + n*K*K;
#pragma HLS RESOURCE variable=ch_in_kk core=Mul_LUT

                    for(mm = 0; mm < Tm; mm++){
                        for(k = 0; k < Tn*K*K; k++)
                        {
#pragma HLS PIPELINE II=1
                            wt_buff[mm][k/(K*K)][(k%(K*K))/K][k%K]=*(w1+base_addr+mm*ch_in_kk+k);
                        }
                    }
                }

void load_dsb(data_t fm_out_buff[Tm][Tr][Tc],
              data_t bias_buff[MAX_LEN/Tm][Tm],
			  unsigned short m,
			  bool use_bias
              ){
                int rr, cc, mm;
                for (rr = 0; rr < Tr; rr++){
                	for(cc = 0; cc < Tc; cc++){
#pragma HLS PIPELINE II=1
                		for(mm=0; mm<Tm; mm++){
#pragma HLS UNROLL
//    		            	if(use_bias)
////    		            		bias_buff[m/Tm][mm]
//    		            		fm_out_buff[mm][rr][cc]=(data_t)0;
//    		            	else
    		            	fm_out_buff[mm][rr][cc]=(data_t)0;
			            }
                    }
                }
            }

void compute_ds(data_t fm_in_buff[Tn][TRin][TCin],
			 	data_t fm_out_buff[Tm][Tr][Tc],
				data_t wt_buff[Tm][Tn][K][K]
			 	 ){

                unsigned short kx, ky, rr, cc, mm;
                for(kx = 0; kx < K; kx++)
                	for(ky = 0;ky < K; ky++)
                		for(rr = 0;rr < Tr; rr++)
                			for(cc = 0; cc < Tc; cc++){
#pragma HLS PIPELINE II=1
                				for(mm = 0; mm < Tm; mm++){
                					data_t mult1 = fm_in_buff[0][rr*S+kx][cc*S+ky]*wt_buff[mm][0][kx][ky];
                					data_t mult2 = fm_in_buff[1][rr*S+kx][cc*S+ky]*wt_buff[mm][1][kx][ky];
                					data_t mult3 = fm_in_buff[2][rr*S+kx][cc*S+ky]*wt_buff[mm][2][kx][ky];
                					data_t mult4 = fm_in_buff[3][rr*S+kx][cc*S+ky]*wt_buff[mm][3][kx][ky];
                					data_t sum1 = mult1+mult2;
                					data_t sum2 = mult3+mult4;
                					data_t sum = sum1+sum2;
                					data_t psum = sum+fm_out_buff[mm][rr][cc];
                					fm_out_buff[mm][rr][cc] = psum;
    				            }
    			            }
						}

void compute_dsout(data_t fm_out_buff[Tm][Tr][Tc],
				   data_t bias_buff[MAX_LEN/Tm][Tm],
				   data_t* in1, data_t* in2, data_t* in3, data_t* in4,
				   data_t* w1,
				   unsigned short m, unsigned short input_row, unsigned input_col, unsigned short ch_in, unsigned short ch_out,
				   unsigned short fm_row, unsigned short fm_col, bool use_bias){

                    data_t fm_in_buff1[Tn][TRin][TCin];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff1 complete dim=1
	                data_t fm_in_buff2[Tn][TRin][TCin];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff2 complete dim=1

	                data_t wt_buff1[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=2
	                data_t wt_buff2[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=2

	                unsigned short ti = 0;
	                load_dsb(fm_out_buff,bias_buff,m,use_bias);
                    load_dsin(fm_in_buff1,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col);
                    load_dsw(wt_buff1,w1,ti,m,ch_in,ch_out);
                    bool pingpong = true;

                    for(ti = Tn; ti < ch_in; ti+=Tn){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
                        if (pingpong)
                        {
                            load_dsin(fm_in_buff2,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col);
                            load_dsw(wt_buff2,w1,ti,m,ch_in,ch_out);
                            compute_ds(fm_in_buff1,fm_out_buff,wt_buff1);
                            pingpong=false;
                        }
                        else{
                            load_dsin(fm_in_buff1,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col);
                            load_dsw(wt_buff1,w1,ti,m,ch_in,ch_out);
                            compute_ds(fm_in_buff2,fm_out_buff,wt_buff2);
                            pingpong=true;
                        }
                    }
                    if(pingpong){
		                compute_ds(fm_in_buff1,fm_out_buff,wt_buff1);
	                }
	                else{
		                compute_ds(fm_in_buff2,fm_out_buff,wt_buff2);
	                }
                }

void store_dsout(data_t fm_out_buff[Tm][Tr][Tc],
                 data_t* out1,
				 unsigned short fm_row, unsigned short fm_col, unsigned short m,
				 unsigned short input_row, unsigned input_col,
				 unsigned short ch_out, bool act
                ){
                    unsigned short mm, rr, cc;
                    unsigned short output_row = input_row/2;
                    unsigned short output_col = input_col/2;
                    int o_size = output_col*output_row;
//                    data_t tmp1,tmp2,tmp3,tmp4;
                    for(mm = 0; mm < Tm; mm++)
    	                for(rr = 0; rr < Tr; rr++)
    		                for(cc = 0; cc < Tc; cc++)
                            {
#pragma HLS PIPELINE II=1
//                                if(act){
//    			    	            tmp1=ds_leaky_relu(fm_out_buff[mm][rr][cc]);
//    			    	            tmp2=ds_leaky_relu(fm_out_buff[mm+1][rr][cc]);
//    			    	           	tmp3=ds_leaky_relu(fm_out_buff[mm+2][rr][cc]);
//    			    	           	tmp4=ds_leaky_relu(fm_out_buff[mm+3][rr][cc]);
//    			                }
//    			                else{
//    			    	            tmp1=fm_out_buff[mm][rr][cc];
//    			    	           	tmp2=fm_out_buff[mm+1][rr][cc];
//    			    	           	tmp3=fm_out_buff[mm+2][rr][cc];
//    			    	           	tmp4=fm_out_buff[mm+3][rr][cc];
//    			                }
    		                	*(out1+(m+mm)*o_size+(fm_row+rr)*output_col+fm_col+cc)=fm_out_buff[mm][rr][cc];
                            }
				}

void next_dsblock(unsigned short r, unsigned short c, unsigned short m,
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
void down_sample(data_t* ds_in1, data_t* ds_in2, data_t* ds_in3, data_t* ds_in4,
                 data_t* ds_w1,
				 data_t* ds_bias,
				 data_t* ds_out1,
				 unsigned short ds_ch_in, unsigned short ds_ch_out,
				 unsigned short ds_input_row, unsigned short ds_input_col, bool ds_act, bool ds_use_bias
				){

                data_t bias_buff[MAX_LEN/Tm][Tm];
//                #pragma HLS ARRAY_PARTITION variable=bias_buff complete dim=2

                data_t fm_out1[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=1
                data_t fm_out2[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=1

                bool pingpong;
                unsigned short r,c,m;
                unsigned short next_r,next_c,next_m;
	            unsigned short out_row = ds_input_row/2;
                unsigned short out_col = ds_input_col/2;

                r = 0;
                c = 0;
                m = 0;
                pingpong = true;
//                if(ds_use_bias)
//                	memcpy((data_t*)bias_buff,(const data_t*)ds_bias,sizeof(data_t)*ds_ch_out);
                compute_dsout(fm_out1,bias_buff,ds_in1,ds_in2,ds_in3,ds_in4,ds_w1,m,ds_input_row,ds_input_col,ds_ch_in,ds_ch_out,r,c,ds_use_bias);
                next_dsblock(r,c,m,next_r,next_c,next_m,out_row,out_col);
                while(true){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
    	            if(pingpong){
    		            compute_dsout(fm_out2,bias_buff,ds_in1,ds_in2,ds_in3,ds_in4,ds_w1,next_m,ds_input_row,ds_input_col,ds_ch_in,ds_ch_out,next_r,next_c,ds_use_bias);
    		            store_dsout(fm_out1,ds_out1,r,c,m,ds_input_row,ds_input_col,ds_ch_out,ds_act);
    		            pingpong = false;
    	            }
    	            else{
    		            compute_dsout(fm_out1,bias_buff,ds_in1,ds_in2,ds_in3,ds_in4,ds_w1,next_m,ds_input_row,ds_input_col,ds_ch_in,ds_ch_out,next_r,next_c,ds_use_bias);
			            store_dsout(fm_out2,ds_out1,r,c,m,ds_input_row,ds_input_col,ds_ch_out,ds_act);
			            pingpong = true;
    	            }
    	            m = next_m;
    	            r = next_r;
    	            c = next_c;
    	            next_dsblock(r,c,m,next_r,next_c,next_m,out_row,out_col);
		            if(next_m >= ds_ch_out)
		                 break;
                    }
                if(pingpong){
    	            store_dsout(fm_out1,ds_out1,r,c,m,ds_input_row,ds_input_col,ds_ch_out,ds_act);
                }
                else{
    	            store_dsout(fm_out2,ds_out1,r,c,m,ds_input_row,ds_input_col,ds_ch_out,ds_act);
                }
            }
