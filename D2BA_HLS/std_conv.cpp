#include "std_conv.h"

#define Tr 16 //height
#define Tc 64 //width
#define Tn 4
#define Tm 32

#define K 3
#define P 1
#define S 1

#define MAX_LEN 512

#define TRin Tr+K-1
#define TCin Tc+K-1

//data_t sc_leaky_relu(data_t x){
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

void load_scin(data_t fm_in_buff[Tn][TRin][TCin],
               data_t* in1, data_t* in2, data_t* in3, data_t* in4,
               unsigned short n, unsigned short fm_row, unsigned short fm_col,
               unsigned short input_row, unsigned input_col, unsigned short ch_in
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

void load_scw(data_t wt_buff[Tm][Tn][K][K],
              data_t* w1, data_t* w2, data_t* w3, data_t* w4,
              unsigned short n, unsigned short m,
			  unsigned short ch_in, unsigned short ch_out
                 ){
                    unsigned short mm, k;
	                int ch_in_kk = ch_in*K*K;
                    int base_addr = m*ch_in_kk + n*K*K;
#pragma HLS RESOURCE variable=ch_in_kk core=Mul_LUT
                    unsigned short num = ((ch_out-m)<Tm)?(ch_out-m):Tm;

                    for(mm = 0; mm < num; mm++){
                        for(k = 0; k<K*K; k++)
                        {
#pragma HLS PIPELINE II=1
                        	unsigned short i = (k%(K*K))/K; // row
                        	unsigned short j = k%K;         // col
                        	wt_buff[mm][0][i][j] = *(w1+base_addr+mm*ch_in_kk+k);
                        	if(n+1 < ch_in)
                        		wt_buff[mm][1][i][j] = *(w2+base_addr+mm*ch_in_kk+9+k);
                        	if(n+2 < ch_in)
                        		wt_buff[mm][2][i][j] = *(w3+base_addr+mm*ch_in_kk+18+k);
                        	if(n+3 < ch_in)
                        		wt_buff[mm][3][i][j] = *(w4+base_addr+mm*ch_in_kk+27+k);
                        }
                    }
				}

void load_scb(data_t fm_out_buff[Tm][Tr][Tc],
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
//    		            	if(use_bias)
//    		            		bias_buff[m/Tm][mm]
//    		            		fm_out_buff[mm][rr][cc] = (data_t)0;
//    		            	else
    		            	fm_out_buff[mm][rr][cc] = (data_t)0;
			            }     
                    }
                }   
            }

void compute_sc(data_t fm_in_buff[Tn][TRin][TCin],
			 	data_t fm_out_buff[Tm][Tr][Tc],
				data_t wt_buff[Tm][Tn][K][K]
				){

                unsigned short kx, ky, rr, cc, nn, mm;
    	        for(ky = 0; ky < K; ky++)
    	        	for(mm = 0; mm < Tm; mm++)
    	        		for(cc = 0; cc < Tc; cc++){
#pragma HLS PIPELINE II=1
    	        			for(kx = 0; kx < K; kx++)
    			            	for(rr = 0; rr < Tr; rr++){
    					            for(nn = 0; nn < Tn; nn++){
    						            data_t mult = fm_in_buff[nn][rr+kx][cc+ky]*wt_buff[mm][nn][kx][ky];
    						            data_t psum = mult+fm_out_buff[mm][rr][cc];
                                        fm_out_buff[mm][rr][cc] = psum;
    					            }
    				            }
    			            }
            }

void compute_scout(data_t fm_out_buff[Tm][Tr][Tc],
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
	                load_scb(fm_out_buff,bias_buff,m,use_bias);
                    load_scin(fm_in_buff1,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col,ch_in);
                    load_scw(wt_buff1,w1,w2,w3,w4,ti,m,ch_in,ch_out);
                    bool pingpong = true;

                    for(ti = Tn; ti < ch_in; ti += Tn){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
                        if (pingpong)
                        {
                            load_scin(fm_in_buff2,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col,ch_in);
                            load_scw(wt_buff2,w1,w2,w3,w4,ti,m,ch_in,ch_out);
                            compute_sc(fm_in_buff1,fm_out_buff,wt_buff1);
                            pingpong = false;
                        }
                        else{
                            load_scin(fm_in_buff1,in1,in2,in3,in4,ti,fm_row,fm_col,input_row,input_col,ch_in);
                            load_scw(wt_buff1,w1,w2,w3,w4,ti,m,ch_in,ch_out);
                            compute_sc(fm_in_buff2,fm_out_buff,wt_buff2);
                            pingpong = true;
                        }
                    }
                    if(pingpong){
		                compute_sc(fm_in_buff1,fm_out_buff,wt_buff1);
	                }
	                else{
		                compute_sc(fm_in_buff2,fm_out_buff,wt_buff2);
	                }    
                }

void store_scout(data_t fm_out_buff[Tm][Tr][Tc],
                 data_t* out1, data_t* out2, data_t* out3, data_t* out4,
                 unsigned short fm_row, unsigned short fm_col, unsigned short m,
				 unsigned short input_row, unsigned input_col,
				 unsigned short ch_out, bool act
                ){
                    unsigned short mm, rr, cc;
                    unsigned short output_row = input_row;
                    unsigned short output_col = input_col;
                    int o_size = output_col*output_row;
//                    data_t tmp1,tmp2,tmp3,tmp4;
                    unsigned short num=((ch_out-m)<Tm)?(ch_out-m):Tm;

                    for(mm=0;mm<num;mm++)
    	                for(rr=0;rr<Tr;rr+=4)
    		                for(cc=0;cc<Tc;cc++)
                            {
#pragma HLS PIPELINE II=1
//    		                	if(act){
//    		                		tmp1=sc_leaky_relu(fm_out_buff[mm][rr][cc]);
//    		                		tmp2=sc_leaky_relu(fm_out_buff[mm][rr+1][cc]);
//    		                		tmp3=sc_leaky_relu(fm_out_buff[mm][rr+2][cc]);
//    		                		tmp4=sc_leaky_relu(fm_out_buff[mm][rr+3][cc]);
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

void next_scblock(unsigned short r, unsigned short c, unsigned short m,
				  unsigned short &next_r, unsigned short &next_c, unsigned short &next_m,
				  unsigned short input_row, unsigned input_col){

    				if(c+Tc >= input_col){
    					if(r+Tr >= input_row){
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

void std_conv(data_t* sc_in1, data_t* sc_in2, data_t* sc_in3, data_t* sc_in4,
              data_t* sc_w1, data_t* sc_w2, data_t* sc_w3, data_t* sc_w4,
              data_t* sc_bias,
              data_t* sc_out1, data_t* sc_out2, data_t* sc_out3, data_t* sc_out4,
              unsigned short sc_ch_in, unsigned short sc_ch_out,
			  unsigned short sc_input_row, unsigned sc_input_col, bool sc_act, bool sc_use_bias
            ){

                data_t bias_buff[MAX_LEN/Tm][Tm];
//                #pragma HLS ARRAY_PARTITION variable=bias_buff complete dim=2

                data_t fm_out1[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=2
                data_t fm_out2[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=2

                bool pingpong;
                unsigned short r, c, m;
                unsigned short next_r, next_c, next_m;
	            unsigned short out_row = sc_input_row;
                unsigned short out_col = sc_input_col;

                r = 0;
                c = 0;
                m = 0;
                pingpong = true;
//                if(sc_use_bias)
//                	memcpy((data_t*)bias_buff,(const data_t*)sc_bias,sizeof(data_t)*sc_ch_out);
                compute_scout(fm_out1,bias_buff, sc_in1,sc_in2,sc_in3,sc_in4, sc_w1,sc_w2,sc_w3,sc_w4, m,sc_input_row,sc_input_col,sc_ch_in,sc_ch_out,r,c,sc_use_bias);
                next_scblock(r,c,m,next_r,next_c,next_m,sc_input_row,sc_input_col);
                while(true){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
    	            if(pingpong){
    		            compute_scout(fm_out2,bias_buff,sc_in1,sc_in2,sc_in3,sc_in4, sc_w1,sc_w2,sc_w3,sc_w4,next_m,sc_input_row,sc_input_col,sc_ch_in,sc_ch_out,next_r,next_c,sc_use_bias);
    		            store_scout(fm_out1,sc_out1,sc_out2,sc_out3,sc_out4,r,c,m,sc_input_row,sc_input_col,sc_ch_out,sc_act);
    		            pingpong = false;
    	            }
    	            else{
    		            compute_scout(fm_out1,bias_buff,sc_in1,sc_in2,sc_in3,sc_in4, sc_w1,sc_w2,sc_w3,sc_w4,next_m,sc_input_row,sc_input_col,sc_ch_in,sc_ch_out,next_r,next_c,sc_use_bias);
			            store_scout(fm_out2,sc_out1,sc_out2,sc_out3,sc_out4,r,c,m,sc_input_row,sc_input_col,sc_ch_out,sc_act);
			            pingpong = true;
    	            }
    	            m = next_m;
    	            r = next_r;
    	            c = next_c;
    	            next_scblock(r,c,m,next_r,next_c,next_m,sc_input_row,sc_input_col);
		            if(next_m >= sc_ch_out)
		                 break;
                    }
                if(pingpong){
    	            store_scout(fm_out1,sc_out1,sc_out2,sc_out3,sc_out4,r,c,m,sc_input_row,sc_input_col,sc_ch_out,sc_act);
                }
                else{
    	            store_scout(fm_out2,sc_out1,sc_out2,sc_out3,sc_out4,r,c,m,sc_input_row,sc_input_col,sc_ch_out,sc_act);
                }
            }


