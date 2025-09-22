#include"basic_op.h"

void top_fun(data_t* in, data_n* fin, data_t* par, data_n* fpar, data_t* weight, data_n* fweight, data_t* bias, data_n* fbias, data_t* out, data_n* fout, data_t* rescale,
			 int ch_in, int ch_out,
		 	 int input_row, int input_col, bool act, bool use_bias,
			 int choice
			){

	XTop_fun_Set_in1(&topf_inst, (long)in);
	XTop_fun_Set_in2(&topf_inst, (long)in);
	XTop_fun_Set_in3(&topf_inst, (long)in);
	XTop_fun_Set_in4(&topf_inst, (long)in);

	XTop_fun_Set_fin1(&topf_inst, (long)fin);
	XTop_fun_Set_fin2(&topf_inst, (long)fin);
	XTop_fun_Set_fin3(&topf_inst, (long)fin);
	XTop_fun_Set_fin4(&topf_inst, (long)fin);

	XTop_fun_Set_par1(&topf_inst, (long)par);
	XTop_fun_Set_par2(&topf_inst, (long)par);
	XTop_fun_Set_par3(&topf_inst, (long)par);
	XTop_fun_Set_par4(&topf_inst, (long)par);

	XTop_fun_Set_fpar1(&topf_inst, (long)fpar);
	XTop_fun_Set_fpar2(&topf_inst, (long)fpar);
	XTop_fun_Set_fpar3(&topf_inst, (long)fpar);
	XTop_fun_Set_fpar4(&topf_inst, (long)fpar);

	XTop_fun_Set_w1_r(&topf_inst, (long)weight);
	XTop_fun_Set_w2_r(&topf_inst, (long)weight);
	XTop_fun_Set_w3_r(&topf_inst, (long)weight);
	XTop_fun_Set_w4_r(&topf_inst, (long)weight);

	XTop_fun_Set_fw1(&topf_inst, (long)fweight);

	XTop_fun_Set_bias(&topf_inst, (long)bias);
	XTop_fun_Set_fbias(&topf_inst, (long)fbias);

	XTop_fun_Set_out1(&topf_inst, (long)out);
	XTop_fun_Set_out2(&topf_inst, (long)out);
	XTop_fun_Set_out3(&topf_inst, (long)out);
	XTop_fun_Set_out4(&topf_inst, (long)out);

	XTop_fun_Set_fout1(&topf_inst, (long)fout);
	XTop_fun_Set_fout2(&topf_inst, (long)fout);
	XTop_fun_Set_fout3(&topf_inst, (long)fout);
	XTop_fun_Set_fout4(&topf_inst, (long)fout);

	XTop_fun_Set_rescale(&topf_inst, (long)rescale);

	XTop_fun_Set_ch_in(&topf_inst,ch_in);
	XTop_fun_Set_ch_out(&topf_inst,ch_out);
	XTop_fun_Set_input_row(&topf_inst,input_row);
	XTop_fun_Set_input_col(&topf_inst,input_col);
	XTop_fun_Set_act(&topf_inst,act);
	XTop_fun_Set_use_bias(&topf_inst,use_bias);
	XTop_fun_Set_choice(&topf_inst,choice);

	XTop_fun_Start(&topf_inst);
	while(XTop_fun_IsDone(&topf_inst)==0);
}


void topf_init(){
	XTop_fun_Initialize(&topf_inst, 0);
}

