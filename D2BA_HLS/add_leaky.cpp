#include "add_leaky.h"

void add_leaky(data_n* al_in1,
			   data_n* al_weight1,
			   data_n* al_out1,
			   int al_size, bool act){

	if(act){
		int i;
		for (i = 0; i < al_size; i++) {
#pragma HLS PIPELINE II=1
			al_out1[i] = al_in1[i] + al_weight1[i];
		}
	}
	else
	{
		const data_n alpha = (data_n)0.1;
		int i;
		for (i = 0; i < al_size; i++) {
#pragma HLS PIPELINE II=1
			if (al_in1[i] >= (data_n)0 )
			{
				al_out1[i] = al_in1[i];
			}
			else
			{
				al_out1[i] = al_in1[i]*alpha;
			}
		}
	}
}
