#include "basic_block.h"

static XTime tEnd, tCur;
static u32 tUsed;

class PreConv{
public:
	StdConv* embedding1;
	StdConv* embedding;
	DownSam* down_sam;

	PreConv(){
		this->embedding1 = new StdConv(1,32,256,256,false,false);
		this->embedding = new StdConv(64,32,256,256,false,false);
		this->down_sam = new DownSam(32,32,256,256,false,false);
	}

	void load_weight(string dir){
		this->embedding1->load_weight(dir+"\\embedding1");
		this->embedding->load_weight(dir+"\\embedding");
		this->down_sam->load_weight(dir+"\\down_sample");
	}

	void forward(data_t* embedding1_in, data_t* mask_out, data_t* embedding_out, data_t* down_sam_out){

		static data_t embedding1_out[32*256*256];
		static data_t concat[64*256*256];

		XTime_GetTime(&tCur);
		this->embedding1->forward(embedding1_in, embedding1_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", PreConv embedding1 end\n";

		int size1 = 32*256*256;
		int size2 = 32*256*256;

		for (int i = 0; i < size1; i++) {
			concat[i] = embedding1_out[i];
		}

		for (int i = 0; i < size2; i++) {
			concat[size1 + i] = mask_out[i];
		}

		Xil_DCacheFlush();
		XTime_GetTime(&tCur);
		this->embedding->forward(concat, embedding_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", PreConv embedding end\n";

		XTime_GetTime(&tCur);
		this->down_sam->forward(embedding_out, down_sam_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", PreConv down_sample end\n";
	}
};

class ConvEncoder{
public:
	StdConv* enconv1;
	DownSam* endown_sam1;
	StdConv* enconv2;
	DownSam* endown_sam2;

	ConvEncoder(){
		this->enconv1 = new StdConv(32,32,128,128,false,false);
		this->endown_sam1 = new DownSam(32,64,128,128,false,false);
		this->enconv2 = new StdConv(64,64,64,64,false,false);
		this->endown_sam2 = new DownSam(64,128,64,64,false,false);

	}

	void load_weight(string dir){
		this->enconv1->load_weight(dir+"\\enconv1");
		this->endown_sam1->load_weight(dir+"\\endown_sam1");
		this->enconv2->load_weight(dir+"\\enconv2");
		this->endown_sam2->load_weight(dir+"\\endown_sam2");
	}

	void forward(data_t* down_sam_out, data_t* enconv1_out, data_t* enconv2_out, data_t* endown_sam2_out){

		static data_t endown_sam1_out[64*64*64];

		XTime_GetTime(&tCur);
		this->enconv1->forward(down_sam_out, enconv1_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", EncoderConv1 end\n";

		XTime_GetTime(&tCur);
		this->endown_sam1->forward(enconv1_out, endown_sam1_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", EncoderDown_Sam1 end\n";

		XTime_GetTime(&tCur);
		this->enconv2->forward(endown_sam1_out, enconv2_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", EncoderConv2 end\n";

		XTime_GetTime(&tCur);
		this->endown_sam2->forward(enconv2_out, endown_sam2_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", EncoderDown_Sam2 end\n";

	}
};

class Encoder{
public:
	LayerNorm* layernorm1;
	PointConv* to_q;
	DeepConv* q_dw;
	PointConv* to_k;
	DeepConv* k_dw;
	PointConv* to_v;
	DeepConv* v_dw;
	Norm* norm_q;
	Norm* norm_k;
	HeadLin* lin;
	SoftMax* sof;
	TransLin* tlin;
	PointConv* proj;
	LayerNorm* layernorm2;
	PointConv* net1_pc;
	DeepConv* net1_dc;
	PointConv* net2_pc;
	DeepConv* net2_dc;
	PointConv* out_conv;
	DownSam* down_sam;
	Add_Leaky* add;
	Add_Leaky* leaky;
	Add_Leaky* leaky1;

	Encoder(){
		this->layernorm1 = new LayerNorm(128,32,32);
		this->to_q = new PointConv(1024,128,128,false,false);
		this->q_dw = new DeepConv(128,32,32,false,false);
		this->to_k = new PointConv(1024,128,128,false,false);
		this->k_dw = new DeepConv(128,32,32,false,false);
		this->to_v = new PointConv(1024,128,128,false,false);
		this->v_dw = new DeepConv(128,32,32,false,false);
		this->norm_q = new Norm(128,32,32);
		this->norm_k = new Norm(128,32,32);
		this->lin = new HeadLin(4,32,1024,32,true,false);
		this->sof = new SoftMax(4,32,32,16);
		this->tlin = new TransLin(4,32,1024,32,false);
		this->proj = new PointConv(1024,128,128,false,false);
		this->layernorm2 = new LayerNorm(128,32,32);
		this->net1_pc = new PointConv(1024,128,128,false,false);
		this->net1_dc = new DeepConv(128,32,32,false,false);
		this->net2_pc = new PointConv(1024,128,128,false,false);
		this->net2_dc = new DeepConv(128,32,32,false,false);
		this->out_conv = new PointConv(1024,256,128,false,false);
		this->down_sam = new DownSam(128,256,32,32,false,false);
		this->add = new Add_Leaky(128*32*32, true);
		this->leaky = new Add_Leaky(128*32*32, false);
		this->leaky1 = new Add_Leaky(256*32*32, false);
	}

	void load_weight(string dir){
		this->layernorm1->load_weight(dir+"\\layernorm1");
		this->to_q->load_weight(dir+"\\to_q");
		this->q_dw->load_weight(dir+"\\q_dw");
		this->to_k->load_weight(dir+"\\to_k");
		this->k_dw->load_weight(dir+"\\k_dw");
		this->to_v->load_weight(dir+"\\to_v");
		this->v_dw->load_weight(dir+"\\v_dw");
		this->lin->load_weight(dir+"\\lin");
		this->tlin->load_weight(dir+"\\tlin");
		this->proj->load_weight(dir+"\\proj");
		this->layernorm2->load_weight(dir+"\\layernorm2");
		this->net1_pc->load_weight(dir+"\\net1_pc");
		this->net1_dc->load_weight(dir+"\\net1_dc");
		this->net2_pc->load_weight(dir+"\\net2_pc");
		this->net2_dc->load_weight(dir+"\\net2_dc");
		this->out_conv->load_weight(dir+"\\out_conv");
		this->down_sam->load_weight(dir+"\\down_sam");
	}

	void forward(data_t* endown_sam2_out, data_n* add_out, data_t* encoder_out){

		static data_n layernorm1_out[128*32*32];
		static data_t to_q_out[128*32*32];
		static data_t q_dw_out[128*32*32];
		static data_t to_k_out[128*32*32];
		static data_t k_dw_out[128*32*32];
		static data_t to_v_out[128*32*32];
		static data_t v_dw_out[128*32*32];

		XTime_GetTime(&tCur);
		this->layernorm1->forward(endown_sam2_out, layernorm1_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder LayerNorm1 end\n";

		XTime_GetTime(&tCur);
		this->to_q->forward(layernorm1_out, to_q_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder to_q end\n";

		XTime_GetTime(&tCur);
		this->q_dw->forward(to_q_out, q_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder q_dw end\n";

		XTime_GetTime(&tCur);
		this->to_k->forward(layernorm1_out, to_k_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder to_k end\n";

		XTime_GetTime(&tCur);
		this->k_dw->forward(to_k_out, k_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder k_dw end\n";

		XTime_GetTime(&tCur);
		this->to_v->forward(layernorm1_out, to_v_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder to_v end\n";

		XTime_GetTime(&tCur);
		this->v_dw->forward(to_v_out, v_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder v_dw end\n";

		static data_n nor_q_out[128*32*32];
		static data_n nor_k_out[128*32*32];

		XTime_GetTime(&tCur);
		this->norm_q->forward(q_dw_out, nor_q_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder Norm_q end\n";

		XTime_GetTime(&tCur);
		this->norm_k->forward(k_dw_out, nor_k_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder Norm_k end\n";

		static data_t lin_out[4*32*32*16];
		static data_n sof_out[4*32*32];
		static data_t tlin_out[128*32*32];
		static data_t proj_out[128*32*32];

		XTime_GetTime(&tCur);
		this->lin->forward(nor_q_out, nor_k_out, lin_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder HeadLinear end\n";

		XTime_GetTime(&tCur);
		this->sof->forward(lin_out, sof_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder SoftMax end\n";

		XTime_GetTime(&tCur);
		this->tlin->forward(v_dw_out, sof_out, tlin_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder TransLinear end\n";

		XTime_GetTime(&tCur);
		this->proj->forward(tlin_out, proj_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder ProjConv end\n";

		static data_n layernorm2_in[128*32*32];
		static data_n layernorm2_out[128*32*32];
		static data_t net1_pc_out[128*32*32];
		static data_t net1_dc_out[128*32*32];
		static data_t net2_pc_out[128*32*32];
		static data_t net2_dc_out[128*32*32];
		static data_t cat_out[256*32*32];
		static data_t conv_out[128*32*32];

		this->add->forward(endown_sam2_out, proj_out, layernorm2_in);

		XTime_GetTime(&tCur);
		this->layernorm2->forward(layernorm2_in, layernorm2_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder LayerNorm2 end\n";

		XTime_GetTime(&tCur);
		this->net1_pc->forward(layernorm2_out, net1_pc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder Net1 PointConv end\n";

		this->leaky->forward(net1_pc_out, nullptr, net1_pc_out);

		XTime_GetTime(&tCur);
		this->net1_dc->forward(net1_pc_out, net1_dc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder Net1 DeepConv end\n";

		XTime_GetTime(&tCur);
		this->net2_pc->forward(layernorm2_out, net2_pc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder Net2 PointConv end\n";

		this->leaky->forward(net2_pc_out, nullptr, net2_pc_out);

		XTime_GetTime(&tCur);
		this->net2_dc->forward(net2_pc_out, net2_dc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder Net2 DeepConv end\n";

		for (int i = 0; i < 128*32*32; i++) {
			cat_out[i] = net1_dc_out[i];
		}

		for (int i = 0; i < 128*32*32; i++) {
			cat_out[128*32*32 + i] = net2_dc_out[i];
		}
		Xil_DCacheFlush();

		this->leaky1->forward(cat_out, nullptr, cat_out);

		XTime_GetTime(&tCur);
		this->out_conv->forward(cat_out, conv_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder ConvOut end\n";

		this->add->forward(conv_out, layernorm2_in, add_out);

		XTime_GetTime(&tCur);
		this->down_sam->forward(add_out, encoder_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Encoder DownSample end\n";

	}
};

//********************************************************
class BottleNeck{
public:
	LayerNorm* layernorm1;
	PointConv* to_q;
	DeepConv* q_dw;
	PointConv* to_k;
	DeepConv* k_dw;
	PointConv* to_v;
	DeepConv* v_dw;
	Norm* norm_q;
	Norm* norm_k;
	HeadLin* lin;
	SoftMax* sof;
	TransLin* tlin;
	PointConv* proj;
	LayerNorm* layernorm2;
	PointConv* net1_pc;
	DeepConv* net1_dc;
	PointConv* net2_pc;
	DeepConv* net2_dc;
	PointConv* out_conv;
	Add_Leaky* add;
	Add_Leaky* leaky;
	Add_Leaky* leaky1;

	BottleNeck(){
		this->layernorm1 = new LayerNorm(256,16,16);
		this->to_q = new PointConv(256,256,256,false,false);
		this->q_dw = new DeepConv(256,16,16,false,false);
		this->to_k = new PointConv(256,256,256,false,false);
		this->k_dw = new DeepConv(256,16,16,false,false);
		this->to_v = new PointConv(256,256,256,false,false);
		this->v_dw = new DeepConv(256,16,16,false,false);
		this->norm_q = new Norm(256,16,16);
		this->norm_k = new Norm(256,16,16);
		this->lin = new HeadLin(8,32,256,32,true,false);
		this->sof = new SoftMax(8,32,32,4);
		this->tlin = new TransLin(8,32,256,32,false);
		this->proj = new PointConv(256,256,256,false,false);
		this->layernorm2 = new LayerNorm(256,16,16);
		this->net1_pc = new PointConv(256,256,256,false,false);
		this->net1_dc = new DeepConv(256,16,16,false,false);
		this->net2_pc = new PointConv(256,256,256,false,false);
		this->net2_dc = new DeepConv(256,16,16,false,false);
		this->out_conv = new PointConv(256,512,256,false,false);
		this->add = new Add_Leaky(256*16*16, true);
		this->leaky = new Add_Leaky(256*16*16, false);
		this->leaky1 = new Add_Leaky(512*16*16, false);
	}

	void load_weight(string dir){
		this->layernorm1->load_weight(dir+"\\layernorm1");
		this->to_q->load_weight(dir+"\\to_q");
		this->q_dw->load_weight(dir+"\\q_dw");
		this->to_k->load_weight(dir+"\\to_k");
		this->k_dw->load_weight(dir+"\\k_dw");
		this->to_v->load_weight(dir+"\\to_v");
		this->v_dw->load_weight(dir+"\\v_dw");
		this->lin->load_weight(dir+"\\lin");
		this->tlin->load_weight(dir+"\\tlin");
		this->proj->load_weight(dir+"\\proj");
		this->layernorm2->load_weight(dir+"\\layernorm2");
		this->net1_pc->load_weight(dir+"\\net1_pc");
		this->net1_dc->load_weight(dir+"\\net1_dc");
		this->net2_pc->load_weight(dir+"\\net2_pc");
		this->net2_dc->load_weight(dir+"\\net2_dc");
		this->out_conv->load_weight(dir+"\\out_conv");
	}

	void forward(data_t* bottleneck_in, data_t* bottleneck_out){

		static data_n layernorm1_out[256*16*16];
		static data_t to_q_out[256*16*16];
		static data_t q_dw_out[256*16*16];
		static data_t to_k_out[256*16*16];
		static data_t k_dw_out[256*16*16];
		static data_t to_v_out[256*16*16];
		static data_t v_dw_out[256*16*16];

		XTime_GetTime(&tCur);
		this->layernorm1->forward(bottleneck_in, layernorm1_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck LayerNorm1 end\n";

		XTime_GetTime(&tCur);
		this->to_q->forward(layernorm1_out, to_q_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck to_q end\n";

		XTime_GetTime(&tCur);
		this->q_dw->forward(to_q_out, q_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck q_dw end\n";

		XTime_GetTime(&tCur);
		this->to_k->forward(layernorm1_out, to_k_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck to_k end\n";

		XTime_GetTime(&tCur);
		this->k_dw->forward(to_k_out, k_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck k_dw end\n";

		XTime_GetTime(&tCur);
		this->to_v->forward(layernorm1_out, to_v_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck to_v end\n";

		XTime_GetTime(&tCur);
		this->v_dw->forward(to_v_out, v_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck v_dw end\n";

		static data_n nor_q_out[256*16*16];
		static data_n nor_k_out[256*16*16];

		XTime_GetTime(&tCur);
		this->norm_q->forward(q_dw_out, nor_q_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck Norm_q end\n";

		XTime_GetTime(&tCur);
		this->norm_k->forward(k_dw_out, nor_k_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck Norm_k end\n";

		static data_t lin_out[8*32*32*4];
		static data_n sof_out[8*32*32];
		static data_t tlin_out[256*16*16];
		static data_t proj_out[256*16*16];

		XTime_GetTime(&tCur);
		this->lin->forward(nor_q_out, nor_k_out, lin_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck HeadLinear end\n";

		XTime_GetTime(&tCur);
		this->sof->forward(lin_out, sof_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck SoftMax end\n";

		XTime_GetTime(&tCur);
		this->tlin->forward(v_dw_out, sof_out, tlin_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck TransLinear end\n";

		XTime_GetTime(&tCur);
		this->proj->forward(tlin_out, proj_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck ProjConv end\n";

		static data_n layernorm2_in[256*16*16];
		static data_n layernorm2_out[256*16*16];
		static data_t net1_pc_out[256*16*16];
		static data_t net1_dc_out[256*16*16];
		static data_t net2_pc_out[256*16*16];
		static data_t net2_dc_out[256*16*16];
		static data_t cat_out[512*16*16];
		static data_t conv_out[256*16*16];

		this->add->forward(bottleneck_in, proj_out, layernorm2_in);

		XTime_GetTime(&tCur);
		this->layernorm2->forward(layernorm2_in, layernorm2_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck LayerNorm2 end\n";

		XTime_GetTime(&tCur);
		this->net1_pc->forward(layernorm2_out, net1_pc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck Net1 PointConv end\n";

		this->leaky->forward(net1_pc_out, nullptr, net1_pc_out);

		XTime_GetTime(&tCur);
		this->net1_dc->forward(net1_pc_out, net1_dc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck Net1 DeepConv end\n";

		XTime_GetTime(&tCur);
		this->net2_pc->forward(layernorm2_out, net2_pc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck Net2 PointConv end\n";

		this->leaky->forward(net2_pc_out, nullptr, net2_pc_out);

		XTime_GetTime(&tCur);
		this->net2_dc->forward(net2_pc_out, net2_dc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck Net2 DeepConv end\n";

		for (int i = 0; i < 256*16*16; i++) {
			cat_out[i] = net1_dc_out[i];
		}

		for (int i = 0; i < 256*16*16; i++) {
			cat_out[256*16*16 + i] = net2_dc_out[i];
		}

		Xil_DCacheFlush();
		this->leaky1->forward(cat_out, nullptr, cat_out);

		XTime_GetTime(&tCur);
		this->out_conv->forward(cat_out, conv_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", BottleNeck ConvOut end\n";

		this->add->forward(conv_out, layernorm2_in, bottleneck_out);
	}
};

class Decoder{
public:
	TransConv* up_sam;
	PointConv* fusion;
	LayerNorm* layernorm1;
	PointConv* to_q;
	DeepConv* q_dw;
	PointConv* to_k;
	DeepConv* k_dw;
	PointConv* to_v;
	DeepConv* v_dw;
	Norm* norm_q;
	Norm* norm_k;
	HeadLin* lin;
	SoftMax* sof;
	TransLin* tlin;
	PointConv* proj;
	LayerNorm* layernorm2;
	PointConv* net1_pc;
	DeepConv* net1_dc;
	PointConv* net2_pc;
	DeepConv* net2_dc;
	PointConv* out_conv;
	Add_Leaky* add;
	Add_Leaky* leaky;
	Add_Leaky* leaky1;

	Decoder(){
		this->up_sam = new TransConv(256,128,16,16,false,true);
		this->fusion = new PointConv(1024,256,128,false,false);
		this->layernorm1 = new LayerNorm(128,32,32);
		this->to_q = new PointConv(1024,128,128,false,false);
		this->q_dw = new DeepConv(128,32,32,false,false);
		this->to_k = new PointConv(1024,128,128,false,false);
		this->k_dw = new DeepConv(128,32,32,false,false);
		this->to_v = new PointConv(1024,128,128,false,false);
		this->v_dw = new DeepConv(128,32,32,false,false);
		this->norm_q = new Norm(128,32,32);
		this->norm_k = new Norm(128,32,32);
		this->lin = new HeadLin(4,32,1024,32,true,false);
		this->sof = new SoftMax(4,32,32,16);
		this->tlin = new TransLin(4,32,1024,32,false);
		this->proj = new PointConv(1024,128,128,false,false);
		this->layernorm2 = new LayerNorm(128,32,32);
		this->net1_pc = new PointConv(1024,128,128,false,false);
		this->net1_dc = new DeepConv(128,32,32,false,false);
		this->net2_pc = new PointConv(1024,128,128,false,false);
		this->net2_dc = new DeepConv(128,32,32,false,false);
		this->out_conv = new PointConv(1024,256,128,false,false);
		this->add = new Add_Leaky(128*32*32, true);
		this->leaky = new Add_Leaky(128*32*32, false);
		this->leaky1 = new Add_Leaky(256*32*32, false);
	}

	void load_weight(string dir){
		this->up_sam->load_weight(dir+"\\up_sam");
		this->fusion->load_weight(dir+"\\fusion");
		this->layernorm1->load_weight(dir+"\\layernorm1");
		this->to_q->load_weight(dir+"\\to_q");
		this->q_dw->load_weight(dir+"\\q_dw");
		this->to_k->load_weight(dir+"\\to_k");
		this->k_dw->load_weight(dir+"\\k_dw");
		this->to_v->load_weight(dir+"\\to_v");
		this->v_dw->load_weight(dir+"\\v_dw");
		this->lin->load_weight(dir+"\\lin");
		this->tlin->load_weight(dir+"\\tlin");
		this->proj->load_weight(dir+"\\proj");
		this->layernorm2->load_weight(dir+"\\layernorm2");
		this->net1_pc->load_weight(dir+"\\net1_pc");
		this->net1_dc->load_weight(dir+"\\net1_dc");
		this->net2_pc->load_weight(dir+"\\net2_pc");
		this->net2_dc->load_weight(dir+"\\net2_dc");
		this->out_conv->load_weight(dir+"\\out_conv");
	}

	void forward(data_t* bottleneck_out, data_n* add_out, data_t* decoder_out){

		static data_t upsam_out[128*32*32];
		static data_t cat_out[256*32*32];
		static data_t fusion_out[128*32*32];

		XTime_GetTime(&tCur);
		this->up_sam->forward(bottleneck_out, upsam_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder UpSample end\n";

		for (int i = 0; i < 128*32*32; i++) {
			cat_out[i] = upsam_out[i];
		}

		for (int i = 0; i < 128*32*32; i++) {
			cat_out[128*32*32 + i] = add_out[i];
		}

		Xil_DCacheFlush();
		XTime_GetTime(&tCur);
		this->fusion->forward(cat_out, fusion_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder Fusion end\n";

		static data_n layernorm1_out[128*32*32];
		static data_t to_q_out[128*32*32];
		static data_t q_dw_out[128*32*32];
		static data_t to_k_out[128*32*32];
		static data_t k_dw_out[128*32*32];
		static data_t to_v_out[128*32*32];
		static data_t v_dw_out[128*32*32];

		XTime_GetTime(&tCur);
		this->layernorm1->forward(fusion_out, layernorm1_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder LayerNorm1 end\n";

		XTime_GetTime(&tCur);
		this->to_q->forward(layernorm1_out, to_q_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder to_q end\n";

		XTime_GetTime(&tCur);
		this->q_dw->forward(to_q_out, q_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder q_dw end\n";

		XTime_GetTime(&tCur);
		this->to_k->forward(layernorm1_out, to_k_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder to_k end\n";

		XTime_GetTime(&tCur);
		this->k_dw->forward(to_k_out, k_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder k_dw end\n";

		XTime_GetTime(&tCur);
		this->to_v->forward(layernorm1_out, to_v_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder to_v end\n";

		XTime_GetTime(&tCur);
		this->v_dw->forward(to_v_out, v_dw_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder v_dw end\n";

		static data_n nor_q_out[128*32*32];
		static data_n nor_k_out[128*32*32];

		XTime_GetTime(&tCur);
		this->norm_q->forward(q_dw_out, nor_q_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder Norm_q end\n";

		XTime_GetTime(&tCur);
		this->norm_k->forward(k_dw_out, nor_k_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder Norm_k end\n";

		static data_t lin_out[4*32*32*16];
		static data_n sof_out[4*32*32];
		static data_t tlin_out[128*32*32];
		static data_t proj_out[128*32*32];

		XTime_GetTime(&tCur);
		this->lin->forward(nor_q_out, nor_k_out, lin_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder HeadLinear end\n";

		XTime_GetTime(&tCur);
		this->sof->forward(lin_out, sof_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder SoftMax end\n";

		XTime_GetTime(&tCur);
		this->tlin->forward(v_dw_out, sof_out, tlin_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder TransLinear end\n";

		XTime_GetTime(&tCur);
		this->proj->forward(tlin_out, proj_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder ProjConv end\n";

		static data_n layernorm2_in[128*32*32];
		static data_n layernorm2_out[128*32*32];
		static data_t net1_pc_out[128*32*32];
		static data_t net1_dc_out[128*32*32];
		static data_t net2_pc_out[128*32*32];
		static data_t net2_dc_out[128*32*32];
		static data_t cat_out1[256*32*32];
		static data_t conv_out[128*32*32];

		this->add->forward(fusion_out, proj_out, layernorm2_in);

		XTime_GetTime(&tCur);
		this->layernorm2->forward(layernorm2_in, layernorm2_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder LayerNorm2 end\n";


		XTime_GetTime(&tCur);
		this->net1_pc->forward(layernorm2_out, net1_pc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder Net1 PointConv end\n";

		this->leaky->forward(net1_pc_out, nullptr, net1_pc_out);

		XTime_GetTime(&tCur);
		this->net1_dc->forward(net1_pc_out, net1_dc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder Net1 DeepConv end\n";



		XTime_GetTime(&tCur);
		this->net2_pc->forward(layernorm2_out, net2_pc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder Net2 PointConv end\n";

		this->leaky->forward(net2_pc_out, nullptr, net2_pc_out);

		XTime_GetTime(&tCur);
		this->net2_dc->forward(net2_pc_out, net2_dc_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder Net2 DeepConv end\n";

		for (int i = 0; i < 128*32*32; i++) {
			cat_out1[i] = net1_dc_out[i];
		}

		for (int i = 0; i < 128*32*32; i++) {
			cat_out1[128*32*32 + i] = net2_dc_out[i];
		}

		Xil_DCacheFlush();

		this->leaky1->forward(cat_out1, nullptr, cat_out1);

		XTime_GetTime(&tCur);
		this->out_conv->forward(cat_out1, conv_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", Decoder ConvOut end\n";

		this->add->forward(conv_out, layernorm2_in, decoder_out);
	}
};

class ConvDecoder{
public:
	TransConv* up_sam1;
	PointConv* pc1;
	PointConv* pc2;
	TransConv* up_sam2;
	PointConv* pc3;
	PointConv* pc4;

	TransConv* out_up_sam;
	Add_Leaky* add;
	StdConv* mapping;

	ConvDecoder(){
		this->up_sam1 = new TransConv(128,64,32,32,false,true);
		this->pc1 = new PointConv(4096,128,64,false,false);
		this->pc2 = new PointConv(4096,64,64,false,false);

		this->up_sam2 = new TransConv(64,32,64,64,false,true);
		this->pc3 = new PointConv(16384,64,32,false,false);
		this->pc4 = new PointConv(16384,32,32,false,false);

		this->out_up_sam = new TransConv(32,32,128,128,false,true);
		this->add = new Add_Leaky(2097152,true);
		this->mapping = new StdConv(32,60,256,256,false,false);
	}

	void load_weight(string dir){
		this->up_sam1->load_weight(dir+"\\up_sam1");
		this->pc1->load_weight(dir+"\\pc1");
		this->pc2->load_weight(dir+"\\pc2");

		this->up_sam2->load_weight(dir+"\\up_sam2");
		this->pc3->load_weight(dir+"\\pc3");
		this->pc4->load_weight(dir+"\\pc4");

		this->out_up_sam->load_weight(dir+"\\out_up_sam");
		this->mapping->load_weight(dir+"\\mapping");
	}

	void forward(data_t* decoder_out, data_t* cat1, data_t* cat2, data_t* cat3, data_t* out){

		static data_t upsam_out1[64*64*64];
		static data_t cat_out1[128*64*64];
		static data_t pc_out1[64*64*64];
		static data_t pc_out2[64*64*64];

		XTime_GetTime(&tCur);
		this->up_sam1->forward(decoder_out, upsam_out1);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder UpSample1 end\n";

		for (int i = 0; i < 64*64*64; i++) {
			cat_out1[i] = upsam_out1[i];
		}

		for (int i = 0; i < 64*64*64; i++) {
			cat_out1[64*64*64 + i] = cat1[i];
		}

		Xil_DCacheFlush();
		XTime_GetTime(&tCur);
		this->pc1->forward(cat_out1, pc_out1);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder PointConv1 end\n";

		XTime_GetTime(&tCur);
		this->pc2->forward(pc_out1, pc_out2);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder PointConv2 end\n";

		static data_t upsam_out2[32*128*128];
		static data_t cat_out2[64*128*128];
		static data_t pc_out3[32*128*128];
		static data_t pc_out4[32*128*128];

		XTime_GetTime(&tCur);
		this->up_sam2->forward(pc_out2, upsam_out2);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder UpSample2 end\n";

		for (int i = 0; i < 32*128*128; i++) {
			cat_out2[i] = upsam_out2[i];
		}

		for (int i = 0; i < 32*128*128; i++) {
			cat_out2[32*128*128 + i] = cat2[i];
		}

		Xil_DCacheFlush();
		XTime_GetTime(&tCur);
		this->pc3->forward(cat_out2, pc_out3);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder PointConv3 end\n";

		XTime_GetTime(&tCur);
		this->pc4->forward(pc_out3, pc_out4);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder PointConv4 end\n";

		static data_t up_sam_out[32*256*256];
		static data_t add_out[32*256*256];

		XTime_GetTime(&tCur);
		this->out_up_sam->forward(pc_out4, up_sam_out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder OutUpSample end\n";

		this->add->forward(cat3, up_sam_out, add_out);

		XTime_GetTime(&tCur);
		this->mapping->forward(add_out, out);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		cout<<"It took "<<tUsed<<"us"<<", ConvDecoder Mapping end\n";
	}
};
