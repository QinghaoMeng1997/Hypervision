#ifndef BASIC_BLOCK_H
#define BASIC_BLOCK_H

#include"basic_op.h"
#define sd_k 3
#define dc_k 3
#define ds_k 4
#define tc_k 2

class StdConv{
public:
    int ch_in;
    int ch_out;
    int in_row;
	int in_col;
	bool act;
	bool use_bias;

    data_t* weight;
    data_t* bias;

    StdConv(int ch_in, int ch_out, int in_row, int in_col, bool act, bool use_bias){
        this->ch_in=ch_in;
        this->ch_out=ch_out;
        this->in_row=in_row;
        this->in_col=in_col;
        this->act=act;
        this->use_bias=use_bias;

        this->weight=new data_t[ch_out*ch_in*sd_k*sd_k];
        this->bias=new data_t[ch_out];
    }

    void forward(data_t* in, data_t* out){
    	top_fun(in, nullptr, nullptr, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr, nullptr,
    			this->ch_in, this->ch_out, this->in_row, this->in_col, this->act, this->use_bias,
				1);
    }
    void load_weight(string dir){

    	string filename;
    	filename=dir+"\\w.bin";
    	read_params(filename,this->weight,this->ch_out*this->ch_in*sd_k*sd_k);
    }
};

class DownSam{
public:
	int ch_in;
	int ch_out;
	int in_row;
	int in_col;
	bool act;
	bool use_bias;

	data_t* weight;
	data_t* bias;

	DownSam(int ch_in, int ch_out, int in_row, int in_col, bool act, bool use_bias){
		this->ch_in=ch_in;
		this->ch_out=ch_out;
		this->in_row=in_row;
		this->in_col=in_col;
		this->act=act;
		this->use_bias=use_bias;

		this->weight=new data_t[ch_out*ch_in*ds_k*ds_k];
		this->bias=new data_t[ch_out];
	}
	void forward(data_t* in, data_t* out){
		top_fun(in, nullptr, nullptr, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr, nullptr,
				this->ch_in, this->ch_out, this->in_row, this->in_col, this->act, this->use_bias,
				2);
	}
	void load_weight(string dir){

	    string filename;
	    filename=dir+"\\w.bin";
	    read_params(filename,this->weight,this->ch_out*this->ch_in*ds_k*ds_k);
	}
};

class TransConv{
public:
	int ch_in;
	int ch_out;
	int in_row;
	int in_col;
	bool act;
	bool use_bias;

	data_t* weight;
	data_t* bias;

	TransConv(int ch_in, int ch_out, int in_row, int in_col, bool act, bool use_bias){
		this->ch_in=ch_in;
		this->ch_out=ch_out;
		this->in_row=in_row;
		this->in_col=in_col;
		this->act=act;
		this->use_bias=use_bias;

		this->weight=new data_t[ch_in*ch_out*tc_k*tc_k];
		this->bias=new data_t[ch_out];
	}
	void forward(data_t* in, data_t* out){
		top_fun(in, nullptr, nullptr, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr, nullptr,
				this->ch_in, this->ch_out, this->in_row, this->in_col, this->act, this->use_bias,
				3);
	}
	void load_weight(string dir){

		string filename;
		filename=dir+"\\w.bin";
		read_params(filename,this->weight,this->ch_in*this->ch_out*tc_k*tc_k);
		filename=dir+"\\b.bin";
		read_params(filename,this->bias,this->ch_out);
	}
};

class DeepConv{
public:
	int ch_in;
	int in_row;
	int in_col;
	bool act;
	bool use_bias;

	data_t* weight;
	data_t* bias;

	DeepConv(int ch_in, int in_row, int in_col, bool act, bool use_bias){
		this->ch_in=ch_in;
		this->in_row=in_row;
		this->in_col=in_col;
		this->act=act;
		this->use_bias=use_bias;

		this->weight=new data_t[ch_in*dc_k*dc_k];
		this->bias=new data_t[ch_in];
	}
	void forward(data_t* in, data_t* out){
		top_fun(in, nullptr, nullptr, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr, nullptr,
				this->ch_in, 0, this->in_row, this->in_col, this->act, this->use_bias,
				4);
	}
	void load_weight(string dir){

		string filename;
		filename=dir+"\\w.bin";
		read_params(filename,this->weight,this->ch_in*dc_k*dc_k);
	}
};

class PointConv{
public:
	int in_row;
	int in_col;
	int out_col;
	bool act;
	bool use_bias;

	data_t* weight;
	data_t* bias;

	PointConv(int in_row, int in_col, int out_col, bool act, bool use_bias){
		this->in_row=in_row;
		this->in_col=in_col;
		this->out_col=out_col;
		this->act=act;
		this->use_bias=use_bias;

		this->weight=new data_t[out_col*in_col];
		this->bias=new data_t[out_col];
	}
	void forward(data_t* in, data_t* out){
		top_fun(in, nullptr, nullptr, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr, nullptr,
				this->in_row, 0, this->in_col, this->out_col, this->act, this->use_bias,
				5);
	}
	void load_weight(string dir){

		string filename;
		filename=dir+"\\w.bin";
		read_params(filename,this->weight,this->out_col*this->in_col);
	}
};

class HeadLin{
public:
	int head;
	int pre_channel;
	int size;
	int feature;
	bool act;
	bool use_bias;

	data_t* weight;
	data_t* bias;
	data_t* rescale;

	HeadLin(int head, int pre_channel, int size, int feature, bool act, bool use_bias){
		this->head=head;
		this->pre_channel=pre_channel;
		this->size=size;
		this->feature=feature;
		this->act=act;
		this->use_bias=use_bias;

		this->weight=new data_t[head*feature*size];
		this->bias=new data_t[head*feature];
		this->rescale=new data_t[head];
	}
	void forward(data_t* in, data_t* weight, data_t* out){
		this->weight = weight;
		top_fun(in, nullptr, nullptr, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr, this->rescale,
				this->head, this->pre_channel, this->size, this->feature, this->act, this->use_bias,
				7);
	}

	void load_weight(string dir){

		string filename;
		filename=dir+"\\r.bin";
		read_params(filename,this->rescale,this->head);
	}
};

class TransLin{
public:
	int head;
	int pre_channel;
	int size;
	int feature;
	bool use_bias;

	data_t* weight;
	data_t* bias;

	TransLin(int head, int pre_channel, int size, int feature, bool use_bias){
		this->head=head;
		this->pre_channel=pre_channel;
		this->size=size;
		this->feature=feature;
		this->use_bias=use_bias;

		this->weight=new data_t[head*feature*pre_channel];
		this->bias=new data_t[head*feature];
	}
	void forward(data_t* in, data_t* weight, data_t* out){
		this->weight = weight;
		top_fun(in, nullptr, nullptr, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr, nullptr,
				this->head, this->pre_channel, this->size, this->feature, false, this->use_bias,
				8);
	}

	void load_weight(string dir){

		string filename;
	}
};

class Norm{
public:
	int channel;
	int row;
	int col;
	data_t* par;

	Norm(int channel, int row, int col){
		this->channel=channel;
		this->row=row;
		this->col=col;
		this->par = new data_t[channel];
	}

	void forward(data_t* in, data_t* out){
		top_fun(nullptr, in, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, par, nullptr,
				this->channel, 0, this->row, this->col, false, false,
				9);

		top_fun(nullptr, in, nullptr, par, nullptr, nullptr, nullptr, nullptr, nullptr, out, nullptr,
				this->channel, 0, this->row, this->col, false, false,
				10);
	}
};

class SoftMax{
public:
	int head;
	int pre_channel;
	int size;
	int folder;

	SoftMax(int head, int pre_channel, int size, int folder){
		this->head=head;
		this->pre_channel=pre_channel;
		this->size=size;
		this->folder=folder;
	}
	void forward(data_t* in, data_t* out){
		top_fun(nullptr, in, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, out, nullptr,
				this->head, this->pre_channel, this->size, this->folder, false, false,
				11);
	}
};

class LayerNorm{
public:
	int ch_in;
	int row;
	int col;

	data_n* par;
	data_n* weight;
	data_n* bias;

	LayerNorm(int ch_in, int row, int col){
		this->ch_in=ch_in;
		this->row=row;
		this->col=col;

		this->weight=new data_n[ch_in];
		this->bias=new data_n[ch_in];
		this->par=new data_n[row*col*2];
	}
	void forward(data_n* in, data_n* out){
		top_fun(nullptr, in, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, this->par, nullptr,
				this->ch_in, 0, this->row, this->col, false, false,
				12);

		top_fun(nullptr, in, nullptr, this->par, nullptr, this->weight, nullptr, this->bias, nullptr, out, nullptr,
				this->ch_in, 0, this->row, this->col, false, false,
				13);

	}
	void load_weight(string dir){
		string filename;
		filename=dir+"\\w.bin";
		read_params_half(filename,this->weight,this->ch_in);

		filename=dir+"\\b.bin";
		read_params_half(filename,this->bias,this->ch_in);
	}
};

class Add_Leaky{
public:
	int size;
	bool act;

	Add_Leaky(int size, bool act){
		this->size=size;
		this->act=act;
	}
	void forward(data_n* in, data_n* weight, data_t* out){
		top_fun(nullptr, in, nullptr, nullptr, nullptr, weight, nullptr, nullptr, nullptr, out, nullptr,
				this->size, 0, 0, 0, this->act, false,
				14);
	}
};
#endif
