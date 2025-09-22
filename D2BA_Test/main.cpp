#include "srnet_small.h"

#pragma GCC optimize(3,"Ofast","inline")
using namespace std;

static string  input_file   = "Srnet_Small\\Img\\img.bin";
static string  mask_file    = "Srnet_Small\\Img\\mask_out.bin";
static string  weight   	= "Srnet_Small";
static string  output_file  = "Srnet_Small\\Out\\out.bin";

int main()
{
    cout << "=====================SRNet_Small=====================" << endl;
    cout << endl;

    SD_Init();
//PreConv
    data_t* embedding1_in = new data_t[1*256*256];
    data_t* mask_out = new data_t[32*256*256];
    data_t* embedding_out = new data_t[32*256*256];
    data_t* down_sam_out = new data_t[32*128*128];

//EncoderConv
    data_t* enconv1_out = new data_t[32*128*128];
    data_t* enconv2_out = new data_t[64*64*64];
    data_t* endown_sam2_out = new data_t[128*32*32];

//Encoder
    data_n* add_out = new data_n[128*32*32];
    data_t* encoder_out = new data_t[256*16*16];


//BottleNeck
    data_t* bottleneck_out = new data_t[256*16*16];

//Decoder
    data_t* decoder_out = new data_t[128*32*32];

 //ConvDecoder
    data_t* out = new data_t[60*256*256];

    cout << "****************Begin Reading Parameters****************" << endl;

    read_params(input_file, embedding1_in, 1*256*256);
    read_params(mask_file, mask_out, 32*256*256);
    topf_init();

    PreConv* preconv=new PreConv();
    ConvEncoder* convencoder = new ConvEncoder();
    Encoder* encoder = new Encoder();
    BottleNeck* bottleneck = new BottleNeck();
    Decoder* decoder = new Decoder();
    ConvDecoder* convdecoder = new ConvDecoder();

    preconv->load_weight(weight+"\\PreConv");
    convencoder->load_weight(weight+"\\ConvEncoder");
    encoder->load_weight(weight+"\\Encoder");
    bottleneck->load_weight(weight+"\\BottleNeck");
    decoder->load_weight(weight+"\\Decoder");
    convdecoder->load_weight(weight+"\\ConvDecoder");

    cout << "***************End Reading Parameters***************" << endl;
    cout << endl;

    Xil_DCacheFlush();

    cout << "------------------Begin PreConv------------------" << endl;
    preconv->forward(embedding1_in, mask_out, embedding_out, down_sam_out);
    cout << "-------------------End PreConv-------------------" << endl;
    cout << endl;

    cout << "----------------Begin ConvEncoder----------------" << endl;
    convencoder->forward(down_sam_out, enconv1_out, enconv2_out, endown_sam2_out);
    cout << "-----------------End ConvEncoder-----------------" << endl;
    cout << endl;

    cout << "------------------Begin Encoder------------------" << endl;
    encoder->forward(endown_sam2_out, add_out, encoder_out);
    cout << "-------------------End Encoder-------------------" << endl;
    cout << endl;

    cout << "-----------------Begin BottleNeck-----------------" << endl;
    bottleneck->forward(encoder_out, bottleneck_out);
    cout << "------------------End BottleNeck------------------" << endl;
    cout << endl;

    cout << "------------------Begin Decoder------------------" << endl;
    decoder->forward(bottleneck_out, add_out, decoder_out);
    cout << "-------------------End Decoder-------------------" << endl;
    cout << endl;

    cout << "-----------------Begin ConvDecoder-----------------" << endl;
    convdecoder->forward(decoder_out, enconv2_out, enconv1_out, embedding_out, out);
    cout << "------------------End ConvDecoder------------------" << endl;
    cout << endl;

    Xil_DCacheInvalidateRange((u32)((long)out&0xffffffe0), 32*((60*256*256*sizeof(data_t))/32+2));
   	write_data(output_file,out,60*256*256);
   	cout << "=======================End=======================" << endl;

    delete [] embedding1_in;
    delete [] mask_out;
    delete [] embedding_out;
    delete [] down_sam_out;
    delete [] enconv1_out;
    delete [] enconv2_out;
    delete [] endown_sam2_out;
    delete [] add_out;
    delete [] encoder_out;
    delete [] bottleneck_out;
    delete [] decoder_out;
    delete [] out;
    return 0;
}
