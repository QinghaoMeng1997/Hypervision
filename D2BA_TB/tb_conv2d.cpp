#include <iostream>
#include <fstream>
using namespace std;

#include "../src/top_fun.h"

//===============================
// config
#define IN_C 32   // input channel
#define IN_H 128
#define IN_W 128

#define K 2
#define P 0
#define S 2

#define OUT_C 32  // output channel
#define OUT_H 256
#define OUT_W 256
//===============================

int tb_conv2d() {
    // file path
	const char* weights_path = "F:/FPGA/Array_Operator/NoC_SRNet/TB/conv2d/weight.txt";
	const char* bias_path    = "F:/FPGA/Array_Operator/NoC_SRNet/TB/conv2d/bias.txt";
	const char* inputs_path  = "F:/FPGA/Array_Operator/NoC_SRNet/TB/conv2d/inputs.txt";
	const char* hls_path     = "F:/FPGA/Array_Operator/NoC_SRNet/TB/conv2d/hls.txt";
    std::ifstream file1(weights_path);
    std::ifstream file_bias(bias_path);
    std::ifstream file2(inputs_path);
    std::ofstream file3(hls_path);
    if (!file1.is_open()) {
        cerr << "Failed to open file: " << weights_path << std::endl;
        return 1;
    }
    if (!file2.is_open()) {
        cerr << "Failed to open file: " << inputs_path << std::endl;
        return 1;
    }
    if (!file3.is_open()) {
       std::cerr << "Failed to open file: " << hls_path << std::endl;
       return 1;
    }
    if (!file_bias.is_open()) {
    	std::cerr << "Failed to open file: " << bias_path << std::endl;
        return 1;
    }

    // define input and weight buffer
       cout<<"TB B1"<<endl;//for debug
       data_t inputs[IN_C][IN_H][IN_W];
       data_t weights[OUT_C][IN_C][K][K];
       cout<<"TB B2"<<endl;//for debug

       // 1.1 read weight file and assign values to the weights array
           for (int n = 0; n < OUT_C; ++n) {
               for (int c = 0; c < IN_C; ++c) {
                   for (int h = 0; h < K; ++h) {
                       for (int w = 0; w < K; ++w) {
                           if (file1.eof()) {
                               std::cerr << "Reached end of file before filling the entire array." << std::endl;
                               return 1;
                           }
                           data_t temp;
                           file1 >> temp;
                           //std::cout << temp << " ";
                           weights[n][c][h][w] = static_cast<data_t>(temp);
                       }
                   }
               }
           }
       cout<<"tb_conv2d B0"<<endl;
       // 1.2 read ifm file and assign values to the inputs array
       for (int c = 0; c < IN_C; ++c) {
           for (int h = 0; h < IN_H; ++h) {
               for (int w = 0; w < IN_W; ++w) {
                   if (file2.eof()) {
                       std::cerr << "Reached end of file before filling the entire array." << std::endl;
                       return 1;
                   }
                   file2 >> inputs[c][h][w];
               }
           }
       }
       // 1.3 read bias
       data_t bias[OUT_C];
       for (int c=0; c<OUT_C; ++c) {
       	if (file_bias.eof()) {
       		std::cerr << "Reached end of file before filling the entire array." << std::endl;
               return 1;
           }
       	file_bias >> bias[c];
           cout<<"bias:"<<bias[c]<<"\n";
       }
       // 2 compute
       // Convert a three-dimensional array to a one-dimensional array
       data_t *in        = reinterpret_cast<data_t*>(inputs);
       data_t *weight    = reinterpret_cast<data_t*>(weights);
       data_t *bias_pt   = reinterpret_cast<data_t*>(bias);

       data_t *output_pt = (data_t*)malloc((OUT_C*OUT_W*OUT_H)*sizeof(data_t));
       cout<<"tb_conv2d B1"<<endl;
       top_fun(in,in,in,in,
    		   nullptr,nullptr,nullptr,nullptr,

			   nullptr,nullptr,nullptr,nullptr,
			   nullptr,nullptr,nullptr,nullptr,

    		   weight,weight,weight,weight,
			   nullptr,

			   bias_pt,
			   nullptr,

			   output_pt,output_pt,output_pt,output_pt,
			   nullptr,nullptr,nullptr,nullptr,

			   nullptr,
			   IN_C,OUT_C,IN_H,IN_W,false,true,3);
       cout<<"tb_conv2d B2"<<endl;
       // 3 Save results
       for (int w = 0; w <OUT_C*OUT_W*OUT_H; ++w) {
       	 data_t data = *((data_t*)output_pt + w);
       	 file3 << data;
       	 //cout<<data<< " ";
            file3<<endl; // Wrap to write next line of data
       }
       // 4 close files
       file1.close();
       file2.close();
       file3.close();
       //
       cout<< "\n Conv2D test successfully!" <<endl;
       
       delete[] inputs;
       delete[] weights;
       delete[] bias_pt;
       delete output_pt;

       return 0;
   }


 int main(){
  	int result=0;
   	cout<<"input shape:"<<IN_C<<" "<<IN_H<<" "<<IN_W<<endl;
   	cout<<"output shape:"<<OUT_C<<" "<<OUT_H<<" "<<OUT_W<<endl;
   	result = tb_conv2d();
   	if(result==0){
   		cout<< "\n Conv2D PASS! :)" <<endl;
   	}else{
   		cout<< "\n Conv2D Failed! :(" <<endl;
   	}
   	return 0;
   }
