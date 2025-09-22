#include <iostream>
#include <fstream>
using namespace std;

#include "../src/top_fun.h"

//===============================
// config
#define channel 128  // input channel
#define size 32*32
//===============================

int tb_LayerNorm() {
    // file path
	const char* weights_path = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\layernorm\\weight.txt";
	const char* bias_path    = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\layernorm\\bias.txt";
    const char* inputs_path  = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\layernorm\\inputs.txt";
    const char* hls_path     = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\layernorm\\hls.txt";
    // open file
    std::ifstream file1(weights_path);
    std::ifstream file_bias(bias_path);
    std::ifstream file2(inputs_path);
//    std::ifstream file_par(par_path);
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
    data_t inputs[channel][size];
    data_t weights[channel];
    cout<<"TB B2"<<endl;//for debug

    // 1.1 read weight file and assign values to the weights array
        for (int c = 0; c < channel; ++c) {
            if (file1.eof()) {
                std::cerr << "Reached end of file before filling the entire array." << std::endl;
                return 1;
            }
            data_t temp;
            file1 >> temp;
            weights[c] = static_cast<data_t>(temp);
        }

    cout<<"tb_layernorm B0"<<endl;

    // 1.2 read ifm file and assign values to the inputs array
    for (int r = 0; r < channel; ++r) {
        for (int c = 0; c < size; ++c) {
            if (file2.eof()) {
                std::cerr << "Reached end of file before filling the entire array." << std::endl;
                return 1;
                }
                file2 >> inputs[r][c];
//                std::cout << inputs[r][c] << " ";
        }
    }
    // 1.3 read bias
    data_t bias[channel];

    for (int r=0; r<channel; ++r) {
    	if (file_bias.eof()) {
    		std::cerr << "Reached end of file before filling the entire array." << std::endl;
            return 1;
        }
    	file_bias >> bias[r];
    }
    // 2 compute
    // Convert a dimensional array to a one-dimensional array
    data_t *in        = reinterpret_cast<data_t*>(inputs);
    data_t *weight    = reinterpret_cast<data_t*>(weights);
    data_t *bias_pt   = reinterpret_cast<data_t*>(bias);

    data_t *output_pt_par = (data_t*)malloc((size*2)*sizeof(data_t));
    data_t *output_pt = (data_t*)malloc((size*channel)*sizeof(data_t));

    cout<<"tb_layernorm B1"<<endl;
    top_fun(in,in,in,in,
    		nullptr,nullptr,nullptr,nullptr,

			nullptr,nullptr,nullptr,nullptr,
			nullptr,
			nullptr,nullptr,nullptr,nullptr,
			nullptr,
			output_pt_par,output_pt_par,output_pt_par,output_pt_par,
			nullptr,nullptr,nullptr,nullptr,
			nullptr,
			channel,0,32,32,false,false,11);

    top_fun(in,in,in,in,
        	nullptr,nullptr,nullptr,nullptr,

			output_pt_par,output_pt_par,output_pt_par,output_pt_par,
    		nullptr,
			weight,weight,weight,weight,
			bias_pt,
   			output_pt,output_pt,output_pt,output_pt,
   			nullptr,nullptr,nullptr,nullptr,
    		nullptr,
    		channel,0,32,32,false,false,12);
    cout<<"tb_layernorm B2"<<endl;

    // 3 Save results
    for (int w = 0; w < channel*size; ++w) {
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

    return 0;
}


int main(){
	int result=0;
	cout<<"input shape:"<<channel<<" "<<size<<" "<<endl;
	cout<<"output shape:"<<channel<<" "<<size<<" "<<endl;
	result = tb_LayerNorm();
	if(result==0){
		cout<< "\n LayerNorm PASS! :)" <<endl;
	}else{
		cout<< "\n LayerNorm Failed! :(" <<endl;
	}
	return 0;
}


