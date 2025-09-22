#include <iostream>
#include <fstream>
using namespace std;

#include "../src/top_fun.h"

//===============================
// config
#define Channel 32
#define IN_R 128   // input channel
#define IN_C 128
#define SIZE IN_R*IN_C
//
//===============================

int tb_LayerNorm() {
    // file path
    const char* inputs_path  = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\normalize\\inputs.txt";
    const char* hls_path     = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\normalize\\hls.txt";
    const char* par_path     = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\normalize\\par.txt";
    // open file
    std::ifstream file1(par_path);
    std::ifstream file2(inputs_path);
    std::ofstream file3(hls_path);

    if (!file1.is_open()) {
    	cerr << "Failed to open file: " << par_path << std::endl;
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

    // define input and weight buffer
    cout<<"TB B1"<<endl;//for debug
    data_t inputs[Channel][SIZE];
    data_t pars[Channel];
    cout<<"TB B2"<<endl;//for debug

    for (int r = 0; r < Channel; ++r) {
    	if (file1.eof()) {
    		std::cerr << "Reached end of file before filling the entire array." << std::endl;
    		return 1;
    	}
    		file1 >> pars[r];
    }
    // 1.2 read ifm file and assign values to the inputs array
    for (int r = 0; r < Channel; ++r) {
        for (int c = 0; c < SIZE; ++c) {
            if (file2.eof()) {
                std::cerr << "Reached end of file before filling the entire array." << std::endl;
                return 1;
                }
                file2 >> inputs[r][c];
        }
    }

    // 2 compute
    // Convert a dimensional array to a one-dimensional array
    data_t *in        = reinterpret_cast<data_t*>(inputs);
    data_t *par       = reinterpret_cast<data_t*>(pars);
    data_t *output_pt = (data_t*)malloc((Channel*SIZE)*sizeof(data_t));

    cout<<"tb_layernorm B1"<<endl;
    top_fun(in,in,in,in,
    		par,par,par,par,
			nullptr,nullptr,nullptr,nullptr,
			nullptr,
    		output_pt,output_pt,output_pt,output_pt,
			nullptr,
		 Channel,0,IN_R,IN_C,false,false,false,7);
    cout<<"tb_layernorm B2"<<endl;
    // 3 Save results
    for (int w = 0; w < Channel*SIZE; ++w) {
    	 data_t data = *((data_t*)output_pt + w);
    	 file3 << data;
    	 //cout<<data<< " ";
         file3<<endl; // Wrap to write next line of data
    }
    // 4 close files
    file2.close();
    file3.close();
    //
    cout<< "\n Conv2D test successfully!" <<endl;
    /*
    // �ͷ��ڴ�
    delete[] inputs;
    delete[] weights;
    delete[] bias_pt;
    delete output_pt;*/

    return 0;
}


int main(){
	int result=0;
	cout<<"input shape:"<<IN_R<<" "<<IN_C<<" "<<endl;
	cout<<"output shape:"<<IN_R<<" "<<IN_C<<" "<<endl;
	result = tb_LayerNorm();
	if(result==0){
		cout<< "\n LayerNorm PASS! :)" <<endl;
	}else{
		cout<< "\n LayerNorm Failed! :(" <<endl;
	}
	return 0;
}


