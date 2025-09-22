#include <iostream>
#include <fstream>
using namespace std;

#include "../src/top_fun.h"

//===============================
// config
#define head 4
#define pre_channel 32
#define size 32
#define folder 16   // input channel
#define item head*pre_channel*size
//===============================

int tb_Softmax() {
    // file path
    const char* inputs_path  = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\softmax\\inputs.txt";
    const char* hls_path     = "F:\\FPGA\\Array_Operator\\NoC_SRNet\\TB\\softmax\\hls.txt";
    // open file
    std::ifstream file1(inputs_path);
    std::ofstream file2(hls_path);
    if (!file1.is_open()) {
        cerr << "Failed to open file: " << inputs_path << std::endl;
        return 1;
    }
    if (!file2.is_open()) {
        cerr << "Failed to open file: " << hls_path << std::endl;
        return 1;
    }
    // define input and weight buffer
    cout<<"TB B1"<<endl;//for debug
    data_t inputs[item][folder];
    cout<<"TB B2"<<endl;//for debug

    // 1.2 read ifm file and assign values to the inputs array
    for (int c = 0; c < item; ++c){
    	for (int r = 0; r < folder; ++r) {
    	    if (file1.eof()) {
    	    	std::cerr << "Reached end of file before filling the entire array." << std::endl;
    	        return 1;
    	    }
    	    file1 >> inputs[c][r];
    	}
    }

    // 2 compute
    // Convert a dimensional array to a one-dimensional array
    data_t *in        = reinterpret_cast<data_t*>(inputs);
    data_t *output_pt = (data_t*)malloc((item)*sizeof(data_t));

    // print in
//    for (int i = 0; i < size; i++) {
//        cout << in[i] << " "<< endl;
//    }
    cout<<"tb_Softmax B1"<<endl;
    top_fun(in, in, in, in,
    		nullptr,nullptr,nullptr,nullptr,

    		nullptr,nullptr,nullptr,nullptr,
			nullptr,nullptr,nullptr,nullptr,

			nullptr,nullptr,nullptr,nullptr,
			nullptr,

			nullptr,
			nullptr,

    		output_pt, output_pt, output_pt, output_pt,
			nullptr,nullptr,nullptr,nullptr,

			nullptr,
			head, pre_channel,size,folder,false,false,10);
    cout<<"tb_Softmax B2"<<endl;
    // 3 Save results
    for (int w = 0; w < item; ++w) {
    	 data_t data = *((data_t*)output_pt + w);
    	 file2 << data;
    	 //cout<<data<< " ";
         file2<<endl; // Wrap to write next line of data
    }
    // 4 close files
    file1.close();
    file2.close();
    //
    cout<< "\n Softmax test successfully!" <<endl;
    /*
    // ÊÍ·ÅÄÚ´æ
    delete[] inputs;
    delete[] weights;
    delete[] bias_pt;
    delete output_pt;*/

    return 0;
}


int main(){
	int result=0;
	cout<<"input shape:"<<item<<" "<<folder<<" "<<endl;
	cout<<"output shape:"<<item<<" "<<" "<<endl;
	result = tb_Softmax();
	if(result==0){
		cout<< "\n Softmax PASS! :)" <<endl;
	}else{
		cout<< "\n Softmax Failed! :(" <<endl;
	}
	return 0;
}


