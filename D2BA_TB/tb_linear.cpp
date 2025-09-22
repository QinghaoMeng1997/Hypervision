#include <iostream>
#include <fstream>
using namespace std;

#include "../src/top_fun.h"

//===============================
// config
#define head 4
#define pre_c 32
#define size 1024
#define feature 32
//===============================

int tb_Linear() {
    // file path
	const char* weights_path = "F:/FPGA/Array_Operator/NoC_SRNet/TB/headlinear/weight.txt";
	const char* bias_path    = "F:/FPGA/Array_Operator/NoC_SRNet/TB/headlinear/bias.txt";
    const char* inputs_path  = "F:/FPGA/Array_Operator/NoC_SRNet/TB/headlinear/inputs.txt";
    const char* hls_path     = "F:/FPGA/Array_Operator/NoC_SRNet/TB/headlinear/hls.txt";
    const char* rescale_path = "F:/FPGA/Array_Operator/NoC_SRNet/TB/headlinear/rescale.txt";

    // open file
    std::ifstream file1(weights_path);
    std::ifstream file_bias(bias_path);
    std::ifstream file2(inputs_path);
    std::ofstream file3(hls_path);
    std::ifstream file4(rescale_path);

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
    if (!file4.is_open()) {
       std::cerr << "Failed to open file: " << rescale_path << std::endl;
       return 1;
    }
    if (!file_bias.is_open()) {
    	std::cerr << "Failed to open file: " << bias_path << std::endl;
        return 1;
    }

    // define input and weight buffer
    cout<<"TB B1"<<endl;//for debug
    data_t inputs[head][pre_c][size];
    data_t weights[head][feature][size];
    cout<<"TB B2"<<endl;//for debug

    // 1.1 read weight file and assign values to the weights array
        for (int r = 0; r < head; ++r) {
            for (int c = 0; c < feature; ++c) {
            	for(int s = 0; s < size; ++s){
            		if (file1.eof()) {
            			std::cerr << "Reached end of file before filling the entire array." << std::endl;
            			return 1;
            		}
            		data_t temp;
            		file1 >> temp;
            		//std::cout << temp << " ";
            		weights[r][c][s] = static_cast<data_t>(temp);

            	}
            }
        }
    cout<<"tb_linear B0"<<endl;

    // 1.2 read ifm file and assign values to the inputs array
    for (int r = 0; r < head; ++r) {
    	for (int c = 0; c < pre_c; ++c) {
    		for(int s = 0; s < size; ++s){
    			if (file2.eof()) {
    				std::cerr << "Reached end of file before filling the entire array." << std::endl;
    				return 1;
    			}
    			data_t temp;
    			file2 >> temp;
    			//std::cout << temp << " ";
    			inputs[r][c][s] = static_cast<data_t>(temp);

    		}
    	}
    }
    // 1.3 read bias
    data_t bias[head*feature];

    for (int r=0; r<head*feature; ++r) {
    	if (file_bias.eof()) {
    		std::cerr << "Reached end of file before filling the entire array." << std::endl;
            return 1;
        }
    	file_bias >> bias[r];
    }

    data_t rescale[head];

    for (int r=0; r<head; ++r) {
    	if (file4.eof()) {
    		std::cerr << "Reached end of file before filling the entire array." << std::endl;
    		return 1;
    	}
        file4 >> rescale[r];
		std::cout << rescale[r] << " "<<endl;

    }
    // 2 compute
    // Convert a dimensional array to a one-dimensional array
    data_t *in        = reinterpret_cast<data_t*>(inputs);
    data_t *weight    = reinterpret_cast<data_t*>(weights);
    data_t *bias_pt   = reinterpret_cast<data_t*>(bias);
    data_t *rescale_pt   = reinterpret_cast<data_t*>(rescale);


    data_t *output_pt = (data_t*)malloc((head*pre_c*feature*16)*sizeof(data_t));

    cout<<"tb_linear B1"<<endl;
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
		 rescale_pt,
		 head,pre_c,size,feature,1,0,6);
    cout<<"tb_linear B2"<<endl;
    // 3 Save results
    for (int w = 0; w < head*pre_c*feature*16; ++w) {
    	 data_t data = *((data_t*)output_pt + w);
    	 file3 << data;
    	 //cout<<data<< " ";
         file3<<endl; // Wrap to write next line of data
    }
    // 4 close files
    file1.close();
    file2.close();
    file3.close();
    file4.close();

    //
    cout<< "\n Conv2D test successfully!" <<endl;
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
	cout<<"input shape:"<<head<<" "<<pre_c<<" "<<size<<" "<<endl;
	cout<<"output shape:"<<head<<" "<<pre_c<<" "<<feature<<" "<<endl;
	result = tb_Linear();
	if(result==0){
		cout<< "\n Linear PASS! :)" <<endl;
	}else{
		cout<< "\n Linear Failed! :(" <<endl;
	}
	return 0;
}


