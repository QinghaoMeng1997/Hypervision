# Hypervision: An on-chip hyperspectral computing microsystem for on-line video-rate computational imaging and sensing

Liheng Bian\*, Qinghao Meng\*, Lianjie Li, Xuan Peng, Zhen Wang, Jiajun Zhao, Zhu Yang and Jun Zhang. (* Equal contributions)

## 📁 Project Structure

##### Hypervision/ 

  ├── D2BA_HLS/              - The D2BA underlying core operator HLS code  
  ├── D2BA_TB/               - The D2BA underlying core operator test bentch code  
  ├── D2BA_Test/             - The D2BA hardware accelerator Vitis integration test code      
  ├── Data_Train/            - The Lite-SRNet hyperspectral reconstruction network training dataset  
  ├── Input_Img/             - The Lite-SRNet hyperspectral reconstruction network tests input images   
  ├── Save_Img/              - The Lite-SRNet hyperspectral reconstruction network tests output image   
  ├── architecture/          - The Lite-SRNet hyperspectral reconstruction network architecture   
  ├── exp/                   - Training results of Lite-SRNet   
  ├── getdataset             - The Lite-SRNet training dataset generates code   
  ├── getdataset_c           - The Lite-SRNet training dataset generates code with reflectance parameters  
  ├── test_lite_srnet        - The Lite-SRNet test code   
  ├── train                  - The Lite-SRNet training code 
  
## 🚀 Quick Start
## 1. System requirements

### 1.1 All software dependencies and operating systems

- The project has been tested on Windows 10 or Ubuntu 20.04.1.

### 1.2 Versions the software has been tested on

- The Lite-SRNet network has been tested on CUDA 12.4, pytorch 2.4.1, torchvision 0.19.1,  python 3.8.20, opencv-python 4.11.0.86.   
- The D2BA HLS code has been tested on Vitis HLS 2020.2, Vivado 2020.2 and Xilinx Vitis 2020.2.

### 1.3 Any required non-standard hardware

- HyperspecI sensor (Nature, 635: 8037, 73-81, 2024), AXU15EG and HyperN computing chip.

## 2. Installation guide

- The code for HLS operator, training and testing can be downloaded at public repository ：https://github.com/QinghaoMeng1997/Hypervision. 
- Due to the massive amount of training dataset, we have packaged it into multiple repositories for storage: https://github.com/bianlab/Hyperspectral-imaging-datase. 
- The mask data file is quite large. Please contact m15890095196@163.com for it.

## 📬 Contact
For questions, please contact:
m15890095196@163.com
Or open an issue on this GitHub repository.
