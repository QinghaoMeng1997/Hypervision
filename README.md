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

The project has been tested on Windows 10 or Ubuntu 20.04.1.

### 1.2 Versions the software has been tested on

The project has been tested on CUDA 12.4, pytorch 2.4.1, torchvision 0.19.1,  python 3.8.20, opencv-python 4.11.0.86. 



## 2. Installation guide

### 2.1 Instructions

- The code for training and testing can be downloaded at public repository ：https://github.com/bianlab/MOCI
- The mask, testing measurements and pre-trained weights can be downloaded from the Google Drive link:https://drive.google.com/drive/folders/1zO8D3iA7adLsSVAfkJf1EhHM2qwzY6TB?usp=drive_link
- Due to the massive amount of training dataset, we have packaged it into multiple repositories for storage: https://github.com/bianlab/Hyperspectral-imaging-dataset



## 3. Program description and testing

### 3.1 Main program and data description

- The model of reconstruction:  `./architecture/PSRNet.py` 

- Pre-trained weights of PSRNet for PHI sensor:   `./model_zoo/psrnet.pth` 

- Calibrated sensing matrix of PHI sensor:   `./MASK/mask.mat` 

- Measurements collected by our single-shot PHI senor:   `./Measurements/` 

- The test and training program :    `train.py` , `test.py` 

  

### 3.2 Model Training of PSRNet

Run the train program on the collected measurements to reconstruct polarizaiton-hyperspectral images in pytorch platform.

Download the training dataset of PHI senor into ` ./Dataset_Train/`. 

The training programs are executed to train the polarization and spectral reconstruction model. 

For training PHI sensor,  execute the following command in the terminal, and the training results will be saved in the ` ./exp/PHI/` folder.

```python
python train.py 
```



### 3.3 Test polarization and hyperspectral reconstruction results in real-world scenes

Run the test program on the collected images to reconstruct polarization and hyperspectral images in pytorch platform.

When the images were collected using our PHI sensors,  the data can be reconstructed by run the following program in the terminal.

```python
python test.py
```

The measurements collected using PHI sensor from the folder  `'./Measurements/images/' `  . And output reconstructed multi-dimensional images  will be saved in  `'./Measurements/Output_PHI/' `  .
