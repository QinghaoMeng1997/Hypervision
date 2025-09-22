#include "sd_io.h"


int SD_Init()
{
    FRESULT rc;

    rc = f_mount(&fatfs,"",0);
    if(rc)
    {
        xil_printf("ERROR : f_mount returned %d\r\n",rc);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

int SD_Transfer_read(char *FileName,u32 DestinationAddress,u32 ByteLength)
{
    FIL fil;
    FRESULT rc;
    UINT br;

    rc = f_open(&fil,FileName,FA_READ);
    if(rc)
    {
        xil_printf("ERROR : f_open returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_lseek(&fil, 0);
    if(rc)
    {
        xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_read(&fil, (void*)DestinationAddress,ByteLength,&br);
    if(rc)
    {
        xil_printf("ERROR : f_read returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_close(&fil);
    if(rc)
    {
        xil_printf(" ERROR : f_close returned %d\r\n", rc);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

int SD_Transfer_write(char *FileName,u32 SourceAddress,u32 ByteLength)
{
    FIL fil;
    FRESULT rc;
    UINT bw;

    rc = f_open(&fil,FileName,FA_CREATE_ALWAYS | FA_WRITE);
    if(rc)
    {
        xil_printf("ERROR : f_open returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_lseek(&fil, 0);
    if(rc)
    {
        xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_write(&fil,(void*) SourceAddress,ByteLength,&bw);
    if(rc)
    {
        xil_printf("ERROR : f_write returned %d\r\n", rc);
        return XST_FAILURE;
    }
    rc = f_close(&fil);
    if(rc){
        xil_printf("ERROR : f_close returned %d\r\n",rc);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

uint16_t float_to_half(float value) {
    uint32_t f_bits = *(uint32_t*)&value; // 将float转换为32位无符号整数
    uint16_t h_bits = 0;

    // 提取符号位，指数和尾数
    uint16_t sign = (f_bits >> 31) & 0x1;
    int16_t exponent = ((f_bits >> 23) & 0xFF) - 127 + 15; // 指数偏移量
    uint32_t mantissa = f_bits & 0x007FFFFF;

    // 处理特殊情况
    if (exponent <= 0) { // 次正规数或0
        if (exponent < -10) {
            h_bits = sign << 15; // 返回符号位
        } else {
            mantissa = (mantissa | 0x00800000) >> (1 - exponent);
            h_bits = (sign << 15) | (mantissa >> 13);
        }
    } else if (exponent >= 31) { // 非数或无穷大
        h_bits = (sign << 15) | 0x7C00;
    } else {
        h_bits = (sign << 15) | (exponent << 10) | (mantissa >> 13);
    }

    return h_bits;
}

float half_to_float(uint16_t h_bits) {
    uint32_t sign = (h_bits >> 15) & 0x1;
    int16_t exponent = (h_bits >> 10) & 0x1F;
    uint32_t mantissa = h_bits & 0x03FF;

    uint32_t f_bits;

    if (exponent == 0) { // 次正规数或0
        if (mantissa == 0) {
            f_bits = sign << 31; // 返回0
        } else {
            exponent = -14;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;
            f_bits = (sign << 31) | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) { // 非数或无穷大
        f_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        f_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return *(float*)&f_bits;
}

void read_params(string filename,data_t* x,int length){
	cout<<"reading "<<filename<<endl;
	float* tmp=new float[length];
	SD_Transfer_read((char*)(filename.c_str()),(long)tmp,(long)(sizeof(float)*length));
    for(int i=0;i<length;i++){
//        x[i]=(data_t)(round(tmp[i]*SCALE));
    	x[i] = float_to_half(tmp[i]);

    }
    delete [] tmp;
}

void read_params_half(string filename,data_n* x,int length){
	cout<<"reading "<<filename<<endl;
	float* tmp=new float[length];
	SD_Transfer_read((char*)(filename.c_str()),(long)tmp,(long)(sizeof(float)*length));
	for(int i=0;i<length;i++){
	//        x[i]=(data_t)(round(tmp[i]*SCALE));
		x[i] = float_to_half(tmp[i]);

	}
	delete [] tmp;
}

void write_data(string filename,data_t* x,int length){
	cout<<"writing "<<filename<<endl;
	float* tmp=new float[length];
	for(int i=0;i<length;i++){
//		tmp[i]=(float)x[i]/(float)SCALE;
		tmp[i]=half_to_float(x[i]);
	}
	SD_Transfer_write((char*)filename.c_str(),(long)tmp,(long)(sizeof(float)*length));
	delete [] tmp;
}






