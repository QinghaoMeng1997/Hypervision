#include<cstdlib>
#include<cstdio>
#include<string>
#include<cmath>
#include<iostream>
#include"ff.h"
#include "xparameters.h"
#include "xdevcfg.h"
using namespace std;

#define SCALE 512
typedef uint16_t data_t;
typedef uint16_t data_n;

static FATFS fatfs;

int SD_Init();

void write_data(string filename,data_t* x,int length);

void read_params(string filename,data_t* x,int length);
void read_params_half(string filename,data_n* x,int length);
float half_to_float(uint16_t h_bits);
uint16_t float_to_half(float value);
