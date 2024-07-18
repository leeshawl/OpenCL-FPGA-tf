#include <assert.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cstring>
#include"CL/opencl.h"
#include"AOCLUtils/aoc1_utils.h"
#include<iostream>
#include<fstream>

#define image_size 784
#define x_size 28*28
#define w_size 784*10
#define b_size 10
#define y_size 10

using namespace aocl_utils;
using namespace std;

//Opencl runtime configuration
static cl_platform_id platform=NULL;
static cl_device_id device=NULL;
static cl_context context=NULL;
static cl_command_queue queue = NULL; 
static cl_kernel mnist_kernel = NULL; 
static cl_program program = NULL;
//Function prototypes
void ReadFloat(const char* filename, cl_float* data);

const char* kernel_name = "mnist_simple";	//定义 kernel 名称	
const char* source_file = "mnist_simple.cl"; //定义 kernel 代码文件的名字
const char* aocx_file = "mnist_simple";	//定义 FPGA编程文件的名字	
//设置测试图片及路径信息,选用测试集的第一张图片,图片中的数字信息为7 
const char * input_file_path="mnist_txt/mnist_img_txt/img_0.txt";
const char* input_label_path = "mnist_txt/mnist_lab_txt/img_lab_0.txt";
cl_int status; 
int main() {
	//Get the OpenCL platform
	clGetPlatformIDs(1, &platform, NULL);
	// Obtain the available OpenCL devices.
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	//Create the context.
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);//Create the command queue .
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	//Create the program,
	std::string binary_file = getBoardBinaryFile(aocx_file, device);
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);//Build the program that was just created,
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);//create the kernel
	mnist_kernel = clCreateKernel(program, kernel_name, &status);//allocate and initialize the input vectors
	cl_float* x, * w, * b;
	x = (cl_float*)alignedMalloc(sizeof(cl_float) * x_size);
	//为输人图像分配空间28X28
	w = (cl_float*)alignedMalloc(sizeof(cl_float) * w_size);
	//为w分配空间 784x10
	b = (cl_f1oat*)alignedMalloc(sizeof(cl_float) * b_size) :	//为bias 分配空间10	
		//create the input buffer
		cl_mem dev_x, dev_w, dev_b, dev_y;
	dev_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * x_size, NULL, &status);
	dev_w = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * w_size, NULL, &status);
	dev_b = clCreateButfer(context, CI_MEM_READ_WRITE, sizeof(ol_tloat) * b_size, NULL, &status);
	dev_y = clCreateButfer(context, CL_MEM_READ_WRITE, sizeof(al_float) * y_size, NULL, &status);
	// load data from text file
	ReadFloat(input_file_path, x);	//输人图像数据
	ReadFloat("w_sim.txt", w);	//输入权值数据	
	ReadFloat("b_sim.txt", b);	//输入偏置数据	
	//load label of input file
	cl_float* input_lab;
	input_lab = (cl_f1oat*)alignedMalloc(sizeof(cl_float) * 1);
	ReadFloat(input_label_path, input_lab);	//输入图像标签	
	printf("label_input is %d\n", (int)(*input_lab));
	//write buffer
	status = clEnqueueWriteBuffer(queue, dev_x, CL_TRUE, 0, sizeof(cl_float) * x_size, x, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, dev_w，CL_TRUE, 0, sizeof(cl_float) * w_size, w, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, dev_b, CL TRUE, 0, sizeof(cl_float) * b_size, b, 0, NULL, NULL);
	//set the arguments
	status = clSetKernelArg(mnist_kernel, 0, sizeof(cl_mem)，(void*) & dev_x);
	status = clSetKernelArg(mnist_kerne1, 1, sizeof(cl_mem), (void*)&dev_w);
	status = clSetKernelArg(mnist_kerne1, 2, sizeof(cl_mem), (void*)&dev_b);
	status = clSetKernelArg(mnist_kerne1, 3, sizeof(cl_mem), (void*)&dev_y);
	//1aunch kernel
	cl_event event_kernel;
	static const size_t GSize[] = { 1 };//mnist_simple的 global size 
	static const size_t WSize[] = { 1 }; //mnist_simple 的 local size
	status = clEnqueueNDRangeKernel(queue，mnist_kernel, 1, 0, GSize, WSize, O, NULL, &event_kernel);
	//read the output
	cl_float* y;
	y = (cl_float*)alignedMalloc(sizeof(cl_float) * y_size);
	status = clEnqueueReadBuffer(queue, dev_y, CL_TRUE, 0, sizeof(cl_float) * y_size, y, o, NULL, NULL);
	//display result
	for (int j = 0; j < y_size; j++)
	{
		printf("j=%i,", j);
		printf("number=%.16f\n", y[j]);
	}
	//display recognaize label
	float tmp = 0;
	char lab;
	for (int j = 0; j < 10; j++)
	{
		if (y[j] > tmp)
		{
			tmp = y[j];
			lab = j;
		}
	}

	printf(" label_recognized=%i \n", lab);

	clFlush(queue);
	clFinish(queue);
	//device side
	clReleaseMemObject(dev_x);
	clReleaseMemObject(dev_w);
	clReleaseMemObject(dev_b);
	clReleaseMemObject(dev_y);
	clReleaseKernel(mnist_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	//hose side 
	free(x);
	free(w);
	free(b);
	free(y);
	free(input_lab);
	return 0;
}

void ReadFloat(const char* filename, cl_float* data)
{
	FILE* fpl;	//定义文件流指针,用于打开读取的文件	
	fpl = fopen(filename, "r+");	//读写方式打开文件	
	int j = 0;
	//逐行读取fp1所指文件中的内容到data中
	while (fscanf(fpl, "%f", &data[j++]) != -1);
	fclose(fpl);	//关闭文件,有打开就要有关闭	
}
void cleanup()
{

}