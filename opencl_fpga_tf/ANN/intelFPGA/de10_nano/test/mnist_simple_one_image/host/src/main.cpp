#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <iostream>
#include <fstream>

#define image_size 784
#define x_size 28*28
#define w1_size 784*100
#define b1_size 100
#define w2_size 100*10
#define b2_size 10
#define y_size 10

using namespace aocl_utils;
using namespace std;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel mnist_kernel = NULL;
static cl_program program = NULL;

// Function prototypes
void ReadFloat(const char* filename, cl_float* data);

const char* kernel_name = "mnist_simple"; // ���� kernel ����
const char* source_file = "mnist_simple.cl"; // ���� kernel �����ļ�������
const char* aocx_file = "mnist_simple"; // ���� FPGA����ļ�������

// ���ò���ͼƬ��·����Ϣ, ѡ�ò��Լ��ĵ�һ��ͼƬ, ͼƬ�е�������ϢΪ7 
const char* input_file_path = "mnist_txt/mnist_img_txt/img_0.txt";
const char* input_label_path = "mnist_txt/mnist_lab_txt/img_lab_0.txt";
cl_int status;

int main() {
    // Get the OpenCL platform
    clGetPlatformIDs(1, &platform, NULL);
    // Obtain the available OpenCL devices.
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    // Create the context.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    // Create the command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    // Create the program
    std::string binary_file = getBoardBinaryFile(aocx_file, device);
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
    // Build the program that was just created
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    // Create the kernel
    mnist_kernel = clCreateKernel(program, kernel_name, &status);

    // Allocate and initialize the input vectors
    cl_float* x = (cl_float*)alignedMalloc(sizeof(cl_float) * x_size);
    cl_float* W1 = (cl_float*)alignedMalloc(sizeof(cl_float) * w1_size);
    cl_float* b1 = (cl_float*)alignedMalloc(sizeof(cl_float) * b1_size);
    cl_float* W2 = (cl_float*)alignedMalloc(sizeof(cl_float) * w2_size);
    cl_float* b2 = (cl_float*)alignedMalloc(sizeof(cl_float) * b2_size);

    // Create the input buffer
    cl_mem dev_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * x_size, NULL, &status);
    cl_mem dev_W1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * w1_size, NULL, &status);
    cl_mem dev_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * b1_size, NULL, &status);
    cl_mem dev_W2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * w2_size, NULL, &status);
    cl_mem dev_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * b2_size, NULL, &status);
    cl_mem dev_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * y_size, NULL, &status);

    // Load data from text file
    ReadFloat(input_file_path, x); // ����ͼ������
    ReadFloat("W1_sim.txt", W1); // ����Ȩֵ����
    ReadFloat("b1_sim.txt", b1); // ����ƫ������
    ReadFloat("W2_sim.txt", W2); // ����Ȩֵ����
    ReadFloat("b2_sim.txt", b2); // ����ƫ������

    // Load label of input file
    cl_float* input_lab = (cl_float*)alignedMalloc(sizeof(cl_float) * 1);
    ReadFloat(input_label_path, input_lab); // ����ͼ���ǩ
    printf("label_input is %d\n", (int)(*input_lab));

    // Write buffer
    status = clEnqueueWriteBuffer(queue, dev_x, CL_TRUE, 0, sizeof(cl_float) * x_size, x, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_W1, CL_TRUE, 0, sizeof(cl_float) * w1_size, W1, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_b1, CL_TRUE, 0, sizeof(cl_float) * b1_size, b1, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_W2, CL_TRUE, 0, sizeof(cl_float) * w2_size, W2, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_b2, CL_TRUE, 0, sizeof(cl_float) * b2_size, b2, 0, NULL, NULL);

    // Set the arguments
    status = clSetKernelArg(mnist_kernel, 0, sizeof(cl_mem), (void*)&dev_x);
    status = clSetKernelArg(mnist_kernel, 1, sizeof(cl_mem), (void*)&dev_W1);
    status = clSetKernelArg(mnist_kernel, 2, sizeof(cl_mem), (void*)&dev_b1);
    status = clSetKernelArg(mnist_kernel, 3, sizeof(cl_mem), (void*)&dev_W2);
    status = clSetKernelArg(mnist_kernel, 4, sizeof(cl_mem), (void*)&dev_b2);
    status = clSetKernelArg(mnist_kernel, 5, sizeof(cl_mem), (void*)&dev_y);

    // Launch kernel
    cl_event event_kernel;
    static const size_t GSize[] = { 1 }; // mnist_simple�� global size 
    static const size_t WSize[] = { 1 }; // mnist_simple �� local size
    status = clEnqueueNDRangeKernel(queue, mnist_kernel, 1, 0, GSize, WSize, 0, NULL, &event_kernel);

    // Read the output
    cl_float* y = (cl_float*)alignedMalloc(sizeof(cl_float) * y_size);
    status = clEnqueueReadBuffer(queue, dev_y, CL_TRUE, 0, sizeof(cl_float) * y_size, y, 0, NULL, NULL);

    // Display result
    for (int j = 0; j < y_size; j++) {
        printf("j=%i, number=%.16f\n", j, y[j]);
    }

    // Display recognized label
    float tmp = 0;
    char lab;
    for (int j = 0; j < 10; j++) {
        if (y[j] > tmp) {
            tmp = y[j];
            lab = j;
        }
    }
    printf("label_recognized=%i \n", lab);

    clFlush(queue);
    clFinish(queue);

    // Device side cleanup
    clReleaseMemObject(dev_x);
    clReleaseMemObject(dev_W1);
    clReleaseMemObject(dev_b1);
    clReleaseMemObject(dev_W2);
    clReleaseMemObject(dev_b2);
    clReleaseMemObject(dev_y);
    clReleaseKernel(mnist_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Host side cleanup
    free(x);
    free(W1);
    free(b1);
    free(W2);
    free(b2);
    free(y);
    free(input_lab);

    return 0;
}

void ReadFloat(const char* filename, cl_float* data) {
    FILE* fpl = fopen(filename, "r+"); // ��д��ʽ���ļ�
    int j = 0;
    // ���ж�ȡfp1��ָ�ļ��е����ݵ�data��
    while (fscanf(fpl, "%f", &data[j++]) != -1);
    fclose(fpl); // �ر��ļ�, �д򿪾�Ҫ�йر�
}

void cleanup() {
    // Cleanup code if needed
}