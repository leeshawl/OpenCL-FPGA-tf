include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h”
#include "AOCLUtils/aocl_utils.h"
#include <iostream>
#include <fstream>
#define total_number_image 101
#define image_size 784
#define x_size 28*28
#define xl_size 30* 30
#define wl_size 36
#define bl_size 4
#define yl_size 196*4
#define w2_size 196*4*50 
#define b2_size 50
#define y2_size 50
#define w3_size 50*10 
#define b3_size 10
#define y3_size 10

using namespace aocl_utils; 
using namespace std;

//OpenCL runtime configuration
static cl_platform_id platform = NULL; 
static cl_device_id device = NULL; 
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel mnist_kernel1, mnist_kerne12, mnist_kerne13 = NULL; 
static cl_program program = NULL;
static cl_mem dev_x, dev_w1, dev_b1, dev_y1, dev_w2, dev_b2, dev_y2, dev_w3, dev_b3, dev_y3 = NULL;
//Function prototypes
void ReadFloat(char* filename, cl_float* data);
double GetKernelExecutionTime(cl_command_queue cmd, cl_event event, char* eventname);

const char* kernel_namel = "conv1";
const char* kernel_name2 = "tel";
const char* kernel_name3 = "fc2";
const char* source_file = "mnist.cl"; 
const char* aocx_file = "mnist";

char input_tile_name[5]; 
char suffix[5] = ",txt"; 
int label_in[1];

cl_int status;

int img_rec_suc = 0;

float correct_ratio = 0;
float kernel1_execution_time[total_number_image]; 
float kerne12_execution_time[total_number_image]; 
float kernel3_execution_time[total_number_image];

int main() {
    //Get the OpencL platform.
    clGetPlatformIDs(l, splatform, NULL);
    //Query the available OpenCL devices,
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL) :
        //Create the context.
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    //Create the command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABL, &status);
    //Create the program.
    std::string binary_file = getBoardBinaryFile(aocx_file, device);
    program = createProgramFromBinary(context, binary_file.c_str()， & device, 1);
    //Build the program that was just created.
    status = clBuildProgram(program, O, NULL, "", NULL, NULL);
    /************/


    double timel = getCurrentTimestamp();

    //cteate the kernel
    mnist_kernell = clCreateKernel(program, kernel_namel, &status);
    mnist_kerne12 = c1CreateKernel(program, kernel_name2, &status);
    mnist_kernel3 = clCreateKernel(program, kernel_name3, &status);
    //allocate and initialize the input vectors
    cl_float* x, * xl, * wl, * bl, * yl, * w2, * b2, * y2, * w3, * b3, * y3;
    x = (cl_float*)alignedMalloc(sizeof(cl_float) * x_size);// 
    x1 = (cl_float*)alignedMalloc(sizeof(cl_float) * x1_size);// 
    wl = (cl_t1oat*)alignedMalloc(sizeof(cl_float) * w1_size);// 
    b1 = (cl_float*)alignedMalloc(sizeof(cl_float) * b1_size) :// 
        yl = (cl_float*)alignedMalloe(sizeof(cl_float) * y1_size) :// 
        w2 = (cl_float*)alignedMa1loc(sizeof(cl_float) * w2_size) :// 
        b2 = (cl_float*)alignedMalloc(sizeof(cl_float) * b2_size);// 
    y2 = (cl_f1oat*)alignedMalloc(sizeof(cl_float) * y2_size);// 
    w3 = (cl_float*)alignedMalloc(sizeof(cl_float) * w3_size) :// 
        b3 = (c1_float*)alignedMalloc(sizeof(cl_float) * b3_size);// 
    y3 = (c1_float*)alignedMalloc(sizeof(cl_float) * y3_size);//

    //create the input buffer

    cl_mem dev_x1, dev_w1, dev_b1, dev_y1, dev_w2, dev_b2, dev_y2, dev_w3, dev_b3, dev_y3;
    dev_x1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * x1_size, NULL, &status);
    dev_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * wl_size, NULL, &status);
    dev_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * bl_size, NULL, &status);
    dev_y1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * y1_size, NULL, &status);
    dev_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * w2_size, NULL, &status);
    dev_b2 = c1CreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * b2_size, NULL, &status);
    dev_y2 = c1CreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * y2_size, NULL, &status);
    dev_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * w3_size, NULL, &status);
    dev_b3 = clcreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_t1oat) * b3_size, NULL, &status);
    dev_y3 = c1CreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * y3_size, NULL, &status);
    //load data from text file
    ReadFloat("w_convl.txt", w1);
    ReadFloat("b_conv1.txt", bl);
    ReadFloat("w_fcl.txt", w2);
    ReadFloat("b_tcl,txt", b2);
    ReadFloat("w_fc2,txt", w3);
    ReadFloat("b_fc2,txt", b3);

    //Write constant buffer
    status = clEnqueueWriteBuffer(queue, dev_w1, CL_TRUE, 0, sizeof(cl_float) * wl_size, wl, O, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_b1, CL_TRUE, 0, sizeof(cl_float) * bl_size, bl, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_w2, CL_TRUE, 0，sizeof(cl_float) * w2_size, w2, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_b2, CL_TRUE, 0，sizeof(cl_float) * b2_size, b2, 0, NULL, NULL);
    status - clEnqueueWriteBuffer(queue, dev_w3, CL_TRUE, 0, sizeof(cl_float) * w3_size, w3, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, dev_b3, CL_TRUE, 0, sizeof(cl_float) * b3_size, b3, 0, NULL, NULL);
    //read the input image file
    for (int img_index = 0; img_index<int(total_number_image); img_index++) {
        sprintf(input_file_name, "&d", img_index);
        char input_file_path[45] = "./mnist_txt/mnist_img_txt/img_";
        char input_lab_path[45] = "./mnist_txt/mnist_lab_txt/img_lab_";
        printf("####################\n");
        printf(" input_filename=\033[7mimg_&s.txt\033[0m\n", input_file name);
        //reading data from text file
        ReadFloat(strcat(strcat(input_file_path, input_file_name), suffix), x);//

        //pading for input data
        for (int row = 0; row < 30; row++)
        {
            for (int col = 0; col < 30; co1++)
            {
                x1[30 * row + co1] = 0.0;
            }
        }
        for (int row = 0; row < 28; row++)
        {
            for (int col = 0; col < 28; co1++)
            {
                x1[28 * row + col + 2 * row] = x[28 * row + co1];
            }
        }

        printf("*******************\n");
        printf("\033[7m \033[40;31m ****read image sucessful********\033[0m\n");
        printf("******************************\n");
        //////////////////////////	
        printf("" * *********************************************\n");
            printf(" *    input_lab_filename=\033[7mimg_1ab_%s.txt \033[0m\n", input_file_name);
        printf("******************************************\n");
        //read the label of input image file

        FILE * fpl;
        fp1 = fopen(strcat(strcat(input_lab_path, input_file_name), suffix), "r");
        fscanf(fp1, "%i", label_in); fclose(fp1);
        printf(*****************************\n");	
            printf(" *\033[7m label of input image=si\033[0m\n", label_in[0]);
        printf("*************************************\n");
        //Write input image buffer
        status = clEnqueueWriteBuffer(queue, dev_x1, CL_TRUE, 0, sizeof(cl_float) * xl_size, x1, 0, NULL, NULL);

        //set the argumenta
        status = clSetKernelArg(mnist_kerne1l, 0, sizeof(cl_mem)，(void*) & dev_x1);
        status = clSetKernelArg(mniet_kernel1, 1, sizeof(cl_mem)，(void*) & dev_w1);
        status = clSetKernelArg(mnist_kerne11, 2, sizeof(cl_mem)，(void*) & dev_bl);
        status = clSetKernelArg(mnist_kerne11, 3, sizeof(c1_mem)，(void*) & dev_y1);

        status = clSetKernelArg(mnist_kernel2, 0, sizeof(c1_mem)，(void*) & dev_y1);
        status = clSetKernelArg(mnist_kernel2, 1, sizeof(c1_mem)，(void*) & dev_w2);
        status = clSetKernelArg(mnist_kernel2, 2, sizeof(cl_mem)，(void*) & dev_b2);
        status = clSetKernelArg(mnist_kernel2, 3, sizeof(cl_mem)，(void*) & dev_y2);

        status = clSetKernelArg(mnist_kerne13, 0, sizeof(c1_mem)，(void*) & dev_y2);
        status = clSetKernelArg(mnist_kernel3, 1, sizeof(cl_mem), (void*)&dev_x3);
        status = clSetKernelArg(mnist_kerne13, 2, sizeof(cl_mem), (void*)&dev_b3);
        status = clSetKernelArg(mnist_kernel3, 3, sizeof(cl_mem), (void*)&dev_y3);
        //launch kernell
        static const size_t GSize1[] = { 4 }; //mnist_simple的 global size 
        static const size_t WSize1[] = { 1 };//mnist_simple的 local size 
        cl_event event_kernell;
        char dim = 1;

        status = clEnqueueNDRangeKernel(queue, mnist_kernel1, dim, 0, GSize1, WSize1, 0, NULL, &event_kernel1);
        /*************count runtime of kerne1**************************/
        double time_sum1 = 0;
        time_suml = GetKernelExecutionTime(queue, event_kernel1, "event_cluster");//
        kernel1_execution_time[img_index] = time_sum1 / 1000000;
        **************/


            //read the output of kernel1
            status = clEnqueueReadBuffer(queue, dev_yl, CL_TRUE, 0, sizeof(cl_float) * yl_size, y1, 0, NULL, NULL);
        //launch kernel2
        static const size_t GSize2[] = { 50 };//mnist_simple的 global size 
        static const size_t WSize2[] = { 1 };//mnist_simple的 local size 
        cl_event event_kernel2;
        status = clEnqueueNDRangeKernel(queue, mnist_kernel2, dim, 0，GSize2, WSize2, 0, NULL, &event_kernel2);

        /**count runtime of kernel****/
        double time_sum2 = 0;
        time_sum2 = GetKernelExecutionTime(queue, event_kerne12, "event_cluster");//	
        kernel2_execution_time[img_index] = time_sum2 / 1000000;

        //read the cutput of kernel2
        status = clEnqueueReadBuffer(queue, dev_y2, CL_TRUE, 0，sizeof(cl_float) * y2_size, y2, 0, NULL, NULL) :
            // launch kernel3
            static const size_t GSize3[] = { 10 }; //mnist_simple的 global size 
        static const size_t WSize3[] = { 1 };//mnist_simple的 local size
        cl_event event_kernel3;

        status = clEnqueueNDRangeKernel(queue, mnist_kernel3, dim, 0, GSize3, WSize3, 0, NULL, &event_kernel3);
        /*************count runtime of kernel**************************/
        double time_sum3 = 0;
        time_sum3 = GetKernelExecutionTime(queue, event_kernel3, "event_cluster");// 
        kernel3_execution_time[img_index] = time_sum3 / 1000000;

        *************************/

            //read the output of kernel3
            status = clEnqueueReadBuffer(queue, dev_y3, CL_TRUE, 0, sizeof(cl_float) * y3_size, y3, 0, NULL, NULL);
        //display result
        for (int j = 0; j < y3_size; j++)
        {
            printf("j=%i,", j);
            printf("number=%.16f\n", y3[j]);
        }

        //display recognaize label
        cl_float tmp = 0;
        int lab;
        for (int j = 0; j < 10; j++)
        {
            if (y3[j] > tmp)
            {
                tmp = y3[j];
                lab = j;
            }
        }
        printf(*********************\n");	
            printf("*	\033[7m label recognized=%i \033[0m\n", 1ab);
        printf("*******************************************\n");
        if (lab == label_in[0])
        {
            img_rec_suc = img_rec_suc + l;
        }
    }

    double time2 = getCurrentTimestamp();

    correct_ratio = float(img_rec_suc) / int(total_number_image);
    printf("*********************************************\n");
    printf("*	img_input_num=ti\n", total_number_image);
    printf("*	img_rec_suc_num=\033[7m%i\033[0m	\n", img_rec_suc);
    printf(" *	correct ratio=\033[7m%f\033[0m	\n", correct_ratio);
    printf("*********************************************\n");


    //print runtime of kernel
    printf("kernel_execution_time is:\n");
    for (int i = 0; i < total_number_image; i++)
        printf("%f\n", kernel1_execution_time[i] + kerne12_execution_time[i] + kernel3_execution_time[i]);
    printf("total_time is:%f ms\n", (time2 - time1) * 1e3);

    //////////////////////////1
    clFlush(queue);
    clFinish(queue);
    //device side
    clReleaseMemObject(dev_x);
    clReleaseMemObject(dev_wl);
    clReleaseMemObject(dev_b1);
    clReleaseMem0bject(dev_y1);
    clReleaseMemObject(dev_w2);
    clReleaseMemObject(dev_b2);
    clReloaseMem0bject(dev_y2);
    clReleaseMemobject(dev_w3);
    clReleaneMemObject(dev_b3);
    clReleaseMemObject(dev_y3);
    clReleaseKernel(mnist_kernel1);
    clReleaseKernel(mnist_kerne12);
    clReleaseKernel(mnist_kerne13);
    clReleaseProgram(program);
    elReleaseCommandQueue(queue);
    clReleaseContext(context);

    //hose side 
    free(x);
    free(wl);
    free(bl);
    free(yl);
    free(w2);
    free(b2);
    free(y2);
    free(w3);
    free(b3);
    free(y3);

    return 0;
}

     //*********************************Added by Page**************************// 
void ReadFloat(char* filename, cl_float* data)
{
    FILE* fpl;
    fp1 = fopen(filename, "r+");
    int j = 0;
    while (fscanf(fpl, "%f", &data[j++]) != -1);
    fclose(fpl);
}

void cleanup() 
{

}

double GetKernelExecutionTime(cl_command_queue cmd, cl_event event, char *eventname)
{
    cl_ulong start, end;
    clFinish(cmd);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ ulong), &end, NULL);
    double runtime = (double)(end - start); 
    return runtime;
}

