# OpenCL-FPGA-tf
OpenCL异构计算，FPGA和TensorFlow神经网络实现手写数字识别项目
## 关于
这是一个基于OpenCL的FPGA加速器项目。OpenCL是一种开放的、新兴的跨平台并行编程语言，可用于GPU和FPGA开发。本项目构建了FPGA+CPU异构计算系统，提供了基于OpenCL的双层人工神经网络和FPGA上的简单卷积神经网络加速器设计。首先，在TensorFlow中实现神经网络模型的训练和测试，然后构建基于Intel FPGA的OpenCL操作平台。最后，在FPGA上实现了手写数字识别并进行了测试。
## 神经网络结构
ANN网络结构为784×100×10，中间含有一个隐藏层。

CNN网络包含一个卷积层一个池化层和两个全连接层，卷积层使用4个3×3×3×1的卷积核，步长为1，激活函数为ReLU。网络包含一个卷积层一个池化层和两个全连接层，卷积层使用4个3×3×3×1的卷积核，步长为1，激活函数为ReLU。
## 步骤
1.神经网络的TensorFlow实现及训练
python my_mnist_simple_fpga.py

运行完成会对测试图片进行测试，并会生成两个txt文件b_sim.txt和w_sim.txt他们分别保存了训练后的偏置和权值，需要写入到FPGA开发板上。

2.kernel代码编写及编译

对kernel进行编译，运行终端命令 aoc device/mnist_simple.cl -o bin/mnist_simple.aocx -board=de10_nano_sharedonly -v -report

会生成两个新文件夹，其中mnist_simple.aocx是FPGA编程文件，需要复制到FPGA开发板上，mnist_simple文件夹下包含了kernel编译相关的信息。

3.host代码编写及编译

host文件在/ANN/intelFPGA/de10_nano/test/mnist_simple_one_image/host/src的main.cpp文件

需要在之前下载的IntelFPGA里的heool_word文件下复制Makefile到mnist_simple_one_image中，然后运行终端命令make

查看mnist_simple_one_image/bin目录下是否有host文件生成

4.移植到FPGA开发板

将b_sim.txt和w_sim.txt以及mnist_txt文件夹中的所以txt文件复制到DE10_nano开发板的SD卡中

5.FPGA运行神经网络

初始化OpenCL运行环境

source ./init_opencl.sh

切换至mnist_simple_one_image目录下运行终端命令

./host
## 工具
DE10_nano 开发板，基于Intel FPGA的片上系统硬件设计平台

Quartus Prime Standard

Intel FPGA SDK for OpenCL

Intel SoC FPGA EDS

DE10_nano BSP

minicom

TensorFlow
