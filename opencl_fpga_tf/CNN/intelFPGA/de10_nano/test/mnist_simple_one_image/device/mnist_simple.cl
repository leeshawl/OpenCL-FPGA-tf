__kernel void mnist_simple(__global const float *restrict dev_x, //30*30, host对28*28的数据进行处理，右侧补两侧0，下端补两行0
                           __global const float *restrict dev_w1, //3*3*1×4
                           __global const float *restrict dev_b1, //4
                           __global float *restrict dev_y1) //14*14*4

{
int ch_id=get_global_id(0) ;//0,1,2,3通道数量channel_num
float result[4][4]={{0.0}};
int add_index[4]={0,1,30,31};
float tmp1[4],tmp2[4],tmp3[4]={0.0};
for (int row=0; row<14;row++)
{
  for(int col=0;col<14; col++)
  {
  for (int k=0;k<4;k++)	//池化窗为2x2=4
  {
  result[ch_id][k]=0;
  for(int i=0;i<3;i++)	//行,纵向
  {
  for(int j=0;j<3;j++)	//列,横向
  {
  result[ch_id][k]=result[ch_id][k]+dev_x[row* 30*2+col*2+30* i+j+add_index[k]]*dev_w1[ch_id+(4* i+12*j)];
  }
  }
result[ch_id][k]=result[ch_id][k]+dev_b1[ch_id];
result[ch_id][k]=((result[ch_id][k]>0)? result[ch_id][k]:0);//Relu函数
}

//pool
tmp1[ch_id]=(result[ch_id][0]>result[ch_id][1])? result[ch_id][0]: result[ch_id][1];
tmp2[ch_id]=(result[ch_id][2]>result[ch_id][3])? result[ch_id][2]: result[ch_id][3];
tmp3[ch_id]=(tmp1[ch_id]>tmp2[ch_id])? tmpI[ch_id]:tmp2[ch_id];

dev_y1[ch_id+row * 14*14+4*col]=tmp3[ch_id];//输出给fc1,格式为14X14X4
}
}
}

__kernel void fc1(__global float * restrict dev_y1,	//14X14X4
                  __global float * restrict dev_w2,	//14X14x4X50
                  __global float * restrict dev_b2,	//50
                  __g1obal float * restrict dev_y2)
{
int ch_id=get_global_id(0);	//0,1,..,49
float result[50]=(0.0);
for(int i=0;i<196*4;i++)
{
result[ch_id]=result[ch_id]+dev_y1[i] * dev_w2[ch_id+i*50];
}
result[ch_id]=result[ch_id]+dev_b2[ch_id];
result[ch_id]=(result[ch_id]>0)? result[ch_id]:0;
dev_y2[ch_id]=result[ch_id];
}

//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void fc2(__global float * restrict dev_y2,//50
                  __global float * restrict dev_w3,	//50*10
                  __global float * restrict dev_b3,	//10
                  __global float * restrict dev_y3)
{
int ch_id-get_global_id(0);	//0,1,...,9
float result[10]={0.0};

for (int i=0;i<50;i++)	//50为fcl通道数量
{
result[ch_id]=result[ch_id]+dev_y2[i]* dev_w3[ch_id+10*1];
}
result[ch_id]=result[ch_id]+dev_b3[ch_id];
dev_y3[ch_id]=result[ch_id];
}