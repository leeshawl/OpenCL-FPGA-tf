__kernel void mnist_simple(__global const float *restrict dev_x, //28*28
                           __global const float *restrict dev_w1, //784*100
                           __global const float *restrict dev_b1, //100
						   __global const float *restrict dev_w2, //784*100
                           __global const float *restrict dev_b2, //10
                           __global float *restrict dev_y2) //10

{
__local float rt1[100];
__local float rt2[10];
__local float y1[100];

for (int k=0;k<10;k++)
{
rt2[k]=0.0;
#pragma unrool 1
for (int thread_id=0;thread_id<100;thread_id++)
{
rt1[thread_id]=0.0;
#pragma unrool 1
for (int i=0;i<784;i++)
{
re1[thread_id]=rt1[thread_id]+dev_x[i]*dev_w1[i*100+thread_id];
}
y1[thread_id]=rt1[thread_id]+dev_b1[thread_id];
y1[thread_id]=(y1[thread_id]>0)?(y1[thread_id]):(0);
rt2[k]=rt2[k]+y1[thread_id]*dev_w2[thread_id*10+k];
dev_y2[k]=rt2[k]+dev_b2[k];
}
}
}