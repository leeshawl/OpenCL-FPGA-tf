#coding: utf-8
import tensorflow as tf
import numpy as np
import sys
sys.path.append("/home/tensorflow/design/my_mnist/")
import pylab
import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples //batch_size
n_batch = 10
#自定义卷积函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

#自定义池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x=tf.placeholder(tf.float32,[None,in_units])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])

#设置第一个卷积层和池化层
w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 4], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[4]))
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#设置第一个全连接层
w_fc1 = tf.Variable(tf.truncated_normal([14 * 14 * 4, 50], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[4]))
h_pool1_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

#设置第二个全连接层
w_fc2 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_out = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_out))

train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init= tf.global_variables_initializer()

correct_prediction=tf.equal(tf.argmax(y_out,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(101):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y_:mnist.train.labels,keep_prob: 1.0})
            print("Iter " + str(epoch) + ", Training Accuracy= " + str(train_acc))
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob: 1.0})
            print("Iter " + str(epoch) + ", Testing Accuracy= " + str(test_acc))

img_in=mnist.test.images[0]
im=img_in.reshape([-1,28])
pylab.imshow(im,cmap='gray')
pylab.show()

w_conv1_val, b_conv1_val = sess.run([w_conv1, b_conv1])
w_fc1_val, bfc1_val = sess.run([w_fc1, b_fc1])
w_fc2_val, bfc2_val = sess.run([w_fc2, b_fc2])

np.savetxt("./conv_txt/w_conv1.txt", w_conv1_val.reshape(-1), fmt="%f", delimiter=",")
np.savetxt("./conv_txt/b_conv1.txt", b_conv1_val.reshape(-1), fmt="%f", delimiter=",")
np.savetxt("./conv_txt/w_fc1.txt", w_fc1_val.reshape(-1), fmt="%f", delimiter=",")
np.savetxt("./conv_txt/b_fc1.txt", bfc1_val.reshape(-1), fmt="%f", delimiter=",")
np.savetxt("./conv_txt/w_fc2.txt", w_fc2_val.reshape(-1), fmt="%f", delimiter=",")
np.savetxt("./conv_txt/b_fc2.txt", bfc2_val)
print("text write sucessful")