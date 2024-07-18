#coding:utf-8
import tensorflow as tf
import sys
sys.path.append("/home/tensorflow/design/my_mnist")
import input_data

mnist = input_data.read_data_sets("/home/tensorflow/design/my_mnist/MNIST_data", one_hot=True)
with tf.Session() as sess:
    import numpy as np
    for i in range(101):
        img_in_i=mnist.test.images[i]
        tag_i=np.argmax(mnist.test.labels[i])
        np.savetxt('./mnist_txt/mnist_img_txt/img_%d.txt' % i, img_in_i.reshape(-1), fmt="%f.31f",delimiter=",")
        np.savetxt('./mnist_txt/mnist_lab_txt/img_lab_%d.txt' % i, img_in_i.reshape(-1), fmt="%f.31f", delimiter=",")
        print("text write sucessful")
