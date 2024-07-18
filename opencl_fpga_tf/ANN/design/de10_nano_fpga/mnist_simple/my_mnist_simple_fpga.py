#coding: utf-8
import tensorflow as tf
import numpy as np
import sys
sys.path.append("/home/tensorflow/design/my_mnist/")
import pylab
import input_data

mnist=input_data.read_data_sets("/home/tensorflow/design/my_mnist/MNIST_data",one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples //batch_size

in_units=784
h1_units=100
x=tf.placeholder(tf.float32,[None,in_units])
y=tf.placeholder(tf.float32,[None,10])

W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))
keep_prob = tf.placeholder(tf.float32)
hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
prediction=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init= tf.global_variables_initializer()

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(101):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob: 1.0})
            print("Iter " + str(epoch) + ", Training Accuracy= " + str(train_acc))
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob: 1.0})
            print("Iter " + str(epoch) + ", Testing Accuracy= " + str(test_acc))

img_in=mnist.test.images[0]
im=img_in.reshape([-1,28])
pylab.imshow(im,cmap='gray')
pylab.show()

w1_sim_val, b1_sim_val, w2_sim_val, b2_sim_val = sess.run([W1, b1, W2, b2])
np.savetxt("w1_sim.txt",w1_sim_val.reshape(-1), fmt="%f.31f", delimiter=",")
np.savetxt("b1_sim.txt", b1_sim_val.reshape(-1), fmt="%f.31f", delimiter=",")
np.savetxt("w2_sim.txt",w2_sim_val.reshape(-1), fmt="%f.31f", delimiter=",")
np.savetxt("b2_sim.txt", b2_sim_val.reshape(-1), fmt="%f.31f", delimiter=",")
print("text write sucessful")