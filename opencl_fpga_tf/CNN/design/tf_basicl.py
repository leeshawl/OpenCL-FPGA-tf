import tensorflow as tf

a=tf.constant(3.0,name="input1")
b=tf.constant(2.0,name="input2")
c=tf.add(a,b,name="my_add_op")
d=tf.multiply(a,c,name="my_mul_op")

sess=tf.Session()
a_val=sess.run(a)
out=sess.run(d)
sess.close()

print(a)
print(a_val)
print(b)
print(c)
print(d)
print(out)