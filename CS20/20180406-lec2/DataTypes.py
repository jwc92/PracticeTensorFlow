import tensorflow as tf
import numpy as np
## zero tensor ##

t_0 = 19
t_0_zeros = tf.zeros_like(t_0)
t_0_ones = tf.ones_like(t_0)

with tf.Session() as sess:
    print(sess.run(t_0_zeros))
    print(sess.run(t_0_ones))

## 1-D tensor with string ##

t_1 = [b"apple", b"peach", b"grape"]
t_1_zeros = tf.zeros_like(t_1)
# t_1_ones = tf.ones_like(t_1)     errors

with tf.Session() as sess:
    print(t_1)
    print(sess.run(t_1_zeros))
#    print(sess.run(t_1_ones))     errors

## 2-D tensor with boolean ##

t_2 = [[True, False, False],[False, False, False],[True, True, True]]
t_2_zeros = tf.zeros_like(t_2)
t_2_ones = tf.ones_like(t_2)

with tf.Session() as sess:
    print(sess.run(t_2_zeros))
    print(sess.run(t_2_ones))



## TF vs NP data types ##


print(tf.int32 == np.int32)


a =tf.ones([2,2], tf.float32)
print(type(a))
with tf.Session() as sess:
    b = sess.run(a)
print(type(b))