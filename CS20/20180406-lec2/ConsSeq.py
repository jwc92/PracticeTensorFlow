import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


## tf.constant ##
print("Costnat....")
a = tf.constant([2,2], name ='a')
b = tf.constant([[0,1],[2,3]], name='b')

x = tf.multiply(a, b, name='mul')

with tf.Session() as sess:
    print(sess.run(x))

## tf.zeros ##
print("Zeros....")
zeros = tf.zeros([2, 3], tf.int32)
input1 = tf.constant([  [0, 1],
                        [2, 3],
                        [4, 5]])
zero_samesize_with_input1 = tf.zeros_like(input1)

with tf.Session() as sess:
    print(sess.run(zeros))
    print(sess.run(zero_samesize_with_input1))

    
## tf.ones ##
print("Ones....")
ones = tf.ones([2, 3], tf.int32)
input2 = tf.constant([  [0, 1],
                        [2, 3],
                        [4, 5]])
ones_samesize_with_input2 = tf.ones_like(input2)

with tf.Session() as sess:
    print(sess.run(ones))
    print(sess.run(ones_samesize_with_input2))

## tf.fill ##
print("fills....")
fill_with_eight = tf.fill([2,3], 8)

with tf.Session() as sess:
    print(sess.run(fill_with_eight))

## Constants as sequences ##
print("Sequence.....")
A = tf.lin_space(10.0, 13.0, 4) # [10. 11. 12. 13.]
B = tf.range(3, 18, 3)          # [3 6 9 12 15]  
C = tf.range(5)                 # [0 1 2 3 4]

with tf.Session() as sess:
    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(C))

## Randomly Generated Constants ##
# tf.random_normal
# tf.truncated_normal
# tf.random_uniform
# tf.random_shuffle
# tf.random_crop
# tf.multinomial
# tf.random_gamma

## seed ##
# tf.set_random_seed(seed)

## Wizard of Div ##
a = tf.constant([2,2], name='a')
b = tf.constant([[0,1],[2,3]], name='b')
with tf.Session() as sess:
    print(sess.run(tf.div(b,a)))             #  [[0 0] [1 1]] 
    print(sess.run(tf.divide(b,a)))          #  [[0 0.5] [1. 1.5]] 
    print(sess.run(tf.truediv(b,a)))         #  [[0 0.5] [1. 1.5]] 
    print(sess.run(tf.floordiv(b,a)))        #  [[0 0] [1 1]] 
#    print(sess.run(tf.realdiv(b,a)))        # only works for real values
    print(sess.run(tf.truncatediv(b,a)))     #  [[0 0] [1 1]] 
    print(sess.run(tf.floor_div(b,a)))       #  [[0 0] [1 1]] 

    