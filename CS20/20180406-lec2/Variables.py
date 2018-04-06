import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# create variables with tf.Variable
s_ = tf.Variable(2, name="scalar")
m_ = tf.Variable([[0, 1],[2, 3]], name="matrix")
W_ = tf.Variable(tf.zeros([784,10]))

# create variables with tf.get_variable
# This is better because unlike tf.constant, tf.Variable is Class 
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0,1 ],[2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

# For example.....
# x = tf.Variable()
# x.initialzier
# x.value
# x.assign
# x.assign_add


print("W....")
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())         # initialize all variables
    sess.run(tf.variables_initializer([s, W]))
    print(sess.run(W))



print("W.eval()....")
## Eval() a variable ##
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W)
    print(W.eval())


print("W.assign()")
# tf.Variable.assign()
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

print("W.assign() in tf.Session")
assign_op = W.assign(100)
two_times = W.assign(2*W)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print(W.eval())
    sess.run(two_times)
    sess.run(two_times)
    sess.run(two_times)
    print(W.eval())


print("assign.add and sub...")
my_var = tf.Variable(10)

with tf.Session() as sess:
    sess.run(my_var.initializer)

    sess.run(my_var.assign_add(10))
    print(sess.run(my_var))
    sess.run(my_var.assign_sub(2))
    print(sess.run(my_var))
    