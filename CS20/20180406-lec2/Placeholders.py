import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.placeholder(tf.float32, shape = [3])
b = tf.constant([5,5,5], tf.float32)

c = a + b

with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))
    



## Normal loading and Lazy loading ##


# Normal loading

x = tf.Variable(10, name='x')
y = tf.Variable(10, name='y')
z = tf.add(x,y)               ## z is created before executing the graph

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)
writer.close()

## Lazy loading

x = tf.Variable(10, name='x')
y = tf.Variable(10, name='y')

writer = tf.summary.FileWriter('./graphs/normal_loading2', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x,y))
writer.close()

