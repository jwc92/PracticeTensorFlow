import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
# name is what will be shown in tensorboard

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(x))
writer.close()

# python3 [yourprogram.py]
# tensorboard --logdir="./graphs" --port 6006
# Then open your browser and go to: http://localhost:6006/