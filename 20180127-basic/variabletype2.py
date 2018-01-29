import tensorflow as tf

value1=5
value2=3
value3=2

ph1 = tf.placeholder(dtype = tf.float32 )
ph2 = tf.placeholder(dtype = tf.float32 )
ph3 = tf.placeholder(dtype = tf.float32 )

result_value = ph1* ph2 + ph3

feed_dict1 = {ph1: value1, ph2: value2, ph3: value3}

sess= tf.Session()
result = sess.run(result_value, feed_dict = feed_dict1)

print(result)
