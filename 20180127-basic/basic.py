
import tensorflow as tf

image = [1,2,3,4,5]
label = [10,20,30,40,50]

ph_image = tf.placeholder(dtype = tf.float32)
ph_label = tf.placeholder(dtype = tf.float32) 

feed_dict = {ph_image:image, ph_label:label}

result_tensor = ph_image + ph_label

sess= tf.Session()	
result = sess.run(result_tensor, feed_dict=feed_dict)
print(result)
