import tensorflow as tf


#ph = tf.placeholder(dtype = tf.float32, shape=(3,3))
##ar = tf.Variable([1,2,3,4,5], dtype = tf.float32)
#const = tf.constant([10, 20, 30 ,40, 50], dtype=tf.float32)

# print (ph)
# print (var)
# print (const)



#result = sess.run(const)
#print (result)

#a = tf.constant([5])
#b = tf.constant([10])
#c = tf.constant([2])

#d=a*b+c
#result=sess.run(d) 
#print(result) 

var1 = tf.Variable( [5] )

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
result2 = sess.run( var1 )

#print(result2)
