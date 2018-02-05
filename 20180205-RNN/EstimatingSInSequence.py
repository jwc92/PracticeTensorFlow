import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

class TimeSeriesData():

    def __init__(self, num_points, xmin, xmax):

        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin,xmax,num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self,x_series):

        return np.sin(x_series)

    def next_batch(self, batch_size,steps,return_batch_ts=False):

        # Grab a random starting point for each batch of data
        rand_start = np.random.rand(batch_size,1)

        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))

        # Create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0,steps+1) * self.resolution

        # Create the Y data for the time series x axis from previous step
        y_batch = np.sin(batch_ts)

        # FORMATING for RNN
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else:
            return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1)


ts_data = TimeSeriesData(250,0,10)

plt.plot(ts_data.x_data,ts_data.y_true)

num_time_steps = 30
y1,y2,ts = ts_data.next_batch(1, 30, return_batch_ts=True)

plt.plot(ts.flatten()[:30],y1.flatten(), '*')

plt.plot(ts_data.x_data, ts_data.y_true,label='sin(t)')
plt.plot(ts.flatten()[1:],y2.flatten(), '*',label='Single Training Instance')
plt.legend()
plt.tight_layout()

plt.show()

# TRAINING DATA
train_inst = np.linspace(5,5+ts_data.resolution*(num_time_steps+1),num_time_steps+1)

plt.title('A TRAINING INSTANCE')

plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'bo', markersize=15, alpha=0.5, label='INSTANCE')
plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]), 'ko', markersize=7,label='TARGET')
plt.legend()

plt.show()


# Create a model
tf.reset_default_graph()

num_inputs = 1
# given x what is y?

num_neurons = 100
num_outputs = 1
learning_rate = 0.001
num_train_iterations = 2000
batch_size = 1

# PLACEHOLDERS

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# RNN CELL LAYER

cell =  tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.GRUCell(num_units=num_neurons,activation=tf.nn.relu)
        ,output_size=num_outputs)

output, states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
loss = tf.reduce_mean(tf.square(output-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):

        X_batch, y_batch = ts_data.next_batch(batch_size,num_time_steps)

        sess.run(train,feed_dict = {X:X_batch, y:y_batch})

        if iteration % 100 ==0:

            mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
            print(iteration, "\tMSE",mse)

    saver.save(sess, "./rnn_time_series_model_codecode")


with tf.Session() as sess:

    saver.restore(sess, "./rnn_time_series_model_codecode")

    X_new = np.sin(np.array(train_inst[:-1].reshape(-1,num_time_steps,num_inputs)))
    y_pred = sess.run(output, feed_dict={X:X_new})


plt.title("TESTING THE MODEL")

# TRAINING INSTANCE
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.3,label='Traning inst')
# TARGET TO PREDICT
plt.plot(train_inst[1:], np.sin(train_inst[1:]),"ko", markersize=10, label='target')
# MODELS PREDICTION
plt.plot(train_inst[1:], y_pred[0,:,0],'r.',markersize=10,label='PREDICTIONS')

plt.xlabel('Time')
plt.legend()
plt.show()
