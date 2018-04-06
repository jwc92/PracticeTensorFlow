import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.rand(len(x_data))

y_true = (0.5*x_data)+5 + noise
x_df = pd.DataFrame(data=x_data,columns=['X Data'])
y_df = pd.DataFrame(data=y_true,columns=['Y'])

my_data = pd.concat([x_df,y_df],axis=1)
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


from sklearn.model_selection import train_test_split

x_train, x_eval, y_train,y_eval = train_test_split(x_data, y_true,test_size = 0.3, random_state=101)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=None,shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=1000,shuffle=False)

eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=8,num_epochs=1000,shuffle=False)

estimator.train(input_fn=input_func,steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)

print('TRANING DATA MATRICES')
print(train_metrics)


print('EVAL MATRICES')
print(eval_metrics)

brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)

predictions = []

for pred in estimator.predict(input_fn=input_fn_predict):
	predictions.append(pred['predictions'])

print(predictions)

my_data.sample(250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(brand_new_data,predictions,'red')

plt.show()
