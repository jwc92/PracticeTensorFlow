import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

cal_housing = pd.read_csv('cal_housing_clean.csv')

# print(cal_housing.head())
# print(cal_housing.columns)
## ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population',
##      'households', 'medianIncome', 'medianHouseValue']

from sklearn.model_selection import train_test_split

cal_housing_train, cal_housing_test = train_test_split(cal_housing, test_size=0.3, random_state=101)

#print(cal_housing_train.shape)
#print(cal_housing_test.shape)

#print(cal_housing.describe())

from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler(copy=True, feature_range=(0,1))
scaler_model.fit(cal_housing_train)
cal_housing_train_transform = scaler_model.transform(cal_housing_train)
cal_housing_test_transform = scaler_model.transform(cal_housing_test)

# print(cal_housing_train)
# print(cal_housing_train_transform)

CH_train = pd.DataFrame(data = cal_housing_train_transform, columns = ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome', 'medianHouseValue'])
CH_test = pd.DataFrame(data = cal_housing_test_transform, columns = ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome', 'medianHouseValue'])
# print(CH_train)
# print(CH_train)

hMA = tf.feature_column.numeric_column('housingMedianAge')
tR = tf.feature_column.numeric_column('totalRooms')
tB = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
hh = tf.feature_column.numeric_column('households')
mI = tf.feature_column.numeric_column('medianIncome')
mHV = tf.feature_column.numeric_column('medianHouseValue')

feat_cols = [hMA, tR, tB, pop, hh, mI]

X_train = CH_train.drop('medianHouseValue',axis=1)
X_test = CH_test.drop('medianHouseValue',axis=1)
y_train = CH_train['medianHouseValue'] 
y_test = CH_test['medianHouseValue']


input_func = tf.estimator.inputs.pandas_input_fn(X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
dnn_model = tf.estimator.DNNRegressor(hidden_units=[10,20,10],feature_columns=feat_cols)
dnn_model.train(input_fn=input_func,steps=10000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
results = dnn_model.evaluate(eval_input_func)

print(results['loss'])

pred_input_func= tf.estimator.inputs.pandas_input_fn(X_test,batch_size=100,shuffle=False)
predict_gen = dnn_model.predict(input_fn=pred_input_func)

predictions = []

for pred in dnn_model.predict(input_fn=pred_input_func):
	predictions.append(pred['predictions'])

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,predictions)**0.5)

plt.plot(predictions, y_test,'*')
plt.show()


