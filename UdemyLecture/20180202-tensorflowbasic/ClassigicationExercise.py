import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

census = pd.read_csv('census_data.csv')

print(census.head())
print(census.columns)

## ['age', 'workclass', 'education', 'education_num', 'marital_status',
##       'occupation', 'relationship', 'race', 'gender', 'capital_gain',
##       'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']

print(census['income_bracket'].unique())

def label_fix(label):
	if label == ' <=50K':
		return 0
	else:
		return 1

census['income_bracket'] = census['income_bracket'].apply(label_fix)

print(census.head())

x_data = census.drop('income_bracket',axis=1)

labels = census['income_bracket'] 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)

age = tf.feature_column.numeric_column('age')
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass',hash_bucket_size=30)
education = tf.feature_column.categorical_column_with_hash_bucket('education',hash_bucket_size=10)
education_num = tf.feature_column.numeric_column('education_num')
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status',hash_bucket_size=10)
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation',hash_bucket_size=10)
relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship',hash_bucket_size=10)
race = tf.feature_column.categorical_column_with_hash_bucket('race',hash_bucket_size=10)
gender = tf.feature_column.categorical_column_with_hash_bucket('gender',hash_bucket_size=10)
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country',hash_bucket_size=20)

feat_cols = [age, workclass ,education , education_num, marital_status, occupation, relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country]

input_func = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=100,num_epochs=1000,shuffle=True)
linear_model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
linear_model.train(input_fn=input_func,steps=1000)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
pred_gen = linear_model.predict(input_fn=pred_fn)
predictions = list((pred_gen))

final_pred = [pred['class_ids'][0] for pred in predictions]

print(final_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test, final_pred))




