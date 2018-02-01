import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.random.randint(0,100,(10,2))

scaler_model = MinMaxScaler()

print(type(scaler_model))

scaler_model.fit(data)
data_transform = scaler_model.transform(data)
print(data_transform)

import pandas as pd
my_data = np.random.randint(0,101,(50,4))
print(my_data)
df = pd.DataFrame(data = my_data, columns = ['f1','f2','f3','label'])

print(df)


X = df[['f1','f2','f3']]
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)

