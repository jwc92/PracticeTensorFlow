import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0,10)
print(x)

y = x**2
plt.plot(x,y,'r--')
plt.xlim(0,4)
plt.ylim(0,10)
plt.title("TITLE")
plt.xlabel('X LABEL')
plt.ylabel('Y LABEL')
plt.show()


mat = np.arange(0,100).reshape(10,10)
plt.imshow(mat,cmap='RdYlGn')
plt.show()

mat2 = np.random.randint(0,1000,(10,10))
plt.imshow(mat2)
plt.colorbar()
plt.show()

df = pd.read_csv('salaries.csv')

df.plot(x='Salary',y='Age',kind='Scatter')
plt.show()
