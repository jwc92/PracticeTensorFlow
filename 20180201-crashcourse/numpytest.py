import numpy as np
my_list = [1,2,3]

arr = np.array(my_list)
print(type(arr))
print(arr)

arr2 = np.arange(0,10)
print(arr2)

arr3 = np.arange(0,11,2)
print(arr3)

arr4 = np.zeros(5)
print(arr4)

arr5 = np.zeros((3,5))
print(arr5)

arr6 = np.ones(3)
print(arr6)

arr7 = np.ones((3,5))
print(arr7)

arr8 = np.linspace(0,11,12)
print(arr8)

arr9 = np.random.randint(0,10)
print(arr9)

arr10 = np.random.randint(0,10,(3,3))
print(arr10)

np.random.seed(101)
A=np.random.randint(0,100,10)
np.random.seed(101)
B=np.random.randint(0,100,10)
print(A)
print(B)

arr11 = np.random.randint(0,100,10)
print(arr11.max())
print(arr11.min())
print(arr11.argmax())
print(arr11.argmin())

arr11.reshape(2,5)
print(arr11)

mat = np.arange(0,100).reshape(10,10)
print(mat)

print(mat[4,3])

print(mat[:,0])

print(mat[0,:])

print(mat[0:3,0:3])
print(mat > 50)
my_filter = mat > 50
print(mat[my_filter])

