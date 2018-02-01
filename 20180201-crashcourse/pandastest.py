import pandas as pd

df = pd.read_csv('salaries.csv')

print(df)

print(df[['Salary','Name']])

print(df['Salary'].max())

print(df.describe())

print(df['Salary'] > 60000)

my_filter = df['Salary'] > 60000

print(df[my_filter])
