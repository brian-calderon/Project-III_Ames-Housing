import os

import pandas as pd

os.chdir("C:/Users/brian/OneDrive/Documents/Academics/Courses/NYC Bootcamp Machine Learning/\
Projects/Project-III/Ames_Housing/Web_app")
print(os.getcwd())

df2 = pd.read_csv("../Ames_Housing/Web_app/assets/Stock_5y.csv")
df2.head()

