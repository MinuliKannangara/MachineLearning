# Data Preprocessing Tools
import numpy as np
import pandas as pd

# Importing the dataset (create a dataframe)
dataset = pd.read_csv('Data.csv')
# create two new entities: features and dependent variable vector, iloc = locate indexes, [rows (: means all the rows)
# , columns (:-1 means all the columns except the last column(index -1))]
x = dataset.iloc[:, :-1].values  # features
y = dataset.iloc[:, -1].values  # dependent variable

# Taking care of missing data
from sklearn.impute import SimpleImputer

# create an object of the SimpleImputer class ( arg1: what are the missing values that need to be addressed ,
# arg2: strategy to handle it)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit the imputer on the DataFrame
# here we only need to select the columns with numerical values and exclude all the string
imputer.fit(x[:, 1:3])

# Apply the transform to the DataFrame
imputer.transform(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data ( convert string data into a numerical representation)
from sklearn.compose import ColumnTransformer  # this is a column transformer class
from sklearn.preprocessing import OneHotEncoder

# Encoding the Independent Variable

# create an object from the ColumnTransformer class (arg1: type of transformation and the column indexes,
# arg2:remainder, columns that shouldn't be transformed
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# in order to train a ML model,we need to call train function by giving x (features). This function expect this X as
# a numpy array. So we need to convert it into a numpy array
x = np.array(ct.fit_transform(x))

# Encoding the Dependent Variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# dependent variable doesn't need to be a numpy array as x.
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# create 4 variables to store the split sets train_test_split( x(features), y(dependent variable), split size (
# test_size = 0.2 (20%)), to ensure that the data is split in the same way every time you run the code. )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature Scaling (put all the data in a same scale)
# here we use standardisation
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
# do the transformation for all the rows and the columns from 3rd column onwards in the x_train matrix.
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
