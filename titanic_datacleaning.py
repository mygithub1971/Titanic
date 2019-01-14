# Importing the libraries
import numpy as np # for numerical, stats, matrix operation
import matplotlib.pyplot as plt # for plotting
import pandas as pd # for data manipulation

def df_info(df): # function to report some general info of a dataframe 
	print("Printing general info:\n")
	print(df.columns) 
#	print(df.dtypes) 
	print(df.shape)
	print("Null check:\n")
	nullcheck = pd.isnull(df).sum() 
	print(nullcheck[nullcheck>0])

# Importing the dataset
#col_names = ["ID","Diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","cancave points","symmetry","fractal"]
df = pd.read_csv('train.csv')

df_info(df)

df = df.drop(['Cabin'], axis = 1)
mean_age = np.mean(df['Age'])
print(mean_age)
#df['Age'] = df['Age'].fillna(mean_age)
df['Age'].fillna(mean_age, inplace = True)
df = df.dropna(axis = 0) 

#use these 2 lines to explore the performance of your bucketing!
#df['Agebuc'] = pd.cut(df['Age'],6)
#df[['Agebuc','Survived']].groupby('Agebuc').mean()

#df.loc[ df['Age'] <= 16, 'Age'] = 0
#df.loc[ df['Age'] > 16, 'Age'] = 1
#df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
#df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
#df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
#df.loc[ df['Age'] > 64, 'Age'] = 4

#df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
#df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
#df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
#df.loc[ df['Fare'] > 31, 'Fare'] = 3

#df['FamilySize'] = df['SibSp'] +  df['Parch'] + 1
#df['IsAlone'] = 0
#df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

labelCols = ['Pclass','Sex']
#labelCols = ['Sex','Age','Fare','IsAlone']
numericCols = ['Age','SibSp', 'Parch', 'Fare']
#numericCols = ['Age','FamilySize', 'Fare']
# numericCols = ['SibSp']
#numericCols = ['SibSp', 'Parch', 'Fare']


X = df.loc[:,labelCols + numericCols]
#X = df.loc[:, labelCols]
y = df.iloc[:,1].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_train.loc[:,'Sex'] = labelencoder.fit_transform(X_train.loc[:,'Sex'])
X_test.loc[:,'Sex'] = labelencoder.transform(X_test.loc[:,'Sex'])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.loc[:,numericCols] = sc.fit_transform(X_train.loc[:,numericCols])
X_test.loc[:,numericCols] = sc.transform(X_test.loc[:,numericCols])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [range(len(labelCols))])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()

#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state = 0).fit(X_train, y_train)
#y_pred = clf.predict(X_test)

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf',random_state = 0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
#regressorDT = DecisionTreeRegressor(random_state = 0)
#regressorRF = RandomForestRegressor(n_estimators = 60, random_state = 0)
#regressorGBM = GradientBoostingRegressor(n_estimators = 100)
#clf = regressorDT.fit(X_train, y_train)
#y_pred = clf.predict(X_test).astype('int64')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(np.mean((y_test == clf.predict(X_test))))
print(np.mean((y_train == clf.predict(X_train))))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=10)
