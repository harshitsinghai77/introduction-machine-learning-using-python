## Introduction
#### In simple words linear regression is predicting the value of a variable Y(dependent variable) based on some variable X(independent variable) provided there is a linear relationship between X and Y

## Libraries Used
#### train_test_split: Split arrays or matrices into random train and test subsets
#### LinearRegression: Ordinary least squares Linear Regression.
#### matplotlib.pyplot: Provides a MATLAB-like plotting framework.

## Use
#### Develop and train the machine learning model to 

## Dataset
#### The folder contains Salary_Data.csv dataset. The X column of the dataset contains the 'Years of Experience' along with corresponding 'Salary' of the employee.

## Code Explaination
Loading the dataset using Pandas library
```python
dataset = pd.read_csv('Salary_Data.csv')
dataset
```

Assigning Year of Experience to 'x' and Salary to 'y' variable
```python
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
```

Using train_test_split library to randomly generate Training set and Testing set from the data and assigning them to variables.
X_train and y_train are the 'Experience' and 'Salary' respectively for the training set and y_train and y_test are the 'Experience' and 'Salary' respectively for the testing set. test_size=1/3 signifies 10 records from 30 records will be used for testing set i.e 1/3 of the total records. 
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=1/3, random_state=0)
```

Using LinearRegression() to import all the function from the library and using '.fit' method find and learn the correlation between the 'Experience' and 'Salary' of the training set.
"regression.predict" takes 'Years of Experience' from the testing set as the parameter and predicts the 'Salary'.The print statement prints the Actual value and the Predicted values.
```python
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train , y_train)

y_predict = regression.predict(X_test)
print('Given values ', y_test)
print('\nPredicted values ',y_predict)
```

Using matplotlib.pyplot to plot the graph
```python
# Plotting the Training set results
plt.scatter(X_train, y_train, color = 'red', label = 'Real values')
plt.plot(X_train, regression.predict(X_train), label = 'Predicted values', color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

```python
#Plotting the Test set results
plt.scatter(X_test, y_test, color = 'red', label = 'Real values')
plt.plot(X_train, regression.predict(X_train), label = 'Predicted values', color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
```
