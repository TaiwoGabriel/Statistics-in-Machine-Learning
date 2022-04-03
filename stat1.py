
# Data Types
# To analyze data, it is important to know what type of data we are dealing with.
#
# We can split the data types into three main categories:
#
# Numerical
# Categorical
# Ordinal
# Numerical data are numbers, and can be split into two numerical categories:
#
# Discrete Data
# - numbers that are limited to integers. Example: The number of cars passing by.
# Continuous Data
# - numbers that are of infinite value. Example: The price of an item, or the size of an item
# Categorical data are values that cannot be measured up against each other. Example: a color value, or any yes/no values.
#
# Ordinal data are like categorical data, but can be measured up against each other. Example: school grades where A is better than B and so on.
#
# By knowing the data type of your data source, you will be able to know what technique to use when analyzing them.

# Mean
# The mean value is the average value.
#
# To calculate the mean, find the sum of all values, and divide the sum by the number of values:
#
# (99+86+87+88+111+86+103+87+94+78+77+85+86) / 13 = 89.77
#
# The NumPy module has a method for this. Learn about the NumPy module in our NumPy Tutorial.
#
# Example
# Use the NumPy mean() method to find the average speed:
import numpy as np
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
my_mean = np.mean(speed)
print(my_mean)


# Median
# The median value is the value in the middle, after you have sorted all the values:
#
# 77, 78, 85, 86, 86, 86, 87, 87, 88, 94, 99, 103, 111
#
# It is important that the numbers are sorted before you can find the median.
#
# The NumPy module has a method for this:
#
# Example
# Use the NumPy median() method to find the middle value:
import numpy as np
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
my_median = np.median(speed)
print(my_median)


# Mode
# The Mode value is the value that appears the most number of times:
#
# 99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 = 86
#
# The SciPy module has a method for this. Learn about the SciPy module in our SciPy Tutorial.
#
# Example
# Use the SciPy mode() method to find the number that appears the most:
from scipy.stats import mode

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
my_mode = mode(speed)
print(my_mode)


# What is Standard Deviation?
# Standard deviation is a number that describes how spread out the values are.
#
# A low standard deviation means that most of the numbers are close to the mean (average) value.
#
# A high standard deviation means that the values are spread out over a wider range.
#
# Example: This time we have registered the speed of 7 cars:
#
# speed = [86,87,88,86,87,85,86]
#
# The standard deviation is:
#
# 0.9
#
# Meaning that most of the values are within the range of 0.9 from the mean value, which is 86.4.
#
# Let us do the same with a selection of numbers with a wider range:
#
# speed = [32,111,138,28,59,77,97]
#
# The standard deviation is:
#
# 37.85
#
# Meaning that most of the values are within the range of 37.85 from the mean value, which is 77.4.
#
# As you can see, a higher standard deviation indicates that the values are spread out over a wider range.
#
# The NumPy module has a method to calculate the standard deviation:
#
# Example
# Use the NumPy std() method to find the standard deviation:

import numpy as np
speed = [86,87,88,86,87,85,86]
m = np.mean(speed)
n = np.std(speed)
print(m)
print(n)


import numpy as np
speed = [32,111,138,28,59,77,97]
m = np.mean(speed)
n = np.std(speed)
print('The mean of the sample is', m)
print('The standard deviation', n)


# Variance
# Variance is another number that indicates how spread out the values are.
#
# In fact, if you take the square root of the variance, you get the standard deviation!
#
# Or the other way around, if you multiply the standard deviation by itself, you get the variance!
# Luckily, NumPy has a method to calculate the variance:
#
# Example
# Use the NumPy var() method to find the variance:
import numpy as np
speed = [32,111,138,28,59,77,97]
m = np.mean(speed)
m1 = np.var(speed)
m2 = np.std(speed)
print('The mean =', m)
print('The variance =', m1)
print('The standard deviation =', m2)


# What are Percentiles?
# Percentiles are used in statistics to give you a number that describes the
# value that a given percent of the values are lower than.
#
# Example: Let's say we have an array of the ages of all the people that lives in a street.
#
# ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
#
# What is the 75. percentile? The answer is 43, meaning that 75% of the people are 43 or younger.
#
# The NumPy module has a method for finding the specified percentile:
#
# Example
# Use the NumPy percentile() method to find the percentiles:

import numpy as np
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

x = np.percentile(ages, 75)
print('The 75th percentile =', x)
print()

# What is the age that 90% of the people are younger than?
import numpy as np
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = np.percentile(ages, 90)
print('The 90th percentile =', x)

# Data Distribution
# Earlier in this tutorial we have worked with very small amounts of data in
# our examples, just to understand the different concepts.
#
# In the real world, the data sets are much bigger, but it can be difficult to
# gather real world data, at least at an early stage of a project.
#
# How Can we Get Big Data Sets?
# To create big data sets for testing, we use the Python module NumPy,
# which comes with a number of methods to create random data sets, of any size.

# Example
# Create an array containing 250 random floats between 0 and 5:
import numpy as np

x = np.random.uniform(0.0, 5.0, 250)
print(x)

# Visualize the data
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0.0, 0.5, 250)
plt.hist(x, 5) # NOTE: the 5 inside the bracket means the number of bars in the histogram
plt.show()


# Big Data Distributions
# An array containing 250 values is not considered very big, but now you know how to create a
# random set of values, and by changing the parameters, you can create the data set as big as you want.
#
# Example
# Create an array with 100000 random numbers, and display them using a histogram with 100 bars:
import numpy as np
import matplotlib.pyplot as plt
x = np.random.uniform(0.0,5.0,100000)
plt.hist(x, 100)
plt.show()


# Normal Data Distribution
# In the previous chapter we learned how to create a completely random array, of a given size, and between two given values.
#
# In this chapter we will learn how to create an array where the values are concentrated around a given value.
#
# In probability theory this kind of data distribution is known as
# the normal data distribution, or the Gaussian data distribution, after the
# mathematician Carl Friedrich Gauss who came up with the formula of this data distribution.
import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(loc=5.0, scale=1.0, size=100000)
#plt.hist (x,100)
plt.boxplot(x,100)
plt.show()
# Note: A normal distribution graph is also known as the bell curve because of it's characteristic shape of a bell.
# Histogram Explained
# We use the array from the numpy.random.normal() method, with 100000 values,  to draw a histogram with 100 bars.
#
# We specify that the mean value is 5.0, and the standard deviation is 1.0.
#
# Meaning that the values should be concentrated around 5.0, and rarely further away than 1.0 from the mean.
#
# And as you can see from the histogram, most values are between 4.0 and 6.0, with a top at approximately 5.0.


# Scatter Plot
# A scatter plot is a diagram where each value in the data set is represented by a dot.
# The Matplotlib module has a method for drawing scatter plots, it needs two arrays of the
# same length, one for the values of the x-axis, and one for the values of the y-axis:
#
# Use the scatter() method to draw a scatter plot diagram:
# # where x array represents the age of each car and y array represents the speed of each car.
import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.xlabel('Car Age')
plt.ylabel('Car Speed')
plt.scatter(x,y)
plt.show()

# Scatter Plot Explained
# The x-axis represents ages, and the y-axis represents speeds.
#
# What we can read from the diagram is that the two fastest cars were both 2 years old,
# and the slowest car was 12 years old.
#
# Note: It seems that the newer the car, the faster it drives, but that could be a coincidence,
# after all we only registered 13 cars.


# Random Data Distributions
# In Machine Learning the data sets can contain thousands-, or even millions, of values.
#
# You might not have real world data when you are testing an algorithm, you might have to
# use randomly generated values.
#
# As we have learned in the previous chapter, the NumPy module can help us with that!
#
# Let us create two arrays that are both filled with 1000 random numbers from a normal data distribution.
#
# The first array will have the mean set to 5.0 with a standard deviation of 1.0.
#
# The second array will have the mean set to 10.0 with a standard deviation of 2.0:

# Example
# A scatter plot with 1000 dots
import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(loc=5.0, scale=1.0, size=1000)
y = np.random.normal(loc=10.0, scale=2.0, size=1000)
plt.xlabel('Car Age')
plt.ylabel('Care Speed')
plt.scatter(x,y)
plt.show()

# Scatter Plot Explained
# We can see that the dots are concentrated around the value 5 on the x-axis, and 10 on the y-axis.
#
# We can also see that the spread is wider on the y-axis than on the x-axis.


# Regression
# The term regression is used when you try to find the relationship between variables.
#
# In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.
#
# Linear Regression
# Linear regression uses the relationship between the data-points to draw a straight line through all them.
#
# This line can be used to predict future values.

# In Machine Learning, predicting the future is very important.
#
# How Does it Work?
# Python has methods for finding a relationship between data-points and to draw a line of linear regression.
# We will show you how to use these methods instead of going through the mathematical formula.
#
# In the example below, the x-axis represents age, and the y-axis represents speed.
# We have registered the age and speed of 13 cars as they were passing a tollbooth.
# Let us see if the data we collected could be used in a linear regression:
#
# Example
# Start by drawing a scatter plot:
import matplotlib.pyplot as plt
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x,y)

def myfunc(x):
    return slope * x + intercept
my_model = list(map(myfunc, x))
plt.scatter(x,y)
plt.plot(x, my_model)
plt.xlabel('Car Age')
plt.ylabel('Car Speed')
plt.show()

# Example Explained
# Import the modules you need.
#
# import matplotlib.pyplot as plt
# from scipy import stats
#
# Create the arrays that represent the values of the x and y axis:
#
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
#
# Execute a method that returns some important key values of Linear Regression:
#
# slope, intercept, r, p, std_err = stats.linregress(x, y)
#
# Create a function that uses the slope and intercept values to return a new value.
# This new value represents where on the y-axis the corresponding x value will be placed:
#
# def myfunc(x):
#   return slope * x + intercept
#
# Run each value of the x array through the function. This will result in
# a new array with new values for the y-axis:
#
# mymodel = list(map(myfunc, x))
#
# Draw the original scatter plot:
#
# plt.scatter(x, y)
#
# Draw the line of linear regression:
#
# plt.plot(x, mymodel)
#
# Display the diagram:
#
# plt.show()


# Correlation Coefficient

# R for Relationship
# It is important to know how the relationship between the values of the x-axis and the values of the y-axis is, if there are no relationship the linear regression can not be used to predict anything.
#
# This relationship - the coefficient of correlation - is called r.
#
# The r value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related.
#
# Python and the Scipy module will compute this value for you, all you have to do is feed it with the x and y values.
#
# Example
# How well does my data fit in a linear regression?
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x,y)
print('The correlation coefficient =',r)
# Note: The result -0.76 shows that there is a relationship, not perfect,
# but it indicates that we could use linear regression in future predictions.


# Predict Future Values
# Now we can use the information we have gathered to predict future values.
#
# Example: Let us try to predict the speed of a 10 years old car.
#
# To do so, we need the same myfunc() function from the example above:
#
# def myfunc(x):
#   return slope * x + intercept
#
# Example
# Predict the speed of a 10 years old car:
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x,y)

def myfunc(x):
    return slope * x + intercept
speed = myfunc(10)
print(speed)
print(r)
print(p)
print(std_err)
print(slope)
print(intercept)


# Bad Fit?
# Let us create an example where linear regression would not be the best method to predict future values.
#
# Example
# These values for the x- and y-axis should result in a very bad fit for linear regression
from scipy import  stats
import matplotlib.pyplot as plt
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x,y)
def myfunc(x):
    return slope * x + intercept
my_model = list(map(myfunc,x))
plt.scatter(x,y)
plt.plot(x, my_model)
plt.show()
print()
# And the r-squared value?
# You should get a very low r-squared value.
print(r)
# The result: 0.013 indicates a very bad relationship, and tells us that this data set
# is not suitable for linear regression.


# Polynomial Regression
# If your data points clearly will not fit a linear regression
# (a straight line through all data points), it might be ideal for polynomial regression.
#
# Polynomial regression, like linear regression, uses the relationship between the
# variables x and y to find the best way to draw a line through the data points.
# How Does it Work?
# Python has methods for finding a relationship between data-points and to
# draw a line of polynomial regression. We will show you how to use these methods
# instead of going through the mathematic formula.
#
# In the example below, we have registered 18 cars as they were passing a certain tollbooth.
#
# We have registered the car's speed, and the time of day (hour) the passing occurred.
#
# The x-axis represents the hours of the day and the y-axis represents the speed:

# Example
# Start by drawing a scatter plot:
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.xlabel('Passing Hours')
plt.ylabel('Car Speed')
plt.scatter(x,y)
plt.show()


# Drawing the line of regression
import numpy as np
import matplotlib.pyplot as plt
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(x,y,3))
myline = np.linspace(1,22,100)
plt.scatter(x,y)
plt.plot(myline, mymodel(myline))
plt.show()

# R-Squared for Polynomial Regression
# It is important to know how well the relationship between the values of the x- and y-axis is,
# if there are no relationship the polynomial regression can not be used to predict anything.
#
# The relationship is measured with a value called the r-squared.
#
# The r-squared value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related.
#
# Python and the Sklearn module will compute this value for you, all you have to do is
# feed it with the x and y arrays:
#
# Example
# How well does my data fit in a polynomial regression?
import numpy as np
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(x,y,3))

r2_output = r2_score(y,mymodel(x))
print('The correlation coefficient value =',r2_output)
# Note: The result 0.94 shows that there is a very good relationship, and we can use
# polynomial regression in future predictions.


# Predict Future Values
# Now we can use the information we have gathered to predict future values.
#
# Example: Let us try to predict the speed of a car that passes the tollbooth at around 17 P.M:
#
# To do so, we need the same mymodel array from the example above:
#
# mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
import numpy as np
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(x,y,3))

speed = mymodel(17)
print('The speed prediction of a car passing the gate at 17PM =',speed)



# Bad Fit?
# Let us create an example where polynomial regression would not be the best method to predict future values.
#
# Example
# These values for the x- and y-axis should result in a very bad fit for polynomial regression:
import numpy as np
import matplotlib.pyplot as plt
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = np.poly1d(np.polyfit(x,y,3))

myline = np.linspace(2,95,100)
plt.scatter(x,y)
plt.plot(myline, mymodel(myline))
plt.show()

# And the r-squared value?
#
# Example
# You should get a very low r-squared value.
import numpy as np
from sklearn.metrics import r2_score
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = np.poly1d(np.polyfit(x,y,3))
speed = r2_score(y,mymodel(x))
print('Correlation coefficient=',speed)
# The result: 0.00995 indicates a very bad relationship, and tells us that this
# data set is not suitable for polynomial regression.


# Multiple Regression
# Multiple regression is like linear regression, but with more than one independent value,
# meaning that we try to predict a value based on two or more variables.
import pandas as pd
from sklearn.linear_model import LinearRegression
df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/cars.csv'
df2 = pd.read_csv(df, delimiter=',')
# Put the data in DataFrame to view it in tabular form
data = pd.DataFrame(df2)
#print(data.shape)
#print(data.describe().transpose())
X = data[["Weight","Volume"]]
y = data["CO2"]
regr = LinearRegression()
regr.fit(X,y) # Training the Linear Regression Model

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
pred_CO2 = regr.predict([[2300, 1300]])
print(pred_CO2)
# NOTE: We have predicted that a car with 1.3 liter engine, and a weight of 2300 kg,
# will release approximately 107 grams of CO2 for every kilometer it drives


# Coefficient
# The coefficient is a factor that describes the relationship with an unknown variable.
#
# Example: if x is a variable, then 2x is x two times. x is the unknown variable, and the number 2 is the coefficient.
#
# In this case, we can ask for the coefficient value of weight against CO2, and for volume against CO2.
# The answer(s) we get tells us what would happen if we increase, or decrease, one of the independent values
# Print the coefficient values of the regression object:
import pandas as pd
from sklearn.linear_model import LinearRegression
df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/cars.csv'
df2 = pd.read_csv(df, delimiter=',')

X = df2[["Weight","Volume"]]
y = df2["CO2"]

regr = LinearRegression()
regr.fit(X, y)
coef = regr.coef_
print(coef)

# The result array represents the coefficient values of weight and volume.
#
# Weight: 0.00755095
# Volume: 0.00780526
#
# These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
#
# And if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.
#
# I think that is a fair guess, but let test it!
#
# We have already predicted that if a car with a 1300cm3 engine weighs 2300kg,
# the CO2 emission will be approximately 107g.

# What if we increase the weight with 1000kg?
import pandas as pd
from sklearn.linear_model import LinearRegression
df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/cars.csv'
df2 = pd.read_csv(df, delimiter=',')

X = df2[["Weight","Volume"]]
print(X)

y = df2["CO2"]

regr = LinearRegression()
regr.fit(X, y)
pred_CO2 = regr.predict([[3300, 1300]])
print(pred_CO2)

# We have predicted that a car with 1.3 liter engine, and a weight of 3300 kg,
# will release approximately 115 grams of CO2 for every kilometer it drives.
#
# Which shows that the coefficient of 0.00755095 is correct:
#
# 107.2087328 + (1000 * 0.00755095) = 114.75968


# Scale Features
# When your data has different values, and even different measurement units, it can be difficult to compare them.
# What is kilograms compared to meters? Or altitude compared to time?
#
# The answer to this problem is scaling. We can scale data into new values that are easier to compare.
#
# Take a look at the table below, it is the same data set that we used in the multiple regression chapter,
# but this time the volume column contains values in liters instead of cm3 (1.0 instead of 1000)
# It can be difficult to compare the volume 1.0 with the weight 790, but if we scale them both into comparable values, we can easily see how much one value is compared to the other.
#
# There are different methods for scaling data, in this tutorial we will use a method called standardization.
#
# The standardization method uses this formula:
#
# z = (x - u) / s
#
# Where z is the new value, x is the original value, u is the mean and s is the standard deviation.
#
# If you take the weight column from the data set above, the first value is 790, and the scaled value will be:
#
# (790 - 1292.23) / 238.74 = -2.1
# If you take the volume column from the data set above, the first value is 1.0, and the scaled value will be:
#
# (1.0 - 1.61) / 0.38 = -1.59
#
# Now you can compare -2.1 with -1.59 instead of comparing 790 with 1.0.
#
# You do not have to do this manually, the Python sklearn module has a method called StandardScaler()
# which returns a Scaler object with methods for transforming data sets.
# Example
# Scale all values in the Weight and Volume columns:

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/cars2.csv'
data = pd.read_csv(df, delimiter=',')

X = data[["Weight", "Volume"]]
scale = StandardScaler()
scaledX = scale.fit_transform(X)
print(scaledX)
# Now predict CO2
y = data["CO2"]

regr = LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])
pred_CO2 = regr.predict([scaled[0]])
print(pred_CO2)


# Machine Learning - Train/Test
# Evaluate Your Model
# In Machine Learning we create models to predict the outcome of certain events,
# like in the previous chapter where we predicted the CO2 emission of a car when
# we knew the weight and engine size.
#
# To measure if the model is good enough, we can use a method called Train/Test.
#
# What is Train/Test
# Train/Test is a method to measure the accuracy of your model.
#
# It is called Train/Test because you split the the data set into two sets: a training set and a testing set.
#
# 80% for training, and 20% for testing.
#
# You train the model using the training set.
#
# You test the model using the testing set.
#
# Train the model means create the model.
#
# Test the model means test the accuracy of the model.
# The x axis represents the number of minutes before making a purchase.
#
# The y axis represents the amount of money spent on the purchase.
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

X = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100)/ X
plt.scatter(X, y)
plt.xlabel("Number of minutes before purchase")
plt.ylabel("Amount of money spent on Purchase")
#plt.show()

# Split Into Train/Test
# The training set should be a random selection of 80% of the original data. The testing set should be the remaining 20%.
train_X = X[:80]
train_y = y[:80]

test_X = X[80:]
test_y = y[80:]

# Display the Training Set
# Display the same scatter plot with the training set:
#plt.scatter(train_X, train_y)
#plt.scatter(test_X, test_y)
#plt.show()

# Fit the Data Set
# What does the data set look like? In my opinion I think the best fit would be a polynomial regression,
# so let us draw a line of polynomial regression.
# design the polynomial model
mymodel = np.poly1d(np.polyfit(train_X, train_y, 4)) # NOTE 4 means power of the equation or degree
# To draw a line through the data points, we use the plot() method of the matplotlib module:
myline = np.linspace(0,6,100)
plt.scatter(train_X, train_y)
plt.plot(myline, mymodel(myline))
plt.show()


# The result can back my suggestion of the data set fitting a polynomial regression,
# even though it would give us some weird results if we try to predict values outside of the data set.
# Example: the line indicates that a customer spending 6 minutes in the shop would make a purchase worth 200.
# That is probably a sign of overfitting.
#
# But what about the R-squared score? The R-squared score is a good indicator of
# how well my data set is fitting the model.
#
# R2
# Remember R2, also known as R-squared?
#
# It measures the relationship between the x axis and the y axis, and the value ranges from 0 to 1,
# where 0 means no relationship, and 1 means totally related.

# R-squared: How well does my training data fit in a polynomial regression?

import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))
print(r2)
# # Let us find the R2 score when using testing data:
r3 = r2_score(test_y, mymodel(test_x))
print(r3)
# Note: The result 0.809 shows that the model fits the testing set as well,
# and we are confident that we can use the model to predict future values.

# Predict Values
# How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?
print("Customer to spend ", mymodel(5).round(1), "dollars")
