# Diving into Descriptive Statistics of The Titanic Dataset

## Introduction to Descriptive Statistics with the Titanic Dataset


So, why do we need to study statistics when dealing with data? Well, statistics is a branch of mathematics dealing with data collection, organization, and interpretation. In data science, we use statistics to extract meaningful insights and knowledge from data.

Statistics helps us deal with the data's complexity by reducing a complex dataset into a simpler summary. It assists in the presentation and visualization of the data, thereby making our data analysis or machine learning model more precise.

Take our current dataset, for instance, which comprises various demographics and passenger information; wouldn't it be interesting to know the average age or to gauge the variety in travelers' fares? Our lesson will focus on extracting these primary statistical features from our dataset, helping us better comprehend the Titanic voyage.

## Overview of Descriptive Statistics

Descriptive statistics summarise and organize the characteristics of a data set. A data set is a collection of responses or observations from a sample or entire population.

In pandas, there's a function called describe(), which calculates the basic statistics for all continuous variables, i.e., types of variables that can take on an infinite number of values within a specific range. It provides the count, mean, standard deviation (std), min, quartiles, and max in its output.

Firstly, let's import the libraries we will be using and load the dataset:

```Python

import seaborn as sns

# Load the dataset
titanic = sns.load_dataset('titanic')

# show the first few rows of data
print(titanic.head())

The output of the head command will be like this:

```Markdown

   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True
````
The describe() function can then be executed as follows:

```Python

# Generate descriptive statistics
titanic_stats = titanic.describe()
print(titanic_stats)
```
The output of the describe() function will be like this:

```Markdown

        survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
```
In this code snippet, the `describe()` function generates descriptive statistics that summarize a dataset's distribution's central tendency, dispersion, and shape, excluding NaN values.

## What Else?

Notice how all the categorical columns, like 'sex' or 'class', are missing in the output. By default, describe() only includes columns with numerical data.

If you want to include all columns, you need to pass include='all' as an argument. Here is how to do it:

```Python

# Generate descriptive statistics
titanic_stats = titanic.describe(include='all')
print(titanic_stats)
```
Note that for categorical variables, the output has different features – unique, top, and freq. 'unique' shows the number of distinct objects in the column, 'top' shows the most frequent object, and 'freq' shows how many times the top object appears in the column.

## Unveiling The Spread

Variability, also known as dispersion, is the extent to which data points differ from the center. Two commonly used measures are the range and interquartile range (IQR).

The range is the difference between a dataset's maximum and minimum values. However, it's sensitive to outliers; extremely high or low values can skew the range. Here's how you calculate the range for the age column of the Titanic dataset:

```Python

# Calculate the numerical data range
age_range = titanic['age'].max() - titanic['age'].min()
print('Age Range:', age_range) # Age Range: 79.58
```
The IQR measures statistical dispersion, or how far apart the data points are. It's the range within which the middle 50% of your data falls. It's a better measure of dispersion than the range because outliers don't affect it. Here's how you can calculate it:

```Python

# Calculate the IQR
Q1 = titanic['age'].quantile(0.25)
Q3 = titanic['age'].quantile(0.75)
IQR = Q3 - Q1
print('Age IQR:', IQR) # Age IQR: 17.875
```
## Determining The Central Position

Central tendency measures help you find the center of your dataset. Mean and median are the most common measures of central tendency.

The mean or average is the most common measure of central tendency. It's the sum of all data points divided by the number of data points.

```Python

# Calculate the mean
mean_age = titanic['age'].mean()
print('Mean Age:', mean_age) # Mean Age: 29.69911764705882
```
The median is the middle score. The scores must be arranged in numerical order to identify the median correctly.

```Python

# Calculate the median
median_age = titanic['age'].median()
print('Median Age:', median_age) # Median Age: 28.0
```
## Wrapping Up

You've just taken your first steps into the realm of descriptive statistics! In this lesson, you've learned about the usefulness of statistics in data analysis and how we can summarize our Titanic dataset via central tendency and dispersion measures.

Hence, understanding these statistical characteristics and central tendencies is significant for making effective predictions about our dataset, offering a sound foundation for building meaningful data visualizations.
