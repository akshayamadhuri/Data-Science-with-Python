# Exploring the California Housing Dataset: An Introduction to Dataset Characteristics and Basic Visualizations

## Introduction and Objectives

Deep Dive into Numpy and Pandas with Housing Data. In this course, you will unlock the secrets of efficient data manipulation and analysis with Numpy and Pandas. We will build your skills from a foundational to an advanced level, strengthening your grasp of Python and preparing you for the world of Data Science.

we will study the California Housing dataset. This important dataset is often used as a benchmark in machine learning and data analysis. It contains detailed information about housing values in California suburbs. The California housing market, due to its high prices and shortages, has been the subject of study for many years. This makes the dataset particularly relevant today. In this lesson, our main objective is to explore the fundamental attributes of this dataset. We aim to understand various attributes such as median income, population, average number of rooms per household, and their influence on house prices. Let's get started!

## Importing the California Housing Dataset

To load the California Housing dataset, we can use the sklearn library, which is powerful, easy to use, and contains many ready-to-use machine learning algorithms. It also comes with a few pre-loaded datasets, including the California Housing dataset. We can load the dataset by simply importing the appropriate sklearn module and calling a function.

```Python

from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
```

After loading the California Housing dataset, we receive the data in a Bunch object; it's similar to a dictionary but with added functionalities. It has keys like data, target, feature_names, each leading to a different part of the dataset. The data key contains all the input features, the target key has the output values we might want to predict (median house values), and feature_names holds the names of the features.

A pandas DataFrame is more familiar and offers more functionality, making it easier to work with. Let's convert our dataset into a pandas DataFrame.

```Python

import pandas as pd

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df["MedHouseValue"] = dataset.target
print(df.head())
"""
   MedInc  HouseAge  AveRooms  ...  Latitude  Longitude  MedHouseValue
0  8.3252      41.0  6.984127  ...     37.88    -122.23          4.526
1  8.3014      21.0  6.238137  ...     37.86    -122.22          3.585
2  7.2574      52.0  8.288136  ...     37.85    -122.24          3.521
3  5.6431      52.0  5.817352  ...     37.85    -122.25          3.413
4  3.8462      52.0  6.281853  ...     37.85    -122.25          3.422

[5 rows x 9 columns]
"""
```

This code converts the different parts of the Bunch object into a Pandas DataFrame. We also add the target values to the dataframe as a new column, MedHouseValue. Now, we have successfully imported our dataset and already explored the first few rows using the head() function!

## Exploring the Basics

Once we've imported the dataset, it's time for some preliminary analysis to get a general understanding of our data.

```Python

print(df.shape) # Output: (20640, 9)
print(df.describe())
"""
             MedInc      HouseAge  ...     Longitude  MedHouseValue
count  20640.000000  20640.000000  ...  20640.000000   20640.000000
mean       3.870671     28.639486  ...   -119.569704       2.068558
std        1.899822     12.585558  ...      2.003532       1.153956
min        0.499900      1.000000  ...   -124.350000       0.149990
25%        2.563400     18.000000  ...   -121.800000       1.196000
50%        3.534800     29.000000  ...   -118.490000       1.797000
75%        4.743250     37.000000  ...   -118.010000       2.647250
max       15.000100     52.000000  ...   -114.310000       5.000010

[8 rows x 9 columns]
"""
```
The shape attribute gives us the dimensions of our dataset - the number of rows and columns. The `describe()` function provides valuable statistical information about each attribute, including the count, mean, standard deviation, and various percentiles. With this information, we can understand the distribution of data across each attribute.

## Investigating Features

To make better use of our dataset, we need to understand the features it includes. Every feature or column in our dataset represents a characteristic of a block group in California. These characteristics include:

`MedInc`: This is the median income for households within a block (scaled and capped at 15 for higher median incomes and at 0.5 for lower median incomes).
`HouseAge`: This is the median house age within a block.
`AveRooms`: This is the average number of rooms in the houses within a block.
`AveBedrms`: This is the average number of bedrooms in the houses within a block.
`Population`: This is the total population within a block.
`AveOccup`: This is the average house occupancy, computed as the total population within a block divided by the number of households.
`Latitude and Longitude`: These are the geographic coordinates of the block groups.
`MedHouseValue`: This is the median house value for households within a block (measured in 100,000s).
Understanding these variables allows us to make more sense of our dataset and prepare it for machine-learning tasks.

## Data Preprocessing Start Point

Knowing your data is crucial, but it is equally important to ensure our data is clean and ready for machine learning models. Data preprocessing involves a wide range of tasks, such as dealing with missing values, data normalization or standardization, handling outliers, etc.

Before we preprocess, we need to check if we have any missing values. With pandas, it's as easy as calling the isnull() function.

```Python

print(df.isnull().sum())
"""
MedInc           0
HouseAge         0
AveRooms         0
AveBedrms        0
Population       0
AveOccup         0
Latitude         0
Longitude        0
MedHouseValue    0
dtype: int64
"""
```
The above line of code checks for missing values in our dataset and sums up the null values found for each attribute. As we can see, our dataframe doesn't contain any missing values.
