# Diving into the Wine Quality Dataset: An In-depth Overview

## Kickoff: Overview of the Wine Quality Dataset

A thorough understanding of your dataset is essential before developing machine learning models. A comprehensive dataset review empowers us to identify potential features that can significantly influence output variables. This process is akin to familiarizing oneself with a novel's characters before delving into the plot; possessing nuanced knowledge of the dataset makes the subsequent modeling phase more coherent.

In the spirit of curiosity, the Wine Quality dataset paves the way for us to explore a real-world problem: determining wine quality based on its physicochemical characteristics. As budding machine learning practitioners, this experience enlivens our learning journey by engaging us in practical applications within an accessible context. So, shall we make a toast to learning and dive right in?

## Meet the Dataset: Wine Quality Dataset
As the name suggests, the Wine Quality dataset encompasses data on wines, specifically, the physicochemical properties of red and white variants of Portuguese "Vinho Verde" wine. The dataset consists of 12 variables, inclusive of quality — the target variable. Here's a quick summary of key columns:

`fixed acidity`

`volatile acidity`

`citric acid`

`residual sugar`

`chlorides`

`free sulfur dioxide`

`total sulfur dioxide`

`density`

`pH`
`sulphates`
`alcohol`
`quality` (score between 0 and 10)

Now, let's learn how to load the dataset. As referred to in the course brief, we'll employ the datasets Python library, which conveniently facilitates the loading of various datasets. This specific dataset is already available in the CodeSignal environment.

```Python

import datasets
import pandas as pd

# Loading Dataset
red_wine = datasets.load_dataset('codesignal/wine-quality', split='red')
white_wine = datasets.load_dataset('codesignal/wine-quality', split='white')
red_wine = pd.DataFrame(red_wine)
white_wine = pd.DataFrame(white_wine)

# Checking the shape of the dataset
print("Red Wine Dataset Shape: ", red_wine.shape) # Red Wine Dataset Shape:  (1599, 12)
print("White Wine Dataset Shape: ", white_wine.shape) # White Wine Dataset Shape:  (4898, 12)
```
In the snippet above, we load the red and white wine datasets separately and subsequently display their respective sizes as an output of the shape function.

More Dataset Insights
Digging deeper, we can examine various features, their types, statistical summaries, and unique value counts for a richer understanding. The Python code below checks the data types of the features.

```Python

# Check Red Wine Dataset data types
print("Red Wine Dataset Data Types:")
print(red_wine.dtypes)
"""
Red Wine Dataset Data Types:
fixed acidity           float64
volatile acidity        float64
citric acid             float64
residual sugar          float64
chlorides               float64
free sulfur dioxide     float64
total sulfur dioxide    float64
density                 float64
pH                      float64
sulphates               float64
alcohol                 float64
quality                 float64
dtype: object
"""

# Check White Wine Dataset data types
print("\nWhite Wine Dataset Data Types:")
print(white_wine.dtypes)
"""
the structure is the same as in the red wine dataset
"""
```
Next, we'll obtain a brief stats summary and unique value count using Python:

```Python

# Describing Red Wine Dataset
print("Red Wine Dataset Description:")
print(red_wine.describe())
"""
Red Wine Dataset Description:
       fixed acidity  volatile acidity  ...      alcohol      quality
count    1599.000000       1599.000000  ...  1599.000000  1599.000000
mean        8.319637          0.527821  ...    10.422983     5.636023
std         1.741096          0.179060  ...     1.065668     0.807569
min         4.600000          0.120000  ...     8.400000     3.000000
25%         7.100000          0.390000  ...     9.500000     5.000000
50%         7.900000          0.520000  ...    10.200000     6.000000
75%         9.200000          0.640000  ...    11.100000     6.000000
max        15.900000          1.580000  ...    14.900000     8.000000

[8 rows x 12 columns]
"""

# Unique values
print("\nUnique values in Red Wine Dataset:")
print(red_wine.nunique())
"""
Unique values in Red Wine Dataset:
fixed acidity            96
volatile acidity        143
citric acid              80
residual sugar           91
chlorides               153
free sulfur dioxide      60
total sulfur dioxide    144
density                 436
pH                       89
sulphates                96
alcohol                  65
quality                   6
dtype: int64
"""

# Describing White Wine Dataset
print("\nWhite Wine Dataset Description:")
print(white_wine.describe())
"""
White Wine Dataset Description:
       fixed acidity  volatile acidity  ...      alcohol      quality
count    4898.000000       4898.000000  ...  4898.000000  4898.000000
mean        6.854788          0.278241  ...    10.514267     5.877909
std         0.843868          0.100795  ...     1.230621     0.885639
min         3.800000          0.080000  ...     8.000000     3.000000
25%         6.300000          0.210000  ...     9.500000     5.000000
50%         6.800000          0.260000  ...    10.400000     6.000000
75%         7.300000          0.320000  ...    11.400000     6.000000
max        14.200000          1.100000  ...    14.200000     9.000000

[8 rows x 12 columns]
"""

# Unique values
print("\nUnique values in White Wine Dataset:")
print(white_wine.nunique())
"""
Unique values in White Wine Dataset:
fixed acidity            68
volatile acidity        125
citric acid              87
residual sugar          310
chlorides               160
free sulfur dioxide     132
total sulfur dioxide    251
density                 890
pH                      103
sulphates                79
alcohol                 103
quality                   7
dtype: int64
"""
```
Executing the above Python script generates a statistical summary for each feature in the dataset and counts the unique values, thus shedding light on the diversity of the datasets.

## Checking for Missing Values

It is crucial to check if our data contain missing values, as these can significantly affect the outcomes of our data analysis and model accuracy. Here's how to check for missing data:

```Python

# Check missing values in Red Wine Dataset
print("Missing values in Red Wine Dataset:")
print(red_wine.isnull().sum()) # There are no null values in all columns


# Check missing values in White Wine Dataset
print("\nMissing values in White Wine Dataset:")
print(white_wine.isnull().sum()) # There are no null values in all columns
```

## A Peek at Data Visualization

Let's delve one step further to better understand our dataset by visualizing the target variable quality. We'll use the matplotlib library to generate histograms of the wine quality for the red and white wine datasets.

```Python

import matplotlib.pyplot as plt

# Plot for Red Wine
plt.hist(red_wine.quality, bins=10, color='red', alpha=0.7)
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Quality Distribution for Red Wine')
plt.show()

# Plot for White Wine
plt.hist(white_wine.quality, bins=10, color='skyblue', alpha=0.7)
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Quality Distribution for White Wine')
plt.show()
```
![image](https://github.com/user-attachments/assets/aa82b0d9-c4b9-4f60-a00c-fe15b1efa6f5)


![image](https://github.com/user-attachments/assets/1071a06f-5d95-45c3-8a28-9fa24bbc1eef)


These histograms visualize the count of wine samples at each quality score, providing insight into how the quality of the wine is distributed.
