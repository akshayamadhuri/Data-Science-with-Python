# Data Cleaning Techniques: Detecting and Handling Missing Data

## Intro to Handling Missing Data

we delve into the topic of handling missing data - a common occurrence in the realm of data cleaning and manipulation. Regardless of the domain, be it retail, healthcare, finance, or any other, dealing with missing data is a crucial step in maintaining the integrity of the dataset and delivering accurate analyses or predictions.

Dealing with missing values is a cornerstone of the data preprocessing pipeline. Data could be missing in real-life scenarios for various reasons - it might not have been collected, perhaps due to human error or system problems. Regardless of why the data is missing, we need to identify and handle these values to ensure that we make accurate and reliable predictions from our data.

## Detecting Missing Values in Pandas

Our first step in handling missing data is to detect those missing values. The Pandas library provides us the isnull() function, which returns a Boolean DataFrame of the same shape as our input, indicating with a True or False whether each individual value is missing.

Using our Titanic dataset as an example, let's demonstrate this process:

```Python

import seaborn as sns

# Load the dataset
titanic_df = sns.load_dataset('titanic')

# Detect missing values 
missing_values = titanic_df.isnull()
print(missing_values.head(10))
"""
   survived  pclass    sex    age  ...   deck  embark_town  alive  alone
0     False   False  False  False  ...   True        False  False  False
1     False   False  False  False  ...  False        False  False  False
2     False   False  False  False  ...   True        False  False  False
3     False   False  False  False  ...  False        False  False  False
4     False   False  False  False  ...   True        False  False  False
5     False   False  False   True  ...   True        False  False  False
6     False   False  False  False  ...  False        False  False  False
7     False   False  False  False  ...   True        False  False  False
8     False   False  False  False  ...   True        False  False  False
9     False   False  False  False  ...   True        False  False  False

[10 rows x 15 columns]
"""
```
Here, we have a DataFrame of the same size as `titanic_df`, but instead of actual data, it holds Boolean values with True indicating the presence of a missing datapoint and False standing for a valid existing data point.

## Counting Missing Values in Each Column

While the step above provides a granular look at our missing data, a more top-level view that is often more useful is the number of missing values in each column. To get this, Pandas provides us with a handy method: sum(). After isnull(), it counts each column's total number of True (i.e., missing) values.

```Python

missing_values_count = titanic_df.isnull().sum()
print(missing_values_count)
"""
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
"""
```
This code calculates and prints the number of missing data points in each column, providing an overview of the completeness of the data in our DataFrame.

## Dealing with Missing Values: Dropping

Before we proceed to the imputation methods, it is important to mention that sometimes the best way to handle missing data is to drop the rows or columns containing them, especially when the data missing is very little and wouldn't impact our analysis or predictions.

Pandas provides the `dropna()` function for this purpose. Here's a demonstration:

```Python

# Copy the original dataset
titanic_df_copy = titanic_df.copy()

# Drop rows with missing values
titanic_df_copy.dropna(inplace=True)

# Check the dataframe
print(titanic_df_copy.isnull().sum())
# There will be no missing values in every column
```
In the given example, we used `inplace=True` to modify the original DataFrame itself.

## Visualizing Missing Data with Seaborn

Visualizing data is often more insightful. Seaborn's heat map function offers a convenient tool to scrutinize missing data visually. It uses different color intensities to represent the presence or absence of data:

```Python

import matplotlib.pyplot as plt
import seaborn as sns

# Detected missing values visualized
plt.figure(figsize=(10,6))
sns.heatmap(titanic_df.isnull(), cmap='viridis')
plt.show()
```
![image](https://github.com/user-attachments/assets/5472b738-fe7c-4a6b-a0cb-8185f6341853)

## Handling Missing Values: Imputation

It's time to handle the detected missing values. One common strategy is to fill in the missing data values, known as "imputation". We can do this in several ways based on the nature and distribution of our data.

In the case of the 'age' variable in our Titanic dataset (which is numerical), we can fill in missing values with either the mean, median, or mode of the available values. Here's the method demonstrated with mean:

```Python

# Impute missing values using mean
titanic_df['age'].fillna(titanic_df['age'].mean(), inplace=True)

# Check the dataframe
print(titanic_df.isnull().sum())
"""
survived         0
pclass           0
sex              0
age              0
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
"""
```
Here, the 'age' column's missing values get filled with the mean age. Suddenly, we no longer have any missing values in our 'age' column!

Another variant of the fillna() method involves forward fill or backward fill, where missing values are filled with the previous or next respective value in the DataFrame:

```Python

# Impute missing values using backward fill
titanic_df['age'].fillna(method='bfill', inplace=True)

# Check the dataframe
print(titanic_df.isnull().sum())
# The output is the same as in the previous example
```
In the above example, each missing value in the `'age'` column is filled with its subsequent value in the DataFrame. Please note again: `inplace=True` means the change should be reflected in the DataFrame itself.

