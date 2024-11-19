# Filtering and Sorting Data with Pandas - Unraveling the Basics

## Welcome to the Next Journey - Data Filtering and Sorting with Pandas

Hello, fellow explorer! we are going to delve into another exciting segment of your data science expedition: Data Filtering and Sorting with Pandas. You'll learn how to narrow down your data to match certain criteria and arrange it in a particular order. This is a fundamental skill when handling data, enabling us to extract valuable information quickly and efficiently.

In the real world, data analysis isn't about dealing with entire datasets but concerning yourself with specific slices of it. For instance, in our Titanic dataset, you might be interested in passengers who survived or those within a certain age group. How about arranging the data based on Fare or Age? That's where data filtering and sorting come into play!

Fundamental Data Filtering in Pandas
Without further ado, let's get into the practical side of things. We'll commence by introducing data filtering, a powerful tool that allows you to extract a subset of your data that meets certain conditions.

Suppose you're interested in data related to passengers who survived the Titanic disaster. How would you extract this data? With Pandas, you can do this using boolean indexing. Here's how it works:

```Python

import seaborn as sns
import pandas as pd

# Load dataset
titanic_df = sns.load_dataset('titanic')

# Filter passengers who survived
survivors = titanic_df[titanic_df['survived'] == 1]
print(survivors.head())

"""
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
8         1       3  female  27.0  ...   NaN  Southampton    yes  False
9         1       2  female  14.0  ...   NaN    Cherbourg    yes  False

[5 rows x 15 columns]
"""
```
In this code, the `titanic_df['survived']` == 1 creates a boolean mask, a sequence of True and False, where True corresponds to passengers who survived and False to those who didn't. When applied to the DataFrame, it returns only the rows where the mask is True, that is, the survivors' data.

## Sorting Data with Pandas

Once we have our filtered data, it's often useful to sort it based on a particular column. For example, we might want to order the survivors' data by age. To do this, we'll use Pandas sort_values() method:

```Python

# Sort survivors by age
sorted_df = survivors.sort_values('age')
print(sorted_df.head())

"""
     survived  pclass     sex   age  ...  deck  embark_town  alive  alone
803         1       3    male  0.42  ...   NaN    Cherbourg    yes  False
755         1       2    male  0.67  ...   NaN  Southampton    yes  False
644         1       3  female  0.75  ...   NaN    Cherbourg    yes  False
469         1       3  female  0.75  ...   NaN    Cherbourg    yes  False
831         1       2    male  0.83  ...   NaN  Southampton    yes  False

[5 rows x 15 columns]
"""
```

The `sort_values()` method arranges the DataFrame in ascending order of the column passed to it as an argument. In our case, it's the age column. The `head() ` function then displays the first 5 rows of the sorted DataFrame.

## Sorting by Multiple Columns

Sometimes, sorting by a single column isn't enough. For instance, what if you want to sort by class and then age within each class? That's where multiple-column sorting comes in. Let's sort our DataFrame first by class ('pclass') in descending order, then by age within each class in ascending order.

```Python

# Sort survivors by class and age
sorted_df = survivors.sort_values(['pclass', 'age'], ascending=[False, True])
print(sorted_df.head())

"""
     survived  pclass     sex   age  ...  deck  embark_town  alive  alone
803         1       3    male  0.42  ...   NaN    Cherbourg    yes  False
469         1       3  female  0.75  ...   NaN    Cherbourg    yes  False
644         1       3  female  0.75  ...   NaN    Cherbourg    yes  False
172         1       3  female  1.00  ...   NaN  Southampton    yes  False
381         1       3  female  1.00  ...   NaN    Cherbourg    yes  False

[5 rows x 15 columns]
"""
```
In this case, we are passing a list of column names to the `sort_values()` function and defining the sort order for each column with `ascending=[False, True]`. This tells pandas to sort by 'pclass' in descending order (from third class to first class) and then sort each class by age in ascending order (from youngest to oldest within each class).

## Employing Multiple Conditions in Data Filtering

However, real-world scenarios often require us to filter data using more complex conditions. For instance, you might want data on female passengers who survived. You can achieve this by combining conditions.

```Python

# Filter female passengers who survived
female_survivors = titanic_df[
    (titanic_df['survived'] == 1) & (titanic_df['sex'] == 'female')
]
print(female_survivors.head())

"""
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
8         1       3  female  27.0  ...   NaN  Southampton    yes  False
9         1       2  female  14.0  ...   NaN    Cherbourg    yes  False

[5 rows x 15 columns]
"""
```
In this code snippet, & stands for the logical AND operator. Thus, the code filters data for passengers who survived ('survived' == 1) and who are female ('sex' == 'female'). The resulting DataFrame, female_survivors, contains information only about women who survived the tragedy.

# Conclusion

And that's it for this session! Give yourself a pat on the back—you've learned how to filter and sort data using pandas. With these skills, you can handle, manipulate, and retrieve data more proficiently.

We covered the basics of data filtering in Pandas using boolean indexing and sorting a DataFrame by a single column. We dove into how to sort by multiple columns and how filtering can employ multiple conditions, providing more flexibility in pinpointing the data you need.



