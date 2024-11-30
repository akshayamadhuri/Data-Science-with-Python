# Mastering Advanced Functions in Pandas: Groupby and Apply for Large-Scale Data Analysis

## Introduction to Mastering Pandas: Advanced Functions

we focus on enhancing your Python skills by exploring the advanced functions that Pandas offers — specifically, the groupby and apply methods.

These tools are central to handling large-scale datasets and simplifying complex data analysis maneuvers. To illustrate this, consider a scenario in an eCommerce business: You want to find the total revenue grouped by different product categories. Here, the groupby function can efficiently sort your large sales data by product categories, and the apply function can help calculate the revenue for these categories. Such manipulations are pivotal for efficient data preprocessing, especially in areas like Machine Learning, where understanding the relationships between different data groups can provide valuable insights.

Our goal for today is threefold: to understand the functionalities of groupby and apply, to recognize their role in data transformation, and most importantly, to apply these tools to tackle complex data analysis problems.

## Deep Dive into the groupby() Method in Pandas

The groupby method plays a crucial role in Pandas. It helps in grouping large data sets based on specified criteria by following a 'split-apply-combine' approach.

To clarify, consider you are an instructor in a school and want to calculate the average score for each of your students in various subjects. The 'split' phase would involve dividing the students based on their subjects. The 'apply' phase calculates the average for each student, and the 'combine' phase compiles these averages against each specific subject.

In coding parlance, the splitting criterion is defined through keys, which can either be a series of labels or an array of the same length as the axis being grouped. Here's a simple demonstration of the groupby method:

```Python

import pandas as pd

# Create a simple dataframe
data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
       'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
       'Sales': [200, 120, 340, 124, 243, 350]}
df = pd.DataFrame(data)

# Apply groupby
df_grouped = df.groupby('Company')
for key, item in df_grouped:
    print("\nGroup Key: {}".format(key))
    print(df_grouped.get_group(key), "\n")
"""
Group Key: FB
  Company Person  Sales
4      FB   Carl    243
5      FB  Sarah    350 


Group Key: GOOG
  Company   Person  Sales
0    GOOG      Sam    200
1    GOOG  Charlie    120 


Group Key: MSFT
  Company   Person  Sales
2    MSFT      Amy    340
3    MSFT  Vanessa    124
"""
```
In the above example, groupby('Company') organizes the DataFrame by its Company column. However, this doesn't display a DataFrame. This is because groupby returns a groupby object that includes many useful methods for performing various operations on these groups. We will explore some of these in the next section.

## Unraveling the groupby() Operations

The pronounced benefit of the groupby method is the variety of operations we can perform on the groupby object. Functions like sum(), mean(), etc., help us simplify the grouped data into more insightful information. Here's how we can use groupby and find out the total sales for each company:

```Python

grouped = df.groupby('Company')
print(grouped.sum())
"""
             Person  Sales
Company                   
FB        CarlSarah    593
GOOG     SamCharlie    320
MSFT     AmyVanessa    464
"""
```
This function will return the sum of all columns (where applicable) for each company in our grouped data. We can effectively dissect our dataset into richer, more insightful information.

## Introduction to the Apply Method in Pandas

Once we've split our DataFrame into different groups, it is time to introduce apply(). This function applies a specific function to every member of a sequence, such as a Series or DataFrame, effectively combining groupby() and apply() to conduct intricate data manipulation tasks.

Here's a simplified instance of the apply method:

```Python

import numpy as np
import pandas as pd

# Create a dataframe
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

# Define a function
def get_sum(row):
    return row.sum()

# Apply the function 
df['sum'] = df[['C', 'D']].apply(get_sum, axis=1)

print(df)
"""
     A      B         C         D       sum
0  foo    one -0.343200  0.184665 -0.158535
1  bar    one  0.058870  1.835614  1.894484
2  foo    two  0.801743 -0.184409  0.617333
3  bar  three  0.935406  0.124109  1.059515
4  foo    two  0.782074  0.583470  1.365544
5  bar    two  0.138934  0.710407  0.849341
6  foo    one  0.364633  1.147963  1.512596
7  foo  three -1.364677  1.719538  0.354861
"""
```
In the example above, we've defined a function, get_sum(), and then used the apply method to apply this function to every row in the dataframe. This operation results in a new 'sum' column which is the sum of 'C' and 'D' for each row.

Leveraging the Power of Apply and Groupby
The apply method can be leveraged most effectively by combining it with groupby. This combination allows us to apply functions not just to each row or column of a DataFrame but also to each group of rows. For instance, let's find the maximum sales for each company:

```Python

print(df.groupby('Company').apply(lambda x: x['Sales'].max()))
"""
Company
FB      350
GOOG    200
MSFT    340
dtype: int64
"""
In this example, groupby('Company') divides our DataFrame by the Company column. Then apply(lambda x: x['Sales'].max()) applies a lambda function to each group, returning the maximum 'Sales' for each company.

Delving into the California Housing Dataset with Advanced Pandas
With the concepts of apply and groupby under our belt, let's dive into the California Housing dataset and extract valuable insights using these functions.

Here is how we import the California Housing dataset:

Python
Copy to clipboard
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Fetch the dataset
data = fetch_california_housing(as_frame=True)

# create a DataFrame
housing_df = pd.DataFrame(data=data.data, columns=data.feature_names)
```
In the above example, fetch_california_housing(as_frame=True) fetches the dataset as a DataFrame. The comprehensive dataset contains houses' values from California, as well as other corresponding features such as median income, average occupancy, etc.

## Advanced Data Analysis

Now, let's apply all our learning to solve a complex problem: calculating the average population for each income category. To do this, we first need to categorize incomes into different categories, which is where the function pd.cut() comes in. It segments and sorts data values into bins. Then groupby() will group our DataFrame by these income categories, and finally, apply() will calculate the average population for each group. Here's the code:

```Python

# Define income category
housing_df['income_cat'] = pd.cut(housing_df['MedInc'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Group by income category and calculate the average population
average_population = housing_df.groupby('income_cat').apply(lambda x: x['Population'].mean())

print(average_population)
"""
income_cat
1    1105.806569
2    1418.232336
3    1448.062465
4    1488.974718
5    1389.890347
dtype: float64
"""
```
In this snippet, pd.cut() segments the median income into different categories, which are labeled from 1 to 5. groupby('income_cat') then groups the DataFrame by these income categories, and apply(lambda x: x['Population'].mean()) calculates the average population for each income category.
