# Engineering New Features for Better Predictions
## Intro to Feature Engineering

Welcome to this lesson on Feature Engineering! Today, we'll explore how to derive new features from our existing data to enhance our predictive models. These derived features could provide more insightful information that our original data might not capture directly.

Feature Engineering is an essential part of machine learning, and it's the process of using domain knowledge to create features that make machine learning algorithms work. Although modern machine learning methods can automatically derive features, manually combining existing features – based on human intuition and industry expertise – can often produce better results.

Why is Feature Engineering vital? Consider this parallel: Artistic talent won't help a painter without paints, and a high-quality dataset may be useless without proper features. The process of Feature Engineering ensures you have the 'right paint' to create your masterpiece!

Let's use the Titanic dataset as an example. We could create a new feature, age_group to categorize age into different groups, or another feature, family_size, by adding sibsp (number of siblings/spouses aboard) and parch (number of parents/children aboard). Let's dive in!

## Creating New Features

We'll start by creating the family_size feature. This is simply the sibsp and parch features added together plus one (the passenger themself). You might be wondering why we are creating the family_size feature. The reason is that sometimes, the size of the family might have a significant impact on the survival chance of a person. For instance, if a person has a big family, they might have gotten confused and lost in the crowd, or they might have tried to look for their family members, delaying their escape.

```Python

# Load the data
import seaborn as sns

titanic_df = sns.load_dataset('titanic')

# Create a new feature, 'family_size'
titanic_df['family_size'] = titanic_df['sibsp'] + titanic_df['parch'] + 1
print(titanic_df.head())
"""
   survived  pclass     sex   age  ...  embark_town  alive  alone family_size
0         0       3    male  22.0  ...  Southampton     no  False           2
1         1       1  female  38.0  ...    Cherbourg    yes  False           2
2         1       3  female  26.0  ...  Southampton    yes   True           1
3         1       1  female  35.0  ...  Southampton    yes  False           2
4         0       3    male  35.0  ...  Southampton     no   True           1

[5 rows x 16 columns]
"""
```
After executing the code above, you'll notice an extra column family_size in the dataset, representing each passenger's family size. For instance, the first passenger (Mr. Owen Harris) has a family size of 2 (one spouse aboard), and Miss. Laina has a family size of 1 (alone).

## Creating Categorical Features

Another common operation in feature engineering is the creation of categorical features. Usually, categories carry more meanings than continuous values. For instance, we could categorize age into different age groups. We can use the cut() function from pandas, which segments and sorts data values into bins. This function is quite efficient for transforming continuous variables into categorical counterparts. An underlying concept of the function is that it uses the values of the input array to determine the appropriate bin for each value.

```Python

# Import pandas
import pandas as pd

# Define the bin edges
age_bins = [0, 12, 18, 30, 45, 100]

# Define the bin labels
age_labels = ['Child', 'Teenager', 'Young Adult', 'Middle Age', 'Senior']

# Create the age group feature
titanic_df['age_group'] = pd.cut(titanic_df['age'], bins=age_bins, labels=age_labels)

# Show the first few rows of the data
print(titanic_df.head())
"""
   survived  pclass     sex   age  ...  alive  alone  family_size    age_group
0         0       3    male  22.0  ...     no  False            2  Young Adult
1         1       1  female  38.0  ...    yes  False            2   Middle Age
2         1       3  female  26.0  ...    yes   True            1  Young Adult
3         1       1  female  35.0  ...    yes  False            2   Middle Age
4         0       3    male  35.0  ...     no   True            1   Middle Age

[5 rows x 17 columns]
"""
```

Here, `pd.cut()` function is used to segregate array elements into different bins. The bins argument defines the bin edges, and the labels argument sets the label names for the resultant bins. In the output, you'll notice a new column, age_group, categorizing passengers into different age groups.

Let's check the distribution of the age_group to verify that the transformation was successful:

```Python

# Check the distribution of the 'age_group' column
print(titanic_df['age_group'].value_counts())
"""
age_group
Young Adult    270
Middle Age     202
Senior         103
Teenager        70
Child           69
Name: count, dtype: int64
"""
```
You'll see that each age group has a specific count of passengers in the dataset belonging to that group.

