# Data Cleaning Techniques: Working with Categorical Data Encoding and Transformation

## Introduction to Encoding and Transforming Categorical Data

By generating numerical representations, we make it possible to build models using datasets that contain categorical variables. This session focuses on introducing you to different types of categorical data encodings, understanding their use, and learning how to apply them.

Understanding categorical variable encoding is essential for a wide array of machine-learning tasks. Sadly, not all algorithms can understand human language the way we do. By converting these text data into numbers, we are translating the data into a format that algorithms can process - and that's what we will cover in this lesson.

Any guesses on the effects that a passenger's gender or embarkation point might have on their survival rates? We address these issues by using different types of encoding techniques to convert the gender and embarkation point details into a form that a machine learning model can understand.

## Gearing Up: Load Libraries and Dataset

While Python provides built-in methods for encoding, the Pandas library shines with its efficiency and simplicity. Let's begin by loading our libraries and dataset.

```Python

import pandas as pd
import seaborn as sns

# Load Titanic dataset
titanic_df = sns.load_dataset('titanic')
```
The above code will load the Titanic dataset and allow us to transform it using different techniques, shown in the following sections.

## Handling Categorical Variables

As part of this session, we mainly consider two categorical variables from the Titanic dataset, sex and embark_town. These columns are in a text format to which our algorithms can't relate. Hence, we use different encoding techniques to solve our problem.

```Python

# Display unique categories in 'sex' and 'embark_town'
print(titanic_df['sex'].unique()) # Output: ['male' 'female']
print(titanic_df['embark_town'].unique()) # Output: ['Southampton' 'Cherbourg' 'Queenstown' nan]
```
This prints out all unique categories within sex and embark_town columns. These categories can be encoded to numbers in a few ways, as shown in the following sections.

## Label Encoding with Pandas

Label encoding converts each category in the variable to a numerical value. You can accomplish this using the factorize() function in Pandas.

```Python

# Label Encoding for 'sex'
titanic_df['sex_encoded'] = pd.factorize(titanic_df['sex'])[0]
print(titanic_df[['sex', 'sex_encoded']].head())
"""
      sex  sex_encoded
0    male            0
1  female            1
2  female            1
3  female            1
4    male            0
"""
```

In this example, the factorize() function assigns numerical values to each category in the sex column. A new column, sex_encoded, is then created to store these encoded values. If you print out the first few records of the sex and sex_encoded columns, you'll see the male and female categories transformed into 0 and 1, respectively.

It is important to note the use of [0] in the code. The factorize() function returns two items: the first is an array containing the encoded labels (the actual numerical representation), and the second is an array containing the unique values. By using [0], we're choosing only to take the first item (the numerical labels), ignoring the unique values.

## One-Hot Encoding with Pandas

One-hot encoding is another common method for encoding categorical variables. It creates a binary column for each category in the variable. This is especially useful when there is no ordinal relationship between the categories, just like in our embark_town example. Pandas' get_dummies() is used for this:

```Python

# One-Hot Encoding for 'embark_town'
encoded_df = pd.get_dummies(titanic_df['embark_town'], prefix='town')
titanic_df = pd.concat([titanic_df, encoded_df], axis=1)
print(titanic_df.head())
"""
   survived  pclass     sex  ...  town_Cherbourg  town_Queenstown  town_Southampton
0         0       3    male  ...           False            False              True
1         1       1  female  ...            True            False             False
2         1       3  female  ...           False            False              True
3         1       1  female  ...           False            False              True
4         0       3    male  ...           False            False              True
"""
```
This script will create three additional columns, town_Cherbourg, town_Queenstown, and town_Southampton for the three categories in embark_town. It assigns 1 to the relevant category and 0 to others, making it easier for algorithms to understand.
