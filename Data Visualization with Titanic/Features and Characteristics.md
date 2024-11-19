# Deep Exploration of the Titanic Dataset: Features and Characteristics

## Insight into Features of the Titanic Dataset
We shall begin our voyage into the dataset by understanding the various attributes of the Titanic dataset.

First, let's briefly go over the features of the Titanic dataset:

`survived`: Whether the passenger survived (0 = No; 1 = Yes).
`pclass`: Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd).
`sex`: Sex of the passenger (male or female).
`age`: Age of the passenger (float number).
`sibsp`: Number of siblings/spouses aboard.
`parch`: Number of parents/children aboard.
`fare`: Passenger fare (in British pounds).
`embarked`: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
... and more!
By discussing these attributes, let's familiarize ourselves with the Titanic dataset available in Seaborn.

```Python

import seaborn as sns

titanic_df = sns.load_dataset('titanic')
print(titanic_df.head())
# This command shows the first five entries of the DataFrame
The output of the head command is in the following table:

Plain text
Copy
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True
```
Each row here represents a different passenger on the ship, while each column corresponds to one of the features described above.

## Diving Deeper: Examining More Characteristics

Our dataset `(titanic_df)` is a Pandas DataFrame, and it comes with many built-in functions that we can use to inspect the data:

`head(n)`: Displays the first n entries of the DataFrame.
`tail(n)`: Displays the last n entries of the DataFrame.
`shape`: Returns the number of rows and columns of the DataFrame.
`info()`: Provides a concise summary of the DataFrame.
`describe()`: Generates descriptive statistics that summarize a dataset's distribution's central tendency, dispersion, and shape.
Each of these functions offers a different perspective on the Titanic dataset:

```Python

# Print the first five entries
print(titanic_df.head())

# Print the last five entries
print(titanic_df.tail())

# Print the shape of the DataFrame
print(titanic_df.shape)
# Output: (891, 15)

# Print a concise summary of the DataFrame
titanic_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
"""

# Print the descriptive statistics of the DataFrame
print(titanic_df.describe())
"""
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
"""
```

The output shows:

The `head` command outputs the first five rows similar to the abovementioned one.
The `tail` command outputs the last five rows of the dataframe.
The `shape` command returns (891, 15), indicating the dataframe has 891 rows and 15 columns.
The `info` command prints a concise summary, including the number of non-null entries for each column.
The `describe` command provides a statistics table for the dataframe's numerical columns.
You will notice from this description that the dataset contains some missing values in features like Age and Embarked, something we will learn to handle in later lessons.

## Deeper Dive with DataFrame Functionality

The `value_counts()` function can also be quite helpful in understanding the distribution of categorical data. For example, if you want to count how many male and female passengers were on the Titanic, you could use this command:

```Python

print(titanic_df['sex'].value_counts())

"""
male      577
female    314
Name: sex, dtype: int64
"""
```
The `nunique()` and `unique()` functions could also come in handy to identify unique entries within your dataset. The former gives the count of unique entries, and the latter gives the actual unique entries.

```Python

# Print the count of unique entries in 'embarked' column
print(titanic_df['embarked'].nunique()) # Output: 3

# Print the unique entries in 'embarked' column
print(titanic_df['embarked'].unique()) # Output: ['S' 'C' 'Q' nan]
```
These additional functions provide functionality to make your exploratory data analysis even more powerful!

## Wrapping Up

We dove into the dataset's content, comprehensively understanding the Titanic passengers and their tragic journey. This deep dive is invaluable in setting the foundation for more advanced data visualizations.

In this, we learned how to:

Load a dataset using Seaborn.
Explore the dataset using the various built-in functions provided by Pandas.
