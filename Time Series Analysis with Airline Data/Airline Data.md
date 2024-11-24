#Exploring the Seaborn Flights Dataset: An Initial Glimpse

## Introduction to the Seaborn Flights Dataset

Our dataset, called the "Flights" dataset, belongs to the Seaborn library. This dataset provides a monthly tally of airline passengers from 1949 to 1960.

### The Flights dataset comprises three distinct columns:

`year`: Represents the year in which the count of passengers was taken.
`month`: Points towards the month in which the passenger count was gathered.
`passengers`: Indicates the number of passengers that traveled in that month of a particular year.
Let's load the dataset in Python. You can easily load this dataset, along with other inbuilt Seaborn datasets, using the `load_dataset()` method as follows:

```Python

import seaborn as sns

# Load the Flights dataset
flights_df = sns.load_dataset('flights')

# Display the first five records
print(flights_df.head())
"""
   year month  passengers
0  1949   Jan         112
1  1949   Feb         118
2  1949   Mar         132
3  1949   Apr         129
4  1949   May         121
"""

# Display the first 10 records
print(flights_df.head(10))

# Display the last five records
print(flights_df.tail())
```
Running the above script will load the "Flights" dataset into a pandas DataFrame and display the first five records, the first ten, and the last 5 records, respectively. As you will see from the output, the dataset contains rows representing individual months over several years, with columns specifying the year, month, and number of passengers.

## Facets of the Dataset

Now, let's delve a little deeper into the structure of our data. Our DataFrame flights_df has a specific shape, i.e., it contains a certain number of rows and columns. You can retrieve this shape using the shape attribute. This attribute returns a tuple representing the dimensionality of the DataFrame. It is used to get the current shape of DataFrame, i.e., (number of rows and columns).

Additionally, you can use the info() method to get a quick description of the data, including the total number of non-null entries and the column data types.

```Python

# Get the dimensions of the dataset
print('Shape of the dataset:', flights_df.shape)
# Output: Shape of the dataset: (144, 3)

# Get more information about the dataset
flights_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 144 entries, 0 to 143
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype   
---  ------      --------------  -----   
 0   year        144 non-null    int64   
 1   month       144 non-null    category
 2   passengers  144 non-null    int64   
dtypes: category(1), int64(2)
memory usage: 2.9 KB
"""
```
This will print out the number of entries, columns, column names, their data types, and the count of non-null entries per column, telling us whether our data has any missing entries. In this case, our dataset is complete and contains no missing values.

## Let's Dig a Bit Deeper

We always want more! It is time we dig a little deeper into the dataset. A quick way to get a summary of the numerical fields in your dataset is to use the `describe()` command. This command provides a statistical summary for numerical columns.

```Python

# Explore basic statistics of the dataset
print(flights_df.describe())
"""
              year  passengers
count   144.000000  144.000000
mean   1954.500000  280.298611
std       3.464102  119.966317
min    1949.000000  104.000000
25%    1951.750000  180.000000
50%    1954.500000  265.500000
75%    1957.250000  360.500000
max    1960.000000  622.000000
"""
```
This command will generate a precise summary of the respective statistics of the DataFrame. You will see from the output that the years range from 1949 to 1960, and the median number of passengers, denoted by the 50% quantile, is around 265.5 - quite insightful already, isn't it?

