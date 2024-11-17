## DataFrame Characteristics

To inspect the structure and properties of a DataFrame, we have a range of functions at our disposal. Here are some commonly used ones:

`df.head(n)`: Returns the first n rows of the DataFrame df.
`df.tail(n)`: Returns the last n rows of the DataFrame df.
`df.shape`: Returns a tuple representing the dimensions (number_of_rows, number_of_columns) of the DataFrame df.
`df.columns`: Returns an index containing column labels of the DataFrame df.
`df.dtypes`: Returns a series with the data type of each column.
Let's see these functions in action below:

```Python

print(df.head(2))  # Print first two rows
print(df.tail(2))  # Print last two rows
print(df.shape)    # Print dimensions of the df (rows, columns): (3, 3)
print(df.columns)  # Print column labels: Index(['Name', 'Age', 'City'], dtype='object')
print(df.dtypes)   # Print data types of each column:
# Name    object
# Age      int64
# City    object
# dtype: object
```
These commands will help us understand the basic shape and type of the DataFrame. `df.head(2)` prints the first 2 rows of the DataFrame, which are {[John, 28, New York], [Anna, 24, Los Angeles]}. `df.tail(2)` prints the last 2 rows which are {[Anna, 24, Los Angeles}, [Peter, 33, Berlin]}. `df.shape` gives us the dimensions of the DataFrame here being (3,3) indicating 3 rows and 3 columns. `df.columns` prints the column names [Name, Age, City] and `df.dtypes` gives us the data types in each column.

