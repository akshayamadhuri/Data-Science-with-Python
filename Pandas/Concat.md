## The Mighty Concat


`Pandas` provides various ways to combine `DataFrames`, one of which is `concat()`. As the name implies, `concat()` combines DataFrame objects along a particular axis.

Let's create a new DataFrame and concatenate it with our existing DataFrame:

```Python

df2 = pd.DataFrame({"Name": ["Megan"], "Age": [34], "City": ["San Francisco"], "IsYouthful": ["No"]})

df_concatenated = pd.concat([df, df2], ignore_index=True)

print(df_concatenated)

"""
    Name  Age           City IsYouthful
0   John   28       New York        Yes
1   Anna   24    Los Angeles        Yes
2  Peter   33         Berlin         No
3  Megan   34  San Francisco         No
"""
```
Did you notice the `ignore_index=True` parameter? When set to True, it resets the index in the resulting `DataFram`e. So, in the resultant DataFrame, the indices are in increasing order starting from 0.

## Locating Elements in a Pandas DataFrame

`Pandas` provides several ways to locate elements in a `DataFrame`.

The simplest way to select a column in a DataFrame is by label:

```Python

print(df['column_name']) # select a single column
print(df[['col1', 'col2']]) # select multiple columns
To select elements within the DataFrame by integer location, we use the iloc method. The iloc indexer is like Python list slicing. This accepts integer inputs and slice notation. The general syntax is df.iloc[row_selection, column_selection]:

For example, if we wish to select the value in the second row (indexed at 1) and the first column (indexed at 0):

Python
Copy
df.iloc[1,0] # Select the value in the second row and the first column (1-based)
df.iloc[:2,:2] # Select the first two rows and columns
```
