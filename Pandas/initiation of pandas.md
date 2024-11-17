## Initiation to Pandas DataFrame

`Pandas DataFrame` is a two-dimensional labeled data structure capable of holding data of various types—integers, floats, strings, Python objects, and more. It's generally the most commonly used Pandas object.

Let's start simply by creating a DataFrame from a dictionary:

```Python

import pandas as pd

data_dict = {"Name": ["John", "Anna", "Peter"],
             "Age": [28, 24, 33],
             "City": ["New York", "Los Angeles", "Berlin"]}

df = pd.DataFrame(data_dict)

print(df)

"""
    Name  Age         City
0   John   28     New York
1   Anna   24  Los Angeles
2  Peter   33       Berlin
"""
```
Each key-value pair in the dictionary corresponds to a column in the resulting `DataFrame`. The key defines the column label, and the corresponding value is a list of column values. The DataFrame constructor takes a dictionary as input and turns it into a two-dimensional table where keys become column names, and values in each key (which should be a list) will be the values for the respective column. Here "John", "Anna", and "Peter" have ages 28, 24, and 33, respectively, and they live in "New York", "Los Angeles", and "Berlin".
