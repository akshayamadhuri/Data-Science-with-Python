## Using λ (Lambda) for DataFrame Manipulation

The `apply()` function in `Pandas` is a versatile tool to manipulate DataFrame values. It allows us to apply a function (either a Python built-in function or a custom function) along the DataFrame's axes (either row-wise or column-wise).

Let's demonstrate this by adding a new column to our DataFrame, which represents whether a person is considered youthful by applying a lambda function to the "Age" column.

Lambda functions, λ (Lambda), in Python, are small anonymous functions that are defined with the lambda keyword. They can take any number of arguments and can only have one expression. They are particularly useful when you need to pass a small function as an argument.

```Python

df["IsYouthful"] = df["Age"].apply(lambda age: "Yes" if age < 30 else "No")
print(df)

"""
    Name  Age         City IsYouthful
0   John   28     New York        Yes
1   Anna   24  Los Angeles        Yes
2  Peter   33       Berlin         No
"""
```
In the above example, we used a lambda function that takes an age as an argument and returns "Yes" if the age is less than 30 and "No" otherwise.
