# Array Manipulations: Indexing, Slicing and Reshaping Numpy Arrays

Numpy arrays can be manipulated in various ways. For instance, we can access specific positions (indexing) and even slices of the array (slicing), or change the shape of the array (reshaping).

Let's play around with these manipulations:

```Python

# Indexing: access the element at the first row, third column
print("Indexed Value: ", np_array[0, 2]) # Indexed Value:  3

# Slicing: access the first row 
print("Sliced Value: ", np_array[0,:]) # Sliced Value:  [1 2 3]

# Reshape the array to 3 rows and 2 columns (only applicable if the reshaped total size equals the original size)
reshaped_array = np_array.reshape(3, 2)
print("Reshaped Array:\n", reshaped_array)
# Reshaped Array:
# [[1 2]
#  [3 4]
#  [5 6]]
```

In the code snippet shown above, indexing is used to access a specific element in the array, slicing is used to access a range of elements, and reshaping is used to reconfigure the layout of the array while keeping the data intact.


