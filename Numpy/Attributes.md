# Attributes of Numpy Arrays

Numpy arrays come equipped with several attributes for gaining more insights:

`ndim` tells us the number of dimensions of the array.

`shape` gives us the size of each dimension.

`size` tells us the total number of elements in the array.

`dtype` tells us the data type of the elements in the array.

## Let's create a 2D array and use these attributes:

```Python

np_array = np.array([[1, 2, 3], [4, 5, 6]])

print("Dimensions: ", np_array.ndim) # Dimensions:  2
print("Shape: ", np_array.shape)     # Shape:  (2, 3)
print("Size: ", np_array.size)       # Size: 6
print("Data Type: ", np_array.dtype) # Data Type:  int64
```

After running these lines, we see that `np_array` is a 2D array (since ndim returns the value 2), has 2 rows and 3 columns (shape returns the tuple (2, 3)), contains 6 elements (size returns 6), and is of integer type (dtype returns 'int64').
