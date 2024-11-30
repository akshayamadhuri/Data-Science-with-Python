# Expanding Horizons: Applications of Numpy and Pandas in Bioinformatics, Astronomy, and Social Networks

## Bioinformatics and Data Manipulation

Bioinformatics combines biology and computer science, providing a platform for analyzing and interpreting complex biological data, particularly genetic data. Bioinformatics is often confronted with vast and intricate datasets that require advanced data manipulation techniques to extract valuable insights.

Consider a realistic illustration of bioinformatics data - DNA sequences. These sequences are strings of characters representing nucleotides labeled as A, T, C, G. Here's a pandas DataFrame that encapsulates the DNA sequences of several genes.

```Python

import pandas as pd

# DNA sequences for several genes
data = {
    "Gene": ["Gene A", "Gene B", "Gene C", "Gene D"],
    "Sequence": ["ATCGTACGA", "CGATCGATG", "TAGCTAG", "CGTAGCTA"]
}

df_genes = pd.DataFrame(data)
print(df_genes)
"""
     Gene   Sequence
0  Gene A  ATCGTACGA
1  Gene B  CGATCGATG
2  Gene C    TAGCTAG
3  Gene D   CGTAGCTA
"""
```

In the DataFrame df_genes, each row corresponds to a distinct gene. For instance, the first row provides information about "Gene A," with the sequence "ATCGTACGA". Now, suppose we wish to determine the length of these sequences. Pandas allow us to fulfill this requirement with relative ease. Let's use the Pandas apply function, a versatile function that applies a function along an axis of the DataFrame. In this case, the function is used to compute the length of each DNA sequence in our DataFrame. We then add this data as a new column, Length, to our DataFrame:

```Python

df_genes['Length'] = df_genes['Sequence'].apply(len)
print(df_genes)
"""
     Gene   Sequence  Length
0  Gene A  ATCGTACGA       9
1  Gene B  CGATCGATG       9
2  Gene C    TAGCTAG       7
3  Gene D   CGTAGCTA       8
"""
```
As we can see, we used the apply function to apply the len function to the 'Sequence' column, calculated the length of each sequence, and added the result to a new column, 'Length'. This form of operation is a staple of data manipulation in bioinformatics.

## Astronomy: Handling Large Datasets

Astronomical research often grapples with large datasets. For instance, astronomical surveys create extensive catalogs of millions to billions of stars. Numpy and Pandas give us the power to manage and manipulate these immense datasets efficiently.

Consider a dataset of star observations that includes their astronomical coordinates (Right Ascension and Declination), magnitudes, and the dates of the observations.

```Python

import numpy as np

# Star dataset (Simulated data for demonstration)
data = {
    "Star_ID": np.arange(1, 5),
    "Right_Ascension": [204.85, 63.70, 305.29, 45.2],
    "Declination": [-29.72, 38.03, -14.78, 7.8],
    "Magnitude": [2.04, 1.25, 3.17, 1.9],
    "Observation_Date": pd.date_range('01/01/2020', periods=4)
}

df_stars = pd.DataFrame(data)
print(df_stars)
"""
   Star_ID  Right_Ascension  Declination  Magnitude Observation_Date
0        1           204.85       -29.72       2.04       2020-01-01
1        2            63.70        38.03       1.25       2020-01-02
2        3           305.29       -14.78       3.17       2020-01-03
3        4            45.20         7.80       1.90       2020-01-04
"""
```
Filtering out stars based on their magnitudes or observation dates is commonplace in Astronomy. Pandas facilitate these requirements, making operations more intuitive and direct. Let's demonstrate this with a simple filter to exclude stars observed before a specific date:

```Python

filter_date = pd.to_datetime('2020-01-02')
filtered_stars = df_stars[df_stars['Observation_Date'] > filter_date]
print(filtered_stars)
"""
   Star_ID  Right_Ascension  Declination  Magnitude Observation_Date
2        3           305.29       -14.78       3.17       2020-01-03
3        4            45.20         7.80       1.90       2020-01-04
"""
```
Here, we use pandas' to_datetime function to convert a string into a datetime object. Then, we use the DataFrame filtering feature from Pandas to identify stars observed after the specified date.

## Data Analysis in Social Networks

Social Network Analysis (SNA) effectively reveals the underlying structure within social networks using network and graph theories. It characterizes network structures using nodes (individual elements in the network) and edges (which represent relationships between these nodes).

Consider the following dataset, which represents social interaction among individuals:

```Python

# Social interaction data (Simulated for demonstration)
data = {
    "Person": ["Alice", "Bob", "Charlie", "Dave"],
    "Friends": [10, 5, 8, 2],
    "Posts": [100, 50, 80, 200]
}

df_social = pd.DataFrame(data)
print(df_social)
"""
    Person  Friends  Posts
0    Alice       10    100
1      Bob        5     50
2  Charlie        8     80
3     Dave        2    200
"""
```
Now, suppose our goal is to compute the average number of posts per friend for each individual. We can accomplish this expeditiously using Pandas. Here is a demonstration of how we introduce a new column into our DataFrame, Posts_per_Friend, which represents the average number of posts per friend:

```Python

df_social['Posts_per_Friend'] = df_social['Posts'] / df_social['Friends']
print(df_social)
"""
    Person  Friends  Posts  Posts_per_Friend
0    Alice       10    100              10.0
1      Bob        5     50              10.0
2  Charlie        8     80              10.0
3     Dave        2    200             100.0
"""
```
We divide the 'Posts' column by the 'Friends' column for each person using vectorized operations from Pandas, which helps us to calculate the average number of posts per friend. We then assign this new calculation to a new column, 'Posts_per_Friend'. Data manipulation of this type is a typical aspect of Social Network Analysis.
