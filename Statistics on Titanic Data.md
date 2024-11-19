 ## Descriptive Statistics
Descriptive statistics are appropriately named, as they provide insights into the main features of our data. Let's start with the Titanic dataset and calculate some basic statistics for the age of passengers: the mean, median, and mode.

```Python

import numpy as np
import pandas as pd
import seaborn as sns

# Load Titanic dataset
titanic_df = sns.load_dataset('titanic')

mean_age = titanic_df['age'].mean()
median_age = titanic_df['age'].median()
mode_age = titanic_df['age'].mode()[0]

print(f"Mean age: {mean_age}") # Mean age: 29.69911764705882
print(f"Median age: {median_age}") # Median age: 28.0
print(f"Mode age: {mode_age}") # Mode age: 24.0
```
The code calculates and displays the mean (average), median (middle value), and mode (most frequently occurring value) of the age column. These are measures of central tendency, and they give us a general picture of the age distribution of passengers aboard the Titanic.

## Measures of Variability: Standard Deviation
Apart from measures of central tendency, there is another important style of measurement in statistics - measures of dispersion (variability). One of the common ways to gauge the variability in a dataset is via the standard deviation, which measures how much the values in a dataset vary around the mean. A super low standard deviation indicates a dataset with values clustered around the mean, while a higher standard deviation represents a wider spread around the mean. For our Titanic dataset, we can calculate the standard deviation of age as follows:

```Python

# Standard deviation
std_dev_age = np.std(titanic_df['age'])

print(f"Standard deviation of age: {std_dev_age}") # Standard deviation of age: 14.516321150817316
```
Running the provided Python code will calculate and print the standard deviation of the age field in the Titanic dataset, thereby giving you a sense of how much the ages of passengers varied.

## Delving Deeper into Data: Quartiles and Percentiles
Let's dig deeper and start looking at the division of data into segments with quartiles and percentiles. Quartiles and percentiles are in essence, a way to cut our data into equal segments. The 25th percentile, for example, is equivalent to the first quartile, and the 75th percentile is the third quartile.

```Python

# Quartiles and percentiles
# Using Numpy
Q1_age_np = np.percentile(titanic_df['age'].dropna(), 25) # dropna is being used to drop NA values
Q3_age_np = np.percentile(titanic_df['age'].dropna(), 75)

print(f"First quartile of age (Numpy): {Q1_age_np}")
print(f"Third quartile of age (Numpy): {Q3_age_np}")

# Output:
# First quartile of age (Numpy): 20.125
# Third quartile of age (Numpy): 38.0

# Using Pandas
Q1_age_pd = titanic_df['age'].quantile(0.25)
Q3_age_pd = titanic_df['age'].quantile(0.75)

print(f"First quartile of age (Pandas): {Q1_age_pd}")
print(f"Third quartile of age (Pandas): {Q3_age_pd}")

# Output:
# First quartile of age (Pandas): 20.125
# Third quartile of age (Pandas): 38.0
```
The executed Python code first calculates and prints the first and third quartiles for the age column of our Titanic dataset using NumPy. It then repeats the calculation using Pandas, giving the same results. With these quartiles, we can immediately understand more about the age distribution of passengers on board the Titanic. For instance, we now know that 50% of passengers were between the ages of `Q1_age_np`(around 20 years old) and `Q3_age_np` (approximately 38 years old).
