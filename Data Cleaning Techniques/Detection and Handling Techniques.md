# Navigating through Outliers: Detection and Handling Techniques

## Setting the Stage: Outlier Detection

how to detect and handle them effectively using Python. As always, we'll use our Titanic dataset to illustrate these concepts.

Why are outliers significant, you might wonder? Outliers are anomalous or unusual values that significantly deviate from other observations. They can adversely impact the performance of our machine-learning models by introducing bias or skewness. Detecting outliers helps us maintain our dataset's integrity by ensuring all data falls within a reasonable range of values.

Going back to our Titanic example. What if some passengers had absurdly high ages, like 800, or an impossible fare of $50,000? We can't just ignore these anomalies. We must deal with them appropriately, ensuring our models learn from accurate, realistic data.

## The Z-score Method
A commonly used method to detect outliers in a dataset is the Z-score method. Given a set of values, the Z-score of a value is the distance between that value and the dataset's mean, expressed in terms of the standard deviation.

A Z-score of 0 indicates that the data point is identical to the mean score. A Z-score of 1.0 indicates a value that is one standard deviation from the mean. Higher Z-scores denote farther (and potentially outlier) values.

Let's use this method to detect any potential outliers in the age feature of our Titanic dataset. We'll only consider positive Z-scores, as negative ages are illogical in our context.

```Python

import numpy as np
import pandas as pd
import seaborn as sns

# Load the dataset
titanic_df = sns.load_dataset('titanic')

# Calculate Z-scores
titanic_df['age_zscore'] = np.abs((titanic_df.age - titanic_df.age.mean()) / titanic_df.age.std())

# Get rows of outliers according to the Z-score method (using a threshold of 3)
outliers_zscore = titanic_df[(titanic_df['age_zscore'] > 3)]
print(outliers_zscore)
"""
     survived  pclass   sex   age  ...  embark_town  alive  alone age_zscore
630         1       1  male  80.0  ...  Southampton    yes   True   3.462699
851         0       3  male  74.0  ...  Southampton     no   True   3.049660

[2 rows x 16 columns]
"""
```

In the code snippet above, the Z-score calculates the distance between each age value and the mean age (titanic_df.age.mean()), in terms of standard deviation (titanic_df.age.std()). We add the results as a new column, age_zscore, into our dataframe. High values (above 3 in our case) are presumed to be potential outliers.

### The IQR Method

Another method to detect outliers is the Interquartile Range (IQR) method. IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile). An outlier is any value that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

Let's detect outliers in the age column of the Titanic dataset using this method:

```Python

# Calculate IQR
Q1 = titanic_df['age'].quantile(0.25)
Q3 = titanic_df['age'].quantile(0.75)
IQR = Q3 - Q1

# Define Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Get rows of outliers according to IQR method
outliers_iqr = titanic_df[(titanic_df['age'] < lower_bound) | (titanic_df['age'] > upper_bound)]
print(outliers_iqr)
"""
     survived  pclass   sex   age  ...  embark_town  alive  alone age_zscore
33          0       2  male  66.0  ...  Southampton     no   True   2.498943
54          0       1  male  65.0  ...    Cherbourg     no  False   2.430103
96          0       1  male  71.0  ...    Cherbourg     no   True   2.843141
116         0       3  male  70.5  ...   Queenstown     no   True   2.808721
280         0       3  male  65.0  ...   Queenstown     no   True   2.430103
456         0       1  male  65.0  ...  Southampton     no   True   2.430103
493         0       1  male  71.0  ...    Cherbourg     no   True   2.843141
630         1       1  male  80.0  ...  Southampton    yes   True   3.462699
672         0       2  male  70.0  ...  Southampton     no   True   2.774301
745         0       1  male  70.0  ...  Southampton     no  False   2.774301
851         0       3  male  74.0  ...  Southampton     no   True   3.049660

[11 rows x 16 columns]
"""
```
Here, we first calculate Q1 and Q3, representing the 25th and 75th percentile of the age field, respectively. The IQR is simply the difference between Q3 and Q1. Outliers are defined as any age below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

## Decision Time: To Keep or Not?

After identifying outliers, you'll have to decide what to do with them—whether to keep them, discard them, or modify them. Regardless of how you identify outliers, applying the most suitable handling technique is crucial.

In data cleaning, there's no one-size-fits-all rule when it comes to dealing with outliers—your decision should depend on the dataset and the specific problem you're working on. Sometimes, removing outliers can improve your model's accuracy. Other times, outliers might be crucial, and removing them could lead to inaccurate models or conclusions.

You might deal with outliers by:

Dropping them:

```Python

# Using the Z-score method
titanic_df = titanic_df[titanic_df['age_zscore'] <= 3]

# Using the IQR method
titanic_df = titanic_df[(titanic_df['age'] >= lower_bound) & (titanic_df['age'] <= upper_bound)]
```
Here, we exclude rows where the age lies in the outlier zone according to the chosen outlier detection method.

Replacing them with another value (mean, median, mode, etc.):

```Python

# using mean
titanic_df.loc[titanic_df['age_zscore'] > 3, 'age'] = titanic_df['age'].mean()

# using median
titanic_df.loc[(titanic_df['age'] < lower_bound) | (titanic_df['age'] > upper_bound), 'age'] = titanic_df['age'].median()
```

In these examples, outliers are replaced by the mean or median value of the age column. The specific age value to use for replacement would depend on the particularities of your dataset.

