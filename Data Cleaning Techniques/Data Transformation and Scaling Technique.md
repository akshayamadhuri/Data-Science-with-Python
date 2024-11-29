# Diving into Data Transformation and Scaling Techniques

## Topic Overview

Data Transformation and Scaling Techniques, an essential constituent of the data cleaning and preprocessing process for machine learning. We will learn how to transform numerical data to different ranges using various scaling techniques, such as `Standard Scaling`, `Min-Max Scaling`, and `Robust Scaling`  .

Data scaling is crucial because machine learning algorithms perform more effectively when numerical features are on the same scale. Without scaling, variables with higher ranges may dominate others in the machine learning models, reducing the model's accuracy.

For example, imagine having two features — age and income — in your Titanic dataset. Age varies between 0 and 100, while income may range from 0 to thousands. A machine learning model could be biased towards income because of its higher magnitude, leading to poor model performance.


## Introduction to Data Scaling

let's briefly discuss three popular techniques to standardize numerical data.

Standard Scaler: It assumes data is normally distributed and scales it to have zero mean and unit variance. It's best used when the data is normally distributed. In other words, when the values of a particular feature follow a bell curve, a Standard Scaler is a good option to standardize the feature.

Min-Max Scaler: Also known as normalization, this technique scales data to range between 0 and 1 (or -1 to 1 if there are negative values). It's commonly used for algorithms that don't assume any distribution of the data. This means if your data doesn't follow a specific shape or form, you might consider using Min-Max Scaler.

Robust Scaler: As its name suggests, this scaler is robust to outliers. It uses the Interquartile Range (IQR) to scale data, and it's suitable when the dataset contains outliers. Outliers are data points that significantly deviate from other observations. They can be problematic because they can affect the results of a data analysis.

There's no "one size fits all" scaler. You'll need to choose the appropriate scaler based on your data's characteristics and your machine-learning algorithm's requirements.

Standard Scaling
We'll start with the Standard Scaler. It scales data based on its mean (μ) and standard deviation (σ), using the formula to calculate the z-score: 
\[ z = \frac{x - \mu}{\sigma} \].

Let's try it on the `age` column of the Titanic dataset:

```Python

import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset and drop rows with missing values
titanic_df = sns.load_dataset('titanic').dropna()

# Initialize the StandardScaler
std_scaler = StandardScaler()

# Fit and transform the 'age' column
titanic_df['age'] = std_scaler.fit_transform(np.array(titanic_df['age']).reshape(-1, 1))

# Check the transformed 'age' column
print(titanic_df['age'].head())
"""
1     0.152082
3    -0.039875
6     1.175852
10   -2.023430
11    1.431795
Name: age, dtype: float64
"""
```
Note how the transformed age values are not easily interpretable. That's because they've been transformed into their respective z-scores. But the important thing to understand is the transformed data is standardized and can be readily included in a machine learning model.

![image](https://github.com/user-attachments/assets/47ced17e-d812-4c42-b653-19f8eea7d438)


Min-Max Scaling
Next, we'll explore Min-Max Scaling, which scales your data to a specified range. The formula used here is: 
xnew=x−xminxmax−xminx new= x max −x min
​
 
x−x 
min
​
 
​
 . This formula essentially resizes your data to fit within the range of 0 to 1.

Let's apply Min-Max Scaler on the fare column:

```Python

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
min_max_scaler = MinMaxScaler()

# Fit and transform the 'fare' column
titanic_df['fare'] = min_max_scaler.fit_transform(np.array(titanic_df['fare']).reshape(-1, 1))

# Check the transformed 'fare' column
print(titanic_df['fare'].head())
"""
1     0.139136
3     0.103644
6     0.101229
10    0.032596
11    0.051822
Name: fare, dtype: float64
"""
```
All fare values are now within the range of 0 to 1, with the smallest fare being 0 and the largest being 1. Intermediate fare values are spread out proportionally between 0 and 1.

## Robust Scaling

Last but not least, we have Robust Scaling useful when dealing with outliers, as it scales data according to its IQR (Inter Quartile Range). Effectively, it's robust against outliers since it uses the IQR, and outliers fall outside the IQR.

Let's apply it to the fare column:

```Python

from sklearn.preprocessing import RobustScaler

# Initialize the RobustScaler
robust_scaler = RobustScaler()

# Fit and transform the 'fare' column
titanic_df['fare'] = robust_scaler.fit_transform(np.array(titanic_df['fare']).reshape(-1, 1))

# Check the transformed 'fare' column
print(titanic_df['fare'].head())
"""
1     0.236871
3    -0.064677
6    -0.085199
10   -0.668325
11   -0.504975
Name: fare, dtype: float64
"""
```
The fare values now reflect how many IQRs are away from the median. This scaling method is resilient to outliers, which effectively become small positive and negative values.
