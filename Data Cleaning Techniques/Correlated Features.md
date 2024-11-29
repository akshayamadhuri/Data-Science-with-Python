# Understanding and Handling Redundant or Correlated Features in Datasets

## Gearing Up

Will focus on tackling redundant or correlated features in a dataset. These features provide similar, overlapping information that can potentially affect the performance of machine learning models. Learning to handle such features is critical to data cleaning and preprocessing.

Why is handling redundant or correlated features necessary, you ask? Here's the reason: Machine learning models are grounded in mathematics, and we need to ensure that the input data doesn't contain multicollinearity, meaning predictors are not independent, as it may cause issues with mathematical calculations. By identifying and eliminating redundant or correlated features, we can ensure that each feature in our dataset offers unique and valuable information that improves the predictive model's performance.

## Correlation: A Quick Introduction

In statistics, correlation is a term that indicates the degree to which two variables move in relation to each other. If two features are highly correlated, they carry similar information.

In the context of our Titanic dataset, let's consider the pclass (passenger class) and fare (ticket cost) columns. Intuitively, passengers belonging to higher class (1st) would have paid a higher fare. Therefore, these two columns are likely to be strongly correlated.

To quantify this relationship, we use the correlation coefficient, a value between -1 and 1. If the correlation coefficient is close to 1, it indicates a strong positive correlation. Conversely, a coefficient near -1 indicates a strong negative correlation. A coefficient close to zero suggests no correlation.

To calculate the correlation between features in our dataset, we use the corr() function from the Pandas library. Let's see how it's done:

```Python

import seaborn as sns

# Load the dataset
titanic_df = sns.load_dataset('titanic')

# Calculate and print the correlation matrix
corr_matrix = titanic_df.corr(numeric_only=True)
print(corr_matrix)
"""
            survived    pclass       age  ...      fare  adult_male     alone
survived    1.000000 -0.338481 -0.077221  ...  0.257307   -0.557080 -0.203367
pclass     -0.338481  1.000000 -0.369226  ... -0.549500    0.094035  0.135207
age        -0.077221 -0.369226  1.000000  ...  0.096067    0.280328  0.198270
sibsp      -0.035322  0.083081 -0.308247  ...  0.159651   -0.253586 -0.584471
parch       0.081629  0.018443 -0.189119  ...  0.216225   -0.349943 -0.583398
fare        0.257307 -0.549500  0.096067  ...  1.000000   -0.182024 -0.271832
adult_male -0.557080  0.094035  0.280328  ... -0.182024    1.000000  0.404744
alone      -0.203367  0.135207  0.198270  ... -0.271832    0.404744  1.000000

[8 rows x 8 columns]
"""
```
Here, titanic_df.corr(numeric_only=True) returns a DataFrame with the correlation coefficients between all pairs of numeric columns in titanic_df.

This correlation matrix displays the relationship between each pair of numerical columns. For instance, the correlation between fare and pclass is -0.549500. This negative sign indicates a negative correlation, meaning that passenger class decreases as the fare increases, which is consistent with our initial assumption.

## Visualization: Using a Heatmap

A correlation matrix can be difficult to read and understand, especially when we have many features. To ease this process, we can visualize the matrix using a heatmap with the help of the seaborn library.

The heatmap() function in Seaborn provides a graphical representation of the correlation matrix where colors represent values:

```Python

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic_df = sns.load_dataset('titanic')

# Calculate the correlation matrix
corr_matrix = titanic_df.corr(numeric_only=True)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')

# Show the plot
plt.show()
```
![image](https://github.com/user-attachments/assets/b1eb1d12-54fb-42ab-a537-e81b676474ac)


In the heatmap, a dark color represents a high negative correlation, and a light color represents a high positive correlation.

For example, fare and pclass are displayed in a dark color, meaning they have a high negative correlation close to -0.55. This observation from the heatmap aligns with our initial assumption: as the passenger class decreases (3rd class to 1st class), the ticket fare increases.

## Handling Redundant or Correlated Features

When two features are highly correlated, they carry similar information; hence, one can be removed without losing important information.

Here's how to drop a column in a Pandas DataFrame:

```Python

# If 'fare' and 'pclass' are highly correlated
clean_df = titanic_df.drop('fare', axis=1)
```
In this case, we are dropping the fare column. The axis=1 parameter indicates that we want to drop a column (for dropping a row, we would have used axis=0). The resulting clean_df DataFrame contains all the original columns except fare.

We removed the fare column because it's highly correlated with pclass, and it's not uncommon for ticket prices to depend on the passenger class.
