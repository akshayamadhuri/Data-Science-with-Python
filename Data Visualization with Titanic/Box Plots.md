# Analyzing Relationships between Passenger Class, Fare, and Survival with Box Plots

## Diving into Box Plots: Passenger Class, Fare, and Survival

Box plots are unique in providing a snapshot of a dataset's distribution and outlier detection, all in one plot!
Box plots are crucial in understanding the Titanic dataset, particularly in discovering relationships between survival rates, passenger classes, and fares. This can answer our central question: How did the passenger class and fare correlate with survival?

## Introducing Box Plots

A box plot, also known as a whisker plot, is a standardized way of displaying the data distribution based on a five-number summary: the minimum, the maximum, the sample median, and the first and third quartiles. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be “outliers” using a method that is a function of the interquartile range.

We can create a box plot using the `boxplot()` function in the Python Seaborn library. First, let's start with pclass (passenger class) against fare:

```Python

import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
titanic_df = sns.load_dataset('titanic')

# Create a box plot
sns.boxplot(x='pclass', y='fare', data=titanic_df)
plt.title('Fares vs Passenger Classes')
plt.show()
```
![image](https://github.com/user-attachments/assets/40e11ac6-0a1f-4c18-a50b-791334cede36)


In the box plot:

The box represents the interquartile range (i.e., 25th to 75th percentile) of the fares in each passenger class.

The line in the middle of the box is the median fare price in that class.

The whiskers (lines extending from the box) represent the fare range within 1.5 times the interquartile range above the upper and lower quartile.

Any points beyond the whiskers can be considered outliers in the fare distribution within each class.

## Adding Another Dimension to Box Plots with `hue`

A great feature of box plots in Seaborn is that it allows you to add a hue parameter to add a third dimension of categorical data. For instance, we can differentiate the passengers who survived from those who didn't on the same pclass vs fare plot:

```Python

sns.boxplot(x='pclass', y='fare', hue='survived', data=titanic_df)
plt.title('Fares vs Passenger Classes Differentiated by Survival')
plt.show()
```
![image](https://github.com/user-attachments/assets/861d4e68-2c41-4293-ae29-2ad80f466b41)


This plot visually compares fares among different passenger classes regarding their survival status, enhancing our grasp of the data.

## Tweaking Your Box Plot

There are many ways you can modify your box plot to better suit your needs, such as:

`orient`: if set to "h", it changes the box plot orientation from vertical to horizontal. Alternatively, you can swap x and y values in the bot plot configuration.
`width`: adjusts the width of the boxes.
`palette`: this modifies the color palette.
`linewidth`: adjusts the width of the line.

Let's try them:

```Python

sns.boxplot(
    x='survived', y='fare',
    hue='pclass', data=titanic_df,
    palette='Set3', linewidth=1.5
)
plt.title('Survival and Passenger Classes by Fare')
plt.show()
```
![image](https://github.com/user-attachments/assets/f143949c-51c4-4b16-8dce-044ca6566089)


## Going Deeper: More on Box Plots

In addition to the above, Seaborn's `boxplot()` function has more parameters that you can use to enhance your box plots and cater them to your needs. Let's dive a bit deeper!

`order`: You can change the order of display of categorical levels by passing the desired order.
`hue_order`: Similar to the order parameter, the hue_order changes the order of display of your hue variable levels.
`color`: If you want all boxes the same color.
`saturation`: Saturation makes patches drawn by the function look darker (if less than 1) or brighter (if greater than 1).
`dodge`: When hue nesting is used, whether elements should be shifted along the categorical axis.
`fliersize`: Size of the markers used to indicate outlier observations.

Here is how these parameters can be used in a sample box plot:

```Python
sns.boxplot(
    x='pclass', y='fare',
    hue='survived',
    data=titanic_df,
    palette='Set3', linewidth=1.5,
    order=[3,1,2], hue_order=[1,0],
    color='skyblue', saturation=0.7,
    dodge=True, fliersize=5
)
plt.title('Fares vs Passenger Classes Differentiated by Survival')
plt.show()
```
![image](https://github.com/user-attachments/assets/baa6587f-369f-4f89-955f-9fc7af5f712b)


