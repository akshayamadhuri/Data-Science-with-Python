## Visualizing Distributions with Histograms Using Seaborn

Histograms are powerful graphical representations that allow us to inspect the underlying frequency distribution (shape) of a continuous or discrete data set. This is particularly useful when we want to visualize the distribution of a variable over a range of values.

Why is understanding the data distribution important, you might ask? In the field of data analytics and statistics, most statistical tests and models assume certain data distribution patterns. Histograms, therefore, are ways for us to validate these assumptions. In other words, knowing our data well sets the stage for more complex analyses later on.

This will take you further into Seaborn's capabilities. We'll cover how to create and customize histograms, offering a sharper lens to inspect our Titanic dataset.

## Diving into Histograms

Let's illustrate a histogram using the passenger ages (age) from `titanic_df`. As we saw in our previous lessons, there were a variety of ages amongst the passengers that should make for an interesting distribution.

Seaborn provides a function called histplot for creating histograms. Here's the basic syntax:

```Python

import seaborn as sns

titanic_df = sns.load_dataset('titanic')

sns.histplot(data=titanic_df, x='age', kde=True)
```
In the code snippet above, we are telling Seaborn to look at the age column from our `titanic_df` DataFrame, and the `kde=True` part is there also to draw a curve of Kernel Density Estimation (KDE) that estimates the probability density function of the variable age (more on this shortly).

This delivers a histogram that shows the distribution of passenger ages. The x-axis represents the ages, and the y-axis represents the number of passengers with ages within the corresponding bin of the histogram.

![image](https://github.com/user-attachments/assets/24ac5037-f376-48af-94e9-a71b65f73604)


## Understanding Kernel Density Estimation (KDE)

You may wonder what the smooth, continuous line overlaying our histogram represents. This smooth line, created by turning on the kde parameter in histplot, is a Kernel Density Estimate (KDE) plot that provides a smooth estimate of our distribution.

The KDE is useful when we want to derive a smooth, continuous function from our discrete observations. Often, this can make the output much more interpretable, aiding in the presentation of our data. However, remember that KDEs are just estimates. The true distribution of your data may be different, especially if you have a small number of observations.

Here is one more example of using KDE in action!

```Python

sns.histplot(data=titanic_df, x='fare', kde=True, color='green')
```
![image](https://github.com/user-attachments/assets/5b0fed87-523a-4374-9c2b-7a2a48a8ca6b)


As you can see, the KDE gives us a smooth curve that fits our observations, providing a pleasing and easy-to-understand representation of our distribution.


## Customizing Your Histogram

Like all plots in Seaborn, histograms are highly customizable. Let's look at improving the readability of our histogram by adding more bins and labeling our axes.

```Python

# Increase the number of bins to 30 (default is 10)
sns.histplot(data=titanic_df, x='age', bins=30, kde=True)

# Give your plot a comprehensive title
plt.title('Age Distribution Among Titanic Passengers')

# Label your axes
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()
```
![image](https://github.com/user-attachments/assets/cce9c6d1-6c00-476d-835f-c750b005d505)


histogram is instantly more legible, offering an improved perception of the underlying distribution. The increased number of bins provides a more defined structure for our data, while the informative title and labels allow us to understand the plot without any additional context.

## Further Customization of Histograms Using Seaborn

Seaborn's histplot function enables the drawing of histograms with rich features. Here are some additional useful parameters that you should know about:

`hue`: This is a very important parameter when you have a categorical column that you want to represent on the histogram. The hue parameter instructs seaborn to color the histogram bars for the age distribution differently depending on the passenger's gender values (i.e., male or female).
`multiple`: This parameter is used with hue to change how the different categories are displayed on the histogram. The default is "layer", but you could set it to `"stack"`, "fill" or "dodge".
`"layer"`: Draw one histogram per variable. Each histogram will represent a separate layer; layers will be superimposed on each other.
`"stack"`: Draw one histogram, stacking the values of each variable on top of the other.
`"fill"`: Draw one histogram, with the area of each filled up to the total height, the cumulating contribution of each variable (like a percentage plot, where the whole plot is 100%).
`"dodge"`: Draw one histogram, but "dodge" them, i.e., move them slightly to the side so each contributes to the overall figure separately and all can be seen.
`palette`: This parameter allows you to change the colors used for the different categories.
`binwidth`: This parameter allows you to set the width of the bins rather than the number of bins. This can be useful for a more direct control of the granularity of the histogram.
`element`: By default, the histogram is made of bars, but you could set this parameter to "step" or "poly" to change the appearance of the histogram.
Let's look at examples demonstrating some of these parameters:

```Python

# A histogram using 'hue', 'multiple', and 'palette'
sns.histplot(data=titanic_df, x='age', hue="sex", multiple="stack", palette="pastel")
```
![image](https://github.com/user-attachments/assets/c0cc64af-56ca-4063-9b20-2c2bbb7a3bf6)


```Python

# A histogram using 'binwidth' and 'element'
sns.histplot(data=titanic_df, x="age", binwidth=1, element="step", color="purple")
```
![image](https://github.com/user-attachments/assets/9ff6184a-13e9-4884-b2b8-5a8526ac18cc)


