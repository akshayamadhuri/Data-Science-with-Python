# Unleashing Aesthetics in Visualization: An Introduction to Seaborn's Styling Capabilities

## Introduction to Seaborn: Aesthetics and Styling

we will dip toes into the Seaborn library, focusing on aesthetics and styling in our plots. we used Matplotlib to create some simple bar plots. we'll see how Seaborn can help us create visually appealing plots effortlessly.

Defining an aesthetic style before creating your plot is an important aspect of any data visualization. The right choice of colors, sizes, and other aesthetic factors can make your plots more engaging, easy to interpret, and effective at conveying your intended insights.

We'll be looking at three essential elements of styling in `Seaborn`: figure style, color palette, and plot size. Think of it this way: you're about to paint a masterpiece, and `Seaborn` provides us with a studio full of tools. Are you ready to create beautiful plots with the Titanic dataset?

## Seaborn: A Quick Introduction

Seaborn is a data visualization library built on top of the Matplotlib library in Python. It offers a high-level and easier-to-use interface, as well as attractive and informative statistical graphics.

Let's start by importing Seaborn and setting the aesthetics for all the plots to whitegrid.

```Python

import seaborn as sns

# Set the seaborn default aesthetic parameters
sns.set(style="whitegrid")
```

The style='whitegrid' parameter in the sns.set() creates a white background with gridlines. Five other preset seaborn themes are available: darkgrid, whitegrid, dark, white, and ticks.

## What else?

You can also set many more aesthetic parameters for your future plots using the `sns.set()` function. Some of the optional parameters that might customize your 

plots include:

`palette`: Set this to any of the Seaborn color palettes or a custom color palette.
`font`: Sets the font for all text in the plot.
`font_scale`: Can be used to scale the size of the font elements.
`color_codes`: If set to True, shorthand notation can be used for colors in the palette (like 'b' for blue).

For example, to set the palette to Blues, the font to Serif and scale it up 1.2 times, you can use:

```Python

sns.set(style="whitegrid", palette="Blues", font="Serif", font_scale=1.2)
```

## Using Seaborn Visualizations

Let's create a barplot for the number of passengers per passenger class using Seaborn.

```Python

import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
titanic_df = sns.load_dataset('titanic')

# Set plot styling
sns.set(style="whitegrid", palette="Blues", font="Serif", font_scale=1.2)

# Create a plot
sns.countplot(x='pclass', data=titanic_df)
plt.show()
```
In the code above, we use sns.countplot() to create a bar plot in Seaborn. It's a simple method that counts the number of occurrences and plots it.

![image](https://github.com/user-attachments/assets/721c5f1d-3d06-4d09-9720-ef5b7ae9344e)


## Customizing Palette with Seaborn

Color plays a major role in creating an attractive and readable chart. Seaborn provides us with a wide range of color palettes that can be easily deployed to any plot. Let's update our previous bar plot to use the coolwarm palette.

```Python

sns.countplot(x='pclass', data=titanic_df, palette='coolwarm')
plt.show()
```
Here, palette='coolwarm' updates the color scheme of the plot. There is a variety of color palettes available in Seaborn. Some other options include Blues, husl, and pastel.

![image](https://github.com/user-attachments/assets/32a108ca-8175-46ff-89e9-ebb478b67fce)


## Adjusting Plots Sizes

Controlling the plot's size can be crucial in certain situations where we must present the results at various scales or dimensions. With Seaborn, adjusting the size is pretty straightforward. Let's resize our previous bar plot to be wider and shorter to see the difference.

```Python

plt.figure(figsize=(12, 6))
sns.countplot(x='pclass', data=titanic_df, palette='coolwarm')
plt.show()
```
![image](https://github.com/user-attachments/assets/1836d13d-dc39-4823-949e-271a72655e46)


In the code above, `plt.figure(figsize=(12, 6))` creates a new figure with the specified width (12) and height (6).

## Advanced Plot Customizations with Seaborn

Seaborn offers many more options to customize your plots. Below, we explore some of them:

## Adding a legend

Legends can provide useful information about the data being represented in your plot. You can easily add a legend to your Seaborn plot by using plt.legend().

## Adding titles and labels

Any plot is incomplete without a meaningful title and well-labeled axes. We can add a title to our plot with `plt.title()`. Likewise, we can add labels to the x-axis and y-axis using `plt.xlabel()` and `plt.ylabel()`, respectively.

Here is how you would implement all these options:

```Python

plt.figure(figsize=(12, 6))
sns.countplot(x='pclass', data=titanic_df, palette='coolwarm')
plt.title('Passenger Class Count')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Passenger Class')
plt.show()
```
![image](https://github.com/user-attachments/assets/4b069b61-fd56-4b21-b00a-60f2614eab97)


## Rotating labels

In cases where the labels on your x-axis or y-axis are long, they may overlap, making them difficult to read. One solution to this issue is to rotate the axis labels. This is achievable with `plt.xticks(rotation=angle)` or `plt.yticks(rotation=angle)` where angle is the degree of rotation.

Here, we will rotate our x-axis labels by 45 degrees:

```Python

plt.figure(figsize=(12, 6))
sns.countplot(x='pclass', data=titanic_df, palette='coolwarm')
plt.title('Passenger Class Count')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Passenger Class')
plt.xticks(rotation=45)
plt.show()
```
![image](https://github.com/user-attachments/assets/07454278-1390-4bd3-8ddc-50e94caa205c)


