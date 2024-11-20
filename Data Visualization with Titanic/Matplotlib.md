# Extending Data Visualization: Enhancing Plots and Analyzing with Matplotlib


## Setting Foot on Matplotlib: Basics of Plotting Categorical Data

we're stepping into the world of data visualization by introducing Matplotlib's visualization tools. We'll be learning the basics of plotting categorical data from our dataset and understanding the insight such visualization can provide.

Data visualization is an essential tool in data analysis—you can communicate complex data structures and uncover relationships, trends, and patterns in the data. It plays a pivotal role in exploratory data analysis, a fundamental skill for all data scientists.

Taking the passengers aboard Titanic as an example, each passenger belonged to a specific gender and a unique passenger class. Can we observe any underlying pattern that might be of interest? Are survival rates higher for a certain gender or passenger class? Or does the embarkation point play a role? We'll address these questions as we traverse the path of data visualization.

## Introduction to Matplotlib

Matplotlib is an extensive library for creating static, animated, and interactive visualizations in Python. To make it versatile across multiple platforms, it offers a MATLAB-like interface.

Let's start by importing the pyplot module of the Matplotlib library:

```Python

import matplotlib.pyplot as plt
```
pyplot provides a high-level interface for creating attractive graphs. To demonstrate this, we'll first analyze the sex column of the Titanic dataset.

We retrieve the counts of each category — male and female — with `value_counts()`, and plotting them is as simple as calling plot() with the argument 'bar':

```Python

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic_df = sns.load_dataset('titanic')

# Count total males and females
gender_data = titanic_df['sex'].value_counts()

# Create a bar chart
gender_data.plot(kind ='bar', title='Sex Distribution')
plt.show()
```
![image](https://github.com/user-attachments/assets/3c1ae4d1-5e74-409f-91c4-a6572844c2da)


## Enhancing Plots: Labels and Title

It's good practice to include a title and labels for the axes to make your plot more understandable. You can achieve this using `xlabel()`, `ylabel()`, and `title()` functions. Let's enhance our plot:

```Python

gender_data = titanic_df['sex'].value_counts()

gender_data.plot(kind ='bar')
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Sex Distribution")
plt.show()
```
![image](https://github.com/user-attachments/assets/17548e5d-3ac2-4797-833e-64644296d441)


In this code, `plt.xlabel("Sex")` adds 'Sex' as the label for the x-axis, `plt.ylabel("Count")` adds 'Count' as the label for the y-axis, and `plt.title("Sex Distribution")` sets 'Sex Distribution' as the title for the plot.

## A Look at Other Categories

Just as we did with the sex column, we can also analyze the pclass (passenger class) and embarked (embarkation point) columns:

```Python

# Passenger class distribution
class_data = titanic_df['pclass'].value_counts()
class_data.plot(kind='bar')
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.title("Passenger Class Distribution")
plt.show()
```
![image](https://github.com/user-attachments/assets/0c894c7e-d0f9-4886-8d15-177c4aa84d96)


```Python

# Embarkation point distribution
embark_data = titanic_df['embarked'].value_counts()
embark_data.plot(kind='bar')
plt.xlabel("Embarkation Point")
plt.ylabel("Count")
plt.title("Embarkation Point Distribution")
plt.show()
```
![image](https://github.com/user-attachments/assets/d52b43de-e023-46a8-8c8d-558aaebb0158)


These plots visualize the count of passengers based on their passenger class and embarked points, giving us some insights about the dataset.

## Customizing Your Plot

Not only does the `plot()` method enable us to generate various types of charts, but it also allows us to adjust many parameters for better visualization.

`color`: Sets the color of the plot.
`alpha`: Sets the transparency level.
`grid`: Whether or not to display grid lines.

Let's experiment with these parameters:

```Python

gender_data.plot(kind='bar', color='skyblue', alpha=0.7, grid=True)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Sex Distribution")
plt.show()
```
![image](https://github.com/user-attachments/assets/aa2eb402-4f2f-4af4-8cec-a2e38aebb40e)
