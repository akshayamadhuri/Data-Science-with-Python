Exploring Multivariate Relationships: Scatter Plots and Correlations in Data Visualization
Entering the Multivariate Analysis Arena: Scatter Plots and Correlation of Variables
Welcome to the next guide on our remarkable voyage! Moving into our discussion on multivariate data visualization, we'll introduce you to scatter plots, one of the most powerful tools for visualizing the relationship between multiple variables. We'll guide you through plotting scatter plots for different variable pairs in our Titanic dataset and stepping further into correlating these variables.

Why is it important to understand the correlation among variables? Imagine, we want to know whether passengers in the higher classes were more likely to survive. Or maybe we are interested in the fare paid for a ticket correlates with the survival on Titanic. Finding correlations among variables will help us generate hypotheses, create insightful visualizations, and eventually enable efficient predictive modeling.

By the end of this lesson, you'll be conversant with how scatter plots and correlation techniques can be used to explore and visualize relationships between different features present in a multivariate dataset.

Scatter `Plots`: An Introduction
A scatter plot is a versatile visualization tool that can disclose the relationship, if any exists, between two variables. Each point on the plot represents an observation in the dataset, with its position along the X and Y axes representing the values of two variables.

Let's initiate with a scatter plot depicting the relationship between age and fare.

Python
Copy to clipboard
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Display Scatter Plot of Age vs Fare
sns.scatterplot(x='age', y='fare', data=titanic)
plt.title("Age vs Fare")
plt.show()
image

In scatterplot() function:

x is for the data along the horizontal axis
y is for the data along the vertical axis
data: it's a required parameter, providing the data source.
Further into Scatter `Plots`
Looking at the scatter plot, there seems to be no apparent correlation between age and fare. But what if we consider another variable - class in our analysis? We might hypothesize that higher class passengers (1st or 2nd) could have paid more fare regardless of age.

Using the hue parameter, we can visualize this by adding color discrimination to our scatter plot. Setting hue='pclass' will provide different colors to data points belonging to different passenger classes:

Python
Copy to clipboard
sns.scatterplot(x='age', y='fare', hue='pclass', data=titanic)
plt.title("Age vs Fare (Separate colors for Passenger Class)")
plt.show()
image

hue: you can think of it as a fourth dimension of data, it can determine the color of data points using an additional variable.

Adding `Markers` and `Sizes`
To add further dimensions to your scatter plot, you can opt for different marker styles for different categories and sizes to represent another numerical variable. Let's try adding styles based on sex and sizes based on fare.

Python
Copy to clipboard
sns.scatterplot(x='age', y='fare', hue='pclass', style='sex', size='fare', sizes=(20, 200), data=titanic)
plt.title("Age vs Fare (Separate markers for Sex and Sizes for Fare)")
plt.show()
Here is what we'll see:

image

Here, style has been used to depict different markers for male and female, and size has been used to give varying point sizes based on the fare. sizes=(20, 200) sets the range of sizes to scale the scatter plot points. By adding both style and size aspects, we achieve a four-variable scatter plot in a two-dimensional space.

style: This attribute will make different marks on the plot for different categories.
size: This attribute can determine the size of a plotting mark using an additional variable. This represents another layer of information, providing you with a 3-dimensional plot.
Correlation of Variables
While scatter plots may visually hint at correlations to quantify the extent of the correlation, we need to move towards correlation coefficients. A correlation coefficient is a numerical measure of the statistical relationship between two variables. The correlation coefficient ranges from -1 to 1 where:

the value of +1 represents an exact positive linear relationship between variables,
the value of -1 represents a perfect negative linear relationship between variables,
the value of 0 suggests no linear relationship between variables.
Let's determine the correlation between all variables in the Titanic dataset. For the same, we'll use the corr() function of pandas:

Python
Copy to clipboard
# Correlation of all numeric variables in the Titanic dataset
corr_vals = titanic.corr(numeric_only=True)
print(corr_vals)
This code outputs:

Markdown
Copy to clipboard
          survived    pclass       age     sibsp     parch      fare
survived  1.000000 -0.338481 -0.077221 -0.035322  0.081629  0.257307
pclass   -0.338481  1.000000 -0.369226  0.083081  0.018443 -0.549500
age      -0.077221 -0.369226  1.000000 -0.308247 -0.189119  0.096067
sibsp    -0.035322  0.083081 -0.308247  1.000000  0.414838  0.159651
parch     0.081629  0.018443 -0.189119  0.414838  1.000000  0.216225
fare      0.257307 -0.549500  0.096067  0.159651  0.216225  1.000000
This code provides the correlation coefficients among all pairs of numerical variables in the dataset.

The corr() function of pandas calculates the pairwise correlation of columns, excluding NA/null values. It operates on Series as well as DataFrame objects. We use numeric_only=True to show correlation only for numeric columns (of int, float, and bool type).
