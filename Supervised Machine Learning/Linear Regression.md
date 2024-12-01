# Mastering Linear Regression: From Theories to Predictions

## Probing Linear Regression

Linear Regression is fundamental to supervised learning. It becomes particularly useful when the target or outcome variable is continuous. Let's illustrate this concept with a simple real-world example: suppose you want to predict the price of a house (the output or dependent variable) based on its size (the input or independent variable). In this case, you would use Linear Regression because both your output and input are continuous.

Along the same lines, we will predict the quality of the wine (a numerical score from 0 to 10, which is continuous) based on several physicochemical properties, such as fixed acidity, volatile acidity, and citric acid, using our dataset.

Linear Regression algorithm optimizes a straight line to encapsulate the relationship accurately between the input and output variables. This line is modeled using a simple equation, y=mx+c, where y is the dependent variable, m is the slope, x is the independent variable, and c is the y-intercept.

## Mathematics behind Linear Regression

At the heart of Linear Regression lies the concept of the cost function and hypothesis, which we'll break down below:

`Hypothesis`: This results in the regression line that can predict the output based on the inputs. If we're trying to predict wine quality based on certain properties, this hypothesis would best fit the linear relationship between our selected properties and the wine's quality. The hypothesis is represented as 
hθ(x)=θ0+θ1x, where θ0 and θ1 are the model's parameters.

`Cost Function` (or Loss Function): This term simply quantifies how wrong our model's predictions are relative to the actual truth. We aim to minimize this function to achieve the most accurate prediction. It's also known as the Mean Squared Error (MSE) and it's given by 

The cost function is:

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2 $$


where m is the total count of observations and the summation over the squared differences (errors) ensures that the higher the error, the greater the cost.

These components come together and can be optimized using the Gradient Descent we learned in the previous lesson. Gradient Descent will painstakingly adjust 
The parameters of the linear regression model are: 

$$ \theta_0 \quad \text{and} \quad \theta_1 $$

to minimize the cost function and derive a line that gives us the lowest possible error or cost.

## Designing Linear Regression Models

Every well-constructed tower needs a solid design, and building a high-performance regression model is no different! Once the foundation (mathematics) of Linear Regression is established, we leverage Python and its powerful scikit-learn library for the implementation.

You can break down the steps to designing a Linear Regression model as follows:

Start by importing the necessary libraries and classes.

Load the dataset and isolate the features (independent variables) and target variables (dependent variables).

Split the data into training and testing parts: the training set for learning and the testing set for evaluating the model's performance. Here, it's crucial to understand that while splitting the data, the test_size argument represents the proportion of the dataset to include in the test set. The random_state argument ensures reproducibility by controlling the shuffling applied to the data before applying the split.

Create the Linear Regression model using scikit-learn's LinearRegression class.

Finally, assess the model using various performance metrics.
Let's implement this in Python and predict some wine quality:

```Python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd

# Load the wine dataset
import datasets
red_wine = datasets.load_dataset('codesignal/wine-quality', split='red')
red_wine = pd.DataFrame(red_wine) 

# Select features and target variable
features = red_wine.drop('quality', axis=1)
target = red_wine['quality']

# Split the dataset into a training set and a testing set
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Instantiate and fit the model
model = LinearRegression()
model.fit(features_train, target_train)

# Predict the test features
predictions = model.predict(features_test)

# Evaluate the model
mse = metrics.mean_squared_error(target_test, predictions)
print('Mean Squared Error:', mse) # Mean Squared Error: 0.39002514396395416
```
To visualize our prediction, let's draw a plot showing the Actual vs Predicted difference:

```Python

import matplotlib.pyplot as plt

# Plot target vs prediction
plt.scatter(target_test, predictions, color='blue')
# Plot the ideal prediction line (with zero error)
plt.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
```
![image](https://github.com/user-attachments/assets/7199337d-f17b-4188-b531-3ad633f31c4c)


## Examining the Model's Performance

That wasn't too bad, was it? But hold on, we're not done yet. It's crucial to check the model's performance by examining the residuals, simply the difference between the actual and predicted values. The smaller the residuals, the better the model performs. We'll look at two key metrics here:

Mean Squared Error (MSE): The average of the squared errors, with larger errors contributing more due to the squaring. This is the cost function we discussed earlier.

Coefficient of Determination (R-squared): This measures the degree of variation in the target variable that our model could predict. It ranges between 0 and 1, with a higher value representing a higher quality of our model.

Here's how to calculate MSE and R-squared in Python:

```Python

r2_score = metrics.r2_score(target_test, predictions)
print('R-squared:', r2_score) # R-squared: 0.4031803412796231
```
## Conclusion

We've unraveled the intricacies of Linear Regression, starting from the basic principles, strolling through the supportive mathematical framework, and finally constructing a fully-functioning model with Python and scikit-learn. Now, you should understand the concepts and workings of Linear Regression, its design, implementation, and application for predictive modeling. Based on this newfound knowledge, you've used the Wine Quality Dataset to predict wine quality based on numerous physicochemical features.
