#Unveiling Logistic Regression: Internals, Design, and Hands-On Implementation with Wine Quality Prediction
## Introduction to Logistic Regression

A key player in the machine learning universe, Logistic Regression is indispensable in supervised learning problems, particularly binary classification.

As you may recall from prior lessons, Linear Regression is effective for regression problems. However, regarding classification problems, Logistic Regression takes the spotlight. We'll understand why as we predict the binary outcomes of wine quality - either good or bad - using our Wine Quality Dataset based on its physicochemical properties. Let's delve into the concept of Logistic Regression, breaking down its theory, internal mechanisms, design, and implementation across various datasets.

## Understanding Logistic Regression

Contrary to its name, Logistic Regression is a classification algorithm used to estimate the probabilities of a binary response based on one or more predictor (also known as independent) variables. It is particularly beneficial for binary outcomes, meaning situations with only two possible results.

Now, let's bring this concept to life by relating it to our Wine Dataset. Our goal is to predict wine quality, which, as you may remember, ranges from 0 to 10. To keep things simple and focus on a binary classification problem, let's classify the wines as good (a quality rating of 7 or above) and not good (a quality rating below 7). Therefore, we will be using Logistic Regression to predict whether the quality of a specific type of wine is 'good' or 'not good' based on its physicochemical features.

In Logistic Regression, all of this is achieved by using a logistic function, which limits the unlimited outcome of the linear equation to a number between 0 and 1. Also known as the Sigmoid function, this logistic function is an S-shaped curve that maps any real-valued number into a value falling within these bounds. The function is defined as follows,

$$ f(x) = \frac{1}{1 + e^{-x}} $$

​
 
In this equation, x represents the output of a linear combination of feature values and their corresponding coefficients,

$$ x = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n $$

In this informative equation: β (Beta) terms are the model's parameters, signifying the influence of each input feature (denoted by X) on the predicted outcome.
X terms represent independent predictor variables. The Math Behind Logistic Regression To understand the intricacies of Logistic Regression, we need to unpack the mathematical marvel that it is. The Logistic or Sigmoid function forms the backbone of Logistic Regression. Once we compute the predicted probability (p) using the Sigmoid function, we can assign classes by defining a threshold (which is generally 0.5):
If p≥0.5 the label for the example is 1 (or Good in our case). 

If p<0.5, the label for the example is 0 (or Not Good in our case).

The next critical component in Logistic Regression is the cost function. Unlike in Linear Regression, we can't use Mean Square Error as the cost function because the Logistic function would introduce a non-linear term into the cost function, making the cost function non-convex anymore. In Logistic Regression, the cost function is defined as:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right] $$

Where: θ represents the parameters we must determine using an optimization algorithm to minimize the cost function.
m is the number of samples.y and x represent the target and input of each sample, respectively.

​
 (x) is the logistic function that computes the predicted probability that 
y
=
1
y=1.
While discussing the cost function, it's crucial to consider optimization algorithms like Gradient Descent used to find the parameters 
θ
θ to minimize this cost.

Disclaimer: in most scenarios, you don't have to remember and implement the cost function yourself, as there are plenty of libraries (e.g., scikit-learn that provide the built-in implementation of the Logistic Regression). However, it's still essential to understand high-level concepts and what's being optimized.

Logistic Regression Model Assumptions
Before we dive deeper, let's discuss the underlying assumptions that guide a Logistic Regression Model. These assumptions serve as rules of thumb when modeling Logistic Regression:

Each observation is independent of others: This means the outcome or probability of success (p in our logistic function) for one example neither influences nor is influenced by the outcomes of other examples.
There is no multicollinearity among explanatory variables: In simple terms, the input variables should not be too highly correlated with each other. Any correlation implies that they carry similar information to the model, which is redundant.
The input variables have a linear relationship with the log odds: Although the outcome in logistic regression is a binary variable, logistic regression stipulates that the input variables are linearly related to the log odds 
log
⁡
p
1
−
p
log 
1−p
p
​
 , and hence, to the logit of the probability, p.
Violating these assumptions may result in inaccurate models and misinterpretations. Therefore, validating these assumptions while modeling Logistic Regression is essential.

Designing Logistic Regression Model
Let's transition from understanding the Logistic Regression concept to its design and implementation. Using Python and scikit-learn, we'll see how to design a Logistic Regression model:

Specify the hypothesis or function the model should learn. In Logistic Regression, this is the Sigmoid function.
Define an error, cost, or loss function we aim to minimize. For Logistic Regression, the cost function is defined as Cross-Entropy Loss.
Define a learning algorithm that optimizes the parameters for the hypothesis to fit the model to the training data. In our case, it's the Gradient Descent algorithm.
Let's look at a quick implementation using scikit-learn:

Python
Copy to clipboard
# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the wine dataset
import datasets
import pandas as pd
red_wine = datasets.load_dataset('codesignal/wine-quality', split='red')
red_wine = pd.DataFrame(red_wine)

# Convert the multi-class problem to a binary one
red_wine['quality'] = red_wine['quality'].apply(lambda x : 1 if x >= 7 else 0)

# Split the dataset into features and target variable
X = red_wine.drop('quality', axis=1)
y = red_wine['quality']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a Logistic Regression object
lr = LogisticRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)

# Print the learned parameters
print(lr.coef_, lr.intercept_)
"""
[[-0.02641816 -3.24280912 -0.04024957  0.07795443 -1.26020881  0.02151089
  -0.01866486 -1.04040183 -2.50766981  2.00156001  0.9266963 ]]
[-1.77875604]
"""
In the script above, we create a LogisticRegression object and use the fit function to train it on the training sets, X_train and y_train. The learned parameters of the Logistic function can be printed as shown in the last line. The coef_ variable gives the coefficients for different features (or 
X
X), while intercept_ provides the intercept term (or 
β
0
β 
0
​
 ).

Implementing Logistic Regression Model: Hands-on Exercise
With the fundamentals and design in place, let's dive into an example to see how the Logistic Regression model predicts the wine quality for our test dataset and evaluates its performance:

Python
Copy to clipboard
# Make predictions on the test dataset
y_pred = lr.predict(X_test)

# Import metrics module for accuracy calculation
from sklearn import metrics

# Model accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
# Accuracy:  0.8875
In the above code block, we use the predict function to predict the wine quality for the test dataset. The accuracy_score() function from the metrics module of the sklearn library is used to calculate the accuracy of our logistic regression model. The function takes in the actual wine qualities and predicted wine qualities and returns the proportion of correct predictions.

Interpreting Coefficients in Logistic Regression
Now that we have our trained Logistic Regression model, we might wonder how to interpret the output of our model. The output of the model includes the coefficients (also known as weights) of each feature and a bias (also known as the intercept). The coefficients represent the log of the odds ratio of the corresponding feature.

For example, if the coefficient of a feature, say pH (with log odds ratio = 
β
β), is 0.5, it indicates that for each unit change in pH, keeping other features constant, the odds of our outcome (whether the wine quality is good) would increase by a factor of 
e
0.5
e 
0.5
 .

Evaluating the Model Performance - Metrics
Evaluating the performance of a model is crucial to assess its usability and reliability. We evaluate our Logistic Regression model's performance using several important metrics. Let's define a few key metrics:

Confusion Matrix: This table describes the performance of a classification model. It's essentially a 
2
×
2
2×2 matrix that visualizes the performance of the regression, representing actual and predicted classifications in terms of true positives, false positives, true negatives, and false negatives.
Accuracy: This is the ratio of correctly predicted observations to total observations. Accuracy = (True Positives + True Negatives) / Total Observations.
Precision: This is the ratio of correctly predicted positive observations to the total predicted positives. Precision = True Positives / (True Positives + False Positives).
Recall (Sensitivity): This is the ratio of correctly predicted positive observations to all observations in the actual class. Recall = True Positives / (True Positives + False Negatives).
F1 Score: This is the weighted average of Precision and recall. F1Score = 2 * Recall * Precision / (Recall + Precision).
ROC-AUC : This is the area under the Receiver Operating Characteristic curve. It indicates how much the model can distinguish between classes.
Here is how these metrics can be calculated using sklearn:

Python
Copy to clipboard
from sklearn import metrics

# Model Accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
# Accuracy:  0.8875

# Model Precision
print("Precision: ", metrics.precision_score(y_test, y_pred))
# Precision:  0.5172413793103449

# Model Recall
print("Recall: ", metrics.recall_score(y_test, y_pred))
# Recall:  0.2727272727272727

# Model F1-Score
print("F1 Score: ", metrics.f1_score(y_test, y_pred))
# F1 Score:  0.3571428571428571

# Model AUC
print("AUC: ", metrics.roc_auc_score(y_test, y_pred))
# AUC:  0.6198930481283422
The accuracy_score() function calculates the model's accuracy by comparing the actual output in y_test with the predicted output in y_pred. Similarly, other functions like precision_score(), recall_score(), f1_score(), and roc_auc_score() calculate their respective metrics.

Conclusion
That's a wrap for the Logistic Regression Model! We've explored the landscape of Logistic Regression, unpacked its internals, understood the designing process, and implemented it on our Wine Quality Dataset to predict wine quality. You've successfully navigated through the critical components, theoretical aspects, and practical application of Logistic Regression. Well done!

We've also dissected the performance evaluation process of our model in detail. This process is instrumental in assessing and enhancing the model's fit and precision. Model performance reveals strengths and areas for improvement, enabling you to create a reliable and efficient model.

