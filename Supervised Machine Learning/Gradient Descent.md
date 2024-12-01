Mastering Gradient Descent: Your Guide to Optimizing Machine Learning Models
Introduction
Hello learners! Until now, we have delved deep into the world of supervised machine learning, using the Wine Quality Dataset as a primary resource. As we proceed, we plan to illuminate the inner workings of the learning process within machine learning models, particularly Gradient Descent.

Gradient Descent is a cornerstone of optimization in machine learning and deep learning. Its function enables the machine learning model to 'learn,' thereby improving itself based on its past performance. As we peel back layers of this lesson, we promise you a more profound understanding of Gradient Descent, its role in machine learning, and its implementation with Python. Buckle up for an exciting educational journey!

Gradient Descent Demystified
Have you ever hiked to the top of a hill and looked down to determine the best route of descent? One potentially disastrous step off a steep cliff is dangerous, while cautiously descending the gentle slopes might cause less harm. The concept of Gradient Descent mirrors this scenario — it, too, sees the value in finding and taking the optimal path or, more precisely, reaching the minimum point.

In machine learning, Gradient Descent can be visualized as a careful navigation downwards until we find the valley between hills. The 'hill' in this context is the cost function, which quantifies our model's error. Through a series of small steps, Gradient Descent refines the cost function by 'walking' down the hill towards the steepest descent until it reaches the lowest possible point at its optimal state.

Mathematics Behind Gradient Descent
Having conceptualized Gradient Descent, let’s delve deeper and uncover the mathematical mechanics that fuel it. At its core, Gradient Descent relies on two key mathematical mechanisms: the Cost Function and the Learning Rate.

The Cost Function (or Loss Function) quantifies the disparity between predicted and expected values, presenting it as a single float number. The type of cost function utilized depends on the challenge at hand. In our Wine Quality dataset, we can define a cost function that computes the difference between our model's predicted quality of wine and the actual quality.

The Learning Rate, symbolized by 
α
α, dictates the size of the steps we take downhill. A lower value of 
α
α results in smaller, more precise steps, while a high value could cause drastic, potentially unstable steps.

From our previous analogy, imagine the hill is symbolized by a function of position, 
g
(
x
)
g(x). Starting at the hill's pinnacle (
x
0
x 
0
​
 ), we revise our position (
x
x) by moving a step proportional to the negative gradient at that location. The gradient 
g
′
(
x
)
g 
′
 (x) is simply the derivative of 
g
(
x
)
g(x), pointing toward the steepest ascent. Conversely, 
−
g
′
(
x
)
−g 
′
 (x) signifies the fastest descending path. We repeat this stepping process until the gradient becomes zero at the minimum point, indicating no further downhill path, i.e., no additional optimization is required.

Advancements in Gradient Descent
Here, an interesting question arises, "Do we always use all data to calculate the gradient?" The answer depends. Gradient Descent has evolved into various versions, depending on the amount of data used in computing the gradient: batch, stochastic, and mini-batch gradient descent.

The original version, batch gradient descent, uses the complete dataset at every step. While this may seem meticulous and comprehensive, it proves extremely inefficient when dealing with substantial datasets housing millions of entries. Imagine watching a movie frame by frame at a snail's pace — it can be painstakingly slow despite its precision.

Implementing Gradient Descent
Now, let's make the Gradient Descent implementation in Python. We start by assigning random values to our model’s parameters. Gradual adjustments to these parameters follow, in each instance computing the cost function, our error, and taking a step towards the steepest slope until our error is minimal or the state is optimized.

Here’s a general outline of how we would implement gradient descent in Python:

Python
Copy to clipboard
def gradient_descent(x, y, theta, alpha, iterations):
    """
    x -- input dataset
    y -- target dataset
    theta -- initial parameters
    alpha -- learning rate
    iterations -- the number of times to execute the algorithm
    """

    m = y.size # number of data points
    cost_list = [] # list to store the cost function value at each iteration
    theta_list = [theta] # list to store the values of theta at each iteration
    
    for i in range(iterations):
        # calculate our prediction based on our current theta
        prediction = np.dot(x, theta)
        
        # compute the error between our prediction and the actual values
        error = prediction - y
        
        # calculate the cost function
        cost = 1 / (2*m) * np.dot(error.T, error)
        
        # append the cost to the cost_list
        cost_list.append(np.squeeze(cost))
        
        # calculate the gradient descent and update the theta
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        
        # append the updated theta to the theta_list
        theta_list.append(theta)
    
    # return the final values of theta, list of all theta, and list of all costs, respectively 
    return theta, theta_list, cost_list
In this code snippet, x represents your input dataset, y is your target dataset, theta indicates your initialized parameters, alpha is your learning rate, and iterations denotes the number of times the optimization algorithm executes to fine-tune the parameters.

Gradient Descent for Wine Quality Prediction: A Hands-On Application
Are you eager to see Gradient Descent in action? Let’s apply it to the Wine Quality Dataset. Using the cost function that computes the error between the actual and predicted wine quality, we can represent this error as a 'hill.' As we journey further into the hill, our error diminishes, optimizing the model's prediction accuracy for wine quality.

Let's focus on one feature for simplicity's sake: alcohol. We will use Python to demonstrate how Gradient Descent can design a model that predicts wine quality based on its alcohol content.

Python
Copy to clipboard
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import datasets

# Load Wine Quality Dataset
red_wine = datasets.load_dataset('codesignal/wine-quality', split='red')
red_wine = pd.DataFrame(red_wine)

# Only consider the 'alcohol' column as a predictive feature for now
x = pd.DataFrame(red_wine['alcohol'])
y = red_wine['quality']

# Splitting datasets into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# We set our parameters to start at 0
theta = np.zeros(x_train.shape[1]).reshape(-1, 1)

# Define the number of iterations and alpha value
alpha = 0.0001
iters = 1000

# Applying Gradient Descent
y_train = np.array(y_train).reshape(-1, 1)
g, theta_list, cost_list = gradient_descent(x_train, y_train, theta, alpha, iters)

print(cost_list)
plt.plot(range(1, iters + 1), cost_list, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.show()
image

In this code, we first extricated the predictor, alcohol, from our Wine Quality Dataset and proceeded to run our Gradient Descent function. In the output, you can see the cost function reducing with each iteration, depicting how Gradient Descent gradually descends the hill, alleviating the cost function and thus enhancing our model's predictions.

Assessing Gradient Descent Performance
The learning rate (alpha) is a critical component in the performance of gradient descent. Striking the right balance can be delicate: if alpha is too large, we might overshoot our optimal point, while if it's too small, we might require an excessive number of iterations to converge, or we might not converge at all.

While this can be adjusted in our code as per requirement, we will later discuss how the ideal alpha is determined empirically by testing various alpha values, leading to the best model performance.
