import numpy as np
import matplotlib.pyplot as plt

#Draws line between two points
def draw(x1, x2):
    ln = plt.plot(x1,x2)

#Sigmoid function
def sigmoid(score):
   return 1/(1+np.exp(-score))

#Calculating the cross entropy of the graph
def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(points*line_parameters)
    cross_entropy = -(1/m)*(np.log(p).T * y + np.log(1-p).T*(1-y))
    return cross_entropy

#Function to calculate the gradient descent over 500 iterations
def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(500):
        p = sigmoid(points*line_parameters)
        #Subtracting this value from the line parameters will provide new parameters with a lower error value
        gradient = (points.T * (p - y))*(alpha/m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1/w2)
    draw(x1, x2)

n_pts = 100 #Number of points we want per class
np.random.seed(0) #Gives us the same random points
bias = np.ones(n_pts)
random_x1_values = np.random.normal(10, 2, n_pts) #Identify center of normal distribution, standard deviation and number of points needed
random_x2_values = np.random.normal(12, 2, n_pts) #Identify center of normal distribution, standard deviation and number of points needed
top_region = np.array([random_x1_values, random_x2_values, bias]).T #Allows for x1 and x2 values of one point to be on each row (top region - diabetic)
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T #Allows for x1 and x2 values of one point to be on each row (bottom region - healthy)
all_points = np.vstack((top_region, bottom_region)) #Stacks both arrays so you get 1 2d array with all the points
line_parameters = np.matrix([np.zeros(3)]).T #We don't want to hard code the weight so we assign the initial value a 0
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

#Weights for random initial line
# w1 = -0.2
# w2 = -.35
# b = 3.5
# line_parameters = np.matrix([w1, w2, b]).T #Make initial line a matrix and turn it into a (3,1) matrix instead of a (1,3)
# x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()]) #Assign two values to x1 which are the two hoizontal values of the line
# x2 = -b / w2 + x1 * (-w1/w2) #Calculates the vertical points of the two horizontal values
# y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1) #Creating an array of 0s and 1s to classify healthy and diabetic people
# linear_combination = all_points*line_parameters #Applies the linear combination to all of the points
# probabilities = sigmoid(linear_combination) #Calculates the probabilites of each point

# #Allows for multiple plots on the same figure
# #"ax" is the axes object and allows us to control everything about the individual plot
# _, ax = plt.subplots(figsize=(4, 4)) #4 inches wide x 4 inches high
# ax.scatter(top_region[:, 0], top_region[:, 1], color='r') #Scatter plot of red points
# ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b') #Scatter plot of blue points
# draw(x1,x2) #Draws the line
# plt.show()

#Allows for multiple plots on the same figure
#"ax" is the axes object and allows us to control everything about the individual plot
_, ax = plt.subplots(figsize=(4, 4)) #4 inches wide x 4 inches high
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()

print(calculate_error(line_parameters, all_points, y))
