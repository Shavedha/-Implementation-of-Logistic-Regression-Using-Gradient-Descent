# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required. 
2. Read the dataset. 
3. Define X and Y array. 
4. Define a function for costFunction,cost and gradient. 
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Y SHAVEDHA
RegisterNumber: 212221230095
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()


plt.show()

    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J,grad
    
X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta  = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1),np.arange(y_min,y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot = np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)),X))
    prob = sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X) == y)
```

## Output:
### Array Value of x
<img width="428" alt="image" src="https://user-images.githubusercontent.com/93427376/233777046-2156fc1a-72dd-4599-947e-a0b12418aad0.png">

### Array Value of y
<img width="308" alt="image" src="https://user-images.githubusercontent.com/93427376/233777082-e36a04e9-72ea-4b1c-ba9d-f80e5c23ff06.png">

### Exam 1- Score graph
<img width="444" alt="image" src="https://user-images.githubusercontent.com/93427376/233777120-d76c83b8-434f-436f-bce5-d85b42257610.png">

### Sigmoid Function Graph
<img width="415" alt="image" src="https://user-images.githubusercontent.com/93427376/233777172-536c28de-aef8-436e-a1cf-2602fe0779a6.png">

### X_train_grad value
<img width="345" alt="image" src="https://user-images.githubusercontent.com/93427376/233777189-66ffabb4-aa1d-4556-a347-21a7e1dc4d0e.png">

### Y_train_grad value
<img width="307" alt="image" src="https://user-images.githubusercontent.com/93427376/233777203-c2134b06-9d71-4668-99bf-ff4b5fba5536.png">

### Print res.x
<img width="356" alt="image" src="https://user-images.githubusercontent.com/93427376/233777223-37464c89-1cba-4c18-a310-179033d585ec.png">

### Decision Boundary grapg for Exam Score
<img width="433" alt="image" src="https://user-images.githubusercontent.com/93427376/233777251-bfdeb1de-1e84-4f7c-8209-60fe312eb034.png">

### Probability value
<img width="217" alt="image" src="https://user-images.githubusercontent.com/93427376/233777262-4d9446ed-1fc6-415b-b48d-17fe38141b09.png">

### Prediction value of mean
<img width="260" alt="image" src="https://user-images.githubusercontent.com/93427376/233777280-808977ea-e392-4fa1-8b8a-bc08e063e8dd.png">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

