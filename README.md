# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:PARAVEZHAA M
RegisterNumber:212225220070

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("50_Startups.csv")


x = data["R&D Spend"].values
y = data["Profit"].values

# Feature scaling (important for gradient descent)
x = (x - np.mean(x)) / np.std(x)


plt.scatter(x, y)
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Profit Prediction")
plt.show()


m = len(y)
X = np.c_[np.ones(m), x]   # shape (m, 2)
y = y.reshape(m, 1)


def computeCost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sq_error = (predictions - y) ** 2
    return (1 / (2 * m)) * np.sum(sq_error)


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        predictions = X.dot(theta)
        error = predictions - y
        theta = theta - (alpha / m) * (X.T.dot(error))
        J_history.append(computeCost(X, y, theta))

    return theta, J_history


theta = np.zeros((2, 1))


theta, J_history = gradientDescent(X, y, theta, alpha=0.01, num_iters=1500)


print(f"h(x) = {theta[0,0]:.2f} + {theta[1,0]:.2f}x")


plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Cost Function using Gradient Descent")
plt.show()


plt.scatter(x, y)
x_vals = np.linspace(min(x), max(x), 100)
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Profit Prediction using Gradient Descent")
plt.show()


*/
```

## Output:
<img width="1491" height="605" alt="Screenshot 2026-02-04 113426" src="https://github.com/user-attachments/assets/f691a070-c66e-4e43-9ab3-81d94ec2ac43" />
<img width="1488" height="564" alt="Screenshot 2026-02-04 113446" src="https://github.com/user-attachments/assets/b79763a3-fe2e-4daf-95f8-8d20d046934b" />
<img width="1489" height="584" alt="Screenshot 2026-02-04 113522" src="https://github.com/user-attachments/assets/08753fbe-1071-4853-8a89-1fb20e22c67e" />






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
