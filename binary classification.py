# Library
import numpy as np

# Iris Dataset with two Species :  Iris-setosa(0) and Iris-versicolor(1)
# It has 4 features named as SepalWidthCm, SepalLengthCm, PetalLengthCm, PetalWidthCm

train_x = np.array(
    [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2],
     [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
     [5.4, 3.7, 1.5, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3],
     [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3],
     [5.2, 2.7, 3.9, 1.4], [4.8, 3.4, 1.6, 0.2], [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2],
     [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
     [5, 2, 3.5, 1], [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3],
     [6.7, 3.1, 4.4, 1.4], [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1]])
train_y = np.array(
    [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0],
     [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
test_x = np.array(
    [[5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5], [4.8, 3.4, 1.9, 0.2],
     [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2], [4.8, 2.9, 4.3, 1.3]])
test_y = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

train_x = train_x.T
train_y = train_y.T
test_x = test_x.T
test_y = test_y.T

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# Sigmoid Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# initilize parameters w and b
def init_with_zeros(input_len):
    w1 = np.zeros((input_len, 1))
    b1 = 0

    assert (w1.shape == (input_len, 1))  # if condition get false then stop execution
    assert (isinstance(b1, int))

    return w1, b1


# Forward and Backward propagation
def propagation(w, b, X, Y):
    m = X.shape[1]  # m is no of training example

    A = sigmoid(np.matmul(w.T, X) + b)  # activtion function

    cost = np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))
    cost = -np.sum(cost) / m
    # print(cost)

    dw = (1 / m) * (np.matmul(X, (A - Y).T))  # backward propagation
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    grads = {"dw": dw, "db": db}

    return grads, cost


# update parameters w, b, db, dw
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagation(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if print_cost:
            print("Cost after iteration %i: %f", (i, cost))  # print cost for every iteration

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


dim = 4
w, b = init_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b), end="\n\n\n\n\n")

grads, cost = propagation(w, b, train_x, train_y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost), end="\n\n\n\n\n")

params, grads, costs = optimize(w, b, train_x, train_y, num_iterations=100, learning_rate=0.009, print_cost=False)
print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))


# Function to predict class of Iris

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.matmul(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] < 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# Perform testing on test_data
print("predictions = " + str(predict(np.array(params["w"]), np.array(params["b"]), test_x)))
print("Test Set / Actual Output", test_y)
# Perform testing on new data

data = np.array([[5.8, 2.7, 3.9, 1.2]]).T
output = np.array(predict(np.array(params["w"]), np.array(params["b"]), data))

#print(str(output))
if(output[0,0] == float(0.)):
    print(str(data.T) + "   This is Iris-setosa - class 0")
else:
    print(str(data.T) + "   This is Iris-versicolor - class 1")