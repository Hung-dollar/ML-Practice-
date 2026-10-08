import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# Input: 4 features
# Hidden layer 1: 5 neurons, ReLU
# Hidden layer 2: 4 neurons, Tanh
# Hidden layer 3: 3 neurons, Sigmoid
# Output layer: 2 neurons, Softmax
# Loss: cross entropy

# numpy implementation
def relu(Z):
    return np.where(Z>0,Z,0)

def relu_derivative(Z):
    return np.where(Z>0,1,0)

def tanh(Z):
    return ( np.exp(Z) - np.exp(-Z) ) / ( np.exp(Z) + np.exp(-Z) )

def tanh_derivative(Z):
    return 1 - tanh(Z)**2

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def softmax(Z):
    n, d = Z.shape
    return np.exp(Z) / np.sum(np.exp(Z), axis = 1, keepdims = True)

def forward(X, W1, b1, W2, b2, W3, b3, W4, b4): 
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2 
    A2 = tanh(Z2)
    Z3 = A2 @ W3 + b3 
    A3 = sigmoid(Z3)
    Z4 = A3 @ W4 + b4
    Y_hat = softmax(Z4)
    return Z1, A1, Z2, A2, Z3, A3, Z4, Y_hat

def mse_loss(Y, Y_hat):
    return np.mean((Y -Y_hat)**2)

def cross_entropy_loss(Y, Y_hat, eps = 1e-12):
    Y_hat = np.clip(Y_hat,eps,1-eps)
    return -np.mean(np.sum(Y * np.log(Y_hat), axis = 1, keepdims = True))
    
def backpropagation(X, W1, b1, W2, b2, W3, b3, W4, b4, Y): 
    b = Y.shape[0]
    Z1, A1, Z2, A2, Z3, A3, Z4, Y_hat = forward(X,W1,b1,W2,b2,W3,b3,W4,b4)

    dZ4 = ( Y_hat - Y ) / b # b x l4

    dW4 = A3.T @ dZ4 # l3 x l4
    db4 = np.sum(dZ4,axis=0,keepdims = True) #1 x l4

    dA3 = dZ4 @ W4.T  # b x l3
    dZ3 = dA3 * sigmoid_derivative(Z3) # b x l3

    dW3 = A2.T @ dZ3 # l2 x l3 
    db3 = np.sum(dZ3,axis=0,keepdims=True) # 1 x l3

    dA2 = dZ3 @ W3.T # b x l2
    dZ2 = dA2 * tanh_derivative(Z2) #b x l2

    dW2 = A1.T @ dZ2 # l1 x l2
    db2 = np.sum(dZ2,axis=0, keepdims = True) #1 x l2

    dA1 = dZ2 @ W2.T # b x l1
    dZ1 = dA1 * relu_derivative(Z1) # b x l1

    dW1 = X.T @ dZ1 # b x l1
    db1 = np.sum(dZ1,axis=0, keepdims = True) # 1 x l1 
    
    return Y_hat, dW1, db1, dW2, db2, dW3, db3, dW4, db4

def train(X, Y, W1, b1, W2, b2, W3, b3, W4, b4, lr = 0.5, epochs = 2000):
    for i in range(epochs):
        Y_hat, dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backpropagation(X, W1, b1, W2, b2, W3, b3, W4, b4, Y)
        loss = cross_entropy_loss(Y,Y_hat)
        if (i%10==0): 
            print(loss)
        W1 -= lr * dW1
        b1 -= lr * db1

        W2 -= lr * dW2
        b2 -= lr * db2

        W3 -= lr * dW3
        b3 -= lr * db3

        W4 -= lr * dW4 
        b4 -= lr * db4

        if i % 200 == 0:
            print(i,loss )
    return W1, b1, W2, b2, W3, b3, W4, b4

# Pytorch implementation
# Activation functions
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z):
        ctx.save_for_backward(Z)
        return torch.where(Z>0,Z,0)

    @staticmethod
    def backward(ctx, grad_out):
        # TODO
        Z, = ctx.saved_tensors
        return grad_out * torch.where(Z>0,1,0)


class MyTanh(torch.autograd.Function):      
    @staticmethod
    def tanh(Z) :
        return  ( torch.exp(Z) - torch.exp(-Z) ) / ( torch.exp(Z) + torch.exp(-Z) )
    @staticmethod
    def forward(ctx, Z):
        ctx.save_for_backward(Z)
        return MyTanh.tanh(Z)
    @staticmethod
    def backward(ctx, grad_out): 
        Z, = ctx.saved_tensors
        return grad_out * (1 - MyTanh.tanh(Z)**2)
        
class MySigmoid(torch.autograd.Function):    
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + torch.exp(-Z))

    @staticmethod
    def forward(ctx, Z): 
        ctx.save_for_backward(Z)
        return MySigmoid.sigmoid(Z)

    @staticmethod
    def backward(ctx, grad_out): 
        Z, = ctx.saved_tensors
        return grad_out * MySigmoid.sigmoid(Z) * (1-MySigmoid.sigmoid(Z))

# Linear layer
class MyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X,W,b)
        return X @ W + b
    @staticmethod
    def backward(ctx, g):
        # TODO: return THREE gradients, in order (X, W, b)
        X, W, b, = ctx.saved_tensors
        dX = g @ W.T
        dW = X.T @ g 
        db = g.sum(dim = 0, keepdims = True)
        return dX, dW, db 


# Loss function
class MySoftmaxCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z, Y):
        P = torch.softmax(Z,dim=1)
        ctx.save_for_backward(P,Y)
        return torch.mean(-torch.sum(Y*torch.log(P),dim=1)) 
    @staticmethod
    def backward(ctx, grad_out):
        P, Y, = ctx.saved_tensors
        return grad_out * (P - Y) / P.shape[0], None
        


def make_params(np_params):      # list of 8 NumPy arrays -> tensors that track gradients
    return [torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in np_params]
def forward_loss(X, Y, params): 
    W1, b1, W2, b2, W3, b3, W4, b4 = params
    Z1 = MyLinear.apply(X,W1,b1)
    A1 = MyReLU.apply(Z1)

    Z2 = MyLinear.apply(A1,W2,b2)
    A2 = MyTanh.apply(Z2)

    Z3 = MyLinear.apply(A2,W3,b3)
    A3 = MySigmoid.apply(Z3)

    Z4 = MyLinear.apply(A3,W4,b4)

    loss = MySoftmaxCE.apply(Z4,Y)
    return loss
    
# Training
def train_torch(X, Y, params, lr, epochs): 
    for i in range(epochs): 
        W1, b1, W2, b2, W3, b3, W4, b4 = params
        loss = forward_loss(X,Y,params)
        if (i%10==0): 
            print(loss)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
                p.grad.zero_()
        
# torch.nn implementation
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4,5)
        self.fc2 = nn.Linear(5,4)
        self.fc3 = nn.Linear(4,3)
        self.fc4 = nn.Linear(3,2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self,X):
        X = self.relu(self.fc1(X))
        X = self.tanh(self.fc2(X))
        X = self.sigmoid(self.fc3(X))
        X = self.fc4(X) 
        return X

def train_nn(model, X, Y, epochs): 
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(X,Y)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1
    )
    for i in range(epochs):
        epoch_loss = 0.0
        for X_batch, Y_batch in loader:
            logits = model(X_batch)
            loss = criterion(logits,Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item() * X_batch.size(0)
        epoch_loss/=len(dataset)
        if i % 10 == 0: 
            print(f"Epoch {i} | Average Loss: {epoch_loss:.4f}")




# Parameters 
n = 10
d0 = 4
d1 = 5
d2 = 4
d3 = 3
d4 = 2

#Init with random samples and weights

np.random.seed(0)
X = np.random.randn(n,d0)
XT = torch.tensor(X)
W1= np.random.randn(d0,d1) 
b1 = np.random.randn(1,d1) 
W2= np.random.randn(d1,d2) 
b2 = np.random.randn(1,d2) 
W3= np.random.randn(d2,d3) 
b3 = np.random.randn(1,d3) 
W4= np.random.randn(d3,d4) 
b4 = np.random.randn(1,d4) 

labels = np.random.randint(0, d4, size=n)
Y = np.eye(d4)[labels]
YT = torch.tensor(Y)
model = MyNetwork()

#Comparison
print("Torch Training: ")
np_params = [W1,b1,W2,b2,W3,b3,W4,b4]
train_torch(XT,YT,make_params(np_params),0.1,200) 
print("Numpy Training: ")
train(X,Y,W1,b1,W2,b2,W3,b3,W4,b4,0.1,200)
XT = torch.tensor(X, dtype=torch.float32)
YT = torch.tensor(labels, dtype=torch.long)
print("NN training: ")
train_nn(model,XT,YT,200)
print("--- Model Parameters ---")
for name, param in model.named_parameters():
    print(f"\nLayer: {name}")
    print(f"Shape: {param.shape}")
    print(f"Values:\n{param.data}")