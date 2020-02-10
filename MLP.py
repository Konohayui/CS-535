"""
INSERT YOUR NAME HERE
"""


from __future__ import division
from __future__ import print_function

import sys
try:
    import _pickle as pickle
except:
    import pickle

import numpy as np

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        # DEFINE __init function
        self.W = W
        self.b = b

    def forward(self, x):
        # DEFINE forward function
        self.x = x
        linear_output = np.dot(self.x, self.W) + self.b
        
        return linear_output

    def backward(self, grad_output):
        # DEFINE backward function
        # Computing the gradients
        dx = np.dot(grad_output, self.W.T)
        dw = np.dot(self.x.T, grad_output)
        db = np.sum(grad_output, axis=0)
        
        return dx, dw, db
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self):
        self.relu_output = None

    def forward(self, x):
        # DEFINE forward function
        self.relu_output = np.maximum(0, x)
        
        return self.relu_output

    def backward(self, grad_output):
        # DEFINE backward function
        # Computing the gradients
        self.relu_output[self.relu_output > 0] = 1
        dx = np.multiply(self.relu_output, grad_output)
        
        return dx

# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self):
        self.predictions = None

    def forward(self, x, y_batch):
        # DEFINE forward function
        self.predictions = 1.0/(1.0+np.exp(-x))
        # this is an error in previous calculation that
        # we couldn't spot it out 
        # and gives negative loss, so add a negative sign.
        loss = -1*(y_batch*np.log(self.predictions+1e-8)+(1.0-y_batch) * np.log(1.0-self.predictions+1e-8))
        
        return self.predictions, loss

    def backward(self, y, grad_output):
        # DEFINE backward function
        # Computing the gradients
        
        return grad_output*(self.predictions-y)

# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):
    def __init__(self, input_dims, hidden_units):
        # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units

        # initialize weights and bias
        self.w1 = np.random.uniform(-1.0, 1.0, size=(input_dims, hidden_units))
        self.b1 = np.random.uniform(-1.0, 1.0, size=(1, hidden_units))
        self.w2 = np.random.uniform(-1.0, 1.0, size=(hidden_units, 1))
        self.b2 = np.random.uniform(-1.0, 1.0, size=(1, 1))

        # initialize layers
        self.linearT1 = LinearTransform(self.w1, self.b1)
        self.linearT2 = LinearTransform(self.w2, self.b2)
        self.relu = ReLU()
        self.SigmoidCE = SigmoidCrossEntropy()

        # initialize momentum
        self.w1_momentum = 0
        self.b1_momentum = 0
        self.w2_momentum = 0
        self.b2_momentum = 0

    # training function
    def train(self, x_batch, y_batch, learning_rate=0.005, momentum=0.8, l2_penalty=0.0):
        # feed forward
        inp = self.linearT1.forward(x_batch)
        act_out = self.relu.forward(inp)
        out = self.linearT2.forward(act_out)
        predictions, loss = self.SigmoidCE.forward(out, y_batch)

        # back-propagation
        loss_back = self.SigmoidCE.backward(y_batch, 1)
        dx2, dw2, db2 = self.linearT2.backward(loss_back)
        relu_back = self.relu.backward(dx2)
        dx1, dw1, db1 = self.linearT1.backward(relu_back)

        # update weights and bias
        self.w1_momentum = momentum * self.w1_momentum - learning_rate * dw1
        self.b1_momentum = momentum * self.b1_momentum - learning_rate * db1
        self.w2_momentum = momentum * self.w2_momentum - learning_rate * dw2
        self.b2_momentum = momentum * self.b2_momentum - learning_rate * db2

        self.w1 += self.w1_momentum
        self.b1 += self.b1_momentum
        self.w2 += self.w2_momentum
        self.b2 += self.b2_momentum

        return predictions, np.mean(loss)

    def evaluate(self, x_batch_val, y_batch_val):
	# INSERT CODE for testing the network
        inp = self.linearT1.forward(x_batch_val)
        relu_out = self.relu.forward(inp)
        out = self.linearT2.forward(relu_out)
        val_pred, val_loss = self.SigmoidCE.forward(out, y_batch_val)
        
        return val_pred, np.mean(val_loss)

if __name__ == '__main__':
    with open(r"cifar_2class_py2.p", "rb") as f:
    	u = pickle._Unpickler(f)    
    	u.encoding = 'latin1'    
    	data = u.load()
    
    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

    num_examples, input_dims = train_x.shape

    # normalize image data
    train_mean = np.mean(train_x, axis=0)
    train_x = (train_x - train_mean)/255
    test_mean = np.mean(test_x, axis=0)
    test_x = (test_x - test_mean)/255
    num_test_examples = test_x.shape[0]
    
    batch_size = 128
    num_epochs = 20
    lr = 1e-3
    hidd_units = 512
    # create network
    mlp = MLP(input_dims=input_dims, hidden_units=hidd_units)
    
    # initialize misc objects
    train_pred = np.zeros((num_examples, 1))
    val_pred = np.zeros((test_x.shape[0], 1))
    num_batches = num_examples//batch_size
    num_batches_val = num_test_examples//batch_size
    
    train_loss_set, train_accuracy_set, val_accuracy_set, val_loss_set = [], [], [], []
    
    # start training
    for epoch in range(num_epochs):
        train_total_loss = 0.0
        val_total_loss = 0.0
        # training batch
        for b in range(num_batches):
            train_x_batch = train_x[batch_size*b:batch_size*(b+1), :]
            train_y_batch = train_y[batch_size*b:batch_size*(b+1), :]
            train_pred[batch_size*b:batch_size*(b+1), :], train_loss = mlp.train(train_x_batch, train_y_batch, learning_rate = lr, momentum = 0.6)
            train_total_loss += train_loss
            
        # eval batch
        for b in range(num_batches_val):
            val_x_batch = test_x[batch_size*b:batch_size*(b+1), :]
            val_y_batch = test_y[batch_size*b:batch_size*(b+1), :]
            val_pred[batch_size*b:batch_size*(b+1), :], val_batch_loss = mlp.evaluate(val_x_batch, val_y_batch)
            val_total_loss += val_batch_loss
            
        # compute & store accuracy
        train_accuracy = np.mean(np.round(train_pred) == train_y)
        train_loss_set.append(train_total_loss)
        train_accuracy_set.append(train_accuracy)
        val_accuracy = np.mean(np.round(val_pred) == test_y)
        val_accuracy_set.append(val_accuracy)
        val_loss_set.append(val_total_loss)
        
        print("Epoch {}/{}  Train Acc: {:.2f}  Train Loss: {:.3f}  Val Acc: {:.2f}  Val Loss: {:.3f}".format(epoch+1, num_epochs, train_accuracy, train_total_loss, val_accuracy, val_total_loss))
