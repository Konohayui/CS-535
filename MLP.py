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
        linear_output = np.dot(self.W.T, self.x) + self.b
        
        return linear_output
    
    def backward(self, grad_output, 
                 learning_rate=0.0, 
                 momentum=0.0, l2_penalty=0.0):
	# DEFINE backward function
        dx = np.dot(self.W.T, grad_output)
        dw = np.dot(self.x.T, grad_output)
        db = np.sum(grad_output, axis = 0)
        
        return dx, dw, db
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    
    def __init__(self):
        self.relu_output = None
    
    def forward(self, x):
	# DEFINE forward function
        self.relu_output = np.maximum(self.x, 0)
        
        return self.relu_output
    
    def backward(self, grad_output, 
                 learning_rate=0.0, momentum=0.0, 
                 l2_penalty=0.0):
    # DEFINE backward function        
        self.relu_output[self.relu_output > 0] = 1
        
        return np.multiply(self.relu_output, grad_output)
    
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self):
        self.predictions = None
        
	def forward(self, x, y):
        # DEFINE forward function
        self.predictions = 1.0/(1.0 + np.exp(-x))
        loss = np.multiply(x, np.log(self.predictions))+np.multiply(1-x, np.log(self.predictions))
        
        return self.predictions, loss
    
	def backward(self, y, grad_output, learning_rate=0.0,
              momentum=0.0, l2_penalty=0.0):
		# DEFINE backward function
        
        return grad_output*(y - self.sigmoid)
    
# ADD other operations and data entries in SigmoidCrossEntropy if needed

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        
        # initialize weights and bias
        self.w1 = np.random.uniform(-1, 1, size = (hidden_units, input_dims))
        self.b1 = np.random.uniform(-1, 1, size = (1, 1))
        self.w2 = np.random.uniform(-1, 1, size = (1, hidden_units))
        self.b2 = np.random.uniform(-1, 1, size = (1, 1))
        
        # initialize layers
        self.linearT1 = LinearTransform(self.w1, self.b1)
        self.relu = ReLU()
        self.linearT2 = LinearTransform(self.w2, self.b2)
        self.loss_func = SigmoidCrossEntropy()
        
        # initialize momentum
        self.momentum_w1 = 0
        self.momentum_b1 = 0
        self.momentum_w2 = 0
        self.momentum_b2 = 0
        
    def train(self, x_batch, y_batch, learning_rate=1e-3, momentum=0, l2_penalty=0):
	# INSERT CODE for training the network
        # forward computation
        inp = self.linearT1.forward(x_batch) # first input linear function
        relu = self.relu.forward(inp) # relu activation
        linear = self.linearT2.forward(relu) # second linear function
        prediction, loss = self.loss_func.forward(linear) # predictions using sigmoid activation
        
        # backward computation
        loss_back = self.loss_func.backward(y_batch, 1)
        dx2, dw2, db2 = self.linearT2.backward(loss_back)
        relu_back = self.relu.back(dx2)
        dx1, dw1, db1 = self.linearT1(relu_back)
        
        # update momentum
        self.momentum_w1 = momentum * self.momentum_w1 - learning_rate * dw1
        self.momentum_b1 = momentum * self.momentum_b1 - learning_rate * db1
        self.momentum_w2 = momentum * self.momentum_w2 - learning_rate * dw2
        self.momentum_b2 = momentum * self.momentum_b2 - learning_rate * db2
        
        self.w1 += self.w1
        self.b1 += self.b1
        self.w2 += self.w2
        self.b2 += self.b2
        
        num_examples = len(y_batch)
        accuracy = (prediction[prediction >= 0.5] - y_batch)/num_example
        
        return accuracy, loss
    
    def evaluate(self, x_batch_val, y_batch_val):
	# INSERT CODE for testing the network
        num_example = len(y)
        inp = self.linearT1.forward(x_batch_val)
        relu = self.relu.forward(inp)
        linear = self.linearT2.forward(relu)
        prediction, val_loss = self.loss_func(linear)
        
        val_accuracy = (prediction[prediction >= 0.5] - y_batch_val)/num_example
        
        return val_accuracy, val_loss
    
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':
    if sys.version_info[0] < 3:
	data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
	data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
	
    num_examples, input_dims = train_x.shape
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
	num_epochs = 10
	num_batches = 1000
    mlp = MLP(input_dims, hidden_units)

    # initialize loss sets
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    
    for epoch in xrange(num_epochs):
        batch_size = num_examples//num_batches
        
	# INSERT YOUR CODE FOR EACH EPOCH HERE
        total_loss = 0.0
        
        for b in xrange(num_batches):
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
			# MAKE SURE TO UPDATE total_loss
            train_x_batch = train_x[batch_size*b:batch_size*(b+1), :]
            train_y_batch = train_y[batch_size*b:batch_size*(b+1), :]
            batch_acc, batch_loss = MLP.train(train_x_batch, train_y_batch)
            total_loss += batch_loss
            
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
            sys.stdout.flush()
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        train_loss.append(total_loss)
        train_acc.append()
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(train_loss,
              100. * train_accuracy,))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(test_loss,
              100. * test_accuracy,))
