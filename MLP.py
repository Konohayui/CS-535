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
    
    def backward(self, grad_output, 
                 learning_rate=0.0, 
                 momentum=0.0, l2_penalty=0.0):
	# DEFINE backward function
        dx = np.dot(grad_output, self.W.T)
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
        self.relu_output = np.maximum(x, 0)
        
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
        loss = y*np.log(self.predictions+1e-10) + (1-y)*np.log(1-self.predictions+1e-10)
        
        return self.predictions, loss
    
    def backward(self, y, grad_output, learning_rate = 0.0,
                momentum = 0.0, l2_penalty = 0.0):
		# DEFINE backward function
        
        return grad_output*(y - self.predictions)
    
# ADD other operations and data entries in SigmoidCrossEntropy if needed

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        
        # initialize weights and bias
        self.w1 = np.random.uniform(-1, 1, size = (input_dims, hidden_units))
        self.b1 = np.random.uniform(-1, 1, size = (1, hidden_units))
        self.w2 = np.random.uniform(-1, 1, size = (hidden_units, 1))
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
        prediction, loss = self.loss_func.forward(linear, y_batch) # predictions using sigmoid activation
        
        # backward computation
        loss_back = self.loss_func.backward(y_batch, 1)
        dx2, dw2, db2 = self.linearT2.backward(loss_back)
        relu_back = self.relu.backward(dx2)
        dx1, dw1, db1 = self.linearT1.backward(relu_back)
        
        # update momentum
        self.momentum_w1 = momentum*self.momentum_w1 - learning_rate*dw1
        self.momentum_b1 = momentum*self.momentum_b1 - learning_rate*db1
        self.momentum_w2 = momentum*self.momentum_w2 - learning_rate*dw2
        self.momentum_b2 = momentum*self.momentum_b2 - learning_rate*db2
        
        self.w1 += self.momentum_w1
        self.b1 += self.momentum_b1
        self.w2 += self.momentum_w2
        self.b2 += self.momentum_b2
        
        return prediction, loss
    
    def evaluate(self, x_batch_val, y_batch_val):
	# INSERT CODE for testing the network
        inp = self.linearT1.forward(x_batch_val)
        relu = self.relu.forward(inp)
        linear = self.linearT2.forward(relu)
        val_pred, val_loss = self.loss_func.forward(linear, y_batch_val)
        
        return val_pred, val_loss
    
# ADD other operations and data entries in MLP if needed
if __name__ == '__main__':
#     if sys.version_info[0] < 3:
#         data = pickle.load(open('cifar_2class_py2.p', 'rb'))
# 	else:
# 	    data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')
	
    with open(r"/kaggle/input/cifar2/cifar_2class_py2.p", "rb") as f:    
    	u = pickle._Unpickler(f)    
    	u.encoding = 'latin1'    
    	data = u.load()

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
	
    num_examples, input_dims = train_x.shape
    num_test_examples = test_x.shape[0]
    
    # normalize data
    train_x = train_x//255
    test_x = test_x//255
    
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epoches = 10
    batch_size = 1000
    hidden_units = 32
    mlp = MLP(input_dims, hidden_units)
    
    # initialize loss sets
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    num_batches_train = num_examples//batch_size
    num_batches_val = num_test_examples//batch_size
    
    for epoch in range(num_epoches):
        # initialize predictions
        train_pred = np.zeros((num_examples, 1))
        test_pred = np.zeros((test_x.shape[0], 1))
        
	# INSERT YOUR CODE FOR EACH EPOCH HERE
        total_loss = 0.0
        val_total_loss = 0.0
        
        for b in range(num_batches_train):
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
			# MAKE SURE TO UPDATE total_loss
            train_x_batch = train_x[batch_size*b:batch_size*(b+1), :]
            train_y_batch = train_y[batch_size*b:batch_size*(b+1), :]
            train_pred[batch_size*b:batch_size*(b+1), :], train_batch_loss = mlp.train(train_x_batch, train_y_batch)
            total_loss += np.sum(train_batch_loss)
        
        for b in range(num_batches_val):
            val_x_batch = test_x[batch_size*b:batch_size*(b+1), :]
            val_y_batch = test_y[batch_size*b:batch_size*(b+1), :]
            test_pred[batch_size*b:batch_size*(b+1), :], val_batch_loss = mlp.evaluate(val_x_batch, val_y_batch)
            val_total_loss += np.sum(val_batch_loss)
            
        train_accuracy = (train_y.flatten() - train_pred[train_pred >= 0.5])/num_examples
        test_accuracy = (test_y.flatten() - test_pred[test_pred >= 0.5])/num_test_examples
        
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        train_loss.append(total_loss)
        train_acc.append(train_accuracy)
        val_loss.append(val_total_loss)
        val_acc.append(test_accuracy)
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(train_loss,
              100. * train_accuracy))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(test_loss,
              100. * test_accuracy))
