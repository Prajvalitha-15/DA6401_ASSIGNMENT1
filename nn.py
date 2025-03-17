import numpy as np
from sklearn.metrics import accuracy_score
import wandb

class NeuralNetwork: 
    def __init__(self, input_dim, hidden_layers, output_dim): 
        self.input_layer = input_dim
        self.rest_layers = hidden_layers + [output_dim]
        self.layers = []
        self.weights = [] 
        self.biases = []
        
        # For momentum-based optimizers
        self.vel_w = []
        self.vel_b = []
        
        # For RMSProp and Adam 
        self.sq_w = []
        self.sq_b = []  
        
        # For Nadam
        self.mom_w = []
        self.mom_b = []

    def train(self, x_train, y_train, x_val = None, y_val = None, epochs = 5, 
              weight_decay = 0, learning_rate = 1e-3, optimizer = 'sgd', batch_size = 16, weight_initialisation = 'random', 
              activation_funtion = 'relu') :
        
        self.initialise_weights(weight_initialisation)

        for epoch in range(epochs) :
            indices = np.arange(x_train.shape[1])
            np.random.shuffle(indices)
            x_train, y_train = x_train[:,indices], y_train[:,indices]
            
            for i in range(0, x_train.shape[1], batch_size):
                x_batch = x_train[:,i:i+batch_size]
                y_batch = y_train[:,i:i+batch_size]
                self.forward(x_batch, activation_funtion)
                self.backprop(x_batch, y_batch, learning_rate, optimizer, activation_funtion)
            
            train_probs = self.forward(x_train, activation_funtion)
            train_loss = self.crossentropy(train_probs, y_train) + 0.5 * weight_decay * sum(np.sum(w ** 2) for w in self.weights)
            train_pred = np.argmax(train_probs, axis = 0)
            train_acc = accuracy_score(np.argmax(y_train, axis = 0), train_pred)

            wandb.log({
                "epoch" : epoch, 
                "train_loss" : train_loss, 
                "train_acc" : train_acc
            })

            if x_val is not None and y_val is not None :
                val_probs = self.forward(x_val, activation_funtion)
                val_loss = self.crossentropy(val_probs, y_val)
                val_pred = np.argmax(val_probs, axis = 0)
                val_acc = accuracy_score(np.argmax(y_val, axis = 0), val_pred)
        
            wandb.log({ 
                "val_loss" : val_loss, 
                "val_acc" : val_acc
            })

        wandb.log({"Accuracy" : val_acc})

    @staticmethod
    def crossentropy(y_pred, y_true) :
        eps = 1e-8 
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
    
    def backprop(self, x, y, learning_rate, optimizer, activation_funtion, weight_decay = 0,
                 beta1=0.9, beta2=0.999, epsilon=1e-8) :
        m = y.shape[1]
        deltas = [self.layers[-1] - y]
        f_derivative = self.sigmoid_der if activation_funtion == 'sigmoid' else self.tanh_der if activation_funtion == 'tanh' else self.relu_der

        for i in range(len(self.weights) - 1, 0, -1) :
            deltas.append(self.weights[i].T.dot(deltas[-1]) * f_derivative(self.layers[i]))
        deltas.reverse()

        for i in range(len(self.weights)) :
            dw = (self.layers[i].dot(deltas[i].T)).T / m + weight_decay * self.weights[i]
            db = np.mean(deltas[i], axis=1, keepdims=True)

            if optimizer == 'sgd':
                self.weights[i] -= learning_rate * dw
                self.biases[i] -= learning_rate * db
            
            elif optimizer == 'momentum':
                self.vel_w[i] = beta1 * self.vel_w[i] - learning_rate * dw
                self.vel_b[i] = beta1 * self.vel_b[i] - learning_rate * db
                self.weights[i] += self.vel_w[i]
                self.biases[i] += self.vel_b[i]
            
            elif optimizer == 'nag':
                prev_w = self.vel_w[i]
                prev_b = self.vel_b[i]
                self.vel_w[i] = beta1 * prev_w - learning_rate * dw
                self.vel_b[i] = beta1 * prev_b - learning_rate * db
                self.weights[i] += -beta1 * prev_w + (1 + beta1) * self.vel_w[i]
                self.biases[i] += -beta1 * prev_b + (1 + beta1) * self.vel_b[i]
            
            elif optimizer == 'rmsprop':
                self.sq_w[i] = beta2 * self.sq_w[i] + (1 - beta2) * (dw ** 2)
                self.sq_b[i] = beta2 * self.sq_b[i] + (1 - beta2) * (db ** 2)
                self.weights[i] -= learning_rate * dw / (np.sqrt(self.sq_w[i]) + epsilon)
                self.biases[i] -= learning_rate * db / (np.sqrt(self.sq_b[i]) + epsilon)
            
            elif optimizer == 'adam' or optimizer == 'nadam':
                self.vel_w[i] = beta1 * self.vel_w[i] + (1 - beta1) * dw
                self.vel_b[i] = beta1 * self.vel_b[i] + (1 - beta1) * db
                self.sq_w[i] = beta2 * self.sq_w[i] + (1 - beta2) * (dw ** 2)
                self.sq_b[i] = beta2 * self.sq_b[i] + (1 - beta2) * (db ** 2)
                if optimizer == 'nadam':
                    dw = beta1 * self.vel_w[i] + (1 - beta1) * dw
                    db = beta1 * self.vel_b[i] + (1 - beta1) * db
                self.weights[i] -= learning_rate * dw / (np.sqrt(self.sq_w[i]) + epsilon)
                self.biases[i] -= learning_rate * db / (np.sqrt(self.sq_b[i]) + epsilon)

    @staticmethod
    def sigmoid_der(x) :
        return x * (1 - x)
    
    @staticmethod 
    def tanh_der(x) :
        return 1 - np.power(x, 2)
    
    @staticmethod
    def relu_der(x) :
        return np.where(x > 0, 1, 0)

    def forward(self, x, f_name) :
        self.layers = [x]
        f = self.sigmoid if f_name == 'sigmoid' else self.tanh if f_name == 'tanh' else self.relu
 
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = f(np.dot(w, x) + b)
            self.layers.append(x)
        
        output = self.softmax(np.dot(self.weights[-1], x) + self.biases[-1])
        self.layers.append(output)
        return output
    
    @staticmethod
    def sigmoid(x) :
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x) :
        return np.tanh(x)
    
    @staticmethod
    def relu(x) :
        return np.maximum(x, 0)
    
    @staticmethod
    def softmax(x) :
        x_dash = x - np.max(x, axis= 0, keepdims= True)
        exp_x = np.exp(x_dash)
        return exp_x / np.sum(exp_x, axis=0, keepdims= True)

    def initialise_weights(self, method) :
        prev = self.input_layer
        
        for layer_size in self.rest_layers :
            scale = 1 / np.sqrt(prev) if method == 'xavier' else 1 

            self.weights.append(np.random.randn(layer_size, prev) * scale)
            self.biases.append(np.zeros((layer_size, 1)))

            self.vel_w.append(np.zeros((layer_size, prev)))
            self.sq_w.append(np.zeros((layer_size, prev)))
            self.mom_w.append(np.zeros((layer_size, prev)))

            self.vel_b.append(np.zeros((layer_size, 1)))
            self.sq_b.append(np.zeros((layer_size, 1)))
            self.mom_b.append(np.zeros((layer_size, 1)))

            prev = layer_size