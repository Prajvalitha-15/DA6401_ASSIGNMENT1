from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from nn import NeuralNetwork
import wandb 

# Loading the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# One-hot encode labels
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Split validation data
val_split = int(0.1 * x_train.shape[0])

# Create validation data
x_val, y_val = x_train[:val_split], y_train[:val_split]
x_train, y_train = x_train[val_split:], y_train[val_split:]

x_train = x_train.T
y_train = y_train.T
x_val = x_val.T
y_val = y_val.T
x_test = x_test.T
y_test = y_test.T

sweep_config = {
    "name": "Neural Network's hyperparameter search",
    "metric": {
        "name": "Accuracy",
        "goal": "maximize"
    },
    "method": "random",
    "parameters": {
        "epochs": {
            "values": [5, 10]
            },
        "hiddenLayers": {
            "values": [3, 4, 5]
            },
        "hiddenLayerSize": {
            "values": [32, 64, 128]
            },
        "weightDecay": {
            "values": [0, 0.0005, 0.5]
            },
        "learningRate": {
            "values": [1e-3, 1e-4]
            },
        "optimizer": {
            "values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]
            },
        "batchSize": {
            "values": [16, 32, 64]
            },
        "weightInitialisation": {
            "values": ["random", "xavier"]
            },
        "activationFunction": {
            "values": ["tanh", "relu", "sigmoid"]
            }
    }
}


def sweep_hyperparameters() :
    
    default = {
        'epochs' : 5, 
        'hiddenLayers' : 3, 
        'hiddenLayerSize' : 32, 
        'weightDecay' : 0, 
        'learningRate' : 1e-3, 
        'optimizer' : 'sgd', 
        'batchSize' : 16, 
        'weightInitialisation' : 'random', 
        'activationFunction' : 'tanh'
    }

    wandb.init(project= "assignment1", entity= "da6401-assignments")
    wandb.init(config= default)

    config = wandb.config 

    epochs = config.epochs  
    hiddenLayers = config.hiddenLayers 
    hiddenLayerSize = config.hiddenLayerSize 
    weightDecay = config.weightDecay 
    learningRate = config.learningRate 
    optimizer = config.optimizer 
    batchSize = config.batchSize 
    weightInitialisation = config.weightInitialisation 
    activationFunction = config.activationFunction

    model = NeuralNetwork(x_train.shape[0], [hiddenLayerSize] * hiddenLayers, y_train.shape[0])

    wandb.run.name = '#'.join(map(str, (epochs, hiddenLayers, hiddenLayerSize, weightDecay, 
                                        learningRate, optimizer, batchSize, weightInitialisation, activationFunction)))
    
    model.train(x_train, y_train, x_val, y_val, epochs, weightDecay, learningRate, optimizer, batchSize, weightInitialisation, activationFunction)

    wandb.run.save()

    return model 

sweepId = wandb.sweep(sweep_config, entity="da6401-assignments", project="assignment1")
wandb.agent(sweepId, sweep_hyperparameters, count=5)
wandb.finish()

api = wandb.Api()
runs = api.runs("da6401-assignments/assignment1")
best_run = min(runs, key=lambda run: run.summary.get("val_loss", float("inf")))

bestrunname = best_run.name 

values = bestrunname.split("#")

epochs = int(values[0])
hiddenLayers = int(values[1])
hiddenLayerSize = int(values[2])
weightDecay = float(values[3])
learningRate = float(values[4])
optimizer = values[5]
batchSize = int(values[6])
weightInitialisation = values[7]
activationFunction = values[8]

print("Best Configuration:\n")
print(f"Epochs: {epochs}")
print(f"Hidden Layers: {hiddenLayers}")
print(f"Hidden Layer Size: {hiddenLayerSize}")
print(f"Weight Decay: {weightDecay}")
print(f"Learning Rate: {learningRate}")
print(f"Optimizer: {optimizer}")
print(f"Batch Size: {batchSize}")
print(f"Weight Initialization: {weightInitialisation}")
print(f"Activation Function: {activationFunction}")

bestmodel = NeuralNetwork(x_train.shape[0], [hiddenLayerSize] * hiddenLayers, y_train.shape[0])

bestmodel.train(x_train, y_train, x_val, y_val, epochs, weightDecay, learningRate, optimizer, batchSize, weightInitialisation, activationFunction)



