- **`q1.ipynb`**: Loads the Fashion-MNIST dataset and plots one sample image for each class, as required in Question 1.  

- **`nn.py`**: Implements a neural network class from scratch, supporting forward and backward propagation with multiple optimization algorithms.  

- **`4567.py`**: Handles Questions 4, 5, 6, and 7. It performs hyperparameter tuning using the `sweep` function, logs validation accuracy, and generates a validation accuracy vs. sweep plot. The best model is selected based on validation accuracy, and the confusion matrix for the test set is logged to Weights & Biases (WandB).