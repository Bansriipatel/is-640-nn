from engine import Value
from nn import Neuron, Layer, MLP

# Example data
x = Value(2.0)  # Input example
y = Value(1.0)  # Target output example

# Create your MLP (customize as needed)
mlp = MLP()  # Make sure to initialize with the right parameters

# Optimization loop
for k in range(100):
    # Forward pass
    output = mlp.forward(x)  # Adjust based on your MLP implementation
    
    # Compute loss (implement your loss function)
    loss = (output - y) ** 2  # Example: Mean Squared Error

    # Backward pass
    loss.backward()

    # Update weights (you may need to implement an optimizer)
    for p in mlp.parameters():  # Make sure your MLP has a method to get parameters
        p.data -= 0.01 * p.grad  # Simple SGD update

    # Print progress
    print(k, loss.data)

