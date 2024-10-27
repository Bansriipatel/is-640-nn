from engine import Value
from nn import Neuron, Layer, MLP

x = Value(2.0)  
y = Value(1.0)  

mlp = MLP()  

# Optimization loop
for k in range(100):
    # Forward pass
    output = mlp.forward(x)  # Adjust based on your MLP implementation
    
    loss = (output - y) ** 2  # Example: Mean Squared Error

    # Backward pass
    loss.backward()

    for p in mlp.parameters():  
        p.data -= 0.01 * p.grad  

    # Print progress
    print(k, loss.data)

