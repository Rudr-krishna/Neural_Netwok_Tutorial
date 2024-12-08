import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the model with different activation functions
class SimpleNN(nn.Module):
    def __init__(self, activation_fn):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.activation = activation_fn

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)


# Training function
def train_model(activation_fn):
    model = SimpleNN(activation_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    # For tracking loss and accuracy
    loss_history = []
    accuracies = []
    gradient_history = []

    # Track gradients
    def record_gradients(model):
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.abs().mean().item())
        return np.mean(grads) if grads else 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss_history.append(running_loss / len(train_loader))
        accuracies.append(correct / total)

        # Record gradients at the end of each epoch
        gradient_history.append(record_gradients(model))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

    return loss_history, accuracies, gradient_history


# Activation functions to test
activation_functions = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
activation_names = ['ReLU', 'Sigmoid', 'Tanh']
loss_histories = {}
accuracies = {}
gradient_histories = {}

# Train and collect results for each activation function
for activation_fn, name in zip(activation_functions, activation_names):
    print(f"Training with {name} activation function...")
    loss_histories[name], accuracies[name], gradient_histories[name] = train_model(activation_fn)

# Plotting results

# 1. Training Loss by Activation Function
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^']  # Markers for each line
for i, (activation, history) in enumerate(loss_histories.items()):
    plt.plot(history, label=f"{activation} (Acc: {accuracies[activation][-1]:.4f})", marker=markers[i])
plt.title("Training Loss by Activation Function")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# 2. Gradient Flow Graph
plt.figure(figsize=(10, 6))
colormap = plt.cm.cividis  # Colorblind-friendly colormap
colors = colormap(np.linspace(0, 1, len(activation_names)))

for i, (activation, gradients) in enumerate(gradient_histories.items()):
    plt.plot(
        gradients,
        label=f"{activation} Gradient Flow",
        marker=markers[i],
        color=colors[i],
        linestyle='-',
    )
plt.title("Gradient Flow Across Epochs by Activation Function")
plt.xlabel("Epoch")
plt.ylabel("Mean Gradient Magnitude")
plt.legend()
plt.grid(True)
plt.show()

