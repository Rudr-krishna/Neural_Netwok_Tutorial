import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Neural network without activation functions
model_no_activation = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128),  # No activation function
    Dense(10, activation='softmax')
])

# Neural network with activation functions
model_with_activation = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),  # ReLU activation function
    Dense(10, activation='softmax')
])

# Compile both models
model_no_activation.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

model_with_activation.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

# Train both models
history_no_activation = model_no_activation.fit(
    x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1
)

history_with_activation = model_with_activation.fit(
    x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1
)

# Evaluate both models
loss_no_activation, acc_no_activation = model_no_activation.evaluate(x_test, y_test, verbose=0)
loss_with_activation, acc_with_activation = model_with_activation.evaluate(x_test, y_test, verbose=0)

print(f"Without Activation Functions: Test Accuracy = {acc_no_activation:.4f}")
print(f"With Activation Functions: Test Accuracy = {acc_with_activation:.4f}")

# Plot training loss comparison
plt.figure(figsize=(12, 6))
plt.plot(history_no_activation.history['loss'], label='Without Activation - Training Loss', linestyle='-', marker='o')
plt.plot(history_with_activation.history['loss'], label='With Activation - Training Loss', linestyle='-', marker='s')
# Validation Loss - Without Activation
plt.plot(history_no_activation.history['val_loss'], label='Without Activation - Validation Loss', linestyle='--', marker='^')
# Validation Loss - With Activation
plt.plot(history_with_activation.history['val_loss'], label='With Activation - Validation Loss', linestyle='--', marker='d')

plt.title("Training and Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()



