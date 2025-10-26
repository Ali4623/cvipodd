import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. Load dataset (MNIST: 28x28 grayscale images of digits)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images: 28x28 → 784
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# 2. Build a simple neural network
model = models.Sequential([
    layers.Input(shape=(784,)),           # Input layer
    layers.Dense(128, activation='relu'), # Hidden layer
    layers.Dense(64, activation='relu'),  # Hidden layer
    layers.Dense(10, activation='softmax')# Output layer (10 classes)
])

# 3. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 5. Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

