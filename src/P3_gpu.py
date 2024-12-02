import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax.numpy as jnp
from jax import grad
from jax.lax import conv_general_dilated
import tensorflow as tf
import time 
import sys
from jax.lib import xla_bridge

print("Number of available GPUs: ", xla_bridge.get_backend().platform)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_train[0]
y_true = x.copy()

# Add salt-and-pepper noise
num_corrupted_pixels = 100
for _ in range(num_corrupted_pixels):
    i, j = np.random.randint(0, x.shape[0]), np.random.randint(0, x.shape[1])
    x[i, j] = np.random.choice([0, 255])

# Normalize images
y_true = y_true.astype(np.float32) / 255.0
x = x.astype(np.float32) / 255.0

# Define loss function
def loss_fn(kernel, x, y_true):
    # use lax_conv_general_dilated to compute the convolution
    y_pred = conv_general_dilated(x.reshape(1, 1, x.shape[0], x.shape[1]), kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1]), (1, 1), 'SAME', (1, 1)).reshape(x.shape)
    return jnp.mean((y_pred - y_true) ** 2)  # Mean squared error

# Initialize kernel
kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])  # Random kernel for horizontal edge detection

# Gradient of the loss function w.r.t. the kernel
loss_grad = grad(loss_fn)

# Training loop
learning_rate = 0.01
# get number of iterations from the command line
num_iterations = int(sys.argv[1])

start = time.time()
losses = []
for i in range(num_iterations):
    gradients = loss_grad(kernel, x, y_true)
    kernel -= learning_rate * gradients  # Update kernel with gradient descent
    # Compute and store the loss
    current_loss = loss_fn(kernel, x, y_true)
    losses.append(current_loss)
    # print(f"Iteration {i}, Loss: {current_loss:.4f}")

# Display denoised image
y_denoised = conv_general_dilated(x.reshape(1, 1, x.shape[0], x.shape[1]), kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1]), (1, 1), 'SAME', (1, 1)).reshape(x.shape)

end = time.time()  
# write the time to a file in append mode
with open(f'../results/gpu_time_{num_iterations}.txt', 'a') as f:
    f.write(str(end-start) + '\n')
# write y_denoised to a file
np.save(f'../results/gpu_denoised_{num_iterations}.npy', y_denoised)
# write losses to a file
np.save(f'../results/gpu_losses_{num_iterations}.npy', losses)