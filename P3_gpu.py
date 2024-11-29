import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax.numpy as jnp
from jax import grad
from jax.lax import conv_general_dilated
import tensorflow as tf
import time 
import sys

# print the number of available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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
print("Time taken: ", end-start)

# write the time to a file in append mode
with open(f'results/gpu_time_{num_iterations}.txt', 'a') as f:
    f.write(str(end-start) + '\n')

# Write the kernel to a file in append mode to a new row
with open(f'results/kernel_{num_iterations}.txt', 'a') as f:
    flattened_kernel = kernel.flatten()
    f.write(','.join(map(str, flattened_kernel)) + '\n')

# Visualize results
plt.figure(figsize=(8, 6))

# Plot loss over iterations
plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

# Display original noisy image
plt.subplot(2, 2, 2)
plt.imshow(x, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')

# Display target clean image
plt.subplot(2, 2, 3)
plt.imshow(y_true, cmap='gray')
plt.title("Target (Clean Image)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(y_denoised, cmap='gray')
plt.title("Denoised Image")
plt.axis('off')

plt.tight_layout()
# write the plot to a file
plt.savefig('gpu_denoising.png')