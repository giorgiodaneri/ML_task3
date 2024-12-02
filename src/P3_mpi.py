from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all but ERROR logs

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad
import sys

def add_noise_parallel(local_dataset, num_corrupted_pixels):
    """
    Add salt-and-pepper noise to the local dataset.
    """
    height, width = local_dataset.shape[1:3]
    for i in range(len(local_dataset)):
        for _ in range(num_corrupted_pixels):
            row, col = np.random.randint(0, height), np.random.randint(0, width)
            local_dataset[i, row, col] = np.random.choice([0, 1])  # Salt-and-pepper noise
    return local_dataset

def loss_fn(kernel, x, y_true):
    """
    Loss function: Mean Squared Error between the prediction and the true values.
    """
    y_pred = jsp.signal.convolve2d(x, kernel, mode='same', boundary='fill')
    return jnp.mean((y_pred - y_true) ** 2)

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Command-line arguments
num_of_images = int(sys.argv[1])  # Number of images
num_iterations = int(sys.argv[2])  # Number of training iterations
num_corrupted_pixels = 100  # Number of corrupted pixels per image

# Load dataset on rank 0
if rank == 0:
    print("INFO: Loading MNIST dataset...")
    (x_train, y_train), _ = mnist.load_data()
    random_indices = np.random.choice(len(x_train), size=num_of_images, replace=False)
    x_train = x_train[random_indices]  # Limit dataset size
    x_train = x_train.astype(np.float32) / 255.0  # Normalize
    y_true = x_train.copy()
    print("INFO: Dataset loaded and normalized.")
else:
    x_train, y_true = None, None

# Broadcast the total number of images
elements = comm.bcast(x_train.shape[0] if rank == 0 else None, root=0)

# Split dataset among ranks
local_data = None
local_targets = None
if rank == 0:
    print("INFO: Splitting dataset...")
    split_data = np.array_split(x_train, size)
    split_targets = np.array_split(y_true, size)
    for target_rank in range(1, size):
        comm.send(split_data[target_rank], dest=target_rank)
        comm.send(split_targets[target_rank], dest=target_rank)
    local_data = split_data[0]
    local_targets = split_targets[0]
else:
    local_data = comm.recv(source=0)
    local_targets = comm.recv(source=0)

# Add noise in parallel
local_data = add_noise_parallel(local_data, num_corrupted_pixels)

# Define kernel and gradient
if rank == 0:
    k = np.random.random((3, 3))
    print("INFO: Initializing kernel...")
all_k = comm.bcast(k if rank == 0 else None, root=0)
kernel = jnp.array(all_k)

# Gradient of the loss function
loss_grad = grad(loss_fn)

# Training loop
learning_rate = 0.001
losses = []
kernels = []

for iteration in range(num_iterations):
    local_gradients = jnp.zeros_like(kernel)
    local_loss = 0.0

    # Compute local gradients and loss
    for i in range(len(local_data)):
        x = jnp.array(local_data[i])
        y_true = jnp.array(local_targets[i])
        local_gradients += loss_grad(kernel, x, y_true)
        local_loss += loss_fn(kernel, x, y_true)
    
    # Average gradients and loss across ranks
    global_gradients = comm.allreduce(local_gradients, op=MPI.SUM) / size
    global_loss = comm.allreduce(local_loss, op=MPI.SUM) / size

    # Update kernel
    kernel -= learning_rate * global_gradients
    kernel = comm.bcast(kernel, root=0)

    if rank == 0:
        losses.append(global_loss)
        print(f"TRAINING: Epoch {iteration}, Loss: {global_loss}")

# Save results on rank 0
if rank == 0:
    np.savetxt("kernel.csv", kernel, delimiter=",")

    plt.figure()
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")

    print("INFO: Kernel and loss curve saved.")
