from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax.numpy as jnp
from jax import grad

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

elements = 0
x_train = None
y_train = None
y_true = None

# Load the MNIST dataset
if rank == 0:
    print("Loading MNIST dataset...")
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[:11]  # Limit data for demonstration
    elements = len(x_train)
    y_true = x_train.copy()

# Broadcast total number of elements to all ranks
elements = comm.bcast(elements, root=0)

if rank == 0:
    print("MNIST dataset loaded.")
    print("Normalizing the dataset...")
    x_train = x_train.astype(np.float32) / 255.0

if rank == 0:
    print("Adding salt-and-pepper noise to the dataset...")
    num_corrupted_pixels = 100
    for i in range(len(x_train)):
        for _ in range(num_corrupted_pixels):
            row, col = np.random.randint(0, 28), np.random.randint(0, 28)
            x_train[i, row, col] = np.random.choice([0, 1])

# Calculate the number of local elements for each rank
local_size = elements // size + (1 if rank < elements % size else 0)

local_data = None
local_targets = None

if rank == 0:
    print("Splitting the dataset...")
    splitted_data = np.array_split(x_train, size)
    splitted_targets = np.array_split(y_true, size)
    print("Sending the dataset to various ranks...")
    for target_rank in range(1, size):
        comm.send(splitted_data[target_rank], dest=target_rank)
        comm.send(splitted_targets[target_rank], dest=target_rank)
    local_data = splitted_data[0]
    local_targets = splitted_targets[0]
else:
    local_data = comm.recv(source=0)
    local_targets = comm.recv(source=0)
    print(f"Rank {rank}: Received data.")

comm.barrier()

# Define convolution function
def convolution_2d(x, kernel):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    padded_x = jnp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output_data = jnp.zeros_like(x)

    for i in range(input_height):
        for j in range(input_width):
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            output_data = output_data.at[i, j].set(jnp.sum(region * kernel))
    return output_data

# Define loss function
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d(x, kernel)
    return jnp.mean((y_pred - y_true) ** 2)

if rank == 0:
    print("Initializing kernel...")
# Initialize kernel
kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])

# Gradient of the loss function
loss_grad = grad(loss_fn)

# Training loop
learning_rate = 0.01
num_iterations = 100
losses = []

if rank == 0:
    print("Training the model...")

for iteration in range(num_iterations):
    local_gradients = jnp.zeros_like(kernel)
    local_loss = 0.0
    for i in range(len(local_data)):
        x = jnp.array(local_data[i])
        y_true = jnp.array(local_targets[i])
        local_gradients += loss_grad(kernel, x, y_true)
        local_loss += loss_fn(kernel, x, y_true)
    
    # Average gradients and losses across all ranks
    global_gradients = comm.allreduce(local_gradients, op=MPI.SUM) / size
    global_loss = comm.allreduce(local_loss, op=MPI.SUM) / size
    
    # Update kernel
    kernel -= learning_rate * global_gradients
    
    if rank == 0:
        print(f"Iteration {iteration + 1}, Loss: {global_loss}")
        losses.append(global_loss)

# Save results on the root process
if rank == 0:
    plt.figure()
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
