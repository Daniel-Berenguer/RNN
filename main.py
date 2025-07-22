import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import GRU_Layer

with open("data/sp500_sequences.npy", "rb") as file:
    X = np.load(file)

with open("data/sp500_labels.npy", "rb") as file:
    Y = np.load(file)


# Convert from labels from prices, to boolean saying if price went up or down next day
prices = Y
Y = np.where(X[:, -1, 3, np.newaxis] < Y, 1, 0)


# Shuffle indices
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Apply shuffle
X_shuffled = X[indices]
Y_shuffled = Y[indices]

# Split index
split_idx = int(0.8 * len(X))

# Train/Test split
X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
Y_train, Y_test = Y_shuffled[:split_idx], Y_shuffled[split_idx:]

# Covert into torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)



# HYPERPARAMETERS
BATCH_SIZE = 32
TRAIN_SIZE = X_train.shape[0]
EPOCHS = 30
lr = 0.04

HSIZE = 64

iterations_per_epoch = TRAIN_SIZE // BATCH_SIZE
iterations = iterations_per_epoch * EPOCHS
plt.style.use("ggplot")

# Create model
model = GRU_Layer(nin=X_train.shape[-1], hSize=HSIZE, bN=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
show=False

for i in range(iterations):
    # Create batch
    ix = torch.randint(0, TRAIN_SIZE, (BATCH_SIZE,))
    X_batch = X_train[ix]
    Y_batch = Y_train[ix]

    # Forward pass

    if i == 999:
        show = True

    out = torch.sigmoid(model.forward(X_batch))

    loss = torch.nn.functional.binary_cross_entropy(out, Y_batch)

    if i == 0 or (i+1) % iterations_per_epoch == 0:
        print("EPOCH Nº: ", (i+1) // iterations_per_epoch)
        print("Loss: ", loss.item())

    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    with torch.no_grad():
        optimizer.step()


## Evaluate on test set
with torch.no_grad():
    out_test = torch.sigmoid(model.forward(X_test, train=False))
    loss_test =  torch.nn.functional.binary_cross_entropy(out_test, Y_test)
    print(f"Test Loss: {loss_test.item()}")


with open("model_file", "wb") as file:
    torch.save(model, file)
