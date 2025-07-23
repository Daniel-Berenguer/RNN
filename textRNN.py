import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.model import Model

with open("data/X_train", "rb") as file:
    X_train = np.load(file)

with open("data/Y_train", "rb") as file:
    Y_train = np.load(file)

with open("data/X_valid", "rb") as file:
    X_valid = np.load(file)

with open("data/Y_valid", "rb") as file:
    Y_valid = np.load(file)


# Shuffle indices
train_ix = np.arange(X_train.shape[0])
valid_ix = np.arange(X_valid.shape[0])
np.random.shuffle(train_ix)
np.random.shuffle(valid_ix)

# Apply shuffle
X_train = X_train[train_ix]
Y_train = Y_train[train_ix]
X_valid = X_valid[valid_ix]
Y_valid = Y_valid[valid_ix]

# Covert into torch tensors
X_train = torch.tensor(X_train, dtype=torch.int8)
X_valid = torch.tensor(X_valid, dtype=torch.int8)
Y_train = torch.tensor(Y_train, dtype=torch.int8)
Y_valid = torch.tensor(Y_valid, dtype=torch.int8)

# Num chars
N = torch.max(X_train).item() + 1

# HYPERPARAMETERS
BATCH_SIZE = 32
TRAIN_SIZE = X_train.shape[0]
EPOCHS = 2
HSIZE = 128


iterations_per_epoch = TRAIN_SIZE // BATCH_SIZE
iterations = iterations_per_epoch * EPOCHS
plt.style.use("ggplot")

# Create model
model = Model(num_chars=N, hSize=HSIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Iterations: ", iterations)

for i in range(80000):

    # Create batch
    ix = torch.randint(0, TRAIN_SIZE, (BATCH_SIZE,))
    X_batch = X_train[ix]
    Y_batch = Y_train[ix].long()

    # Forward pass
    logits = model.forward(X_batch)

    loss = F.cross_entropy(logits, Y_batch)

    if i == 0 or (i+1) % 100 == 0:
        print(f"EPOCH Nº: {(i+1) // iterations_per_epoch} (iteration: {i})")
        print("Loss: ", loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    with torch.no_grad():
        optimizer.step()


## Evaluate on test set
with torch.no_grad():
    logits_valid = model.forward(X_valid, train=False)
    Y_valid = Y_valid.long()
    loss_test =  F.cross_entropy(logits_valid, Y_valid)
    print(f"Test Loss: {loss_test.item()}")


with open("model_weights", "wb") as file:
    torch.save(model.state_dict(), file)
