import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.modelTokenized import Model

with open("data/X_train_tokenized", "rb") as file:
    X_train = torch.load(file)

with open("data/Y_train_tokenized", "rb") as file:
    Y_train = torch.load(file)

with open("data/X_valid_tokenized", "rb") as file:
    X_valid = torch.load(file)

with open("data/Y_valid_tokenized", "rb") as file:
    Y_valid = torch.load(file)


# Shuffle indices
train_ix = torch.randperm(X_train.shape[0])
valid_ix = torch.randperm(X_valid.shape[0])

# Apply shuffle
X_train = X_train[train_ix]
Y_train = Y_train[train_ix]
X_valid = X_valid[valid_ix]
Y_valid = Y_valid[valid_ix]


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
model = Model(num_tokens=N, hSize=HSIZE)
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
    print(f"Validation Loss: {loss_test.item()}")


with open("model_weights_tokenized", "wb") as file:
    torch.save(model.state_dict(), file)
