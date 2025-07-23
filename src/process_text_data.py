import numpy as np

with open("data/ptb.train.txt", "r") as file:
    train = file.read()

with open("data/ptb.test.txt", "r") as file:
    test = file.read()

with open("data/ptb.valid.txt", "r") as file:
    valid = file.read()

data = [train, test, valid]

for i in range(len(data)):
    data[i] = data[i].replace("#", "")
    data[i] = data[i].replace("\/", "|")
    data[i] = data[i].replace("<unk>", "")
    data[i] = data[i].replace("\*", "")
    data[i] = data[i].replace("$ N", "V")
    data[i] = data[i].replace("N ", "")
    data[i] = data[i].replace("N", "")

chars = set()

for lines in data:
    for c in lines:
        chars.add(c)

chars_to_int = dict()
int_to_chars = []

for i, c in enumerate(chars):
    int_to_chars.append(c)
    chars_to_int[c] = i

import pickle

with open("int_to_chars", "wb") as file:
    pickle.dump(int_to_chars, file)

tokens = [[], [], []]

N = len(chars)

for i in range(len(data)):
    for c in data[i]:
        tokens[i].append(chars_to_int[c])

npy_tokens = [np.array(x, dtype=np.int8) for x in tokens]

print(npy_tokens[0].shape)

SEQUENCE_LENGTH = 30

processed_data = []
filenames = ["train", "test", "valid"]

for j in range(len(npy_tokens)):
    sequences = np.zeros((len(npy_tokens[j]) - SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=np.int8)
    labels = np.zeros((len(npy_tokens[j]) - SEQUENCE_LENGTH), dtype=np.int8)


    for i in range(SEQUENCE_LENGTH, npy_tokens[j].shape[0]):
        labels[i - SEQUENCE_LENGTH] = npy_tokens[j][i]
        sequences[i - SEQUENCE_LENGTH] = npy_tokens[j][i-SEQUENCE_LENGTH:i]

    processed_data.append((sequences, labels))

    with open(f"data/X_{filenames[j]}", "wb") as file:
        np.save(file, sequences)

    with open(f"data/Y_{filenames[j]}", "wb") as file:
        np.save(file, labels)