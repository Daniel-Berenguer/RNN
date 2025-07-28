import torch
from tokenizer import Tokenizer

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

tokenizer = Tokenizer()
tokenizer.load()

print("A")

tokens = [torch.tensor(tokenizer.encode(text), dtype=torch.int16) for text in data]

N = tokenizer.N
SEQUENCE_LENGTH = 30

processed_data = []
filenames = ["train", "test", "valid"]

for j in range(len(tokens)):
    sequences = torch.zeros((len(tokens[j]) - SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=torch.int16)
    labels = torch.zeros((len(tokens[j]) - SEQUENCE_LENGTH), dtype=torch.int16)


    for i in range(SEQUENCE_LENGTH, tokens[j].shape[0]):
        labels[i - SEQUENCE_LENGTH] = tokens[j][i]
        sequences[i - SEQUENCE_LENGTH] = tokens[j][i-SEQUENCE_LENGTH:i]

    processed_data.append((sequences, labels))

    with open(f"data/X_{filenames[j]}_tokenized", "wb") as file:
        torch.save(sequences, file)

    with open(f"data/Y_{filenames[j]}_tokenized", "wb") as file:
        torch.save(labels, file
        
        )