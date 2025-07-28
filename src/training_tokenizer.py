from tokenizer import Tokenizer

with open("data/ptb.train.txt") as file:
    text = file.read()

# Removes unwanted things from text
text = text.replace("\/", "|")
text = text.replace("<unk>", "")
text = text.replace("\*", "")
text = text.replace("$ N", "V")
text = text.replace("N ", "")
text = text.replace("N", "")


chunk = text[500:800]
print(chunk)

tokenizer = Tokenizer(1024)
tokenizer.train(text)
tokenizer.save()

enc = tokenizer.encode(chunk)
print(tokenizer.decode(enc))