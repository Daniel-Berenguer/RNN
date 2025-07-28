import pickle

class Tokenizer:
    def __init__(self, N=0):
        self.chars_to_tokens = dict()
        self.tokens_to_chars = []
        self.merges = dict()
        self.N = N

    def save(self):
        with open("tokens", "wb") as file:
            values = (self.chars_to_tokens, self.tokens_to_chars,
                      self.merges, self.N)
            pickle.dump(values, file)
    
    def load(self):
        with open("tokens", "rb") as file:
            self.chars_to_tokens, self.tokens_to_chars, self.merges, self.N = pickle.load(file)
        


    def get_stats(self, tokenized_text):
        # returns dict with token pair -> count
        counts = dict()

        for pair in zip(tokenized_text, tokenized_text[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def merge(self, tokenized_text, pair, index):
        # Updates the tokenized text to include the new merged token
        i = 0
        new_tokens = []
        
        while i < len(tokenized_text):
            if (i < len(tokenized_text) - 1) and (tokenized_text[i] == pair[0]) and (tokenized_text[i+1] == pair[1]):
                new_tokens.append(index)
                i += 2

            else:
                new_tokens.append(tokenized_text[i])
                i += 1

        return new_tokens

    def train(self, text):
        # Resets data
        self.chars_to_tokens = dict()
        self.tokens_to_chars = []
        self.merges = dict()
        
        # First gets initial apparitions and tokenizes the text
        tokenized_text = []
        for c in text:
            if c not in self.chars_to_tokens:
                self.chars_to_tokens[c] = len(self.tokens_to_chars)
                self.tokens_to_chars.append(c)
            tokenized_text.append(self.chars_to_tokens[c])

        # Generates new token until we have N tokens
        while len(self.tokens_to_chars) < self.N:
            counts = self.get_stats(tokenized_text)
            pair = max(counts, key=counts.get)
            index = len(self.tokens_to_chars)
            chars = self.tokens_to_chars[pair[0]] + self.tokens_to_chars[pair[1]]

            self.tokens_to_chars.append(chars)
            self.chars_to_tokens[chars] = index

            self.merges[pair] = index
            tokenized_text = self.merge(tokenized_text, pair, index)
            print(len(self.tokens_to_chars))

        print(self.tokens_to_chars)

    def encode(self, text):
        # Tokenize just characters
        tokenized_text = []
        for c in text:
            tokenized_text.append(self.chars_to_tokens[c])

        # Deals with merges
        while len(tokenized_text) >= 2:
            counts = self.get_stats(tokenized_text)
            pair = min(counts, key= lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                # No merges left to make, ended
                break
            index = self.merges[pair]
            print(index)
            tokenized_text = self.merge(tokenized_text, pair, index)

        text = ""

        for t in tokenized_text:
            text += self.tokens_to_chars[t] + "|"

        return tokenized_text
    

    def decode(self, tokenized_text):
        text = ""
        for t in tokenized_text:
            text += self.tokens_to_chars[t]

        return text
