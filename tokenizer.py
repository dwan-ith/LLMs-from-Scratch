#sample training data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print("Training Corpus:")
for doc in corpus:
    print(doc)

# Initialize vocabulary with unique characters
unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)

vocab = list(unique_chars)
vocab.sort()

# Add a special end-of-word token
end_of_word = "</w>"
vocab.append(end_of_word)

print("Initial Vocabulary:")
print(vocab)
print(f"Vocabulary Size: {len(vocab)}")

# Pre-tokenize the corpus 
word_splits = {}
for doc in corpus:
    words = doc.split(' ')
    for word in words:
        if word:
            char_list = list(word) + [end_of_word]
            word_tuple = tuple(char_list) #Lists cant be used as dictionary keys. But you can't change tuple once it's created-so they can be used as dictionary keys because of that.
            if word_tuple not in word_splits:
                 word_splits[word_tuple] = 0
            word_splits[word_tuple] += 1 #Word frequency

print("\nPre-tokenized Word Frequencies:")
print(word_splits)


# Helper Function: `get_pair_stats`
# This function takes the current word splits and calculates the frequency of each adjacent pair of symbols (across the entire corpus).
import collections
