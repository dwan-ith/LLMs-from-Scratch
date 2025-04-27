#STEP1
#Sample training data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print("Training Corpus:")
for doc in corpus:
    print(doc)

#STEP2
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

def get_pair_stats(splits):
    """Counts the frequency of adjacent pairs in the word splits."""

pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq # Add the frequency of the word to the pair count
    return pair_counts

# Helper Function: `merge_pair`
# This function takes a specific pair that we want to combine and outputs the current `splits`

def merge_pair(pair_to_merge, splits):
    """Merges the specified pair in the word splits."""
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        while i < len(symbols):
            # If the current and next symbol match the pair to merge
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                new_symbols.append(merged_token)
                i += 2 # Skip the next symbol
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[tuple(new_symbols)] = freq # Use the updated symbol list as the key
    return new_splits

#STEP3
# Iterative BPE Merging Loop
# Now we perform the core BPE training. We'll loop for a fixed number of merges
# --- BPE Training Loop Initialization ---
num_merges = 15
merges = {}
current_splits = word_splits.copy() # Start with initial word splits
print("\n--- Starting BPE Merges ---")
print(f"Initial Splits: {current_splits}")
print("-" * 30)

for i in range(num_merges):
    print(f"\nMerge Iteration {i+1}/{num_merges}")

    # 1. Calculate Pair Frequencies
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge.")
        break
    # Optional: Print top 5 pairs for inspection
    sorted_pairs = sorted(pair_stats.items(), key=lambda item: item[1], reverse=True)
    print(f"Top 5 Pair Frequencies: {sorted_pairs[:5]}")
    # 2. Find Best Pair
    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Found Best Pair: {best_pair} with Frequency: {best_freq}")
    # 3. Merge the Best Pair
    current_splits = merge_pair(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merging {best_pair} into '{new_token}'")
    print(f"Splits after merge: {current_splits}")

    # 4. Update Vocabulary
    vocab.append(new_token)
    print(f"Updated Vocabulary: {vocab}")

    # 5. Store Merge Rule
    merges[best_pair] = new_token
    print(f"Updated Merges: {merges}")

    print("-" * 30)

#STEP4
#Review Final Results
# --- BPE Merges Complete ---
print("\n--- BPE Merges Complete ---")
print(f"Final Vocabulary Size: {len(vocab)}")
print("\nLearned Merges (Pair -> New Token):")
# Pretty print merges
for pair, token in merges.items():
    print(f"{pair} -> '{token}'")

print("\nFinal Word Splits after all merges:")
print(current_splits)

print("\nFinal Vocabulary (sorted):")
# Sort for consistent viewing
final_vocab_sorted = sorted(list(set(vocab))) # Use set to remove potential duplicates if any step introduced them
print(final_vocab_sorted)
