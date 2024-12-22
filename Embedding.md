# Understanding Word Embeddings in Sentiment Analysis

## Introduction to Word Embeddings

Word embeddings are a fundamental concept in natural language processing that allow us to convert words into meaningful numerical representations. In our sentiment analysis model, we use pre-trained GloVe (Global Vectors for Word Representation) embeddings to capture the semantic relationships between words.

## The Code Implementation

Let's examine the core components of our word embedding implementation:

```python
embedding_dim = 50  # Sets the dimensionality of our word vectors

embedding_index = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.array(values[1:], dtype='float32')
        embedding_index[word] = coefficients
```

## Understanding the Dimensional Space

Think of words as points in a 50-dimensional space. Just as we use two numbers (latitude and longitude) to locate a point on a map, we represent each word using 50 numbers. These numbers aren't randomly assigned – they're carefully calculated values that position similar words close together in this high-dimensional space.

### Example of Word Relationships
- The word "good" appears near "great" and "excellent"
- "Movie" is positioned close to "film" and "cinema"
- Similar concepts cluster together in this 50-dimensional space

## The GloVe File Format

Each line in the GloVe file contains a word followed by its 50-dimensional vector. Here's an example format:

```
movie 0.123 -0.456 0.789 ... (47 more numbers)
```

### Processing Steps
1. Split each line into the word and its vector values
2. Convert string numbers to floating-point numbers
3. Store in a dictionary with words as keys and vectors as values

## Creating the Embedding Matrix

The next step involves creating a matrix that maps our vocabulary to these vector representations:

```python
word_index = tokenizer.word_index  # Dictionary mapping words to indices
embedding_matrix = np.zeros((max_vocab_size, embedding_dim))

for word, i in word_index.items():
    if i < max_vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
```

### Matrix Structure
- Rows: 20,000 (one for each word in vocabulary)
- Columns: 50 (one for each dimension of word vectors)
- Each row contains the pre-trained vector for the corresponding word

### Example
If "movie" has index 17 in our vocabulary, its 50-dimensional vector is stored in row 17 of the embedding_matrix.

## Implementation in the Model

The embedding layer is created with the following code:

```python
Embedding(max_vocab_size, embedding_dim,
          input_length=max_sequence_length,
          weights=[embedding_matrix],
          trainable=False)
```

### Layer Functionality
1. Accepts word indices as input
2. Looks up corresponding rows in embedding matrix
3. Returns 50-dimensional vectors
4. trainable=False preserves pre-trained relationships

## Why This Approach Is Powerful

### Semantic Understanding
- Words with similar meanings have similar vectors
- Relationships between words are preserved numerically
- Model can generalize to new word combinations

### Benefits for Sentiment Analysis
1. Rich Semantic Information
   - Each word brings its contextual relationships
   - Similar words have similar representations

2. Improved Generalization
   - Model can understand words in new contexts
   - Rare words have meaningful representations if in GloVe data

3. Efficient Learning
   - Pre-trained vectors capture existing language knowledge
   - Model can focus on learning sentiment patterns

## Processing Example

When analyzing a review like "This movie is good":
1. Each word is converted to its index
2. Each index retrieves a 50-dimensional vector
3. These vectors contain rich semantic information
4. The model processes these vectors to determine sentiment

## Conclusion

Word embeddings form the foundation of our sentiment analysis model by converting text into meaningful numerical representations. By using pre-trained GloVe embeddings, we leverage existing knowledge about word relationships and meanings, allowing our model to better understand and analyze sentiment in text.

This approach combines the power of large-scale pre-training with the specific requirements of sentiment analysis, creating a robust system for understanding and classifying text sentiment.