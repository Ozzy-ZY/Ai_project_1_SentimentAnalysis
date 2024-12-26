# Deep Learning Sentiment Analysis Model: A Comprehensive Explanation

## Introduction to the System

This code implements a sophisticated deep learning model for sentiment analysis, specifically designed to analyze movie reviews from the IMDB dataset and determine whether they express positive or negative sentiment. The system combines several modern deep learning techniques, including word embeddings, bidirectional LSTMs, and various optimization strategies.

## Library Imports and Their Purpose

The code begins with importing necessary libraries:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
```

TensorFlow is the primary deep learning framework we're using. It provides both low-level operations and high-level APIs through Keras. The Sequential model allows us to build our neural network layer by layer, while the imported layers serve specific purposes:
- Embedding: Converts words to dense vectors
- LSTM: Processes sequences of data
- Dense: Creates fully connected neural network layers
- Dropout: Helps prevent overfitting
- Bidirectional: Allows layers to process sequences in both directions

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

These imports handle the training process:
- Adam is an adaptive optimization algorithm
- EarlyStopping prevents overfitting by monitoring training progress
- ReduceLROnPlateau adjusts the learning rate during training

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
```

These utilities handle data preprocessing:
- pad_sequences standardizes text length
- Tokenizer converts text to numbers
- train_test_split divides data into training and testing sets
- numpy handles numerical operations
- pandas manages data structures
- pickle saves objects for later use

## System Setup and GPU Detection

```python
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow Version: ", tf.__version__)
```

This code checks for available GPUs and displays the TensorFlow version. GPUs can dramatically speed up training by processing many calculations in parallel. Modern deep learning typically requires GPU acceleration for practical training times.

## Data Loading and Preprocessing

```python
data = pd.read_csv('IMDB Dataset.csv')
```

The dataset is loaded from a CSV file. The IMDB dataset contains movie reviews labeled as positive or negative, making it perfect for binary sentiment classification.

```python
texts = data['review'].values
labels = data['sentiment'].replace(['positive', 'negative'], [1, 0]).values
```

This code separates the reviews (features) from their sentiment labels (targets). The labels are converted from text ('positive'/'negative') to binary values (1/0) for machine learning purposes.

## Text Preprocessing Configuration

```python
max_vocab_size = 20000
max_sequence_length = 150
```

These parameters define crucial constraints:
- max_vocab_size limits our vocabulary to the 20,000 most frequent words, balancing between coverage and computational efficiency
- max_sequence_length standardizes all reviews to 150 words, either truncating longer reviews or padding shorter ones

## Tokenization Process

```python
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
```

The Tokenizer converts text to numbers by:
1. Creating a vocabulary from the training texts
2. Assigning unique indices to each word
3. Using "<OOV>" (Out Of Vocabulary) to handle unknown words

```python
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
```

This converts the text reviews into numerical sequences:
1. Each word is replaced by its vocabulary index
2. Sequences are padded to ensure uniform length
3. Padding is added at the end ('post') of shorter sequences

## Tokenizer Persistence

```python
with open('the_better_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

The tokenizer is saved to disk so it can be used later to process new reviews consistently. This is crucial for deployment, as any new text must be processed exactly like the training data.

## Data Splitting

```python
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)
```

The data is split into training (80%) and testing (20%) sets. The random_state ensures reproducibility by using the same random split every time.

## Word Embeddings

```python
embedding_dim = 50
```

This sets the dimensionality of our word vectors to match the pre-trained GloVe embeddings we'll use.

```python
embedding_index = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.array(values[1:], dtype='float32')
        embedding_index[word] = coefficients
```

This loads pre-trained GloVe word embeddings, which capture semantic relationships between words based on their usage patterns in a large corpus of text. Each word is represented by a 50-dimensional vector.

```python
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_vocab_size, embedding_dim))

for word, i in word_index.items():
    if i < max_vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
```

This creates an embedding matrix for our vocabulary:
1. Initialize an empty matrix of size (vocab_size × embedding_dim)
2. For each word in our vocabulary, find its pre-trained embedding
3. Copy the embedding vector into our matrix at the word's index

## Model Architecture

The model is built as a sequential stack of layers:

```python
model = Sequential([
    Embedding(max_vocab_size, embedding_dim,
              input_length=max_sequence_length,
              weights=[embedding_matrix],
              trainable=False),
```

The Embedding layer:
- Converts word indices to dense vectors
- Is initialized with our pre-trained embeddings
- Keeps embeddings fixed during training (trainable=False)
- Expects sequences of length max_sequence_length

```python
    Bidirectional(LSTM(128, return_sequences=True)),
```

First LSTM layer:
- Uses 128 units to learn sequence patterns
- Is bidirectional to process text in both directions
- Returns full sequences for the next layer
- Allows the model to capture long-range dependencies

```python
    Dropout(0.3),
```

First Dropout layer:
- Randomly deactivates 30% of connections during training
- Prevents overfitting by forcing redundant learning
- Makes the model more robust

```python
    Bidirectional(LSTM(64)),
```

Second LSTM layer:
- Uses 64 units (smaller than first layer)
- Creates a "funnel" architecture
- Outputs only the final state
- Further processes the sequence information

```python
    Dense(32, activation='relu'),
```

Dense layer:
- Fully connected layer with 32 neurons
- Uses ReLU activation for non-linearity
- Helps in final feature processing

```python
    Dropout(0.3),
```

Second Dropout layer:
- Provides additional regularization
- Prevents overfitting in the dense layers

```python
    Dense(1, activation='sigmoid')
])
```

Output layer:
- Single neuron for binary classification
- Sigmoid activation outputs probability between 0 and 1
- Above 0.5 indicates positive sentiment, below 0.5 negative

## Model Compilation

```python
model.compile(optimizer=Adam(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])
```

This configures the training process:
- Adam optimizer adapts learning rates for each parameter
- Binary crossentropy is the standard loss function for binary classification
- Accuracy metric provides intuitive performance measurement

## Training Callbacks

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

Early stopping prevents overfitting:
- Monitors validation loss
- Stops if no improvement for 3 epochs
- Restores the best weights found

```python
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2
)
```

Learning rate reduction:
- Monitors validation loss
- Halves learning rate if no improvement for 2 epochs
- Helps fine-tune the model

## Model Training

```python
epochs = 15
batch_size = 64
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr]
)
```

The training process:
- Runs for up to 15 epochs
- Processes data in batches of 64 samples
- Uses validation data to monitor progress
- Applies early stopping and learning rate reduction
- Returns training history for analysis

## Model Persistence and Evaluation

```python
model.save("A_better_model.h5")
```

Saves the entire model:
- Architecture
- Weights
- Training configuration
- Optimizer state

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

Final evaluation:
- Tests model on held-out test data
- Provides unbiased performance estimate
- Reports accuracy as percentage

## Practical Usage

This model can be used to analyze the sentiment of new movie reviews by:
1. Loading the saved tokenizer
2. Processing new text using the same tokenization steps
3. Loading the saved model
4. Getting predictions for the processed text

The system is designed to be both accurate and practical, with careful attention to preventing overfitting and ensuring consistent preprocessing for new data.
## DataSet Used 
[Kaggle.com](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
## Word Embedding Link
[Kaggle.com](https://www.kaggle.com/datasets/adityajn105/glove6b50d)
## References
- [Embedding](https://keras.io/api/layers/core_layers/embedding/)
- [Bidirectional LSTM "Long Short Term Memory"](https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/)
