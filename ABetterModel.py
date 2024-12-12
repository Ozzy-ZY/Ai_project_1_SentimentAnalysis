import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow Version: ", tf.__version__)

# Load dataset
data = pd.read_csv('IMDB Dataset.csv')

# Preprocess data
texts = data['review'].values
labels = data['sentiment'].replace(['positive', 'negative'], [1, 0]).values

# Tokenization and padding
max_vocab_size = 20000
max_sequence_length = 150

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Save the tokenizer
with open('the_better_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Load pre-trained embeddings (e.g., GloVe)
# Set embedding dimensions to match GloVe file
embedding_dim = 50  # Matches glove.6B.50d.txt dimensions

# Load pre-trained GloVe embeddings
embedding_index = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.array(values[1:], dtype='float32')  # GloVe vector
        embedding_index[word] = coefficients

# Create embedding matrix
word_index = tokenizer.word_index  # Vocabulary from the tokenizer
embedding_matrix = np.zeros((max_vocab_size, embedding_dim))  # Initialize matrix

for word, i in word_index.items():
    if i < max_vocab_size:  # Ensure index is within vocabulary size
        embedding_vector = embedding_index.get(word)  # Look up GloVe embedding
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Assign to matrix

# Use the embedding matrix in the Embedding layer
model = Sequential([
    Embedding(max_vocab_size, embedding_dim, input_length=max_sequence_length,
              weights=[embedding_matrix], trainable=False),  # Freeze pre-trained embeddings
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model with AdamW optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Train the model
epochs = 15
batch_size = 64
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])

# Save the trained model
model.save("A_better_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
