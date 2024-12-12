import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle

# Load the model saved in .h5 format
model = load_model("sentiment_analysis_model.h5")
# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Example input text
new_texts = ["I loved the movie, it was fantastic!", "the movie is not good"]

# Convert the text to sequences using the loaded tokenizer
new_sequences = tokenizer.texts_to_sequences(new_texts)
max_vocab_size = 20000  # Increase vocabulary size
max_sequence_length = 150  # Increase sequence length for better context
# Pad the sequences to ensure uniform length
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length, padding='post')

# Make predictions
predictions = model.predict(new_padded_sequences)
print(predictions)
# Convert probabilities to binary sentiment labels
predicted_labels = ['positive' if p >= 0.5 else 'negative' for p in predictions]

# Print results
for text, label in zip(new_texts, predicted_labels):
    print(f"Text: {text}\nPredicted Sentiment: {label}\n")

