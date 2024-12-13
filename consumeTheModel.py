import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model saved in .h5 format
model = load_model("A_better_model.h5")

# Load the tokenizer
with open('the_better_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define constants
max_vocab_size = 20000  # Vocabulary size
max_sequence_length = 150  # Sequence length for padding

def predict_sentiment(single_text):
    sequence = tokenizer.texts_to_sequences([single_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')

    prediction = model.predict(padded_sequence)
    sentiment_label = 'positive' if prediction[0][0] >= 0.5 else 'negative'
    return sentiment_label

if __name__ == "__main__":
    print("Enter a review to analyze its sentiment (type 'exit' to quit):")
    while True:
        text = input("Enter a review: ")
        if text.lower() == 'exit':
            print("Exiting sentiment analysis. Goodbye!")
            break
        sentiment = predict_sentiment(text)
        print(f"Predicted Sentiment: {sentiment}\n")
