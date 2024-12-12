from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import uvicorn

# Load the model and tokenizer
MODEL_PATH = "A_better_model.h5"
TOKENIZER_PATH = "the_better_tokenizer.pickle"

try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Create FastAPI app
app = FastAPI()

# Request body model
class SentimentRequest(BaseModel):
    text: str

# Preprocessing function
def preprocess_text(text: str):
    # Tokenize and pad the text
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=5000, padding='post')
    return padded

# Predict endpoint
@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    try:
        processed_text = preprocess_text(request.text)
        prediction = model.predict(processed_text)
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        return {"text": request.text, "sentiment": sentiment, "confidence": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Run the server
if __name__ == "__main__":
    uvicorn.run(app)
