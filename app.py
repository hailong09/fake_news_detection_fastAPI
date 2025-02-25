from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
# Initialize the app
app = FastAPI()

# Load Hugging Face model and tokenizer (TensorFlow version)
model_name = "hldo/my-fake-news-model"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define request format


class TextRequest(BaseModel):
    inputs: str


# Endpoint to get predictions
@app.post("/predict")
async def predict(request: TextRequest):
    text = request.inputs
    # Tokenize the input text (TensorFlow tensors)
    inputs = tokenizer(text, return_tensors="tf",
                       truncation=True, padding=True, max_length=512)
    # Get model predictions (TensorFlow)
    outputs = model(**inputs)
    logits = outputs.logits
    # Use TensorFlow's argmax
    predicted_class = tf.argmax(logits, axis=-1).numpy().item()
    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
