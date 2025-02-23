import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HF_API_KEY")

login(api_key)

model_name = "hldo/my-fake-news-model"


# Load the fine-tuned model and tokenizer

classifier = pipeline("text-classification", model=model_name)
# Initialize the app
app = FastAPI()


# Define request format
class TextRequest(BaseModel):
    inputs: str


# Endpoint to get predictions
@app.post("/predict")
async def predict(request: TextRequest):
    text = request.inputs
    print(text)
    result = classifier(text)
    return result
