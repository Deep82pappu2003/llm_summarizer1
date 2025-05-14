from fastapi import FastAPI, UploadFile, File
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

app = FastAPI()

# Load BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Simple extractive summary (just first 3 sentences)
def extractive_summary(text: str, num_sentences: int = 3) -> str:
    sentences = text.split('. ')
    return '. '.join(sentences[:num_sentences])

# Abstractive summary using BART
def abstractive_summary(text: str) -> str:
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.post("/summarize/")
async def summarize_text(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")

    # Step 1: Extractive summary
    extractive = extractive_summary(text)

    # Step 2: Abstractive summary
    final_summary = abstractive_summary(extractive)

    return {"summary": final_summary}
