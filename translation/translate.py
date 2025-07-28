from transformers import MarianMTModel, MarianTokenizer
import torch
import logging
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

logging.basicConfig(level=logging.INFO)

et_en_model_name = "Helsinki-NLP/opus-mt-et-en"
en_et_model_name = "Helsinki-NLP/opus-mt-en-et"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

et_en_tokenizer = MarianTokenizer.from_pretrained(et_en_model_name)
et_en_model = MarianMTModel.from_pretrained(et_en_model_name).to(torch_device)

en_et_tokenizer = MarianTokenizer.from_pretrained(en_et_model_name)
en_et_model = MarianMTModel.from_pretrained(en_et_model_name).to(torch_device)

def translate_et_to_en(text, max_length=512, chunk_overlap=100):
    logging.info(f"Translating from Estonian to English: {text[:100]}...")

    sentences = sent_tokenize(text)
    inputs = et_en_tokenizer(sentences, return_tensors=None, add_special_tokens=False)["input_ids"]
    translated_chunks = []

    logging.info(f"Input length: {len(inputs)} tokens. Chunking into segments of {max_length} with overlap of {chunk_overlap}.")
    for i in range(0, len(inputs), max_length - chunk_overlap):
        chunk_ids = inputs[i:i + max_length]
        chunk_text = et_en_tokenizer.decode(chunk_ids, skip_special_tokens=True)
        
        tokens = et_en_tokenizer([chunk_text], return_tensors="pt", truncation=True).to(torch_device)
        output = et_en_model.generate(**tokens, max_new_tokens=256, no_repeat_ngram_size=3, repetition_penalty=1.2)
        translated_text = et_en_tokenizer.decode(output[0], skip_special_tokens=False)
        
        translated_chunks.append(translated_text)

    logging.info(f"Translation complete. Output length: {len(translated_chunks)} chunks.")
    logging.debug(f"Output chunks: {translated_chunks}")

    return " ".join(translated_chunks)

def translate_en_to_et(text):
    tokens = en_et_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(torch_device)
    output = en_et_model.generate(**tokens)
    return en_et_tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG)

    sample_text_et = "Tere tulemast Riigiteataja juturobotisse!"
    sample_text_en = "Welcome to the Riigiteataja chatbot!"

    translated_to_en = translate_et_to_en(sample_text_et)
    print(f"Translated from Estonian to English: {sample_text_et} = {translated_to_en}")

    translated_to_et = translate_en_to_et(sample_text_en)
    print(f"Translated from English to Estonian: {sample_text_en} = {translated_to_et}")

    test_translation = "Kui Arno isaga koolimajja jõudis, olid tunnid juba alanud..."
    print(f"Test translation from Estonian to English: {test_translation} = {translate_et_to_en(test_translation)}")