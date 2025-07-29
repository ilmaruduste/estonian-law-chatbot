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
model_cache = {}

et_en_tokenizer = MarianTokenizer.from_pretrained(et_en_model_name)
et_en_model = MarianMTModel.from_pretrained(et_en_model_name).to(torch_device)

en_et_tokenizer = MarianTokenizer.from_pretrained(en_et_model_name)
en_et_model = MarianMTModel.from_pretrained(en_et_model_name).to(torch_device)

import re
from nltk.tokenize import sent_tokenize


def load_model(source_lang: str, target_lang: str):
    model_key = f"{source_lang}-{target_lang}"
    if model_key not in model_cache:
        model_name = f"Helsinki-NLP/opus-mt-{model_key}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(torch_device)
        model_cache[model_key] = (tokenizer, model)
    return model_cache[model_key]


def translate(text: str, source_lang: str, target_lang: str, max_length: int = 512) -> str:
    logging.debug(f"Translating from {source_lang} to {target_lang} (first 100 chars): {text[:100]!r}")
    tokenizer, model = load_model(source_lang, target_lang)

    paragraphs = re.split(r'\n{2,}', text)
    translated_paragraphs = []

    for para in paragraphs:
        if not para.strip():
            translated_paragraphs.append("")
            continue

        lines = para.splitlines()
        all_sentences = [sent for line in lines for sent in sent_tokenize(line) if sent.strip()]
        translated_sentences = []

        for sent in all_sentences:
            if len(sent) > max_length:
                logging.warning(f"Skipping long sentence: {sent[:50]}...")
                continue
            inputs = tokenizer(sent, return_tensors="pt").to(torch_device)
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True
            )
            translated_sentences.append(tokenizer.decode(output[0], skip_special_tokens=True))

        # Reconstruct paragraphs
        rebuilt_lines = []
        idx = 0
        for line in lines:
            if not line.strip():
                rebuilt_lines.append("")
                continue
            num = len(sent_tokenize(line))
            rebuilt_lines.append(" ".join(translated_sentences[idx:idx+num]))
            idx += num

        translated_paragraphs.append("\n".join(rebuilt_lines))

    return "\n\n".join(translated_paragraphs)


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG)

    et_text = "Tere tulemast Riigiteataja juturobotisse!"
    en_text = "Welcome to the Riigiteataja chatbot!"

    logging.debug(translate(et_text, "et", "en"))
    logging.debug(translate(en_text, "en", "et"))

    test_translation = "Kui Arno isaga koolimajja jõudis, olid tunnid juba alanud..."
    logging.debug(f"Test translation from Estonian to English: {test_translation} = {translate(test_translation, 'et', 'en')}")