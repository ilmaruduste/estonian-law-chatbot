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

import re
from nltk.tokenize import sent_tokenize

def translate_et_to_en(text, max_length=512):
    logging.info(f"Translating from Estonian to English (first 100 chars): {text[:100]!r}")

    # Split by double newlines or multiple newlines to detect paragraphs
    paragraphs = re.split(r'\n{2,}', text)
    translated_paragraphs = []

    for para_num, para in enumerate(paragraphs):

        if para_num % 10 == 0:
            logging.info(f"Translating paragraph {para_num + 1}/{len(paragraphs)}: {para[:50]!r}")

        if not para.strip():
            translated_paragraphs.append("")  # Preserve blank paragraph
            continue

        # Split paragraph into lines, then into sentences
        lines = para.splitlines()
        all_sentences = []
        line_map = []  # Track which sentence belongs to which line

        for line in lines:
            if not line.strip():
                continue
            sentences = sent_tokenize(line)
            all_sentences.extend(sentences)
            line_map.extend([len(translated_paragraphs)] * len(sentences))

        # Translate each sentence
        translated_sentences = []
        for sent in all_sentences:
            if len(sent) > max_length:
                logging.warning(f"Sentence exceeds max length ({len(sent)} / {max_length}): {sent[:50]}...")
                continue

            inputs = et_en_tokenizer(sent, return_tensors="pt").to(torch_device)

            output = et_en_model.generate(
                **inputs,
                max_new_tokens=512,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True
            )
            translated_text = et_en_tokenizer.decode(output[0], skip_special_tokens=True)
            translated_sentences.append(translated_text)

        # Reconstruct lines from translated sentences
        rebuilt_lines = []
        current_sentence_idx = 0
        for line in lines:
            if not line.strip():
                rebuilt_lines.append("")
                continue
            num_sentences = len(sent_tokenize(line))
            translated_line = " ".join(translated_sentences[current_sentence_idx:current_sentence_idx + num_sentences])
            rebuilt_lines.append(translated_line)
            current_sentence_idx += num_sentences

        translated_paragraph = "\n".join(rebuilt_lines)
        translated_paragraphs.append(translated_paragraph)

    final_translation = "\n\n".join(translated_paragraphs)
    logging.info(f"Translation complete. Output has {len(translated_paragraphs)} paragraphs.")
    return final_translation



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