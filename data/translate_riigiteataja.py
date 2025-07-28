import os
import sys
import pymupdf4llm
import logging

from translation.translate import translate_et_to_en

def translate_riigiteataja(test_document = None):
    """
    Translates the contents of the Riigiteataja PDFs from Estonian to English and vice versa.
    """
    docs_dir = "data/riigiteataja_pdfs"
    translated_docs_dir = "data/riigiteataja_translated"

    os.makedirs(translated_docs_dir, exist_ok=True)

    md_read = pymupdf4llm.LlamaMarkdownReader()

    filename_list = os.listdir(docs_dir)
    logging.info(f"Found {len(filename_list)} files in {docs_dir}")
    logging.info(f"Filenames: {filename_list}")

    if test_document:
        filename_list = [f for f in filename_list if test_document in f]
        logging.info(f"Filtered filenames: {filename_list}")

    for filename in filename_list:
        if not filename.endswith(".pdf"):
            continue
        logging.info(f"Processing file: {filename}")
        data = md_read.load_data(os.path.join(docs_dir, filename))
        logging.info(f"Loaded {len(data)} pages from {filename}")

        joined_text = "\n".join([page.to_dict()["text"] for page in data])
        logging.info(f"Joined text length: {len(joined_text)} characters")
        logging.info(f"First 50 characters: {joined_text[:50]}")

        # Translate Estonian to English
        translated_text = translate_et_to_en(joined_text)
        logging.info(f"Translated text length: {len(translated_text)} characters")
        logging.info(f"First 50 characters of translated text: {translated_text[:50]}")

        new_filename = filename.replace(".pdf", ".txt")
        with open(os.path.join(translated_docs_dir, f"en_{new_filename}"), "w", encoding="utf-8") as f:
            f.write(translated_text)
        logging.info(f"Translated {filename} to English and saved as en_{new_filename}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    translate_riigiteataja("põhiseadus")
    logging.info("Translation completed. Check the 'riigiteataja_translated' directory for results.")