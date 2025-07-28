import os
import requests
import logging
from lxml.html import fromstring

def scrape_riigiteataja(url, folder=None):
    """
    Scrape the Riigiteataja website for legal documents.
    
    Args:
        url (str): The URL of the Riigiteataja page to scrape.
        
    Returns:
        str: The content of the page.
    """
    try:
        original_url_response = requests.get(url)
        title = fromstring(original_url_response.content).findtext('.//title')
        logging.info(f"Scraped title: {title}")

        pdf_file_name = title.split("–")[0] + " (" + os.path.basename(url) + ").pdf"
        
        if "pdf" not in url:
            pdf_url = url + ".pdf"

        logging.info(f"Downloading PDF from {pdf_url}")
        pdf_response = requests.get(pdf_url)


        pdf_response.raise_for_status()  # Raise an error for bad responses


        if pdf_response.status_code == 200:
            # Save in specified folder or current working directory
            if folder:
                os.makedirs(folder, exist_ok=True)
                filepath = os.path.join(folder, pdf_file_name)
            else:
                filepath = os.path.join(os.getcwd(), pdf_file_name)

            with open(filepath, 'wb') as file:
                file.write(pdf_response.content)
            logging.info(f"PDF saved successfully at {filepath}\n")

        return pdf_response.text
    

    except requests.RequestException as e:
        print(f"An error occurred while scraping: {e}")
        return None

if __name__ == '__main__':
    # URL from which pdfs to be downloaded
    logging.getLogger().setLevel(logging.INFO)
    estonian_URLs = ['https://www.riigiteataja.ee/akt/111042025002',
            'https://www.riigiteataja.ee/akt/105072025009',
            'https://www.riigiteataja.ee/akt/131122024048',
            'https://www.riigiteataja.ee/akt/106072023031',
            'https://www.riigiteataja.ee/akt/119122024004'
        ]

    # english_URLs = ['https://www.riigiteataja.ee/en/eli/521052015001',
    #         'https://www.riigiteataja.ee/en/eli/529122024005',
    #         'https://www.riigiteataja.ee/en/eli/523122015007',
    #         'https://www.riigiteataja.ee/en/eli/527032019002',
    #         'https://www.riigiteataja.ee/en/eli/523012015008'
    #        ]

    for url in estonian_URLs:
        scrape_riigiteataja(url, folder='riigiteataja_pdfs')