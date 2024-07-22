# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import os
import hashlib
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException
import logging

class ImageScraper:
    def __init__(self, query, directory, num_pages=30):
        # Initialize the ImageScraper with search query, save directory, and number of pages to scrape
        self.query = query
        self.directory = directory
        self.num_pages = num_pages
        # Set the base URL for Bing image search
        self.base_url = "https://www.bing.com/images/search"
        # Set user agent to mimic a browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Set to store hashes of downloaded images to avoid duplicates
        self.downloaded_images = set()
        # Create a session object for persistent connections
        self.session = requests.Session()
        # Set up logging
        self.setup_logging()
        # Create the directory to save images
        self.create_directory()

    def setup_logging(self):
        # Configure logging to display time, log level, and message
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def create_directory(self):
        # Create the directory to save images if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

    def fetch_images(self):
        # Use tqdm to create a progress bar for fetching pages
        with tqdm(total=self.num_pages, desc=f"Fetching pages for '{self.query}'") as pbar:
            for page in range(self.num_pages):
                # Set parameters for the Bing image search
                params = {
                    "q": self.query,
                    "first": page * 10,
                    "count": 10
                }
                try:
                    # Send a GET request to Bing
                    response = self.session.get(self.base_url, params=params, headers=self.headers, timeout=10)
                    response.raise_for_status()
                    # Parse and download images from the response
                    self.parse_and_download_images(response.content)
                except RequestException as err:
                    # Log any errors that occur during the request
                    self.logger.error(f"Error fetching page {page}: {err}")
                # Update the progress bar
                pbar.update(1)

    def parse_and_download_images(self, html_content):
        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")
        # Find all img tags
        images = soup.find_all("img")
        # Extract image URLs
        image_urls = [image.get("src") or image.get("data-src") for image in images if image.get("src") or image.get("data-src")]

        # Use ThreadPoolExecutor for concurrent downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit download tasks to the executor
            futures = [executor.submit(self.download_image, url) for url in image_urls]
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()

    def download_image(self, image_url, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.session.get(image_url, timeout=10)
                response.raise_for_status()
                image_hash = hashlib.md5(response.content).hexdigest()
                
                if image_hash not in self.downloaded_images:
                    self.downloaded_images.add(image_hash)
                    # Cambiar el nombre de la imagen para incluir la consulta
                    image_name = f"{self.query}_{len(self.downloaded_images)}.jpg"
                    image_path = os.path.join(self.directory, image_name)
                    
                    with open(image_path, "wb") as file:
                        file.write(response.content)
                    self.logger.info(f"Image saved: {image_name}")
                else:
                    self.logger.info(f"Duplicate image: {image_url}")
                return
            except RequestException as e:
                self.logger.warning(f"Error downloading {image_url}, attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to download image after {max_retries} attempts: {image_url}")
                time.sleep(5)

def main():
    queries = ["bolsa de plastico", "vidrio roto","envase de tetrapak", "bateria usada", "celular viejo", "juguete de plastico roto"]
    directory = "dataset/train/inorganic"
    
    # Crear el directorio una sola vez
    os.makedirs(directory, exist_ok=True)
    
    for query in queries:
        # Pasar el mismo directorio para todas las consultas
        scraper = ImageScraper(query, directory)
        scraper.fetch_images()

if __name__ == "__main__":
    main()
