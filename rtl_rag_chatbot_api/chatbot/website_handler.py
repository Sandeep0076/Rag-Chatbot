import logging
import os
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from weasyprint import HTML
from weasyprint.logger import LOGGER
from webdriver_manager.chrome import ChromeDriverManager

# Suppress WeasyPrint warnings
LOGGER.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]


class WebsiteHandler:
    def __init__(self, base_handler=None):
        """Initialize the WebsiteHandler.

        Args:
            base_handler: Optional BaseRAGHandler instance to use for PDF processing
        """
        self.base_handler = base_handler
        self._driver = None  # Lazy-loaded Selenium WebDriver

    def _get_browser_headers(self):
        """Return headers that mimic a browser to bypass simple anti-bot measures."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

    def url_to_pdf(self, url: str, output_path: str = None) -> str:
        """Convert a website to PDF using WeasyPrint.

        Args:
            url: The URL to convert to PDF
            output_path: Optional path to save the PDF. If None, a temporary file is created.

        Returns:
            Path to the generated PDF file
        """
        try:
            # Generate a unique filename if output_path is not provided
            if not output_path:
                os.makedirs("local_data", exist_ok=True)
                file_id = str(uuid.uuid4())[:8]
                domain = url.split("//")[-1].split("/")[0].replace(".", "_")
                output_path = f"local_data/{file_id}_{domain}.pdf"

            print(f"Converting {url} to PDF at {output_path}")

            # Fetch the HTML content with browser headers
            response = requests.get(
                url, headers=self._get_browser_headers(), timeout=15, verify=False
            )
            response.raise_for_status()

            # Convert HTML to PDF
            html = HTML(string=response.text)
            html.write_pdf(output_path)

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise ValueError(f"Failed to create PDF or PDF is empty: {output_path}")

            print(f"Successfully converted {url} to PDF: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error converting {url} to PDF: {str(e)}")
            raise

    def get_text_from_url(self, url: str) -> str:
        """Extract text directly from a URL using requests and BeautifulSoup.

        Args:
            url: The URL to extract text from

        Returns:
            Extracted text content
        """
        try:
            # Make the request with browser headers
            response = requests.get(
                url, headers=self._get_browser_headers(), timeout=10, verify=False
            )
            response.raise_for_status()

            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get text
            text = soup.get_text(separator="\n")

            # Clean up text: remove extra whitespace and empty lines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return f"Error fetching content: {str(e)}"

    def _get_selenium_driver(self):
        """Initialize and return a Selenium WebDriver instance with Chrome."""
        if self._driver is None:
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            # Add user agent to mimic a real browser
            chrome_options.add_argument(
                f"user-agent={self._get_browser_headers()['User-Agent']}"
            )

            # Initialize the WebDriver
            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service, options=chrome_options)

        return self._driver

    def get_text_with_selenium(self, url: str) -> tuple[str, str]:
        """Extract text from a JavaScript-heavy website using Selenium.

        Args:
            url: The URL to extract text from

        Returns:
            Tuple of (extracted_text, page_title)
        """
        try:
            driver = self._get_selenium_driver()

            # Navigate to the URL
            print(f"Loading {url} with Selenium...")
            driver.get(url)

            # Wait for the page to load (adjust timeout as needed)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Give JavaScript some extra time to execute
            time.sleep(5)

            # Get the page title
            title = driver.title

            # Get the page content
            body = driver.find_element(By.TAG_NAME, "body")
            text = body.text

            print(f"Successfully extracted content with Selenium. Title: {title}")
            return text, title

        except Exception as e:
            print(f"Error extracting with Selenium: {str(e)}")
            raise

    def get_vectorstore_from_url(self, *urls) -> List[Document]:
        """Fetch content from one or more URLs using multiple extraction strategies.

        This method accepts variable arguments, so you can pass one or more URL strings:
        - get_vectorstore_from_url("https://example.com")  # Single URL
        - get_vectorstore_from_url("https://example.com", "https://another.com")  # Multiple URLs

        For each URL, it tries different approaches in the following order:
        1. Direct text extraction with requests/BeautifulSoup
        2. Selenium for JavaScript-heavy sites
        3. PDF conversion as a last resort

        Args:
            *urls: One or more URL strings to fetch content from

        Returns:
            A list of Document objects containing the content from all URLs
        """
        # Check if any URLs were provided
        if not urls:
            raise ValueError("At least one URL must be provided")

        # If only one URL is provided, process it directly
        if len(urls) == 1:
            return self._get_vectorstore_from_url(urls[0])

        # For multiple URLs, process each one and combine results
        all_documents = []

        for url in urls:
            if not isinstance(url, str):
                print(f"Warning: Skipping non-string URL: {url}")
                continue

            try:
                print(f"Processing URL: {url}")
                documents = self._get_vectorstore_from_url(url)
                all_documents.extend(documents)
                print(f"Successfully processed URL: {url}")
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                # Add an error document so we know which URL failed
                all_documents.append(
                    Document(
                        page_content=f"Error processing URL: {str(e)}",
                        metadata={"source": url, "error": str(e)},
                    )
                )

        return all_documents

    def _get_vectorstore_from_url(self, url: str) -> List[Document]:
        """Internal implementation for fetching content from a single URL.

        This method tries different approaches in the following order:
        1. Direct text extraction with requests/BeautifulSoup
        2. Selenium for JavaScript-heavy sites
        3. PDF conversion as a last resort

        Args:
            url: The URL to fetch content from

        Returns:
            A list containing a Document with the page content and metadata
        """
        try:
            text = None
            title = None

            # Strategy 1: Try direct text extraction first
            try:
                text = self.get_text_from_url(url)
                # If the text contains error messages or is very short, it likely failed
                if "Error fetching content" in text or len(text.strip()) < 100:
                    raise ValueError(
                        "Direct text extraction failed or returned insufficient content"
                    )

                # Extract title using BeautifulSoup
                response = requests.get(
                    url, headers=self._get_browser_headers(), timeout=10, verify=False
                )
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.text if soup.title else "No title"

            except Exception as e:
                print(f"Direct extraction failed: {str(e)}. Trying Selenium...")

                # Strategy 2: Try Selenium for JavaScript-heavy sites
                try:
                    text, title = self.get_text_with_selenium(url)
                    if not text or len(text.strip()) < 100:
                        raise ValueError(
                            "Selenium extraction returned insufficient content"
                        )

                except Exception as selenium_error:
                    print(
                        f"Selenium extraction failed: {str(selenium_error)}. Trying PDF conversion..."
                    )

                    # Strategy 3: Try PDF conversion as a last resort
                    pdf_path = self.url_to_pdf(url)

                    # Use the base_handler to extract text from the PDF if available
                    if self.base_handler:
                        text = self.base_handler.extract_text_from_pdf(pdf_path)
                    else:
                        # Fallback to using pdfminer directly if base_handler not available
                        from pdfminer.high_level import extract_text

                        text = extract_text(pdf_path)

                    # Clean up the temporary PDF file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

                    # Use URL as title if we don't have it from direct extraction
                    title = url.split("//")[-1].split("/")[0]

            # Create a Document object
            document = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": title,
                    "language": "en",  # Default to English
                },
            )

            result = [document]
            print(result)
            return result

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            # Return a document with the error message
            return [
                Document(
                    page_content=f"Error fetching content: {str(e)}",
                    metadata={"source": url, "error": str(e)},
                )
            ]


url1 = "https://en.wikipedia.org/wiki/API"
url2 = "https://python.langchain.com/docs/integrations/document_loaders/web_base/"
url3 = "https://www.w3schools.com/js/js_api_intro.asp"
url4 = "https://www.langchain.com/langgraph"
url5 = "https://platform.openai.com/docs/guides/images?api-mode=responses"
url6 = "https://www.rtl.de/"

if __name__ == "__main__":
    # Test the unified function with both single and multiple URLs
    handler = WebsiteHandler()
    try:
        # Test with a single URL
        print("\n=== Testing with a single URL ===")
        single_result = handler.get_vectorstore_from_url(url6)
        print(f"Single URL result: {len(single_result)} document processed")
        doc = single_result[0]
        print(f"Source: {doc.metadata['source']}")
        print(f"Title: {doc.metadata.get('title', 'No title')}")
        print(f"Content (first 100 chars): {doc.page_content[:500]}...")

        # # Test with multiple URLs as separate arguments
        # print("\n=== Testing with multiple URLs as separate arguments ===")
        # results = handler.get_vectorstore_from_url(url1, url3, url5)  # Pass multiple URLs as separate arguments
        # print(f"Multiple URLs results: {len(results)} documents processed")
        # for i, doc in enumerate(results):
        #     print(f"\nDocument {i+1}:")
        #     print(f"Source: {doc.metadata['source']}")
        #     print(f"Title: {doc.metadata.get('title', 'No title')}")
        #     print(f"Content (first 100 chars): {doc.page_content[:100]}...")
    finally:
        # Clean up Selenium driver if it was initialized
        if handler._driver:
            handler._driver.quit()
