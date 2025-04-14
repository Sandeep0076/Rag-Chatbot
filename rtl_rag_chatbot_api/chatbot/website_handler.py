import logging
import os
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import trafilatura
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

    def extract_content_with_trafilatura(
        self, url: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract clean content from a URL using trafilatura.

        Trafilatura is specialized in extracting main content from web pages,
        removing boilerplate, navigation, ads, etc.

        Args:
            url: The URL to extract content from

        Returns:
            A tuple of (extracted_text, title) or (None, None) if extraction failed
        """
        try:
            # Download the webpage content with proper headers
            response = requests.get(
                url, headers=self._get_browser_headers(), timeout=15, verify=False
            )
            response.raise_for_status()
            html_content = response.text

            # Get the title from the HTML using BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            title = soup.title.text.strip() if soup.title else None

            # Use the simplest possible call to trafilatura
            # This is the most reliable approach based on their documentation
            try:
                # Try the simplest approach first
                extracted_text = trafilatura.extract(html_content)

                if not extracted_text:
                    # If that fails, try with minimal parameters
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        extracted_text = trafilatura.extract(downloaded)
            except Exception as e:
                print(f"Trafilatura extraction error: {str(e)}")
                extracted_text = None

            # If we couldn't extract text, return None
            if not extracted_text:
                print(f"Trafilatura couldn't extract content from {url}")
                return None, None

            return extracted_text, title

        except Exception as e:
            print(f"Error extracting content with trafilatura from {url}: {str(e)}")
            return None, None

    def get_text_from_url(self, url: str) -> str:
        """Extract text directly from a URL using requests and BeautifulSoup.

        This is a legacy method kept for backward compatibility.
        The preferred method is now extract_content_with_trafilatura.

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

        # Clean up URLs to handle any potential newlines or formatting issues
        cleaned_urls = []
        for url_item in urls:
            if not isinstance(url_item, str):
                print(f"Warning: Skipping non-string URL: {url_item}")
                continue

            # Split by newlines in case multiple URLs are in a single string
            if "\n" in url_item:
                sub_urls = [u.strip() for u in url_item.split("\n") if u.strip()]
                cleaned_urls.extend(sub_urls)
            else:
                cleaned_urls.append(url_item.strip())

        # If only one URL is provided, process it directly
        if len(cleaned_urls) == 1:
            return self._get_vectorstore_from_url(cleaned_urls[0])

        # For multiple URLs, process each one and combine results
        all_documents = []

        for url in cleaned_urls:
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

    def _detect_language(self, text: str) -> str:
        """Detect the language of the extracted text.

        Currently supports basic detection for German vs. English.

        Args:
            text: The text to analyze

        Returns:
            Language code ('en' or 'de')
        """
        if not text or len(text) < 50:
            return "en"  # Default to English for short texts

        try:
            # Use a simple heuristic for German detection
            german_markers = [
                "der",
                "die",
                "das",
                "und",
                "für",
                "ist",
                "nicht",
                "mit",
                "Sie",
                "sind",
            ]
            text_lower = text.lower()
            german_count = sum(
                1 for word in german_markers if f" {word} " in text_lower
            )

            # If at least 3 German markers are found
            if german_count >= 3:
                return "de"
        except Exception:
            pass  # Ignore errors in language detection

        return "en"  # Default to English

    def _extract_with_trafilatura(
        self, url: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract content using trafilatura.

        Args:
            url: The URL to extract from

        Returns:
            Tuple of (text, title) or (None, None) if extraction failed
        """
        print(f"Attempting trafilatura extraction for {url}...")

        try:
            text, title = self.extract_content_with_trafilatura(url)

            # Validate extraction quality
            if text and len(text.strip()) > 100:
                print(f"Trafilatura extraction successful for {url}")
                return text, title

            raise ValueError("Trafilatura extraction returned insufficient content")
        except Exception as e:
            print(f"Trafilatura extraction failed: {str(e)}")
            return None, None

    def _extract_with_selenium(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract content using Selenium for JavaScript-heavy sites.

        Args:
            url: The URL to extract from

        Returns:
            Tuple of (text, title) or (None, None) if extraction failed
        """
        print(f"Attempting Selenium extraction for {url}...")

        try:
            text, title = self.get_text_with_selenium(url)

            # Validate extraction quality
            if text and len(text.strip()) > 100:
                print(f"Selenium extraction successful for {url}")
                return text, title

            raise ValueError("Selenium extraction returned insufficient content")
        except Exception as e:
            print(f"Selenium extraction failed: {str(e)}")
            return None, None

    def _extract_with_beautifulsoup(
        self, url: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract content using BeautifulSoup (legacy approach).

        Args:
            url: The URL to extract from

        Returns:
            Tuple of (text, title) or (None, None) if extraction failed
        """
        print(f"Attempting BeautifulSoup extraction for {url}...")

        try:
            # Get text using the legacy method
            text = self.get_text_from_url(url)
            title = None

            # Validate extraction quality
            if "Error fetching content" in text or len(text.strip()) < 100:
                raise ValueError(
                    "BeautifulSoup extraction returned insufficient content"
                )

            # Get title
            try:
                response = requests.get(
                    url, headers=self._get_browser_headers(), timeout=10, verify=False
                )
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.text if soup.title else None
            except Exception:
                # If title extraction fails, don't let it prevent text extraction
                pass

            print(f"BeautifulSoup extraction successful for {url}")
            return text, title
        except Exception as e:
            print(f"BeautifulSoup extraction failed: {str(e)}")
            return None, None

    def _extract_with_pdf(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract content by converting to PDF first.

        Args:
            url: The URL to extract from

        Returns:
            Tuple of (text, title) or (None, None) if extraction failed
        """
        print(f"Attempting PDF conversion extraction for {url}...")
        pdf_path = None

        try:
            # Convert to PDF
            pdf_path = self.url_to_pdf(url)
            text = None

            # Extract text from PDF
            if self.base_handler:
                text = self.base_handler.extract_text_from_pdf(pdf_path)
            else:
                from pdfminer.high_level import extract_text

                text = extract_text(pdf_path)

            # Use domain as title
            title = url.split("//")[-1].split("/")[0]

            if text and len(text.strip()) > 100:
                print(f"PDF conversion extraction successful for {url}")
                return text, title

            raise ValueError("PDF extraction returned insufficient content")
        except Exception as e:
            print(f"PDF conversion extraction failed: {str(e)}")
            return None, None
        finally:
            # Clean up temporary PDF file
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

    def _create_document(
        self, text: str, title: str, url: str, language: str = "en"
    ) -> Document:
        """Create a Document object with the extracted content and metadata.

        Args:
            text: The extracted text content
            title: The title of the content
            url: The source URL
            language: The detected language code (default: 'en')

        Returns:
            A Document object with content and metadata
        """
        return Document(
            page_content=text,
            metadata={
                "source": url,
                "title": title if title else url,  # Use URL as fallback if no title
                "language": language,
            },
        )

    def _create_error_document(self, url: str, error: Exception) -> Document:
        """Create a Document object for extraction errors.

        Args:
            url: The source URL that failed
            error: The exception that occurred

        Returns:
            A Document object with error information
        """
        return Document(
            page_content=f"Error fetching content: {str(error)}",
            metadata={"source": url, "error": str(error)},
        )

    def cleanup(self):
        """Clean up resources used by the WebsiteHandler.

        This method ensures that the Selenium WebDriver is properly closed.
        """
        if self._driver:
            try:
                self._driver.quit()
            except Exception as e:
                print(f"Error closing Selenium driver: {str(e)}")
            finally:
                self._driver = None

    def _get_vectorstore_from_url(self, url: str) -> List[Document]:
        """Internal implementation for fetching content from a single URL.

        This method tries different extraction strategies in sequence until one succeeds.

        Args:
            url: The URL to fetch content from

        Returns:
            A list containing a Document with the page content and metadata
        """
        try:
            # Try each extraction strategy in sequence
            # Strategy 1: Trafilatura (best quality)
            text, title = self._extract_with_trafilatura(url)

            # Strategy 2: Selenium for JavaScript-heavy sites
            if not text:
                text, title = self._extract_with_selenium(url)

            # Strategy 3: BeautifulSoup (legacy approach)
            if not text:
                text, title = self._extract_with_beautifulsoup(url)

            # Strategy 4: PDF conversion as last resort
            if not text:
                text, title = self._extract_with_pdf(url)

            # If all extraction methods failed
            if not text:
                raise ValueError("All extraction methods failed")

            # Detect language
            language = self._detect_language(text)

            # Create and return document
            document = self._create_document(text, title, url, language)
            return [document]

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return [self._create_error_document(url, e)]


url1 = "https://en.wikipedia.org/wiki/API"
url2 = "https://python.langchain.com/docs/integrations/document_loaders/web_base/"
url3 = "https://www.w3schools.com/js/js_api_intro.asp"
url4 = "https://langchain-ai.github.io/langgraph/tutorials/introduction/"
url5 = "https://platform.openai.com/docs/guides/images?api-mode=responses"
url6 = ("https://www.rtl.de/",)
url7 = (
    "https://www.quora.com/What-is-it-that-we-dont-understand-or-know-about-black-holes"
)
url8 = "https://www.helicone.ai/llm-cost"
url9 = "https://yourgpt.ai/tools/openai-and-other-llm-api-pricing-calculator"

if __name__ == "__main__":
    # Test the unified function with both single and multiple URLs
    handler = WebsiteHandler()
    try:
        # Test with a single URL‚
        print("\n=== Testing with a single URL ===")
        single_result = handler.get_vectorstore_from_url(url9)
        print(f"Single URL result: {len(single_result)} document processed")
        doc = single_result[0]
        print(f"Source: {doc.metadata['source']}")
        print(f"Title: {doc.metadata.get('title', 'No title')}")
        print(f"Content : {doc.page_content}")
        # print(f"Content (first 2000 chars): {doc.page_content[:2000]}...")

        # # Test with multiple URLs as separate arguments
        # print("\n=== Testing with multiple URLs as separate arguments ===")
        # results = handler.get_vectorstore_from_url(url4, url5, url6)  # Pass multiple URLs as separate arguments
        # print(f"Multiple URLs results: {len(results)} documents processed")
        # for i, doc in enumerate(results):
        #     print(f"\nDocument {i+1}:")
        #     print(f"Source: {doc.metadata['source']}")
        #     print(f"Title: {doc.metadata.get('title', 'No title')}")
        #     print(f"Content : {doc.page_content}")
    finally:
        # Clean up Selenium driver if it was initialized
        if handler._driver:
            handler._driver.quit()
