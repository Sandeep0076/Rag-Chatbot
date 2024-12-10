import base64
import logging  # Import logging module
import os
from typing import Any, Dict, List

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def encode_image(image_path: str) -> str:
    logging.info(f"Encoding image from path: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def create_payload(encoded_image: str) -> Dict[str, Any]:
    logging.info("Creating payload for image analysis")
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an advanced AI image analyzer."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Analyze the image and provide a comprehensive and structured response based on the content type
(text, graph,charts,tables etc):
General Overview:

Describe the main content: Briefly summarize what the image represents, whether it's primarily text, a
graph,chart table etc
and the key subject matter.
Text Content:

Transcription: Accurately transcribe all visible text while preserving  structure.
Contextual Interpretation: Provide a brief interpretation or summary of the textual content.
Highlight key points, themes, or arguments presented.
Graph or Chart:
extract complete information of table in JSON.
If possible convert the graph, chart values to a table and write all information in json form.

Structure Description:
Identify the type of graph or chart.
Describe the axes, including labels, units, and scale.
Detail any legends, titles, and annotations.
Data Extraction:
Identify value of each entry on x-axis or value of each entry on y-axis.Write down all the numerical
values or percentage for each entry.
List all labeled data points, entities, or categories shown on the graph.
For each labeled point or entity on x-axis provide its corresponding y-axis values as accurately as possible.
Describe trends, patterns, or correlations present in the data.
Extract and explain significant comparisons (e.g., differences between any two entity values).
Quantify key data points, such as specific values or comparisons (e.g., "India's GDP is higher than Mexico's by X%").
Analysis:
Discuss the implications of the data, highlight key insights, and note any anomalies or outliers.
Provide potential interpretations, discussing what the data suggests or indicates in a broader context.
Table Content:

Table Structure Recreation:
Recreate the table structure, identifying rows, columns, headers, and any hierarchical data.
Describe the main subject or purpose of the table.
Data Representation:

Identify and summarize key trends, patterns, or significant data points.
Interpretation and Analysis:
Provide a brief analysis of the table's content, noting significant findings, insights, or any anomalies.
Discuss potential conclusions or implications drawn from the data, and any limitations observed.
Key Insights and Summary:
Highlight Findings: Summarize the most important findings, insights, and correlations identified across text, graph,
 or table content.
Contextual Discussion: Provide context for the data, discuss potential conclusions, and address any limitations
or uncertainties in the interpretation.
                    """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4000,
    }


def analyze_single_image(image_path: str, api_key: str, endpoint: str) -> str:
    logging.info(f"Analyzing single image: {image_path}")
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    encoded_image = encode_image(image_path)
    payload = create_payload(encoded_image)

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        logging.info(f"Image analysis successful for: {image_path}")
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logging.error(f"Failed to make the request for {image_path}. Error: {e}")
        return f"Failed to make the request for {image_path}. Error: {e}"
    except KeyError:
        logging.error(
            f"Unexpected response format for {image_path}. Full response: {response.json()}"
        )
        return f"Unexpected response format for {image_path}. Full response: {response.json()}"


def analyze_images(image_path: str) -> List[Dict[str, Any]]:
    logging.info(f"Starting analysis for images in path: {image_path}")
    API_KEY = os.environ.get("AZURE_LLM__MODELS__GPT_4_OMNI__API_KEY")
    ENDPOINT = construct_endpoint_url()

    if not API_KEY or not ENDPOINT:
        logging.error("API_KEY or ENDPOINT environment variables are not set.")
        raise ValueError("API_KEY or ENDPOINT environment variables are not set.")

    result = analyze_single_image(image_path, API_KEY, ENDPOINT)
    logging.info(f"Image analysis completed for: {image_path}")
    return [{"filename": os.path.basename(image_path), "analysis": result}]


def construct_endpoint_url():
    logging.info("Constructing endpoint URL")
    # Retrieve environment variables
    base_url = os.getenv("AZURE_LLM__MODELS__GPT_4_OMNI__ENDPOINT", "").rstrip("/")
    deployment = os.getenv("AZURE_LLM__MODELS__GPT_4_OMNI__DEPLOYMENT", "")
    api_version = os.getenv("AZURE_LLM__MODELS__GPT_4_OMNI__API_VERSION", "")

    # Construct the endpoint URL
    endpoint = f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    return endpoint


# if __name__ == "__main__":
#     IMAGE_FOLDER = "processed_data/images/BYjxOmR.png"
#     RESULT_PATH = "processed_data/image_analysis_results.json"

#     results = analyze_images(IMAGE_FOLDER)

#     Save the results to a JSON file
#     with open(RESULT_PATH, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=4)

#     print(f"Image analysis results saved to {RESULT_PATH}")
