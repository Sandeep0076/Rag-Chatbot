import base64
import json
import os
from typing import Any, Dict, List

import requests


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def create_payload(encoded_image: str) -> Dict[str, Any]:
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
Analyze the image and provide a structured response with the following sections:

1. Overview: Briefly describe the main content of the image.
2. Text Content: Accurately transcribe all visible text, maintaining its original format.
3. Visual Elements:
   - For graphs or charts: Describe the type, structure, key data points, trends, and relationships.
   - For tables: Recreate the table structure, identify the main subject, and note any trends.
   Include a complete JSON representation of the table.
4. Data Insights: Highlight key findings, significant insights, correlations, and any anomalies.
5. Context and Implications: Provide context for the data and discuss potential conclusions and limitations.

Use clear section titles and ensure all table data is in a valid JSON format.
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
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    encoded_image = encode_image(image_path)
    payload = create_payload(encoded_image)

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"Failed to make the request for {image_path}. Error: {e}"
    except KeyError:
        return f"Unexpected response format for {image_path}. Full response: {response.json()}"


def analyze_images(image_path: str) -> List[Dict[str, Any]]:
    API_KEY = os.environ.get("AZURE_LLM__MODELS__GPT_4_VISION__API_KEY")
    ENDPOINT = os.environ.get("AZURE_LLM__MODELS__GPT_4_VISION__ENDPOINT")

    if not API_KEY or not ENDPOINT:
        raise ValueError("API_KEY or ENDPOINT environment variables are not set.")

    result = analyze_single_image(image_path, API_KEY, ENDPOINT)
    return [{"filename": os.path.basename(image_path), "analysis": result}]


if __name__ == "__main__":
    IMAGE_FOLDER = "processed_data/images"
    RESULT_PATH = "processed_data/image_analysis_results.json"

    results = analyze_images(IMAGE_FOLDER)

    # Save the results to a JSON file
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Image analysis results saved to {RESULT_PATH}")
