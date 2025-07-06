import asyncio
import base64
import logging
import os
from typing import Any, Dict, Optional

import aiohttp
from vertexai.generative_models import Part

from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    logging.info(f"Encoding image from path: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def create_analysis_prompt() -> str:
    """Create the analysis prompt for both GPT-4 and Gemini models."""
    return """Analyze the image and provide a comprehensive and structured response based on the content type
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
or uncertainties in the interpretation. If the image does not contain text, graph, chart or table,
 describe all the even the minute details and ovservations  what the image shows.
Do not write anything that you cannot see the image."""


def create_gpt4_payload(encoded_image: str, temperature: float = 0.7) -> Dict[str, Any]:
    """Create payload for GPT-4-OMNI model."""
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
                        "text": create_analysis_prompt(),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            },
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": 4000,
    }


async def analyze_single_image_gpt4(
    image_path: str, api_key: str, endpoint: str, temperature: float = 0.7
) -> Dict[str, Any]:
    """Analyze a single image using GPT-4.1 model."""
    try:
        logging.info(f"Analyzing image with GPT-4.1: {image_path}")
        encoded_image = encode_image(image_path)
        logging.info("Image encoded successfully for GPT-4.1")

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        payload = create_gpt4_payload(encoded_image, temperature)
        logging.info(f"Making request to GPT-4.1 endpoint: {endpoint}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint, headers=headers, json=payload
            ) as response:
                logging.info(f"GPT-4.1 API Response status: {response.status}")
                response_json = await response.json()

                if response.status != 200:
                    error_msg = f"GPT-4.1 API error: {response_json.get('error', 'Unknown error')}"
                    logging.error(error_msg)
                    return {"error": error_msg}

                analysis = response_json["choices"][0]["message"]["content"]
                logging.info(f"GPT-4.1 analysis successful for {image_path}")
                return {"analysis": analysis}
    except Exception as e:
        error_msg = (
            f"Failed to analyze image with GPT-4.1 for {image_path}. Error: {str(e)}"
        )
        logging.error(error_msg)
        return {"error": error_msg}


async def analyze_single_image_gemini(
    image_path: str, gemini_handler: GeminiHandler, temperature: float = 0.1
) -> Dict[str, Any]:
    """Analyze a single image using Gemini Pro model."""
    try:
        logging.info(f"Analyzing image with Gemini: {image_path}")
        if not gemini_handler:
            error_msg = "GeminiHandler instance is required for Gemini model"
            logging.error(error_msg)
            return {"error": error_msg}

        if not gemini_handler.generative_model:
            logging.info("Initializing Gemini model")
            gemini_handler.initialize("gemini-2.5-flash")

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            logging.info(f"Image read successfully for Gemini: {image_path}")

        image_part = Part.from_data(data=image_data, mime_type="image/jpeg")
        prompt = create_analysis_prompt()

        logging.info("Making request to Gemini API")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_handler.generative_model.generate_content(
                [prompt, image_part], generation_config={"temperature": temperature}
            ),
        )

        logging.info(f"Image analysis successful for: {image_path}")
        return {"analysis": response.text}
    except Exception as e:
        error_msg = (
            f"Failed to analyze image with Gemini for {image_path}. Error: {str(e)}"
        )
        logging.error(error_msg)
        return {"error": error_msg}


async def analyze_images(
    image_path: str,
    model: str = "gpt4-omni",  # Default to only using GPT-4 (Azure)
    gemini_handler: Optional[GeminiHandler] = None,
) -> Dict[str, Any]:
    """Analyze images using GPT-4.1 for unified Azure approach.
    Note: Gemini analysis is kept in codebase but disabled as part of unified Azure approach.
    """
    try:
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logging.error(error_msg)
            return {"error": error_msg}

        results = {}
        tasks = []

        # Only use GPT-4.1 analysis task for the unified Azure approach
        endpoint = construct_endpoint_url()
        api_key = os.getenv("AZURE_LLM__MODELS__GPT_4_1__API_KEY")
        if not api_key:
            error_msg = (
                "AZURE_LLM__MODELS__GPT_4_1__API_KEY environment variable not set"
            )
            logging.error(error_msg)
            return {"error": error_msg}
        tasks.append(
            analyze_single_image_gpt4(image_path, api_key, endpoint, temperature=0.7)
        )

        # Note: Gemini analysis is disabled as part of unified Azure approach
        # The code for analyze_single_image_gemini is kept for reference only

        # Run analysis
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results - only Azure GPT-4.1 is used
        gpt4_result = completed_tasks[0]
        results["gpt4_analysis"] = (
            gpt4_result["analysis"]
            if "analysis" in gpt4_result
            else gpt4_result["error"]
        )

        return results

    except Exception as e:
        error_msg = f"Error in analyze_images: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}


def construct_endpoint_url() -> str:
    """Construct the endpoint URL for GPT-4.1 model."""
    base_url = os.getenv("AZURE_LLM__MODELS__GPT_4_1__ENDPOINT", "").rstrip("/")
    deployment = os.getenv("AZURE_LLM__MODELS__GPT_4_1__DEPLOYMENT", "")
    api_version = os.getenv("AZURE_LLM__MODELS__GPT_4_1__API_VERSION", "")

    # Construct the endpoint URL
    endpoint = f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    # Extract the deployment name from the endpoint

    if not endpoint:
        raise ValueError("AZURE_ENDPOINT environment variable not set")

    return endpoint
