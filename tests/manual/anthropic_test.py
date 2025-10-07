import vertexai
from anthropic import AnthropicVertex

from configs.app_config import Config


def main() -> None:
    # Load Anthropic configuration from Config
    cfg = Config()
    project_id = cfg.anthropic.project
    location = cfg.anthropic.location
    model = cfg.anthropic.model_sonnet

    print(f"Using project: {project_id}, location: {location}, model: {model}")

    # Initialize Vertex AI (ADC)
    vertexai.init(project=project_id, location=location)

    # Create Anthropic Vertex client and ask a simple question
    client = AnthropicVertex(region=location, project_id=project_id)
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "Give me the top five countries with the highest GDP.",
            }
        ],
    )

    # Print the text response (fallback to JSON if structure differs)
    try:
        print(message.content[0].text)
    except Exception:
        print(message.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
