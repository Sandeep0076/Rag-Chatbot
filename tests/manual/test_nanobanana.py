import vertexai
from google import genai
from google.genai import types

from configs.app_config import Config


def generate():
    print("🚀 Starting Gemini 2.5 Flash Image test with Vertex AI authentication...")
    print("=" * 60)

    try:
        # Load configs to get Vertex AI project and location
        print("📋 Loading configuration...")
        configs = Config()
        print(
            f"✅ Config loaded - Project: {configs.gemini.project}, Location: {configs.gemini.location}"
        )

        # Initialize Vertex AI with project and location (uses default credentials)
        print("\n🔐 Initializing Vertex AI authentication...")
        vertexai.init(project=configs.gemini.project, location=configs.gemini.location)
        print("✅ Vertex AI initialized successfully")

        # Create client using Vertex AI authentication (no API key needed)
        # Note: genai.Client requires explicit project and location when using vertexai=True
        print("\n🔌 Creating genai client...")
        client = genai.Client(
            vertexai=True,
            project=configs.gemini.project,
            location=configs.gemini.location,
        )
        print("✅ Client created successfully")

        model = "gemini-2.5-flash-image"
        print(f"\n🎨 Using model: {model}")

        # Add a simple prompt for image generation with image specifications in the prompt
        prompt = (
            "Generate a beautiful 1:1 PNG sunset image over mountains, 1024x1024 pixels"
        )
        print(f"📝 Prompt: {prompt}")

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
        )

        print("\n🔄 Generating content (streaming)...")
        print("-" * 60)

        response_text = ""
        image_count = 0

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if hasattr(chunk, "text") and chunk.text:
                response_text += chunk.text
                print(chunk.text, end="", flush=True)

            # Check for image data in chunk
            if hasattr(chunk, "parts"):
                for part in chunk.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image_count += 1
                        print(
                            f"\n🖼️  Image {image_count} received (size: {len(part.inline_data.data)} bytes)"
                        )

        print("\n" + "-" * 60)
        print("\n✅ Generation complete!")
        if response_text:
            print(f"📄 Text response length: {len(response_text)} characters")
        if image_count > 0:
            print(f"🖼️  Images received: {image_count}")
        else:
            print("⚠️  No images received in response")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    return True


if __name__ == "__main__":
    success = generate()
    exit(0 if success else 1)
