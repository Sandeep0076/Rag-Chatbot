# New functions to extract from main() to reduce complexity
import requests
import streamlit as st

# Define API URL as it's used across functions
API_URL = "http://localhost:8080"


def display_app_header():
    """Display the app header with title and logo."""
    st.markdown(
        (
            "<div style='background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%); "
            "padding: 2rem 1rem 1rem 1rem; border-radius: 1rem; "
            "box-shadow: 0 4px 16px rgba(30,58,138,0.07); margin-bottom: 1rem; text-align: center;'>"
            "<span style='font-size: 2.5rem; font-weight: 700; color: #fff; letter-spacing: 1px;'>"
            "RTL-Deutschland RAG Chatbot"
            "</span>"
            "<div style='font-size: 1.2rem; font-weight: 400; color: #e0e7ef; margin-top: 0.5rem;'>"
            "Chat with your PDFs, Images, Tables, and more"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def validate_image_model():
    """Validate that the selected model is an image generation model and return model info."""
    current_model = st.session_state.model_choice

    # Make sure model_types exists in session state
    if "model_types" not in st.session_state:
        st.session_state.model_types = {
            "text": [],
            "image": ["dall-e-3", "imagen", "Dalle + Imagen"],
        }

    # Verify it's an image model
    is_image_model = current_model in st.session_state.model_types.get("image", [])

    if not is_image_model:
        # If the current model is not an image model, show a warning
        st.warning(
            f"The current model '{current_model}' is not an image generation model. "
            "Please select an image model from the sidebar."
        )
        # Get the first image model from the available models
        available_image_models = st.session_state.model_types.get("image", [])
        if available_image_models:
            suggested_model = available_image_models[0]
            st.info(f"Suggested model: {suggested_model}")
        return None

    return current_model


def display_model_information(current_model):
    """Display information and guidance for the selected image model."""
    with st.expander("Model information"):
        st.info(f"Current model: {current_model}")
        st.write("You can change the model in the sidebar.")

        # Show prompt guide for all image models
        st.subheader("Image Generation Prompt Guide")

        # Add links and general guidance
        if current_model == "Dalle + Imagen":
            st.markdown("**Compare results from both models with the same prompt:**")
            st.markdown(
                "[DALL-E Prompt Guide](https://platform.openai.com/docs/guides/images) | "
                "[Vertex AI Imagen Prompt Guide]("
                "https://cloud.google.com/vertex-ai/generative-ai/docs/image/img-gen-prompt-guide)"
            )
        elif "imagen" in current_model.lower():
            st.markdown(
                "[Vertex AI Imagen Prompt Guide]("
                "https://cloud.google.com/vertex-ai/generative-ai/docs/image/img-gen-prompt-guide) - "
                "Learn how to craft effective prompts for better image generation results."
            )
        elif "dall-e" in current_model.lower():
            st.markdown(
                "[DALL-E Prompt Guide](https://platform.openai.com/docs/guides/images) - "
                "Learn how to craft effective prompts for image generation."
            )

        # Common tips for all models
        st.markdown(
            """**Pro Tips for Better Image Generation Results:**

            The first thing to think about with any prompt is the subject: the object, person, animal,
            or scenery you want an image of.

            Context and background: Just as important is the background or context in which the subject
            will be placed. Try placing your subject in a variety of backgrounds. For example, a studio with
            a white background, outdoors, or indoor environments.

            Style: Finally, add the style of image you want. Styles can be general (painting, photograph,
            sketches) or very specific (pastel painting, charcoal drawing, isometric 3D).

            - Be detailed and specific in your descriptions
            - Mention lighting, style, and viewpoint for more control
            - Use artistic terms like 'digital art', 'photorealistic', or 'watercolor'
            - Mention camera details for photographic looks (e.g., 'shot with a DSLR camera')

            """
        )


def get_image_generation_inputs():
    """Get user inputs for image generation."""
    # Add a text area for the prompt
    prompt = st.text_area(
        "Enter a detailed description of the image you want to generate:",
        placeholder="A futuristic cityscape with flying cars and neon lights, digital art",
        height=150,
    )

    # Add a select box for image size
    size_options = {
        "Square (1024x1024)": "1024x1024",
        "Portrait (1024x1792)": "1024x1792",
        "Landscape (1792x1024)": "1792x1024",
    }
    size_selection = st.selectbox(
        "Select image size:", options=list(size_options.keys()), index=0
    )
    selected_size = size_options[size_selection]

    return prompt, selected_size


def setup_dual_image_display_css():
    """Add CSS for controlling dual image display and fullscreen icons."""
    st.markdown(
        """
    <style>
    /* Hide fullscreen button on the right image */
    .imagen-container button[title="View fullscreen"] {
        display: none !important;
    }

    /* Custom fullscreen button for the left image */
    .custom-fullscreen-btn {
        position: absolute;
        top: 8px;
        left: 8px;
        background: rgba(0, 0, 0, 0.4);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 4px 8px;
        cursor: pointer;
        z-index: 100;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def display_dalle_image(dalle_result, col):
    """Display the DALL-E generated image in the given column."""
    with col:
        st.subheader("DALL-E 3")
        if dalle_result.get("success"):
            dalle_url = dalle_result.get("image_url")

            # Add a unique class to the container for the DALL-E image
            st.markdown('<div class="dalle-container">', unsafe_allow_html=True)
            st.image(dalle_url, caption="Generated with DALL-E 3")
            st.markdown("</div>", unsafe_allow_html=True)

            # Add download button for DALL-E image
            download_url = dalle_url
            st.markdown(
                f"""<a href='{download_url}'
                download='dalle_image.png' target='_blank'>
                 <button style='background-color: #4CAF50;
                 color: white; padding: 10px 15px;
                 border: none; border-radius: 4px;
                 cursor: pointer;'>
                Download DALL-E Image</button></a>""",
                unsafe_allow_html=True,
            )
        else:
            st.error(
                f"Failed to generate DALL-E image: "
                f"{dalle_result.get('error', 'Unknown error')}"
            )


def display_imagen_image(imagen_result, prompt, col):
    """Display the Imagen generated image in the given column."""
    with col:
        st.subheader("Imagen")
        if imagen_result.get("success"):
            imagen_url = imagen_result.get("image_url")
            is_base64 = imagen_result.get("is_base64", False)

            # Add a unique class to the container for the Imagen image
            st.markdown('<div class="imagen-container">', unsafe_allow_html=True)
            st.image(imagen_url, caption="Generated with Imagen")
            st.markdown("</div>", unsafe_allow_html=True)

            # Add download button for Imagen image
            # with matching style to DALL-E button
            if is_base64 and imagen_url.startswith("data:image/png;base64,"):
                base64_data = imagen_url.split(",")[1]
                # Create a hidden download link that will be triggered by the button
                download_link_id = f"imagen_download_{hash(prompt)}"
                st.markdown(
                    f"""
                    <a id="{download_link_id}" style="display:none;"></a>
                    <button style="background-color: #4CAF50; color: white;
                    padding: 10px 15px; border: none;
                    border-radius: 4px; cursor: pointer;"
                    onclick="
                        const link = document.getElementById('{download_link_id}');
                        link.href = 'data:image/png;base64,{base64_data}';
                        link.download = 'imagen_image.png';
                        link.click();
                    ">
                    Download Imagen Image</button>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error(
                f"Failed to generate Imagen image: "
                f"{imagen_result.get('error', 'Unknown error')}"
            )


def display_combined_model_results(result, prompt):
    """Display results from both DALL-E and Imagen models side by side."""
    st.success("Images generated successfully!")

    # Add CSS for controlling the full-screen icon placement
    setup_dual_image_display_css()

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Get results for each model
    dalle_result = result.get("dalle_result", {})
    imagen_result = result.get("imagen_result", {})

    # Display each image in its column
    display_dalle_image(dalle_result, col1)
    display_imagen_image(imagen_result, prompt, col2)


def display_single_model_results(result, current_model):
    """Display results from a single image generation model."""
    st.success("Image generated successfully!")

    # Check if it's a base64 image (from Imagen) or a URL (from DALL-E)
    is_base64 = result.get("is_base64", False)
    image_url = result["image_url"]

    # Display the image
    st.image(image_url, caption=f"Generated with {current_model}")

    # Add appropriate download options
    if is_base64:
        # For base64 images, we need to provide a way to download
        # Extract the base64 data without the prefix
        if image_url.startswith("data:image/png;base64,"):
            base64_data = image_url.split(",")[1]
            st.download_button(
                label="Download Image",
                data=base64_data,
                file_name="generated_image.png",
                mime="image/png",
            )
    else:
        # For URL-based images (DALL-E), use the original approach
        download_url = image_url
        st.markdown(
            (
                (
                    f"""<a href='{download_url}'
            download='generated_image.png' target='_blank'>
            <button style='background-color: #4CAF50;
            color: white; padding: 10px 15px;
            border: none; border-radius: 4px;
            cursor: pointer;'>
            Download Image</button></a>"""
                )
            ),
            unsafe_allow_html=True,
        )


def generate_image(current_model, prompt, selected_size):
    """Call the appropriate API endpoint to generate an image."""
    API_URL = "http://localhost:8080"

    with st.spinner(
        f"Generating your image with {current_model}...this may take a moment"
    ):
        try:
            # Determine which API endpoint to call based on model choice
            if current_model == "Dalle + Imagen":
                # Call the combined endpoint for both models
                response = requests.post(
                    f"{API_URL}/image/generate-combined",
                    json={
                        "prompt": prompt,
                        "size": selected_size,
                        "n": 1,
                        "model_choice": "",  # Not needed for combined endpoint
                    },
                )
            else:
                # Call the single model endpoint
                response = requests.post(
                    f"{API_URL}/image/generate",
                    json={
                        "prompt": prompt,
                        "size": selected_size,
                        "n": 1,
                        "model_choice": current_model,
                    },
                )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.text}")
                return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None


def handle_image_generation():
    """Handle the image generation functionality."""
    # Get the current model
    current_model = validate_image_model()
    if not current_model:
        return

    # Show title with current model
    st.markdown(
        f'<div class="subheader">Generate images using {current_model}</div>',
        unsafe_allow_html=True,
    )

    # Check if username is provided
    if not st.session_state.username:
        st.error("Username is required. Please enter a username in the sidebar.")
        return

    # Get user inputs
    prompt, selected_size = get_image_generation_inputs()

    # Display model information
    display_model_information(current_model)

    # Add a generate button
    if st.button("Generate Image", type="primary"):
        if not prompt:
            st.error("Please enter a prompt to generate an image.")
        else:
            # Generate the image
            result = generate_image(current_model, prompt, selected_size)

            # Display results
            if result:
                if current_model == "Dalle + Imagen" and result.get("success"):
                    display_combined_model_results(result, prompt)
                elif result.get("success"):
                    display_single_model_results(result, current_model)
                else:
                    st.error(
                        f"Failed to generate image: {result.get('error', 'Unknown error')}"
                    )
