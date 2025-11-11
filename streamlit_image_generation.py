# New functions to extract from main() to reduce complexity
import requests
import streamlit as st

# Define API URL as it's used across functions
API_URL = "http://localhost:8080"


def display_app_header():
    """Display the app header with modern neumorphic design."""
    st.markdown(
        (
            "<div class='main-header'>"
            "<h1>RTL-Deutschland RAG Chatbot</h1>"
            "<div style='font-size: 0.9rem; font-weight: 400; color: var(--color-text-muted); margin-top: 0.2rem;'>"
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
    """Get user inputs for image generation: prompt and size."""
    # Get the current model
    current_model = st.session_state.model_choice

    # Create a text area for the prompt
    prompt = st.text_area(
        "Enter a prompt describing the image you want to generate:",
        placeholder="Example: A photo of a cat in space...",
        height=100,
    )

    # Create a horizontal layout with columns for size and number options
    col1, col2 = st.columns(2)

    # Add a dropdown for size selection
    with col1:
        # Define the available image sizes
        image_sizes = [
            "1024x1024",
            "1024x1792",
            "1792x1024",
        ]
        selected_size = st.selectbox("Select image size:", image_sizes)

    # Add a dropdown for number of images only for Imagen model
    num_images = 1  # Default for DALL-E and combined
    with col2:
        # The combined model has a specific name "Dalle + Imagen"
        is_combined = current_model == "Dalle + Imagen"

        # Check if this is a pure Imagen model by looking for 'imagen' in the model name
        # This will match 'imagen', 'imagen-3.0-generate-002', 'imagen-1.5-pro-002', etc.
        is_pure_imagen = "imagen" in current_model.lower() and not is_combined

        if is_pure_imagen:
            # Only show number selection for pure Imagen model
            num_images = st.selectbox("Number of images:", [1, 2, 3, 4], index=0)
        else:
            # For DALL-E or combined, show a static message since only 1 image is allowed
            st.info("DALL-E 3 supports generating 1 image per request")

    return prompt, selected_size, num_images


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
            # Extract structured error if available
            error_code = dalle_result.get("code") or dalle_result.get("error_code")
            error_key = dalle_result.get("key") or dalle_result.get("error_key")
            error_msg = dalle_result.get("message") or dalle_result.get(
                "error", "Unknown error"
            )
            error_details = dalle_result.get("details") or dalle_result.get(
                "error_details"
            )

            if error_code and error_key:
                st.error(f"DALL-E Error {error_code}: {error_key} - {error_msg}")
            else:
                st.error(f"Failed to generate DALL-E image: {error_msg}")

            # Display detailed error information if available
            if error_details:
                with st.expander("🔍 Error Details"):
                    for key, value in error_details.items():
                        st.text(f"{key}: {value}")


def display_imagen_image(imagen_result, prompt, col):
    """Display the Imagen generated image(s) in the given column."""
    with col:
        st.subheader("Imagen")
        if imagen_result.get("success"):
            # Check if we have multiple images or a single image
            image_urls = imagen_result.get("image_urls", [])
            if not image_urls and imagen_result.get("image_url"):
                # Legacy format with single image_url
                image_urls = [imagen_result.get("image_url")]

            is_base64 = imagen_result.get("is_base64", False)

            # Display all images with a counter
            for i, imagen_url in enumerate(image_urls):
                # Add a unique class to the container for each Imagen image
                st.markdown('<div class="imagen-container">', unsafe_allow_html=True)
                st.image(
                    imagen_url,
                    caption=f"Generated with Imagen (Image {i + 1}/{len(image_urls)})",
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # Add download button for Imagen image
                if is_base64 and imagen_url.startswith("data:image/png;base64,"):
                    base64_data = imagen_url.split(",")[1]
                    # Create a unique download link ID for each image
                    download_link_id = f"imagen_download_{hash(prompt)}_{i}"
                    st.markdown(
                        f"""
                        <a id="{download_link_id}" style="display:none;"></a>
                        <button style="background-color: #4CAF50; color: white;
                        padding: 10px 15px; border: none;
                        border-radius: 4px; cursor: pointer;"
                        onclick="
                            const link = document.getElementById('{download_link_id}');
                            link.href = 'data:image/png;base64,{base64_data}';
                            link.download = 'imagen_image_{i + 1}.png';
                            link.click();
                        ">
                        Download Imagen Image {i + 1}</button>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<br>", unsafe_allow_html=True
                    )  # Add space between download buttons
        else:
            # Extract structured error if available
            error_code = imagen_result.get("code") or imagen_result.get("error_code")
            error_key = imagen_result.get("key") or imagen_result.get("error_key")
            error_msg = imagen_result.get("message") or imagen_result.get(
                "error", "Unknown error"
            )
            error_details = imagen_result.get("details") or imagen_result.get(
                "error_details"
            )

            if error_code and error_key:
                st.error(f"Imagen Error {error_code}: {error_key} - {error_msg}")
            else:
                st.error(f"Failed to generate Imagen image: {error_msg}")

            # Display detailed error information if available
            if error_details:
                with st.expander("🔍 Error Details"):
                    for key, value in error_details.items():
                        st.text(f"{key}: {value}")


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
    # Model type detection moved to where it's actually used

    # Get image URLs - handle both single image and multiple images
    image_urls = result.get("image_urls", [])
    if not image_urls and result.get("image_url"):
        # Legacy format with single image_url
        image_urls = [result.get("image_url")]

    # DALL-E only generates one image, Imagen can generate multiple
    if len(image_urls) > 1:
        st.markdown(f"### Generated {len(image_urls)} images with {current_model}")

    # Display each image with its own download button
    for i, image_url in enumerate(image_urls):
        # Add a container to visually separate multiple images
        if len(image_urls) > 1:
            st.markdown(f"#### Image {i + 1}/{len(image_urls)}")

        # Display the image
        st.image(image_url, caption=f"Generated with {current_model}")

        # Add appropriate download options
        if is_base64:
            # For base64 images (Imagen), provide download button
            if image_url.startswith("data:image/png;base64,"):
                base64_data = image_url.split(",")[1]
                st.download_button(
                    label=f"Download Image {i + 1 if len(image_urls) > 1 else ''}",
                    data=base64_data,
                    file_name=f"generated_image_{i + 1 if len(image_urls) > 1 else '1'}.png",
                    mime="image/png",
                )
        else:
            # For URL-based images (DALL-E), use HTML download link
            download_url = image_url
            file_name = f"generated_image_{i + 1 if len(image_urls) > 1 else '1'}.png"
            button_text = (
                f"Download Image {i + 1}" if len(image_urls) > 1 else "Download Image"
            )

            st.markdown(
                f"""<a href='{download_url}' download='{file_name}' target='_blank'>
                <button style='background-color: #4CAF50; color: white; padding: 10px 15px;
                border: none; border-radius: 4px; cursor: pointer;'>
                {button_text}</button></a>""",
                unsafe_allow_html=True,
            )

        # Add spacing between images if there are multiple
        if len(image_urls) > 1:
            st.markdown("---")


def _parse_image_error_response(response):
    """Parse error response from image generation API and extract meaningful error message."""
    try:
        error_data = response.json()
        if isinstance(error_data, dict):
            # Check for structured error format
            code = error_data.get("code") or error_data.get("error_code")
            key = error_data.get("key") or error_data.get("error_key")
            message = error_data.get("message")

            if code and key and message:
                return f"Error {code}: {key} - {message}"

            # Fallback to detail or error fields
            if "detail" in error_data:
                detail = error_data["detail"]
                if isinstance(detail, dict):
                    detail_code = detail.get("code") or detail.get("error_code")
                    detail_key = detail.get("key") or detail.get("error_key")
                    detail_message = detail.get("message")
                    if detail_code and detail_key and detail_message:
                        return f"Error {detail_code}: {detail_key} - {detail_message}"
                    return detail.get("message", str(detail))
                return str(detail)

            return error_data.get("error", response.text)
        return response.text
    except Exception:
        return response.text


def generate_image(
    current_model, prompt, selected_size, num_images, prompt_history=None
):
    """Call the appropriate API endpoint to generate an image."""
    API_URL = "http://localhost:8080"

    with st.spinner(
        f"Generating your image with {current_model}...this may take a moment"
    ):
        try:
            # Build prompt array: history + current prompt (similar to chat endpoint)
            if prompt_history and len(prompt_history) > 0:
                prompt_array = prompt_history + [prompt]
            else:
                prompt_array = [prompt]

            # DALL-E 3 can only generate 1 image per request
            # Note: n=1 enforcement for DALL-E is handled in the API call below

            # Determine which API endpoint to call based on model choice
            if current_model == "Dalle + Imagen":
                # Call the combined endpoint for both models
                # For combined, DALL-E gets 1 image, Imagen gets requested number
                response = requests.post(
                    f"{API_URL}/image/generate-combined",
                    json={
                        "prompt": prompt_array,  # Send as array
                        "size": selected_size,
                        "n": num_images,  # Imagen will use this, backend will override DALL-E to 1
                        "model_choice": "",  # Not needed for combined endpoint
                    },
                )
            else:
                # For individual models
                # If using DALL-E, force n=1, otherwise use selected number
                actual_n = 1 if "dall-e" in current_model.lower() else num_images

                # Call the single model endpoint
                response = requests.post(
                    f"{API_URL}/image/generate",
                    json={
                        "prompt": prompt_array,  # Send as array
                        "size": selected_size,
                        "n": actual_n,
                        "model_choice": current_model,
                    },
                )

            if response.status_code == 200:
                return response.json()
            else:
                # Parse structured error response
                error_msg = _parse_image_error_response(response)
                st.error(f"Image generation failed: {error_msg}")
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

    # Initialize image prompt history in session state
    if "image_prompt_history" not in st.session_state:
        st.session_state.image_prompt_history = []

    # Get user inputs
    prompt, selected_size, num_images = get_image_generation_inputs()

    # Display model information
    display_model_information(current_model)

    # Show prompt history if available
    if st.session_state.image_prompt_history:
        st.info(
            f"**Previous prompts:** {len(st.session_state.image_prompt_history)} prompts in history"
        )
        with st.expander("View prompt history"):
            for i, hist_prompt in enumerate(st.session_state.image_prompt_history):
                st.text(f"{i + 1}. {hist_prompt}")

    # Add a generate button
    if st.button("Generate Image", type="primary"):
        if not prompt:
            st.error("Please enter a prompt to generate an image.")
        else:
            # Generate the image with prompt history
            result = generate_image(
                current_model,
                prompt,
                selected_size,
                num_images,
                st.session_state.image_prompt_history,
            )

            # Display results
            if result:
                if current_model == "Dalle + Imagen" and result.get("success"):
                    display_combined_model_results(result, prompt)
                elif result.get("success"):
                    display_single_model_results(result, current_model)

                    # Add successful prompt to history
                    final_prompt = result.get("final_prompt", prompt)
                    st.session_state.image_prompt_history.append(final_prompt)

                    # Show context information if used
                    if result.get("used_context"):
                        context_type = result.get("context_type", "unknown")
                        if context_type == "modification":
                            st.success(
                                "✅ Modified previous image (preserved original style)"
                            )
                        elif context_type == "new_request":
                            st.success("✅ Created new image (ignored previous context)")
                        else:
                            st.success(
                                f"✅ Used context from previous prompts "
                                f"(method: {result.get('rewrite_method', 'unknown')})"
                            )
                        st.info(f"**Final prompt:** {final_prompt}")
                else:
                    # Extract structured error information
                    error_code = result.get("code") or result.get("error_code")
                    error_key = result.get("key") or result.get("error_key")
                    error_msg = result.get("message") or result.get(
                        "error", "Unknown error"
                    )
                    error_details = result.get("details") or result.get("error_details")

                    if error_code and error_key:
                        st.error(f"Error {error_code}: {error_key} - {error_msg}")
                    else:
                        st.error(f"Failed to generate image: {error_msg}")

                    # Display detailed error information if available
                    if error_details:
                        with st.expander("🔍 Error Details"):
                            for key, value in error_details.items():
                                st.text(f"{key}: {value}")

    # Add clear history button
    if st.session_state.image_prompt_history and st.button("Clear Prompt History"):
        st.session_state.image_prompt_history = []
        st.success("Prompt history cleared!")
        st.rerun()
