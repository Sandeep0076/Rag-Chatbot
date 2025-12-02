# New functions to extract from main() to reduce complexity
import base64
import logging

import requests
import streamlit as st

# Define API URL as it's used across functions
API_URL = "http://localhost:8080"

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
            "image": ["dall-e-3", "imagen", "NanoBanana", "Dalle + Imagen"],
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
        elif "nanobanana" in current_model.lower():
            st.markdown(
                "[Gemini Image Generation Guide](https://ai.google.dev/gemini-api/docs/image-generation) - "
                "Learn how to craft effective prompts for Gemini image generation."
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
    """Get user inputs for image generation: prompt, size, and optional input image."""
    # Get the current model
    current_model = st.session_state.model_choice

    # Check if this is NanoBanana (supports image editing)
    is_nanobanana = "nanobanana" in current_model.lower()

    # Image upload section for NanoBanana only
    uploaded_images = None
    input_image_base64 = None

    if is_nanobanana:
        st.markdown("### 🎨 Image Editing (Optional)")
        st.info(
            "💡 NanoBanana supports image-to-image editing! Upload 1-3 images to modify them together,"
            " or leave empty for text-to-image generation."
        )

        uploaded_images = st.file_uploader(
            "Upload images to edit (optional):",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Upload 1-3 images to modify together based on your prompt. Max 10MB per image.",
        )

        if uploaded_images is not None and len(uploaded_images) > 0:
            # Limit to 3 images
            if len(uploaded_images) > 3:
                st.error(
                    f"❌ Too many images uploaded ({len(uploaded_images)}). Maximum 3 images allowed."
                )
                uploaded_images = None
                input_image_base64 = None
            else:
                # Process all uploaded images
                input_image_base64 = []
                valid_images = True

                for idx, uploaded_image in enumerate(uploaded_images):
                    # Validate file size (10MB limit)
                    file_size_mb = uploaded_image.size / (1024 * 1024)
                    if file_size_mb > 10:
                        st.error(
                            f"❌ Image {idx + 1} size ({file_size_mb:.2f}MB) exceeds 10MB limit. "
                            "Please upload a smaller image."
                        )
                        valid_images = False
                        break
                    else:
                        # Convert to base64
                        image_bytes = uploaded_image.read()
                        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")

                        # Determine MIME type
                        mime_type = uploaded_image.type
                        image_data_uri = f"data:{mime_type};base64,{base64_encoded}"
                        input_image_base64.append(image_data_uri)

                        # Show preview
                        st.image(
                            uploaded_image,
                            caption=f"Reference Image {idx + 1} for Editing ({file_size_mb:.2f}MB)",
                            use_column_width=True,
                        )

                        # Reset file pointer for potential re-reading
                        uploaded_image.seek(0)

                if valid_images:
                    if len(input_image_base64) == 1:
                        st.success(
                            f"✅ {len(input_image_base64)} image uploaded successfully"
                        )
                    else:
                        st.success(
                            f"✅ {len(input_image_base64)} images uploaded successfully"
                        )
                else:
                    input_image_base64 = None

    # Create a text area for the prompt
    has_images = input_image_base64 is not None and (
        (isinstance(input_image_base64, list) and len(input_image_base64) > 0)
        or (isinstance(input_image_base64, str) and input_image_base64)
    )
    prompt_label = (
        "Enter a prompt to modify the image(s):"
        if (is_nanobanana and has_images)
        else "Enter a prompt describing the image you want to generate:"
    )
    prompt_placeholder = (
        "Example: Take the couple from the first picture and make them stand near that stone building..."
        if (is_nanobanana and has_images)
        else "Example: A photo of a cat in space..."
    )

    prompt = st.text_area(
        prompt_label,
        placeholder=prompt_placeholder,
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

    # Add a dropdown for number of images only for Imagen and NanoBanana models
    num_images = 1  # Default for DALL-E and combined
    with col2:
        # The combined model has a specific name "Dalle + Imagen"
        is_combined = current_model == "Dalle + Imagen"

        # Check if this is a pure Imagen or NanoBanana model
        # This will match 'imagen', 'imagen-3.0-generate-002', 'imagen-1.5-pro-002', etc.
        is_pure_imagen = "imagen" in current_model.lower() and not is_combined

        if is_pure_imagen:
            # Show number selection for Imagen models (1-4)
            num_images = st.selectbox("Number of images:", [1, 2, 3, 4], index=0)
        elif is_nanobanana:
            # Show number selection for NanoBanana models (1-10)
            num_images = st.selectbox(
                "Number of images:", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=0
            )
            if num_images > 4:
                st.info(
                    f"🚀 NanoBanana supports generating up to 10 images per request (selected: {num_images})"
                )
        else:
            # For DALL-E or combined, show a static message since only 1 image is allowed
            st.info("DALL-E 3 supports generating 1 image per request")

    return prompt, selected_size, num_images, input_image_base64


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
    current_model,
    prompt,
    selected_size,
    num_images,
    prompt_history=None,
    input_image_base64=None,
    session_id=None,
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

                # Build request payload
                payload = {
                    "prompt": prompt_array,  # Send as array
                    "size": selected_size,
                    "n": actual_n,
                    "model_choice": current_model,
                }

                # Add optional parameters for NanoBanana image editing
                if input_image_base64:
                    payload["input_image_base64"] = input_image_base64
                if session_id:
                    payload["session_id"] = session_id

                # Call the single model endpoint
                response = requests.post(
                    f"{API_URL}/image/generate",
                    json=payload,
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


def _initialize_image_session_state():
    """Initialize session state variables for image generation."""
    if "image_prompt_history" not in st.session_state:
        st.session_state.image_prompt_history = []
    if "image_generation_history" not in st.session_state:
        st.session_state.image_generation_history = []
    if "image_session_id" not in st.session_state:
        import uuid

        st.session_state.image_session_id = str(uuid.uuid4())


def _display_prompt_history():
    """Display the prompt history if available."""
    if st.session_state.image_prompt_history:
        st.info(
            f"**Previous prompts:** {len(st.session_state.image_prompt_history)} prompts in history"
        )
        with st.expander("View prompt history"):
            for i, hist_prompt in enumerate(st.session_state.image_prompt_history):
                st.text(f"{i + 1}. {hist_prompt}")


def _display_image_history_for_nanobanana(is_nanobanana, current_model):
    """Display image history specifically for NanoBanana models."""
    if is_nanobanana and st.session_state.image_generation_history:
        st.info(
            f"**Generated images:** {len(st.session_state.image_generation_history)} images in session"
        )
        with st.expander("View previous generated images"):
            for i, img_data in enumerate(st.session_state.image_generation_history):
                st.image(
                    img_data, caption=f"Generated Image {i + 1}", use_column_width=True
                )


def _prepare_input_image_for_nanobanana(is_nanobanana, input_image_base64):
    """Prepare the final input image(s) for NanoBanana, using history if needed."""
    final_input_image = input_image_base64

    if is_nanobanana:
        # Check if this is a follow-up edit (has prompt history)
        has_prompt_history = (
            hasattr(st.session_state, "image_prompt_history")
            and st.session_state.image_prompt_history
        )

        # For follow-up edits, prioritize last generated image over uploaded image(s)
        if has_prompt_history and st.session_state.image_generation_history:
            # Use the most recent generated image as reference for follow-up edits
            # Convert to list format for consistency
            final_input_image = [st.session_state.image_generation_history[-1]]
            st.info("ℹ️ Using the last generated image as reference for modification")
        elif not input_image_base64 and st.session_state.image_generation_history:
            # No uploaded image, but we have history - use last generated image
            final_input_image = [st.session_state.image_generation_history[-1]]
            st.info("ℹ️ Using the last generated image as reference for modification")
        elif input_image_base64:
            # Normalize to list format if it's a single string
            if isinstance(input_image_base64, str):
                final_input_image = [input_image_base64]
            # If it's already a list, keep it as is
            elif isinstance(input_image_base64, list):
                final_input_image = input_image_base64

    return final_input_image


def _update_prompt_history(result, prompt):
    """Update the prompt history after successful image generation."""
    final_prompt = result.get("final_prompt", prompt)
    st.session_state.image_prompt_history.append(final_prompt)
    return final_prompt


def _update_image_history_for_nanobanana(is_nanobanana, result):
    """Update image generation history for NanoBanana models."""
    if is_nanobanana:
        image_urls = result.get("image_urls", [])
        if image_urls:
            # Store the first generated image for future reference
            st.session_state.image_generation_history.append(image_urls[0])
            # Keep only last 5 images to avoid memory issues
            if len(st.session_state.image_generation_history) > 5:
                st.session_state.image_generation_history = (
                    st.session_state.image_generation_history[-5:]
                )


def _display_operation_type_info(result):
    """Display information about the operation type (text-to-image vs image-to-image)."""
    operation_type = result.get("operation_type", "text_to_image")
    reference_used = result.get("reference_image_used", False)

    if operation_type == "image_to_image":
        st.success("🎨 Image-to-image editing completed!")
        if reference_used:
            st.info("✅ Used uploaded/previous image as reference")
    else:
        st.success("🖼️ Text-to-image generation completed!")


def _display_context_information(result, final_prompt):
    """Display context information if context was used in generation."""
    if result.get("used_context"):
        context_type = result.get("context_type", "unknown")
        if context_type == "modification":
            st.success("✅ Modified previous image (preserved original style)")
        elif context_type == "new_request":
            st.success("✅ Created new image (ignored previous context)")
        else:
            st.success(
                f"✅ Used context from previous prompts "
                f"(method: {result.get('rewrite_method', 'unknown')})"
            )
        st.info(f"**Final prompt:** {final_prompt}")


def _handle_successful_generation(result, current_model, prompt, is_nanobanana):
    """Handle successful image generation results."""
    if current_model == "Dalle + Imagen" and result.get("success"):
        display_combined_model_results(result, prompt)
    elif result.get("success"):
        display_single_model_results(result, current_model)

        # Add successful prompt to history
        final_prompt = _update_prompt_history(result, prompt)

        # Store generated images in session history for NanoBanana
        _update_image_history_for_nanobanana(is_nanobanana, result)

        # Show operation type information
        _display_operation_type_info(result)

        # Show context information if used
        _display_context_information(result, final_prompt)


def _handle_generation_error(result):
    """Handle error cases in image generation."""
    # Extract structured error information
    error_code = result.get("code") or result.get("error_code")
    error_key = result.get("key") or result.get("error_key")
    error_msg = result.get("message") or result.get("error", "Unknown error")
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


def _handle_generation_button_click(
    current_model, prompt, selected_size, num_images, input_image_base64, is_nanobanana
):
    """Handle the image generation button click event."""
    if not prompt:
        st.error("Please enter a prompt to generate an image.")
        return

    # Prepare input image for NanoBanana
    final_input_image = _prepare_input_image_for_nanobanana(
        is_nanobanana, input_image_base64
    )

    logger.info(f"Generating image with {current_model}")

    # Generate the image with prompt history and optional input image
    result = generate_image(
        current_model,
        prompt,
        selected_size,
        num_images,
        st.session_state.image_prompt_history,
        final_input_image,
        st.session_state.image_session_id,
    )

    # Handle results
    if result:
        if result.get("success"):
            _handle_successful_generation(result, current_model, prompt, is_nanobanana)
        else:
            _handle_generation_error(result)


def _handle_clear_history_button():
    """Handle the clear history button click event."""
    if (
        st.session_state.image_prompt_history
        or st.session_state.image_generation_history
    ) and st.button("Clear History"):
        st.session_state.image_prompt_history = []
        st.session_state.image_generation_history = []
        # Generate new session ID
        import uuid

        st.session_state.image_session_id = str(uuid.uuid4())
        st.success("History cleared and new session started!")
        st.rerun()


def _validate_prerequisites():
    """Validate prerequisites for image generation."""
    current_model = validate_image_model()
    if not current_model:
        logger.warning("Image model validation failed")
        return None

    if not st.session_state.username:
        logger.warning("Username not provided")
        st.error("Username is required. Please enter a username in the sidebar.")
        return None

    return current_model


def _setup_ui_components(current_model):
    """Setup UI components for image generation."""
    # Show title with current model
    st.markdown(
        f'<div class="subheader">Generate images using {current_model}</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state for image generation
    _initialize_image_session_state()

    # Get user inputs (now includes input_image_base64)
    inputs = get_image_generation_inputs()
    return inputs


def _display_interface_elements(current_model):
    """Display interface elements including model info and history."""
    # Display model information
    display_model_information(current_model)

    # Show prompt history if available
    _display_prompt_history()

    # Show image history for NanoBanana
    is_nanobanana = "nanobanana" in current_model.lower()
    _display_image_history_for_nanobanana(is_nanobanana, current_model)

    return is_nanobanana


def _handle_action_buttons(
    current_model, prompt, selected_size, num_images, input_image_base64, is_nanobanana
):
    """Handle action buttons for image generation and history clearing."""
    # Add a generate button
    if st.button("Generate Image", type="primary"):
        _handle_generation_button_click(
            current_model,
            prompt,
            selected_size,
            num_images,
            input_image_base64,
            is_nanobanana,
        )

    # Add clear history button
    _handle_clear_history_button()


def handle_image_generation():
    """Handle the image generation functionality."""
    # Validate prerequisites
    current_model = _validate_prerequisites()
    if not current_model:
        logger.error("Prerequisites validation failed, exiting")
        return

    # Setup UI components and get inputs
    (prompt, selected_size, num_images, input_image_base64) = _setup_ui_components(
        current_model
    )

    # Display interface elements
    is_nanobanana = _display_interface_elements(current_model)

    # Handle action buttons
    _handle_action_buttons(
        current_model,
        prompt,
        selected_size,
        num_images,
        input_image_base64,
        is_nanobanana,
    )
