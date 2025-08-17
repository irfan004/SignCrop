import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Set a consistent padding value
PADDING = 50

def process_signature(image_bytes):
    """
    Processes the uploaded signature image to detect, crop, and create a clean,
    solid black signature on a white background.

    Args:
        image_bytes: The byte representation of the uploaded image.

    Returns:
        A tuple containing:
        - original_pil (PIL.Image): The original uploaded image.
        - processed_white_bg (PIL.Image): The cleaned signature on a white background.
    """
    # 1. Load the image with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # 2. Preprocessing and Signature Detection
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological Closing to remove noise and close gaps
    kernel = np.ones((5,5), np.uint8)
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Find Contours and Bounding Box
    contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        st.error("No signature detected. Please try another image with better contrast.")
        return None, None

    # Combine all contour bounding boxes
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # 4. Crop the Signature with Padding
    img_h, img_w, _ = img_cv.shape
    crop_x1 = max(0, x_min - PADDING)
    crop_y1 = max(0, y_min - PADDING)
    crop_x2 = min(img_w, x_max + PADDING)
    crop_y2 = min(img_h, y_max + PADDING)
    
    # We only need the shape of the cropped area for the new canvas
    cropped_shape = (crop_y2 - crop_y1, crop_x2 - crop_x1, 3)
    mask_cropped = img_closed[crop_y1:crop_y2, crop_x1:crop_x2]

    # 5. Create the Clean White Background Version
    # Create a white canvas of the same size as the cropped area
    white_bg = np.full(cropped_shape, 255, dtype=np.uint8)

    # --- KEY CHANGE: Plotting in Black ---
    # Instead of copying original pixels, paint the signature area solid black (0,0,0)
    white_bg[mask_cropped != 0] = (0, 0, 0)
    # --- END OF KEY CHANGE ---
    
    # Convert from BGR (OpenCV) to RGB (Pillow)
    processed_white_bg = Image.fromarray(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))

    return original_pil, processed_white_bg


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Signature Cleaner")

st.title("✒️ Signature Cleaner")
st.markdown("Upload a photo of your signature. The app will automatically crop it and provide a clean version with a white background.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    
    with st.spinner('✨ Processing your signature...'):
        original, white_bg = process_signature(image_bytes)

    if original:
        st.success("✅ Processing complete!")
        
        # Display images in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 1. Original Image")
            st.image(original, use_container_width=True)

        with col2:
            st.markdown("### 2. Cleaned Signature")
            st.image(white_bg, use_container_width=True)

            # Convert PIL image to bytes for download
            buf = io.BytesIO()
            white_bg.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Cleaned Signature (PNG)",
                data=byte_im,
                file_name="signature_cleaned_black.png",
                mime="image/png",
            )
else:
    st.info("☝️ Upload an image to get started.")

