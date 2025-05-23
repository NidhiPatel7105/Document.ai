import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import cv2
import json

# Set up Streamlit page
st.set_page_config(page_title="Nidhi's Document Extractor", layout="wide")
st.title("📄 Nidhi's Document Extraction Clone")

uploaded_file = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for OpenCV
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Load EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # OCR result: [([[x1,y1], [x2,y2], ...], text, confidence), ...]
    results = reader.readtext(image_np)

    extracted_text = ""
    text_boxes = []

    for bbox, text, conf in results:
        if conf > 0.4:
            extracted_text += text + " "
            # Draw rectangle
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            image_cv = cv2.polylines(image_cv, [pts], True, (0, 255, 0), 2)
            text_boxes.append({
                "text": text,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Annotated Image")
        st.image(image_cv, channels="BGR", use_column_width=True)

    with col2:
        st.subheader("🧾 Extracted Text")
        st.text_area("Text", extracted_text.strip(), height=300)

        st.subheader("📦 JSON Output")
        st.json(text_boxes)