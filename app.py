import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

# Load trained YOLOv8 model
model = YOLO("best.pt")  # Use your trained YOLO model

st.title("🦷 Dental Bone Loss Detection")
st.write("Upload a dental X-ray image to detect bone loss using YOLOv8.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    results = model(temp_path)
    annotated = results[0].plot()

    st.image(annotated, caption="🦷 Detected Bone Loss", use_column_width=True)

    with st.expander("📊 Detection Details"):
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            st.write(f"Class: {model.names[int(class_id)]}, Confidence: {score:.2f}")
