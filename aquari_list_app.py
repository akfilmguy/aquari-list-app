import streamlit as st
import pandas as pd
import cv2
import os
import tempfile
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Font

# Page settings
st.set_page_config(page_title="Aquari-List", layout="wide", page_icon="📺")

# Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0492c2;
        }
        .title-text {
            font-family: 'American Typewriter', serif;
            font-size: 36pt;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }
        .footer-text {
            font-size: 10pt;
            text-align: center;
            margin-top: 60px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">Aquari-List</div>', unsafe_allow_html=True)

# Upload section
video_file = st.file_uploader("Upload your video file", type=["mp4", "mov"])
csv_file = st.file_uploader("Upload your CSV file with timecodes", type=["csv"])

if video_file and csv_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, video_file.name)
        csv_path = os.path.join(tmpdir, csv_file.name)

        with open(video_path, "wb") as f:
            f.write(video_file.read())
        with open(csv_path, "wb") as f:
            f.write(csv_file.read())

        df = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        def timecode_to_seconds(tc):
            h, m, s, f = map(int, str(tc).split(":")) if isinstance(tc, str) else (0, 0, 0, 0)
            return h * 3600 + m * 60 + s + f / fps

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

        descriptions = []

        for idx, row in df.iterrows():
            try:
                start_tc = row['Time Code In']
                end_tc = row['Time Code Out']
                mid_sec = (timecode_to_seconds(start_tc) + timecode_to_seconds(end_tc)) / 2.0
                frame_num = int(mid_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    descriptions.append("Failed to read frame.")
                    continue
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Proper BLIP call
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)
                descriptions.append(caption)
            except Exception as e:
                descriptions.append(f"Caption error: {str(e)}")

        df["Description"] = descriptions

        output_excel = os.path.join(tmpdir, "output.xlsx")
        df.to_excel(output_excel, index=False)

        with open(output_excel, "rb") as f:
            st.download_button("📥 Download Excel with Captions", f, file_name="Aquari-List_Descriptions.xlsx")

st.markdown('<div class="footer-text">©JWright 2025, All rights reserved.</div>', unsafe_allow_html=True)
