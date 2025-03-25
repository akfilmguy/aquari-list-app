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
from openpyxl.styles import Alignment, Font
from openpyxl.utils.dataframe import dataframe_to_rows

# Page config
st.set_page_config(page_title="Aquari-List", layout="wide", page_icon="📺")

# Styling
st.markdown("""
    <style>
        .stApp { background-color: #0492c2; }
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

# Upload fields
video_file = st.file_uploader("Upload your video file", type=["mp4", "mov"])
csv_file = st.file_uploader("Upload your CSV file with timecodes", type=["csv"])
include_images = st.checkbox("Include images in Excel export")

# Metadata section
st.subheader("Optional Metadata")
meta1, meta2, meta3, meta4, meta5 = st.columns(5)
project_name = meta1.text_input("Project Name")
episode_number = meta2.text_input("Episode Number")
cut_version = meta3.text_input("Cut Version")
network = meta4.text_input("Network")
date = meta5.text_input("Date")

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
            try:
                h, m, s, f = map(int, str(tc).split(":"))
                return h * 3600 + m * 60 + s + f / fps
            except:
                return 0

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

        descriptions = []
        image_paths = []

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
                    image_paths.append(None)
                    continue
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if img.mode != "RGB":
                    img = img.convert("RGB")

                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)
                descriptions.append(caption)

                # Save image for embedding
                frame_path = os.path.join(tmpdir, f"frame_{idx}.png")
                img.thumbnail((320, 180))
                img.save(frame_path)
                image_paths.append(frame_path)
            except Exception as e:
                descriptions.append(f"Caption error: {str(e)}")
                image_paths.append(None)

        df["Description"] = descriptions

        output_excel = os.path.join(tmpdir, "output_with_metadata.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "Aquari-List"

        metadata = [
            ["Project Name", project_name],
            ["Episode Number", episode_number],
            ["Cut Version", cut_version],
            ["Network", network],
            ["Date", date],
            []
        ]

        for row in metadata:
            ws.append(row)

        headers = list(df.columns)
        if include_images:
            headers.append("Frame")
        ws.append(headers)

        for i, row in df.iterrows():
            row_data = list(row.values)
            if include_images:
                row_data.append("")
            ws.append(row_data)

        if include_images:
            for idx, path in enumerate(image_paths):
                if path:
                    img = XLImage(path)
                    img.width, img.height = 160, 90
                    cell = f"{chr(65 + len(df.columns))}{idx + len(metadata) + 2}"
                    ws.add_image(img, cell)
                    ws.row_dimensions[idx + len(metadata) + 2].height = 70

        wb.save(output_excel)

        with open(output_excel, "rb") as f:
            st.download_button("📥 Download Excel with Captions", f, file_name="Aquari-List_Descriptions.xlsx")

st.markdown('<div class="footer-text">©JWright 2025, All rights reserved.</div>', unsafe_allow_html=True)
