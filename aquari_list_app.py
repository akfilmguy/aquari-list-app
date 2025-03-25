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

st.set_page_config(page_title="Aquari-List", layout="wide", page_icon="📺")

st.title("Aquari-List")

video_file = st.file_uploader("Upload video", type=["mp4", "mov"])
csv_file = st.file_uploader("Upload CSV", type=["csv"])

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
            h, m, s, f = map(int, tc.split(':'))
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

                inputs = processor(images=img, return_tensors="pt").to("cpu")
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
            st.download_button("Download Excel with Descriptions", f, file_name="Aquari-List_Descriptions.xlsx")


# ========== Uploads and Options ==========
st.markdown("""<div style='display: flex; justify-content: center;'>""", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    video_file = st.file_uploader("Upload Video File", type=["mp4", "mov"])
    start_trim = st.number_input("Start Trim Time (seconds)", min_value=0.0, value=0.0, step=1.0)
    output_format = st.selectbox("Select output format", ["Excel (.xlsx)", "CSV (.csv)"])
    include_images = st.checkbox("Include images in file")

with col2:
    csv_file = st.file_uploader("Upload CSV with Timecodes", type=["csv"])
    end_trim = st.number_input("End Trim Time (seconds, 0 = no trim)", min_value=0.0, value=0.0, step=1.0)
    frame_mode = st.selectbox("Frame to use from timecode range", ["Midpoint", "Start", "End"])

st.markdown("""</div>""", unsafe_allow_html=True)

# Metadata
with st.container():
    st.subheader("Optional Metadata")
    meta1, meta2, meta3, meta4, meta5 = st.columns(5)
    project_name = meta1.text_input("Project Name")
    episode_number = meta2.text_input("Episode Number")
    cut_version = meta3.text_input("Cut Version")
    network = meta4.text_input("Network")
    date = meta5.text_input("Date")

# ========== Run ==========
if video_file and csv_file and st.button("Generate Descriptions and Export"):
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, video_file.name)
        csv_path = os.path.join(tmpdir, csv_file.name)
        out_xlsx = os.path.join(tmpdir, "AquariList_Output.xlsx")
        out_csv = os.path.join(tmpdir, "AquariList_Output.csv")
        frame_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        with open(video_path, "wb") as f:
            f.write(video_file.read())
        with open(csv_path, "wb") as f:
            f.write(csv_file.read())

        df = pd.read_csv(csv_path)

        def is_valid_timecode(tc):
            parts = str(tc).split(":")
            return len(parts) == 4 and all(part.isdigit() for part in parts)

        df = df[df['Time Code In'].apply(is_valid_timecode) & df['Time Code Out'].apply(is_valid_timecode)].copy()

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        auto_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = st.number_input("Frames Per Second (FPS)", value=float(auto_fps), step=0.01)
        st.info(f"Auto-detected FPS: {auto_fps:.2f}")

        def timecode_to_seconds(tc, fps):
            h, m, s, f = map(int, tc.split(":"))
            return h * 3600 + m * 60 + s + f / fps

        # Load BLIP model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

        descriptions = []
        image_paths = []
        error_count = 0

        progress_bar = st.progress(0, text="Processing frames...")

        for idx, row in df.iterrows():
            try:
                start_sec = timecode_to_seconds(row['Time Code In'], fps) - 3600
                end_sec = timecode_to_seconds(row['Time Code Out'], fps) - 3600

                if end_trim > 0:
                    if start_sec < start_trim or end_sec > end_trim:
                        descriptions.append("Trimmed")
                        image_paths.append(None)
                        continue

                if frame_mode == "Midpoint":
                    target_sec = (start_sec + end_sec) / 2
                elif frame_mode == "Start":
                    target_sec = start_sec
                else:
                    target_sec = end_sec

                middle_frame = int(target_sec * fps)

                if middle_frame < 0 or middle_frame >= frame_count:
                    descriptions.append("Frame out of range")
                    image_paths.append(None)
                    error_count += 1
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                success, frame = cap.read()

                if not success:
                    descriptions.append("Frame not readable")
                    image_paths.append(None)
                    error_count += 1
                    continue

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Save resized frame
                frame_path = os.path.join(frame_dir, f"frame_{idx}.png")
                img.thumbnail((320, 180))
                img.save(frame_path)
                image_paths.append(frame_path)

                # Generate caption
                inputs = processor(images=[img], return_tensors="pt", padding=True).to("cpu")
                with torch.no_grad():
                    output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)
                descriptions.append(caption)

            except Exception as e:
                descriptions.append(str(e))
                image_paths.append(None)
                error_count += 1

            progress_bar.progress((idx + 1) / len(df), text=f"Processing frame {idx + 1}/{len(df)}")

        progress_bar.empty()
        cap.release()
        df["Description"] = descriptions

        if output_format == "Excel (.xlsx)":
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
                headers += ["Frame"]
            ws.append(headers)

            for i, row in df.iterrows():
                row_data = list(row.values)
                if include_images:
                    row_data += [""]
                ws.append(row_data)

            if include_images:
                for idx, path in enumerate(image_paths):
                    if path:
                        img = XLImage(path)
                        img.width, img.height = 160, 90
                        cell = f"{chr(65 + len(df.columns))}{idx + len(metadata) + 2}"
                        ws.add_image(img, cell)
                        ws.row_dimensions[idx + len(metadata) + 2].height = 70

            wb.save(out_xlsx)
            with open(out_xlsx, "rb") as f:
                st.download_button("Download Excel File", f, file_name="AquariList_Output.xlsx")
        else:
            df.to_csv(out_csv, index=False)
            with open(out_csv, "rb") as f:
                st.download_button("Download CSV File", f, file_name="AquariList_Output.csv")

        st.success(f"✅ Export complete. Total processed: {len(df)}, Errors: {error_count}")

# ========== Footer ==========
st.markdown('<div class="footer-text">©JWright 2025, All rights reserved.</div>', unsafe_allow_html=True)
