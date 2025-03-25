# 🐠 Aquari-List

**Aquari-List** is a smart video analysis tool that reads timecoded video segments from a CSV and uses AI to generate natural-language descriptions of selected frames. It's designed for editors, producers, and archivists who need quick, accurate descriptions of footage.

![screenshot](https://via.placeholder.com/800x400?text=Aquari-List+App+Demo)

---

## ✨ Features

- 🎞 Upload a video file (`.mp4`, `.mov`)
- 📋 Upload a timecoded CSV with `Time Code In` and `Time Code Out`
- 🧠 AI-generated captions using [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- 🖼 Frame images embedded directly into an Excel export (optional)
- 🧾 Choose between Excel or CSV output
- 🎛 Set trim range, FPS, and frame selection mode
- 🎨 Clean UI with branding, dark mode toggle, and cerulean theme

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/aquari-list-app.git
cd aquari-list-app