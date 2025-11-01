# 🧠 Assistive OCR-TTS v2.0.2

### Empowering Visually Impaired Students Through Real-Time Classroom Assistance

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![Status](https://img.shields.io/badge/Status-Ongoing-yellow)

---

## 📘 Project Overview

**Assistive OCR-TTS v2.0.2** is an **AI-powered assistive system** designed to help **visually impaired students** hear what’s written on classroom blackboards in real time. The system captures live camera footage, extracts text using **Optical Character Recognition (OCR)**, and converts it to **speech** through a multilingual **Text-to-Speech (TTS)** engine.

This project is part of **NAIN 2.0**, a **Karnataka State Government–funded initiative** that encourages innovation among engineering students by solving real-world societal problems through technology.

---

## 🎯 Objectives

* Enable visually impaired students to understand written content from classroom blackboards.
* Support **multilingual OCR and TTS**, including **English, Hindi, Kannada, and Marathi**.
* Deliver **real-time, high-accuracy OCR** and **natural-sounding speech output**.
* Integrate the system into a **wearable smart-glasses device** powered by **NVIDIA Jetson Orin Nano**.

---

## 🧩 Core Features

✅ **Live Text Capture** — Real-time video input via high-definition camera
✅ **Accurate OCR** — Advanced text extraction using PaddleOCR / Tesseract
✅ **Multilingual Support** — Dynamic text translation before TTS conversion
✅ **Text-to-Speech (TTS)** — Converts recognized text to audio in user’s preferred language
✅ **Web-Based Control Interface** — Adjust camera, brightness, and preview settings in real-time
✅ **Optimized for Jetson Orin Nano** — GPU-accelerated inference for faster processing

---

## ⚙️ System Architecture

1. **Camera Input** — Captures live classroom blackboard feed
2. **OCR Engine** — Detects and converts text to digital format
3. **Language Processor** — Translates text to selected language
4. **TTS Engine** — Converts text into natural audio output
5. **Audio Output Module** — Plays audio through integrated headset or earphones

---

## 🧠 Technologies Used

| Category         | Technology / Library                         |
| ---------------- | -------------------------------------------- |
| OCR              | PaddleOCR, pytesseract                       |
| Object Detection | YOLOv8 (ultralytics)                         |
| TTS              | Coqui-TTS, eSpeak-NG                         |
| Web Framework    | Flask / FastAPI                              |
| Hardware         | NVIDIA Jetson Orin Nano, Arducam 4K HDR MIPI |
| Audio            | sounddevice, simpleaudio                     |
| Helpers          | OpenCV, NumPy, Symspellpy, RapidFuzz         |

---

## 🧰 Hardware Requirements

* **NVIDIA Jetson Orin Nano Developer Kit**
* **Arducam 4K HDR MIPI (B0586)** or **e-con Systems e-CAM86_CUONX (IMX678)**
* **Stereo Audio Output Device** (Bone conduction or Bluetooth-enabled)
* **5V Power Supply / Battery Module**

---

## 💻 Software Requirements

* **Python 3.11+**
* **Flask / FastAPI**
* **OpenCV**, **NumPy**, **PaddleOCR**, **pyttsx3**, **Coqui-TTS**, **sounddevice**

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python 1ocr_assistive_tech.py
```

Access the web interface:

```
http://localhost:5000
```

---

## 🧪 Current Progress (v2.0.2)

* ✅ Implemented in-memory frame storage
* ✅ Integrated MJPEG live streaming
* ✅ Added live color preview
* ✅ Enhanced real-time OCR accuracy
* ✅ Optimized TTS pipeline
* 🚧 In progress: Handwritten OCR, Bluetooth integration, and glasses-mounted device testing

---

## 🚀 Future Enhancements

* Integration with **e-con Systems IMX678 camera** for better low-light performance
* Real-time **handwriting detection** from classroom boards
* **Bluetooth communication** between Jetson and wearable device
* Offline multilingual translation module

---

## 🏫 Funding & Acknowledgment

This project is funded under the **NAIN (New Age Innovation Network) 2.0 Program**,
**Department of Electronics, IT, Bt and S&T**, **Government of Karnataka**.

Developed by **Ctrl Developers**,
Department of Computer Science and Engineering,
KLE Institute of Technology, Hubballi.

---

## 📬 Contact

**Project Lead:** Ronit Bongale
**GitHub:** [RonitRB](https://github.com/RonitRB)
**Email:** [ronitbongale@gmail.com](mailto:ronitbongale@gmail.com)
