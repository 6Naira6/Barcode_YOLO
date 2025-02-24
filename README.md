# Barcode Detection with YOLO

## Overview

This project implements a **Barcode Detection** system using the **YOLO (You Only Look Once)** object detection framework. The system accurately detects and localizes barcodes in images and videos.

## Features

- **Real-time Barcode Detection**: Leveraging YOLO for fast, real-time detection.
- **Pre-trained Model Included**: Provided model (`best.pt`) allows immediate barcode detection.
- **Python-based Implementation**: Easy to run, modify, and integrate with other applications.

## Repository Contents

- `Barcode_1.py`: Barcode detection using YOLO.
- `Barcode_2.py`: Additional barcode detection functionalities.
- `Straight.py`: Image preprocessing or result post-processing.
- `best.pt`: Pre-trained YOLO model optimized for barcode detection.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/6Naira6/Barcode_YOLO.git
   cd Barcode_YOLO
   ```

2. **Set Up Environment:**
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download YOLOv5 Repository:**
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cp best.pt yolov5/
   ```

## Usage

Navigate to the YOLOv5 directory and use the following commands:

- **Detect barcodes in an image:**
  ```bash
  python detect.py --weights best.pt --img 640 --conf 0.25 --source path/to/your/image.jpg
  ```

- **Detect barcodes in a video:**
  ```bash
  python detect.py --weights best.pt --img 640 --conf 0.25 --source path/to/your/video.mp4
  ```

- **Use Webcam:**
  ```bash
  python detect.py --weights best.pt --img 640 --conf 0.25 --source 0
  ```

Detection results are saved in `runs/detect`.

## Train Custom Model

To train the YOLO model with your custom barcode dataset:

1. **Prepare Your Dataset:**
   - Organize images and annotations in YOLO format.
   - Create a YAML file (`barcode.yaml`) specifying dataset paths.

2. **Train the Model:**
   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data barcode.yaml --weights yolov5s.pt
   ```
---

*Ensure you have necessary permissions to distribute the `best.pt` model and associated data.*

