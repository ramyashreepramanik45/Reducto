# Reducto

## Introduction
This repository contains code and artifacts for the paper ["Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics"](https://dl.acm.org/doi/10.1145/3387514.3405874), published in the ACM Digital Library. 

**Reducto** is a system designed for efficient video summarization by selecting keyframes based on visual similarity and content variation. It employs clustering and frame difference techniques to reduce redundancy while preserving essential information. This repository has been evaluated for its effectiveness in video compression and summarization.

---

## Getting Started

<!-- ### Directory Structure
```bash
.
├── assets 
│   ├── Bitrates                    # Bitrates of all videos
│   ├── F2s                         # Calibrated outputs
│   ├── GroundTruths                # YOLOv5x ground truths
│   ├── GroundTruths_TileLevel      # StrongSORT-YOLO ground truths for calibration
│   ├── Runtimes                    # Speed (fps) pickle files 
│   │   ├── On Nano
│   │   ├── On RPi4
│   ├── labels                      # Labels to calculate accuracy
│   ├── rates.pkl                   
│   ├── ratios.pkl                   
│   ├── UnremovedTileFrameSnip.png  
│   ├── tileRemovedFrameSnip1.png   
│
├── baselines
│   ├── CloudSeg                    # CloudSeg codes
│   ├── DDS                         # DDS artifacts
│   ├── Reducto                     # Reducto implementation
│   ├── StaticTileRemoval           # STR codes
│   ├── README.md                   
│              
├── src        
│   ├── GT                          # StrongSORT-YOLO codebase
│   ├── scripts                     # Scripts to run TileClipper
│   ├── calibrate.py                # TileClipper calibration script
│   ├── capture.sh                  # Live tiled video encoding from camera 
│   ├── detect_for_groundtruth.py   # Generate labels/GT using YOLOv5x
│   ├── get_results.ipynb           # Generates all plots
│   ├── live_client.py              # Camera-side code during live experiments
│   ├── live_server.py              # Server-side code during live experiments
│   ├── metric_calculator.py        # Performance metric calculation
│   ├── tileClipper.py              # TileClipper's source code
│   ├── ratios_withVideoName.pkl    
│   ├── requirements.txt            # Python dependencies
│
├── utils                           # Additional scripts and utilities
│
└── videos                          # Available after downloading and extracting the dataset
``` -->

---

## Overview
This repository includes two primary components:

### 1. **YOLOv5 Text File Generation**
- Detects objects in each frame of a video and stores the results in a text file.
- Used as input for the frame filtering process.

### 2. **Reducto Frame Filtering (Counting Query)**
- Processes frames to retain only those necessary for achieving the required accuracy.
- Uses KMeans clustering and frame difference techniques for dynamic threshold selection.

---

## Installation
```bash
# Clone the repository
git clone https://github.com/ramyashreepramanik45/Reducto
cd Reducto  # Navigate to the project directory

# Install required Python libraries
pip install -r src/requirements.txt
```

---

## Usage
### Running Reducto on Videos
Reducto has two types of queries:

### **1. Counting Query**
This query counts the number of objects in each video frame.

#### Step 1: Generate YOLOv5 Detection Text File
Run the following command to generate object detections for each frame:
```bash
python yolovTxtGen.py
```
To change the video name and output text file path, modify the `yolovTextgen` function in `yolovTxtGen.py`:
```python
yolovTextgen(video_name, output_txt_path)
```

**Parameters:**
- `video_name`: Path to the input video file.
- `output_txt_path`: Path to save the generated text file with detected objects.

**Example:**
```python
yolovTextgen("input.mp4", "detections.txt")
```

#### Step 2: Perform Frame Filtering
Run the Reducto frame filtering function to process frames based on detected objects:
```bash
python reducto_couting.py
```
To change the parameters, modify the `driver` function in `reducto_couting.py`:
```python
driver(pathVid, pathYolo, accPassed, pathForRes, saveSegments, saveFile)
```

**Parameters:**
- `pathVid`: Path to the input video file.
- `pathYolo`: Path to the YOLO-generated text file.
- `accPassed`: Required accuracy threshold.
- `pathForRes`: Path to store the output filtered frames.
- `saveSegments`: Boolean flag to save video segments.
- `saveFile`: Path to store accuracy results in a JSON file.

**Example:**
```python
driver("input.mp4", "detections.txt", 0.85, "filtered_frames/", True, "results.json")
```

---

### **2. Bounding Box Query**
This query extracts bounding boxes of objects in the video.

#### Step 1: Generate DeepSORT Text File from Video
Run the following command to generate bounding box information:
```bash
python deepsort2txt.py
```
Modify the arguments in `deepsort2txt.py` to customize the process:
```python
process_video(video_path)
fill_missing_frames(INPUT_PATH, OUTPUT_PATH)
```

**Parameters:**
- `video_path`: Path to the input video file.
- `INPUT_PATH`: Path to input CSV file.
- `OUTPUT_PATH`: Path to output CSV file.

#### Step 2: Perform Frame Filtering
Run the Reducto frame filtering function for bounding box queries:
```bash
python Reducto_boundingbox.py
```
Modify the `driver` function in `Reducto_boundingbox.py` to adjust parameters:
```python
driver(pathVid, pathYolo, accPassed, pathForRes, saveSegments, saveFile)
```

**Example:**
```python
driver("input.mp4", "detections.txt", 0.85, "filtered_frames/", True, "results.json")
```

---

## Conclusion
Reducto is a powerful tool for video summarization and object detection, providing efficient resource usage while maintaining accuracy. Follow the steps above to get started with Reducto and optimize your video analytics workflow.
