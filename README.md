# Reducto
## Introduction
This repository contains codes/artifacts for the paper ["Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics"](https://dl.acm.org/doi/10.1145/3387514.3405874), published at ACM DIGITAL LIBRARY. Reducto is a system designed for efficient video summarization by selecting keyframes based on visual similarity and content variation. It employs clustering and frame difference techniques to reduce redundancy while preserving essential information. This repository has been evaluated for effectiveness in video compression and summarization.


## Getting Started

### 1) Directory Structure
```
└── assets 
|   ├── Bitrates                    : Bitrates of all videos
|   ├── F2s                         : Calibrated outputs
|   ├── GroundTruths                : Yolov5x Groundtruths
|   ├── GroundTruths_TileLevel      : StrongSORT-Yolo groundtruths for calibration
|   ├── Runtimes                    : Speed (fps) pickle files 
|   |   ├── On Nano
|   |   ├── On RPi4
|   |
|   ├── labels                      : Labels to calculate accuracy
|   ├── rates.pkl                   
|   ├── ratios.pkl                   
|   ├── UnremovedTileFrameSnip.png               
|   ├── tileRemovedFrameSnip1.png                  
|
└── baselines
|   ├── CloudSeg                    : CloudSeg codes
|   ├── DDS                         : DDS artifacts
|   ├── Reducto                     : Reducto implementation
|   ├── StaticTileRemoval           : STR codes
|   ├── README.md                   
|              
└── src        
|   ├── GT                          : StrongSORT-Yolo codebase
|   ├── scripts                     : Scripts to run TileClipper
|   ├── calibrate.py                : For TileClipper calibration
|   ├── capture.sh                  : Live tiled video encoding from camera 
|   ├── detect_for_groundtruth.py   : Generate labels/GT using  Yolov5x
|   ├── get_results.ipynb           : Generates all plots.
|   ├── live_client.py              : Camera-side code during live experiment
|   ├── live_server.py              : Server-side code during live experiment
|   ├── metric_calculator.py        : Base code for performance metric calculation
|   ├── tileClipper.py              : TileClipper's source code.
|   ├── ratios_withVideoName.pkl
|   ├── requirements.txt
|
└── utils                           : Has addon scripts and codes.  
|
└── videos                          : Available after downloading and extracting the dataset

```

### 2) Overview
This repository includes two primary components:

1. **YOLOv5 Text File Generation**:
   - Detects objects in each frame of a video and stores the results in a text file.
   - Used as input for the frame filtering process.

2. **Reducto Frame Filtering (Counting Query)**:
   - Takes the detected object text file and processes frames to retain only those necessary for achieving the required accuracy.
   - Uses KMeans clustering and frame difference techniques for dynamic threshold selection.



### 3) Dependency Install
```bash
$> git clone https://github.com/ramyashreepramanik45/Reducto
$> cd Reducto                      # for bash
$> pip install -r src/requirements.txt    # installs python libraries
```


### 5) RUsage
Reducto operates on videos. The `videos/` folder . Reducto have 2 Query , for both query demonstration is shown below. Run Reducto on it as:

#### a) Counting Query
This query counts the number of object in frame of video.

##### Step 1) Generate YOLOv5 Detection Text File 
Run the following function to generate object detections for each frame:

###### Execution
```bash
$> python reducto_counting.py
```


```bash
yolovTextgen(video_name, output_txt_path)
```

###### Parameters
video_name: Path to the input video file.
output_txt_path: Path to save the generated text file with detected objects.

###### Example
```bash
yolovTextgen("input.mp4", "detections.txt")
```

##### Step 2) Perform Frame Filtering
Run the Reducto frame filtering function to process frames based on detected objects and accuracy requirements:

```bash
driver(pathVid, pathYolo, accPassed, pathForRes, saveSegements, saveFile)
```

###### Parameters
pathVid: Path to the input video file.
pathYolo: Path to the YOLO-generated text file.
accPassed: Required accuracy threshold.
pathForRes: Path to store the output filtered frames.
saveSegements: Boolean flag to save video segments.
saveFile: Path to store accuracy results in a JSON file.

###### Example
```bash
driver("input.mp4", "detections.txt", 0.85, "filtered_frames/", True, "results.json")
```


#### b) Bounding Box Query

##### Step 1) Generate YOLOv5 Detection Text File 
Run the following function to generate object detections for each frame:

```bash
yolovTextgen(video_name, output_txt_path)
```

###### Parameters
video_name: Path to the input video file.
output_txt_path: Path to save the generated text file with detected objects.

###### Example
```bash
yolovTextgen("input.mp4", "detections.txt")
```

##### Step 2) Perform Frame Filtering
Run the Reducto frame filtering function to process frames based on detected objects and accuracy requirements:

```bash
driver(pathVid, pathYolo, accPassed, pathForRes, saveSegements, saveFile)
```

###### Parameters
pathVid: Path to the input video file.
pathYolo: Path to the YOLO-generated text file.
accPassed: Required accuracy threshold.
pathForRes: Path to store the output filtered frames.
saveSegements: Boolean flag to save video segments.
saveFile: Path to store accuracy results in a JSON file.

###### Example
```bash
driver("input.mp4", "detections.txt", 0.85, "filtered_frames/", True, "results.json")
```
