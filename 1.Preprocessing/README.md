# Preprocessing Pipeline

This folder contains the preprocessing pipeline for the AKTIVES dataset. The pipeline processes raw video data and expert annotations to prepare them for feature extraction and machine learning analysis.

## 🏗️ Pipeline Overview

The preprocessing pipeline consists of three main stages that process data sequentially:

```
Raw Data → Label Extraction → Frames Extraction → Face and Upper body extraction
```

## 📁 File Structure

```
1.Preprocessing/
├── main.py                   # Main pipeline orchestrator
├── label_extraction.py       # Expert label processing
├── video_processing.py       # Video frame extraction
├── face_extraction.py        # Face detection and extraction
└── experiments/              # Experimental scripts and variations
```

## 🔧 Components

### 1. Label Extraction (`label_extraction.py`)

**Purpose**: Processes expert annotations from Excel files to create standardized CSV labels.

**Key Features**:
- **Multi-expert consensus**: Combines annotations from multiple experts
- **Label normalization**: Standardizes stress and reaction labels
- **Condition mapping**: Maps Turkish condition names to English equivalents
- **Game standardization**: Normalizes game name variations

**Output**: 
- `Processed Data/Expert Majority/` - Consensus labels for each participant-game combination

**Supported Conditions**:
- Brachial Plexus Injury (`Brachial Pleksus`)
- Dyslexia (`Disleksi`) 
- Intellectual Disability (`Mental Retardasyon`)
- Typical Development (`Normal Gelişim`)

**Supported Games**:
- CatchAPet 
- LeapBall

**Reference**: This label extraction approach was inspired by the [AKTIVES Dataset 2022 repository](https://github.com/hiddenslate-dev/aktives-dataset-2022/tree/main), which provides the original implementation for expert label synchronization and validation. 

### 2. Video Processing (`video_processing.py`)

**Purpose**: Extracts frames from video files at specific time intervals based on expert annotations.

**Key Features**:
- **Temporal alignment**: Extracts frames aligned with expert annotation timestamps
- **Interval-based extraction**: Creates 10-second interval groups
- **Frame organization**: Organizes frames by participant, game, and time interval
- **Automatic cleanup**: Removes existing frame data before processing

**Output**:
- `Processed Data/Frames/` - Organized frame collections for each participant-game-interval

**Frame Structure**:
```
Frames/
├── C1_CP_frames/                    # Participant C1, CatchAPet
│   ├── C1_CP_10/                    # 10-second interval at 10s
│   │   ├── second_1.png             # Frame at 1 second
│   │   ├── second_2.png             # Frame at 2 seconds
│   │   └── ...                      # ... up to 10 seconds
│   └── C1_CP_20/                    # 10-second interval at 20s
└── C1_LP_frames/                    # Participant C1, LeapBall
```

### 3. Face Extraction (`face_extraction.py`)

**Purpose**: Detects and extracts face regions from extracted frames using MediaPipe.

**Key Features**:
- **MediaPipe integration**: Uses Google's MediaPipe for robust face detection
- **Child face selection**: Intelligent selection of the most likely child face
- **Multi-face handling**: Processes cases with multiple detected faces
- **Padding control**: Adds configurable padding around detected faces
- **Quality scoring**: Uses position and size metrics to select optimal faces

**Output**:
- `Processed Data/Face Data/` - Face crops organized by participant, game, and interval

**Face Selection Algorithm**:
- **Position score**: Faces closer to image center get higher scores
- **Size score**: Larger faces (likely closer to camera) get higher scores
- **Combined scoring**: 70% size weight + 30% position weight



## 🚀 Usage

### Quick Start

Run the complete preprocessing pipeline from the **repository root** (`main.py` uses the current
working directory as the data root, so the condition folders and `Processed Data/` resolve correctly):

```bash
python 1.Preprocessing/main.py
```

### Step-by-Step Execution

If you want to run stages individually:

```python
from label_extraction import LabelExtractor
from video_processing import VideoProcessor
from face_extraction import FaceExtractor

# Step 1: Process labels
label_extractor = LabelExtractor(root_dir)
label_extractor.process_all_labels()

# Step 2: Extract video frames
video_processor = VideoProcessor(root_dir)
video_processor.process_all_videos()

# Step 3: Extract faces
face_extractor = FaceExtractor(root_dir)
face_extractor.process_all_faces()
```

### Input Data Structure

The pipeline expects the following directory structure:

```
Root Directory/
├── Brachial Pleksus/
│   ├── NameParticipant1/
│   │   ├── CatchAPet/
│   │   │   └── video.mp4
│   │   └── LeapBall/
│   │       └── video.mp4
│   └── NameParticipant2/
├── Disleksi/
├── Mental Retardasyon/
├── Normal Gelişim/
└── Processed Data/  # Created automatically
```

## 📊 Output Structure

After running the pipeline, you'll have:

```
Processed Data/
├── Expert Majority/           # Consensus expert labels
├── Frames/                   # Extracted video frames
└── Face Data/                # Extracted face crops
```


## 🔗 Next Steps

After preprocessing, the data is ready for:

1. **Feature Extraction**: DINOv2 embeddings, emotion probabilities, landmarks
2. **Temporal Aggregation**: Creating interval-based feature representations
3. **Machine Learning**: Training stress classification models


---

**Note**: This preprocessing pipeline is designed specifically for the AKTIVES dataset structure.
