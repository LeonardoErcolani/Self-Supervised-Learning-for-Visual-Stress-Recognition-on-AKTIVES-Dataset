# Feature Extraction Pipeline

Extracts visual features from preprocessed frames for stress detection analysis.

## 🏗️ Overview

```
Preprocessed Frames → Feature Extraction → Temporal Aggregation → ML-Ready Features
```

## 📁 Structure

```
2.Feature_Extraction/
├── DINOv2_Feature_Extraction/    # Self-supervised visual features
│   ├── 2_Full_Feature_Extraction/    # Full frame analysis
│   ├── 2_Face_Feature_Extraction/    # Face-only analysis
│   └── 2_UpperBody_Feature_Extraction/ # Upper body analysis
├── Emotion_Extraction/            # FER emotion probability extraction
├── Landmark_Extraction/           # MediaPipe facial and pose landmarks
└── temporal_utils.py              # Temporal aggregation utilities
```

## 🔧 Feature Types

### 1. DINOv2 Features
- **Model**: Meta AI DINOv2 Vision Transformer
- **Dimensions**: 768 features per frame
- **Regions**: Full frame, face-only, upper body
- **Output**: `Processed Data/DinoV2 Features Extracted/`

### 2. Emotion Features
- **Model**: Pre-trained facial emotion recognition
- **Features**: 6 emotion probability distributions
- **Classes**: Angry, Fear, Happy, Neutral, Sad, Surprise (disgust excluded, following the FER model)
- **Output**: `Processed Data/FER_Emotion_Features/`

### 3. Expression & Pose Landmarks
- **Tool**: MediaPipe facial and pose analysis
- **Features**: 468 facial landmarks + 33 pose landmarks
- **Output**: `Processed Data/Expression_Pose_Analysis/`

## ⏱️ Temporal Strategies

- **Baseline 10 Seconds**: Single frame at 10th second
- **Average per Interval**: Average features across interval
- **Triple Intervals**: Three samples per interval (early, middle, late)
- **Overlapping Intervals**: 5-second sliding windows

## 🚀 Usage

> Run all commands from the **repository root** (scripts use paths relative to it, e.g. `Processed Data/...`).

### Extract DINOv2 Features
```bash
python 2.Feature_Extraction/DINOv2_Feature_Extraction/2_Full_Feature_Extraction/dinov2_feature_extractor.py
python 2.Feature_Extraction/DINOv2_Feature_Extraction/2_Full_Feature_Extraction/create_all_temporal_features.py
```

### Extract Emotions
```bash
python 2.Feature_Extraction/Emotion_Extraction/fer_emotion_extractor.py
python 2.Feature_Extraction/Emotion_Extraction/create_all_temporal_features.py
```

### Extract Landmarks
```bash
python 2.Feature_Extraction/Landmark_Extraction/facial_expression_pose_analyzer.py
python 2.Feature_Extraction/Landmark_Extraction/create_all_temporal_features.py
```

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 📊 Output Structure

```
Processed Data/
├── DinoV2 Features Extracted/
│   ├── all_features_at_sec_10.csv           # Baseline
│   ├── all_features_per_interval.csv        # Average
│   ├── all_features_triple_intervals.csv    # Triple
│   └── all_features_overlapping_intervals.csv # Overlapping
├── FER_Emotion_Features/
└── Expression_Pose_Analysis/
```
