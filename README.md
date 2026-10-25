# A Self-Supervised Learning approach for Stress Recognition in Children with Special Needs

## 🎯 Research Context

This work addresses the critical challenge of stress recognition during therapeutic activities for children with special needs. Stress can significantly impact a child's ability to engage in physiotherapy or rehabilitation exercises, making early detection essential for optimizing treatment outcomes.


The study builds upon the AKTIVES dataset, a multimodal resource containing physiological signals and video recordings of children with different special needs during serious game therapy, available at [A physiological signal database of children with different special needs for stress recognition](https://www.nature.com/articles/s41597-023-02272-2#Sec10).

**Dataset**: 
Three expert occupational therapists assigned stress Labels as Stress/No Stress each 10 seconds, each label is representative of the 10 seconds interval.

AKTIVES dataset with 52 participants with various conditions:
- Typical Development (20 participants)
- Intellectual Disabilities (15 participants)  
- Dyslexia (13 participants)
- Brachial Plexus Injury (4 participants)


## 🏗️ Technical Architecture

The pipeline implements a novel self-supervised learning approach that addresses the limitations of traditional supervised methods:

```
Raw Videos → Preprocessing → Feature Extraction → Training → Results
    ↓              ↓              ↓                 ↓         ↓
  Expert         Frame      DINOv2/Emotion         ML       Stress
 Annotations   Extraction      /Landmarks         Models   Classification
```

**Innovation**: Uses DINOv2 Vision Transformer for self-supervised visual representation learning, capturing temporal variations that supervised CNNs miss.

## 📁 Repository Structure

```
├── 1.Preprocessing/           # Video processing and label synchronization
├── 2.Feature_Extraction/      # DINOv2, emotions, landmarks
├── 3.Training/                # ML training pipeline (train.py + experiments_config.json)
└── Traditional Methods/       # Baselines: FER, ResNet18/VGG-Face CNN, LSTM
```

## 📦 Data & Model Weights (not included)

This repository contains **code only**. To keep it lightweight and to respect the privacy of the
participants (children with special needs), no videos, frames, face images, feature CSVs, or labels
are distributed here. Running the pipeline regenerates everything under a local `Processed Data/`
folder (created relative to the repository root).

To reproduce the results you need to obtain the inputs separately:

- **AKTIVES dataset** (raw videos + expert annotations): available from the original publication,
  [Coşkun et al. (2023), *Scientific Data*](https://www.nature.com/articles/s41597-023-02272-2#Sec10).
  Place the condition folders (`Brachial Pleksus/`, `Disleksi/`, `Mental Retardasyon/`, `Normal Gelişim/`)
  at the repository root.
- **FER model weights** (`models/adults_8.h5`, `models/kids_8.h5`, `models/mixed_1.h5`): download from the
  original FER repository, [Stress Recognition from Facial Images in Children](https://github.com/FidanVural/Stress-Recognition-from-Facial-Images-in-Children),
  and place them in a `models/` folder at the repository root.

**Run order** (each step writes into `Processed Data/`; run all commands from the repository root):

1. `python 1.Preprocessing/main.py` — labels → frames → face crops
2. Feature extraction (DINOv2, emotions, landmarks) — see `2.Feature_Extraction/README.md`
3. `python 3.Training/train.py` — runs every experiment in `experiments_config.json`

## 🔬 Feature Extraction Methods

### 1. DINOv2 Self-Supervised Features (Primary Innovation)
- **Model**: Meta AI DINOv2 Vision Transformer
- **Dimensions**: 768 features per frame
- **Regions**: Full frame, face-only, upper body
- **Advantage**: Captures temporal variations without manual supervision
- **Performance**: Best overall results across all experiments
- **Reference**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | [Hugging Face Implementation](https://huggingface.co/facebook/dinov2)

### 2. Emotion Features (FER)
- **Model**: Pre-trained facial emotion recognition
- **Features**: 6 emotion probability distributions
- **Classes**: Angry, Fear, Happy, Neutral, Sad, Surprise (disgust excluded, following the FER model)
- **Integration**: Combines effectively with DINOv2 features
- **Reference**: The model used is a pre-trained VGG-Face model found at [Stress Recognition from Facial Images in Children](https://github.com/FidanVural/Stress-Recognition-from-Facial-Images-in-Children)

### 3. Expression & Pose Landmarks
- **Tool**: MediaPipe facial and pose analysis
- **Features**: 468 facial landmarks + 33 pose landmarks
- **Metrics**: Eye aspect ratio, mouth aspect ratio, head tilt, shoulder alignment, face symmetry
- **Use Case**: Geometric features for stress expression analysis
- **Reference**: [MediaPipe](https://mediapipe.dev/)

## ⏱️ Temporal Representation Strategies

The pipeline explores multiple temporal aggregation approaches to capture stress dynamics:

### 1. Baseline 10 Seconds
- **Description**: Single frame at 10th second of each interval
### 2. Average per Interval  
- **Description**: Averaging all features from each 10 frame interval into one.

### 3. Triple Intervals
- **Description**: Three samples per interval (early, middle, late)
### 4. Overlapping Intervals
- **Description**: Three overlapping intervals of 5-seconds.



## 🚀 Quick Start Guide

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

> All commands are run from the **repository root** (scripts use paths relative to it, e.g. `Processed Data/...`).

### 2. Preprocessing Pipeline
```bash
python 1.Preprocessing/main.py

# Optional extras:
# python 1.Preprocessing/experiments/upper_body_extraction.py       # upper-body crops
# python 1.Preprocessing/experiments/label_extraction_unanimous.py  # unanimous-agreement labels
```

### 3. Feature Extraction
```bash
# DINOv2 features (full frame). Face/upper-body variants live in the sibling folders.
python 2.Feature_Extraction/DINOv2_Feature_Extraction/2_Full_Feature_Extraction/dinov2_feature_extractor.py
python 2.Feature_Extraction/DINOv2_Feature_Extraction/2_Full_Feature_Extraction/create_all_temporal_features.py

# Emotions and landmarks (each extractor has a create_all_temporal_features.py for the temporal variants)
python 2.Feature_Extraction/Emotion_Extraction/fer_emotion_extractor.py
python 2.Feature_Extraction/Emotion_Extraction/create_all_temporal_features.py
python 2.Feature_Extraction/Landmark_Extraction/facial_expression_pose_analyzer.py
python 2.Feature_Extraction/Landmark_Extraction/create_all_temporal_features.py
```

### 4. Training & Evaluation
```bash
# Runs every experiment defined in 3.Training/experiments_config.json, in paper order.
python 3.Training/train.py
```
Edit `3.Training/experiments_config.json` to add, remove, or reorder experiments. Results are written to `Results/`.

## 📋 System Requirements

- **Python**: 3.8+ required
- **GPU**: GPU with CUDA support (recommended for DINOv2)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: SSD for faster I/O operations
- **Dependencies**: PyTorch, TensorFlow, OpenCV, MediaPipe, Scikit-learn

## 📚 Documentation

- [Preprocessing Pipeline](1.Preprocessing/README.md) - Video processing and label synchronization
- [Feature Extraction](2.Feature_Extraction/README.md) - Multimodal feature extraction pipeline
- [Training Pipeline](3.Training/README.md) - Unified ML training and evaluation

## 🔗 Citation

If you use this work, please cite:

**This Repository:**

**Preprint:**
_Ercolani, Leonardo and Semerci, Yusuf Can and Erol Barkana, Duygun and Hortal, Enrique, A Self-Supervised Learning Approach for Visual Stress Recognition in Children with Special Needs. Available at SSRN: http://dx.doi.org/10.2139/ssrn.7606964_

```bibtex
@article{SSLaktives2025,
  title={A Self-Supervised Learning approach for Stress Recognition in Children with Special Needs},
  author={Ercolani L.},
  journal={Journal Name},
  year={2025}
}
```

**AKTIVES Dataset:**
```bibtex
@article{coskun2023aktives,
  title={A physiological signal database of children with different special needs for stress recognition},
  author={Coşkun, Buket and Ay, Sevket and Erol Barkana, Duygun and Bostanci, Hilal and Uzun, İsmail and Oktay, Ayse Betul and Tuncel, Basak and Tarakci, Devrim},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={382},
  year={2023},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41597-023-02272-2}
}
```

**Key Technologies:**
- **DINOv2**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | [Hugging Face Implementation](https://huggingface.co/facebook/dinov2)
- **FER Model**: [Stress Recognition from Facial Images in Children](https://github.com/FidanVural/Stress-Recognition-from-Facial-Images-in-Children)
- **MediaPipe**: [Machine Learning Solutions](https://mediapipe.dev/)


**Research Note**: This work achieves state-of-the-art performance in stress recognition for children with special needs using visual data in a transparent, reliable, and reproducible way. The self-supervised learning approach significantly outperforms traditional supervised methods.

**Acknowledgments**: The preprocessing pipeline, particularly the label extraction and expert consensus approach, was inspired by the [AKTIVES Dataset repository](https://github.com/hiddenslate-dev/aktives-dataset-2022/tree/main) by hiddenslate-dev, which provides the original implementation for expert label synchronization and validation.

## 📚 References

### Datasets
- **AKTIVES Dataset**: [Coşkun et al. (2023)](https://www.nature.com/articles/s41597-023-02272-2#Sec10) - A physiological signal database of children with different special needs for stress recognition

### Core Technologies
- **DINOv2**: [Oquab et al. (2023)](https://arxiv.org/abs/2304.07193) - Learning Robust Visual Features without Supervision | [Hugging Face Implementation](https://huggingface.co/facebook/dinov2)
- **FER Model**: [Vural et al.](https://github.com/FidanVural/Stress-Recognition-from-Facial-Images-in-Children) - Stress Recognition from Facial Images in Children
- **MediaPipe**: [Google Research](https://mediapipe.dev/) - Machine Learning Solutions for Computer Vision

