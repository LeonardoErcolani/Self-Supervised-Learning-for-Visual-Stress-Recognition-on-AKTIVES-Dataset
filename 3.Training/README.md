# Training Pipeline

This folder contains the unified training pipeline for stress classification experiments using the AKTIVES dataset. The pipeline consolidates all training functionality into a single, flexible system that can run any combination of features, temporal strategies, and machine learning models.

## 🏗️ Pipeline Overview

The training pipeline provides a unified approach to stress classification experiments:

```
Feature Data → Preprocessing → Model Training → Evaluation → Results
```

## 📁 File Structure

```
3.Training/
├── README.md                    # This file
├── train.py                     # Runs every experiment listed in experiments_config.json
├── experiments_config.json      # The full list of experiments (edit this to add/remove/reorder)
└── unified_experiment_runner.py # Core training engine and utilities
```

## 🔧 Components

### 1. Main Training Script (`train.py`)

**Purpose**: User-friendly interface to run experiments.

**Key Features**:
- **Pre-configured experiments** loaded from `experiments_config.json`
- **Simple experiment selection** and modification
- **Clear output formatting** with progress tracking
- **Error handling** and troubleshooting guidance

**Experiment Types**:
- **Single features**: DINOv2, Emotions, Landmarks
- **Combined features**: DINOv2 + Emotions, DINOv2 + Landmarks
- **Multiple temporal strategies**: Baseline, Average, Triple, Overlapping
- **PCA options**: With/without dimensionality reduction

### 2. Unified Experiment Runner (`unified_experiment_runner.py`)

**Purpose**: Core training engine that handles all experiment execution.

**Key Features**:
- **Data preprocessing** and validation
- **Participant-based splitting** for robust evaluation
- **Multiple ML models** with consistent evaluation
- **Temporal feature handling** for different aggregation strategies
- **Comprehensive metrics** and visualization
- **Automatic result saving** and organization


## 🚀 Usage

### Quick Start

Run all experiments defined in `experiments_config.json` (run from the **repository root** so the `Processed Data/...` paths resolve):

```bash
python 3.Training/train.py
```

### Customizing Experiments

The experiment list lives in `3.Training/experiments_config.json`. Add, remove, or reorder
entries there — `train.py` runs them top to bottom. Each entry is a JSON object:

```json
{
    "name": "My_Custom_Experiment",
    "type": "single" , // or "combined"
    "input_csv": "path/to/features.csv",
    "feature_type": "DINOv2", // or "Emotions", "Expression_Pose"
    "temporal_strategy": "Overlapping intervals", // or single, average or triple
    "use_pca": true
}
```

### Experiment Configuration Options

#### **Single Feature Experiments**
```json
{
    "name": "Experiment_Name",
    "type": "single",
    "input_csv": "path/to/features.csv",
    "feature_type": "DINOv2",
    "temporal_strategy": "Overlapping intervals",
    "use_pca": true
}
```

#### **Combined Feature Experiments**
```json
{
    "name": "Combined_Experiment",
    "type": "combined",
    "dinov2_csv": "path/to/dinov2_features.csv",
    "second_csv": "path/to/second_features.csv",
    "second_feature_type": "Emotions",  // or "Expression_Pose"
    "temporal_strategy": "Average per interval",
    "use_pca": true
}
```

## ⚙️ Configuration Options

### **Feature Types**
- **"DINOv2"**: Full frame DINOv2 features (768 dimensions)
- **"Face DINOv2"**: Face-only DINOv2 features (768 dimensions)
- **"Upper Body DINOv2"**: Upper body DINOv2 features (768 dimensions)
- **"Emotions"**: FER emotion probabilities (6 dimensions)
- **"Expression_Pose"**: Facial landmarks and pose features (9+ dimensions)

### **Temporal Strategies**
- **"Baseline 10 seconds"**: Single frame at 10th second
- **"Average per interval"**: Average features across interval
- **"Triple intervals"**: Three samples per interval (early, middle, late)
- **"Overlapping intervals"**: 5-second overlapping windows
- **"Single interval"**: Custom interval selection

### **PCA Options**
- **True**: Apply PCA dimensionality reduction (10 components for DINOv2)
- **False**: Use original feature dimensions

## 🤖 Machine Learning Models

The pipeline automatically trains and evaluates multiple models:

### **Model Types**
1. **Logistic Regression** - Linear classification with regularization
2. **SVM (Support Vector Machine)** - Kernel-based classification
3. **Random Forest** - Ensemble tree-based classification
4. **XGBoost** - Optimized gradient boosting implementation
5. **XGBoost (Regularized)** - XGBoost with stronger regularization


## 📊 Data Processing

### **Preprocessing Steps**
1. **Data Loading**: Load feature CSV files
2. **Feature Selection**: Identify and extract feature columns
3. **Data Cleaning**: Handle missing values, infinite values, outliers
4. **Baseline Removal**: Exclude calm period data (seconds 10, 20, 30, 460, 470, 480)
5. **Feature Scaling**: Standardize features using StandardScaler
6. **Class Balancing**: Apply SMOTE for imbalanced stress/no-stress classes

### **Participant-Based Splitting**
- **Stratified splitting** by condition and dominant stress
- **No data leakage** between train and test sets
- **Consistent evaluation** across all experiments

## 📈 Evaluation Metrics

### **Performance Metrics**
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Focal Loss**: Advanced loss function for imbalanced classes

### **Output Files**
- **Confusion matrices**: Visual classification results
- **Classification reports**: Detailed per-class metrics
- **Results CSV**: Comprehensive experiment results
- **Model performance**: Best model identification

## 📁 Output Structure

After training, results are organized as:

```
Results/
├── DINOv2_Results/                    # DINOv2 feature results
│   ├── Experiment_Name/
│   │   ├── confusion_matrix_*.png     # Confusion matrices
│   │   ├── experiment_results.csv     # Detailed results
│   │   └── classification_reports/    # Per-model reports
├── Combined_DINOv2_Emotions_Results/  # Combined feature results
├── Combined_DINOv2_Expression_Pose_Results/
└── All_Classification_Reports/         # Consolidated reports
```

## 🔍 Troubleshooting

### **Common Issues**

1. **File not found errors**: Ensure preprocessing and feature extraction completed
2. **Memory errors**: Reduce batch size or use smaller feature sets
3. **CUDA errors**: Check GPU availability and PyTorch installation
4. **Import errors**: Install required dependencies

### **Debug Mode**

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Performance Optimization**

- **GPU acceleration**: Use CUDA for faster training
- **Batch processing**: Process experiments in parallel if memory allows
- **Feature selection**: Use PCA for high-dimensional features
- **Memory management**: Clear GPU cache between large experiments

## 📋 Requirements

### **Python Dependencies**
```bash
pip install scikit-learn
pip install xgboost
pip install pandas numpy
pip install matplotlib seaborn
pip install imbalanced-learn
pip install torch torchvision
pip install opencv-python
```

### **System Requirements**
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Storage**: SSD for faster I/O operations
- **Python**: 3.8+ required

## 📚 References

- **Scikit-learn**: [Machine Learning in Python](https://scikit-learn.org/)
- **XGBoost**: [Gradient Boosting Library](https://xgboost.readthedocs.io/)
- **SMOTE**: [Synthetic Minority Over-sampling](https://imbalanced-learn.org/)
- **PyTorch**: [Deep Learning Framework](https://pytorch.org/)

## 🔗 Integration

### **Input Dependencies**
- **Preprocessing pipeline**: Must complete before training
- **Feature extraction**: All feature types must be extracted
- **Temporal features**: Temporal aggregation must be completed

### **Output Usage**
- **Model selection**: Use best performing models for deployment
- **Feature analysis**: Analyze which features contribute most to performance
- **Temporal analysis**: Understand stress patterns over time
- **Cross-validation**: Validate results across different data splits

