# Melanoma Classification and CBIR System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)](https://www.kaggle.com/)

A comprehensive deep learning system for melanoma classification and content-based image retrieval (CBIR), implementing an ensemble of CNNs and Vision Transformers with state-of-the-art optimizations.

---

## 📊 Results

### Classification Performance
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| DenseNet121 | 94.10% | 93.45% | 94.80% | 94.12% | 0.9885 |
| InceptionV3 | 92.25% | 93.60% | 90.70% | 92.12% | 0.9752 |
| Xception | 93.65% | 93.10% | 94.25% | 93.67% | 0.9846 |
| ViT | 87.20% | 88.50% | 85.80% | 87.13% | 0.9421 |
| **Ensemble** | **94.80%** | **94.15%** | **95.50%** | **94.82%** | **0.9901** |

### CBIR Performance (mAP@5)
| Model | mAP | Improvement |
|-------|-----|-------------|
| DenseNet121 | **1.0000** | +5.04% vs baseline |
| InceptionV3 | **0.9975** | +20.53% vs baseline |
| Xception | 0.9780 | +6.09% vs baseline |
| ViT | 0.8819 | +12.80% vs baseline |
| **Multi-model Fusion** | **0.9840** | **+3.02% vs baseline** ✅ |

**Baseline Comparison:**
- Classification: 94.80% (target: 95.25%, gap: -0.45%)
- CBIR: **0.9840** (target: 0.9538, **BEAT by +3.17%**) ✅

---

## 🚀 Features

### Core Functionality
- ✅ **Multi-Model Ensemble**: DenseNet121, InceptionV3, Xception, Vision Transformer
- ✅ **Classification**: Binary melanoma detection with ensemble learning
- ✅ **CBIR**: Content-based image retrieval with multi-model fusion
- ✅ **Statistical Validation**: Bootstrap confidence intervals (n=100)
- ✅ **Production Ready**: Complete pipeline from training to deployment

### Key Improvements
- 🔧 **InceptionV3 BatchNorm Fix**: Corrected missing BatchNormalization layer (+1.05% accuracy)
- 🎲 **Mixup Augmentation**: Data augmentation with α=0.2 for better generalization
- 🏷️ **Label Smoothing**: Regularization with ε=0.1 to prevent overfitting
- 🔍 **Random Search Ensemble**: Optimized weights using 2000 trials with early stopping
- ⚡ **Mixed Precision Training**: FP16 for 2-3× speedup

### Outputs Generated
- 📊 **6 Tables**: Transfer learning performance, ensemble results, CBIR metrics, bootstrap CI
- 📈 **8 Figures**: Dataset distribution, confusion matrices, ROC curves, PR curves, top-5 retrievals
- 💾 **4 Models**: Trained .h5 files for all architectures
- 📝 **Training Logs**: CSV files with epoch-by-epoch metrics

---

## 📁 Project Structure

```
melanoma-Detection/
├── Melanoma_COMPLETE_FINAL.ipynb    # Main notebook (82 cells)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── LICENSE                          # MIT License
│
├── outputs/                         # Generated outputs (after running)
│   ├── models/                      # Trained model files
│   │   ├── DenseNet121_best.h5
│   │   ├── InceptionV3_best.h5
│   │   ├── Xception_best.h5
│   │   └── ViT_best.h5
│   │
│   ├── results/                     # Tables and metrics
│   │   ├── table_2_transfer_learning.csv
│   │   ├── table_3_ensemble.csv
│   │   ├── table_4_cbir_individual.csv
│   │   ├── table_5_cbir_fusion.csv
│   │   ├── table_6_accuracy_ci.csv
│   │   ├── table_7_cbir_map_ci.csv
│   │   └── *_training.csv (4 files)
│   │
│   ├── figures/                     # Visualizations (300 DPI)
│   │   ├── figure_2_dataset_distribution.png
│   │   ├── figure_9_confusion_matrices.png
│   │   ├── figure_10_roc_curves.png
│   │   ├── figure_11_ensemble_results.png
│   │   ├── figure_12_cbir_pr_curves.png
│   │   ├── figure_13_cbir_top5_individual.png
│   │   ├── figure_14_cbir_pr_fusion.png
│   │   └── figure_15_cbir_top5_fusion.png
│   │
│   └── features/                    # Extracted features for CBIR
│       └── *_features.npz (4 files)
│
└── data/                            # Dataset (not included)
    ├── train/
    │   ├── Benign/
    │   └── Malignant/
    └── test/
        ├── Benign/
        └── Malignant/
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- GPU with CUDA support (recommended: P100 or T4)
- 16GB+ RAM
- 10GB+ disk space

### Option 1: Kaggle (Recommended)
1. Upload `Melanoma_COMPLETE_FINAL.ipynb` to Kaggle
2. Add dataset: [melanoma-cancer-dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
3. Set accelerator to **GPU** (P100 or T4)
4. Click **Run All**

### Option 2: Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/melanoma-classification-cbir.git
cd melanoma-classification-cbir

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place dataset in data/ directory following the structure above
```

### Requirements
```txt
tensorflow==2.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
tqdm>=4.66.0
```

---

## 🎯 Usage

### Quick Start
```python
# 1. Load the notebook
jupyter notebook Melanoma_COMPLETE_FINAL.ipynb

# 2. Run all cells (Ctrl+A, Shift+Enter)
# The notebook will automatically:
#   - Train 4 models
#   - Evaluate classification performance
#   - Compute ensemble predictions
#   - Extract features for CBIR
#   - Generate all tables and figures
#   - Calculate bootstrap confidence intervals
```

### Training Time
| Component | Time (P100 GPU) | Time (T4 GPU) |
|-----------|-----------------|---------------|
| DenseNet121 | ~1h 07min | ~1h 30min |
| InceptionV3 | ~56min | ~1h 15min |
| Xception | ~1h 47min | ~2h 15min |
| ViT | ~59min | ~1h 20min |
| Feature Extraction | ~20min | ~30min |
| CBIR Evaluation | ~10min | ~15min |
| **Total** | **~4h 48min** | **~6h 30min** |

### Memory Requirements
- **Training**: ~10-12 GB GPU memory (with mixed precision)
- **Inference**: ~4-6 GB GPU memory
- **CPU RAM**: ~16 GB recommended

---

## 🧠 Model Architectures

### 1. DenseNet121
```python
Base: DenseNet121 (ImageNet pretrained)
Freeze: First 121 layers
Custom Head:
  - Conv2D(256) → MaxPool → Conv2D(256) → MaxPool → Conv2D(128) → MaxPool
  - GlobalAveragePooling2D
  - Dense(512, ReLU) → Dropout(0.5)
  - Dense(256, ReLU) → Dropout(0.3)  # Feature layer for CBIR
  - Dense(1, Sigmoid)

Optimizer: Adamax (LR=0.001)
Loss: Binary Crossentropy with Label Smoothing (0.1)
```

### 2. InceptionV3
```python
Base: InceptionV3 (ImageNet pretrained)
Freeze: First 150 layers
Custom Head:
  - Conv2D(512) → MaxPool → Conv2D(512) → MaxPool
  - Conv2D(256) → MaxPool → Conv2D(128) → MaxPool
  - BatchNormalization  # CRITICAL FIX
  - GlobalAveragePooling2D
  - Dense(512, ReLU) → Dropout(0.5)
  - Dense(256, ReLU) → Dropout(0.3)  # Feature layer for CBIR
  - Dense(1, Sigmoid)

Optimizer: Adam (LR=0.001)
Loss: Binary Crossentropy with Label Smoothing (0.1)
```

### 3. Xception
```python
Base: Xception (ImageNet pretrained)
Freeze: First 80 layers
Custom Head:
  - Conv2D(256) → MaxPool → Conv2D(256) → MaxPool → Conv2D(128) → MaxPool
  - GlobalAveragePooling2D
  - Dense(512, ReLU) → Dropout(0.5)
  - Dense(256, ReLU) → Dropout(0.3)  # Feature layer for CBIR
  - Dense(1, Sigmoid)

Optimizer: Adam (LR=0.001)
Loss: Binary Crossentropy with Label Smoothing (0.1)
```

### 4. Vision Transformer (ViT)
```python
Input: 224×224×3
Patch Size: 16×16
Projection Dim: 64
Architecture:
  - Patches → PatchEncoder
  - 8× TransformerBlock:
      * MultiHeadAttention (4 heads, key_dim=64)
      * MLP (projection_dim × 2, GELU)
  - LayerNormalization → Flatten
  - Dense(2048, GELU) → Dropout(0.5)
  - Dense(1024, GELU) → Dropout(0.5)
  - Dense(256, ReLU)  # Feature layer for CBIR
  - Dense(1, Sigmoid)

Optimizer: Adam (LR=0.001)
Loss: Binary Crossentropy with Label Smoothing (0.1)
```

---

## 🎨 Data Augmentation

### Training Augmentation
```python
- Rotation: ±15°
- Width/Height Shift: 10%
- Shear: 0.2
- Zoom: 0.1
- Horizontal Flip: True
- Vertical Flip: True
- Brightness: [0.8, 1.2]
- Fill Mode: Reflect
```

### Mixup Augmentation
```python
Alpha: 0.2 (Beta distribution)
Formula:
  mixed_image = λ × image_i + (1-λ) × image_j
  mixed_label = λ × label_i + (1-λ) × label_j
```

---

## 🔍 CBIR System

### Feature Extraction
- **Layer**: Dense(256) before final output
- **Normalization**: L2 normalization
- **Dimension**: 256-D feature vectors

### Similarity Metric
```python
Cosine Similarity: cos(θ) = (A · B) / (||A|| × ||B||)
```

### Multi-Model Fusion
```python
Fusion Strategy: Weighted feature concatenation
Weights: Based on individual CBIR mAP performance
  - DenseNet121: 0.2738 (mAP=1.0000)
  - InceptionV3: 0.2731 (mAP=0.9975)
  - Xception: 0.2678 (mAP=0.9780)
  - ViT: 0.2414 (mAP=0.8819)

Fused Feature: Σ(feature_i × weight_i) → L2 normalize
```

### Retrieval Process
1. Extract query image features
2. Calculate cosine similarity with all database images
3. Sort by similarity (descending)
4. Return top-K most similar images

---

## 📊 Evaluation Metrics

### Classification
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Area under ROC curve

### CBIR
- **Average Precision (AP)**: Mean precision across relevant retrievals
- **Mean Average Precision (mAP)**: Mean of AP across all queries
- **Top-K Accuracy**: Percentage of relevant images in top-K results

### Statistical Validation
```python
Bootstrap Confidence Interval (95%):
  CI = mean ± 1.96 × (std / √n)
  n = 100 iterations
```

---

## 🏆 Key Achievements

### Novel Contributions
1. **Architecture Fix**: Identified and corrected missing BatchNormalization in InceptionV3 (+1.05% accuracy)
2. **Efficient Ensemble**: Random Search optimization with early stopping (60-80% fewer iterations)
3. **Enhanced Augmentation**: Mixup + Label Smoothing for improved generalization
4. **Superior CBIR**: Multi-model fusion achieving +3.17% over baseline

### Performance Highlights
- ✅ **CBIR Excellence**: Beat baseline by 3.17% (0.9840 vs 0.9538)
- ✅ **Perfect Retrieval**: DenseNet121 achieved 1.0000 mAP@5
- ✅ **Efficient Training**: 60% faster than expected (4.8h vs 10-12h)
- ✅ **Production Ready**: Complete pipeline with all outputs

---

## 📈 Reproducibility

### Fixed Random Seeds
```python
SEED = 42
numpy.random.seed(SEED)
tensorflow.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
```

### Deterministic Operations
- ✅ Fixed train/test split
- ✅ Deterministic data augmentation (seeded)
- ✅ Reproducible random search (seeded)
- ✅ Consistent bootstrap sampling (seeded)

### Environment
- **TensorFlow**: 2.15.0
- **CUDA**: 11.8+ (for GPU)
- **cuDNN**: 8.6+ (for GPU)
- **Python**: 3.10.12

---

## 🐛 Troubleshooting

### Common Issues

#### 1. AttributeError: 'generator' object has no attribute 'samples'
**Solution**: Use the latest version (`Melanoma_COMPLETE_FINAL.ipynb`) which implements `MixupGenerator` class.

#### 2. Out of Memory (OOM)
**Solutions**:
- Reduce `BATCH_SIZE` to 16 or 8
- Disable mixed precision: Remove `keras.mixed_precision.set_global_policy('mixed_float16')`
- Train models sequentially instead of all at once

#### 3. Slow Training
**Solutions**:
- Enable mixed precision (should be on by default)
- Use GPU with higher compute capability
- Reduce `EPOCHS` or enable early stopping

#### 4. CUDA/GPU Not Detected
**Solutions**:
```python
# Check GPU availability
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Enable memory growth
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 🔮 Future Work

### Planned Improvements
- [ ] **Test-Time Augmentation**: Ensemble predictions across augmented versions
- [ ] **Advanced Architectures**: EfficientNetV2, ConvNeXt, Swin Transformer
- [ ] **Attention Mechanisms**: Integrate attention maps for interpretability
- [ ] **Cross-Validation**: K-fold validation for robust performance estimation
- [ ] **Deployment**: Flask/FastAPI API for real-time inference
- [ ] **Mobile Optimization**: TensorFlow Lite conversion for edge devices

### Research Directions
- **Self-Supervised Learning**: Leverage unlabeled skin lesion images
- **Multi-Task Learning**: Joint training for classification + segmentation
- **Federated Learning**: Privacy-preserving distributed training
- **Explainable AI**: Grad-CAM, SHAP for clinical interpretability

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{melanoma_classification_cbir_2025,
  title = {Melanoma Classification and CBIR System: An Ensemble Approach},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/melanoma-classification-cbir},
  note = {Ensemble of CNNs and Vision Transformers for melanoma detection and content-based image retrieval}
}
```

### References

**Baseline Paper**:
```
Transfer Learning-Based Ensemble of CNNs and Vision Transformers 
for Accurate Melanoma Diagnosis and Image Retrieval
Diagnostics 2025, 15, 1928
https://doi.org/10.3390/diagnostics15151928
```

**Key Methods**:
- **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
- **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 👥 Authors

- **Rofiq Samanhudi** - *Initial work* - [YourGitHub](https://github.com/yourusername)

See also the list of [contributors](https://github.com/yourusername/melanoma-classification-cbir/contributors) who participated in this project.

---

## 🙏 Acknowledgments

- Dataset providers for the melanoma cancer dataset
- TensorFlow and Keras teams for excellent deep learning frameworks
- Kaggle for providing free GPU resources
- Research community for baseline papers and methodologies
- Open-source contributors for inspiration and tools

---

## 📞 Contact

**Your Name**
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

**Project Link**: [https://github.com/yourusername/melanoma-classification-cbir](https://github.com/yourusername/melanoma-classification-cbir)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/melanoma-classification-cbir&type=Date)](https://star-history.com/#yourusername/melanoma-classification-cbir&Date)

---

<div align="center">

**Made with ❤️ for advancing melanoma detection**

[⬆ Back to Top](#melanoma-classification-and-cbir-system)

</div>
