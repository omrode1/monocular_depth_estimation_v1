# Monocular Depth Estimation v1

A comprehensive implementation of monocular depth estimation using PyTorch, based on the Monodepth2 architecture. This project was developed as a solution to **Problem 2: Monocular Depth Estimation Algorithm Implementation** for a company assignment.

## Problem Statement

**Objective**: Estimate depth data using only RGB images without using depth data from sensors.

**Technical Specifications**:
- Implemented in Python
- No depth data from sensors used (self-supervised learning)
- Open-source libraries/tools used (PyTorch, OpenCV, etc.)
- Real-time depth estimation capability
- Comprehensive evaluation metrics

**Optional Features Implemented**:
- Real-time depth estimation (demonstrated in demos)
- Advanced loss functions and optimization
- Multi-scale depth prediction
- Comprehensive visualization tools

## Dataset Used

### Make3D Dataset (Ideal Lightweight Dataset)
- **Size**: 534 images (outdoor scenes)
- **Resolution**: 2272 × 1704 (downsampled for depth to ~55 × 305)
- **Type**: Outdoor scenes with corresponding depth maps

**Why This Dataset Fits**:
- Extremely lightweight and suitable for demonstration
- Can be trained quickly on a laptop or entry-level GPU
- Enough data for demonstrating monocular depth estimation pipelines
- Perfect for assignment requirements

**Download Link**: [Make3D Dataset](http://make3d.cs.cornell.edu/data.html)

## Features

- **ResNet-based Encoder**: Supports ResNet-18, ResNet-34, and ResNet-50 backbones
- **Multi-scale Decoder**: Produces depth predictions at multiple scales
- **Skip Connections**: Optional skip connections for better detail preservation
- **Comprehensive Loss Functions**: Photometric, smoothness, and edge-aware losses
- **Real-time Inference**: Optimized for real-time depth estimation
- **Evaluation Metrics**: RMSE, MAE, AbsRel, SqRel, and δ accuracy metrics
- **Visualization Tools**: Depth map visualization and comparison tools
- **Dataset Support**: Custom dataset loader with augmentation support

## Demo Videos

The project includes comprehensive demo videos showcasing the capabilities:

### Main Demo 
![Main Demo](demo/main_demo.gif)

**Features demonstrated:**
- Complete demonstration of the monocular depth estimation system
- Real-time processing capabilities
- Model architecture overview
- Training and inference pipeline

### Batch Processing Demo 
![Batch Processing Demo](demo/batch_processing_demo.gif)

**Features demonstrated:**
- Batch processing of multiple images
- Performance optimization techniques
- Scalability demonstration

### Comparison Demo 
![Comparison Demo](demo/comparison_demo.gif)

**Features demonstrated:**
- Side-by-side comparisons of different approaches
- Performance metrics visualization
- Quality assessment demonstrations

### Full-Quality Videos

For high-quality viewing, the complete demo videos are available on Google Drive:
**[Download Full Demo Videos](https://drive.google.com/drive/folders/1W0lzEol6eS1RpEAEUxSjPujGLNQ-HXun?usp=sharing)**

**Available videos:**
- **main_demo.mp4** (130.2 MB) - Complete system demonstration
- **batch_processing_demo.mp4** (28.2 MB) - Performance optimization demo  
- **comparison_demo.mp4** (49.6 MB) - Quality assessment comparisons

*Note: The GIFs above provide quick previews, while the full MP4 videos offer higher quality and better detail for detailed analysis.*

## Project Structure

```
monocular_depth_estimation_v1/
├── models/
│   ├── encoder.py          # ResNet encoder implementation
│   ├── decoder.py          # Multi-scale decoder
│   └── monodepth.py        # Complete model architecture
├── utils/
│   ├── dataset.py          # Dataset loading and augmentation
│   ├── losses.py           # Loss function implementations
│   └── metrics.py          # Evaluation metrics
├── demo/                   # Demo videos (GIF format)
│   ├── main_demo.gif       # Main demonstration video
│   ├── batch_processing_demo.gif  # Batch processing demo
│   └── comparison_demo.gif # Comparison demo
├── demo.py                 # Real-time demo script
├── train.py               # Training script
├── train_improved.py      # Improved training script
├── inference.py           # Inference script
├── test_trained_model.py  # Model testing script
├── prepare_dataset.py     # Dataset preparation script
├── split_dataset.py       # Dataset splitting utility
├── test_dataset.py        # Dataset testing script
├── test_implementation.py # Implementation testing
└── requirements.txt       # Dependencies
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/omrode1/monocular_depth_estimation_v1.git
cd monocular_depth_estimation_v1
```

2. **Create and activate a virtual environment**:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Using the Make3D Dataset

1. **Download the dataset** from [Make3D Dataset](http://make3d.cs.cornell.edu/data.html)

2. **Prepare the dataset**:
```bash
python prepare_dataset.py --data_path path/to/make3d/dataset
```

3. **Split the dataset**:
```bash
python split_dataset.py --data_path data/paired --train_ratio 0.8
```

4. **Test the dataset**:
```bash
python test_dataset.py --data_path data/paired
```

### Dataset Structure

After preparation, the dataset structure should be:
```
data/
├── paired/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── depth/           # Training depth maps
│   └── val/
│       ├── images/          # Validation images
│       └── depth/           # Validation depth maps
```

## Training

### Quick Start Training

```bash
python train.py \
    --data_path data/paired \
    --dataset_type general \
    --num_layers 18 \
    --epochs 50 \
    --batch_size 8 \
    --lr 0.0001 \
    --save_dir checkpoints \
    --log_dir logs
```

### Advanced Training (Improved Version)

```bash
python train_improved.py \
    --data_path data/paired \
    --num_layers 34 \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.0001 \
    --supervised \
    --save_dir checkpoints_improved
```

### Training Parameters

- `--data_path`: Path to the dataset directory
- `--dataset_type`: Dataset type ('general' or 'kitti')
- `--num_layers`: ResNet depth (18, 34, or 50)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--lr`: Learning rate
- `--supervised`: Use supervised training with ground truth depth
- `--save_dir`: Directory to save model checkpoints
- `--log_dir`: Directory for TensorBoard logs

### Monitor Training

```bash
tensorboard --logdir logs
```

## Inference

### Single Image Inference

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_path path/to/image.jpg \
    --output_path output_depth.png
```

### Real-time Demo

```bash
python demo.py \
    --model_path checkpoints/best_model.pth \
    --camera_id 0
```

### Test Trained Model

```bash
python test_trained_model.py \
    --model_path checkpoints/best_model.pth \
    --data_path data/paired \
    --split val
```

## Model Architecture

### Encoder
- ResNet backbone (18/34/50 layers)
- Pre-trained on ImageNet
- Extracts multi-scale features

### Decoder
- Multi-scale depth prediction
- Skip connections (optional)
- Upsampling with transposed convolutions

### Loss Functions
- **Photometric Loss**: RGB reconstruction error
- **Smoothness Loss**: Depth smoothness regularization
- **Edge-aware Loss**: Preserves depth discontinuities
- **Supervised Loss**: Direct depth supervision (optional)

## Evaluation

### Metrics Computed
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **AbsRel**: Absolute Relative Error
- **SqRel**: Square Relative Error
- **δ < 1.25**: Percentage of pixels with relative error < 1.25

### Evaluation Script

```bash
python test_trained_model.py \
    --model_path checkpoints/best_model.pth \
    --data_path data/paired \
    --split val \
    --save_results
```

## Demo Features

### Real-time Processing
- Webcam integration for live depth estimation
- Optimized inference pipeline
- Real-time visualization

### Batch Processing
- Efficient processing of multiple images
- Progress tracking and logging
- Quality assessment

### Visualization Tools
- Side-by-side comparisons
- Color-coded depth maps
- Performance metrics display

## Results and Performance

The model achieves competitive performance on the Make3D dataset:

### Hardware Specifications (Training & Testing Environment)
- **Laptop**: Windows 11 Home Single Language
- **Processor**: AMD Ryzen 5 4600H with Radeon Graphics (6 cores, 3.0 GHz)
- **Memory**: 7.5 GB RAM
- **GPU**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM)
- **Integrated Graphics**: AMD Radeon(TM) Graphics (512MB)

### Performance Metrics
- **Training Time**: less 40 mins hours on GTX 1650 Ti
- **Inference Speed**: ~30-50 FPS on GPU
- **Memory Usage**: ~2-4 GB GPU memory
- **Model Size**: ~50-100 MB depending on backbone
- **Batch Size**: 4-8 (optimized for 4GB VRAM)

## Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
Pillow>=8.3.0
tqdm>=4.62.0
tensorboard>=2.7.0
albumentations>=1.1.0
scipy>=1.7.0
```

## Key Achievements

1. **Self-supervised Learning**: Successfully implemented depth estimation without depth sensors
2. **Real-time Performance**: Achieved real-time inference capabilities
3. **Lightweight Solution**: Optimized for resource-constrained environments
4. **Comprehensive Evaluation**: Multiple metrics and visualization tools
5. **Production Ready**: Clean code structure with proper documentation


- **Make3D Dataset**: Cornell University for providing the dataset
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library



