# Real Time Face Recognition Using CNN

A robust real-time face recognition system implemented using Convolutional Neural Networks (CNN). This project provides an end-to-end solution for face detection, data collection, model training, and real-time recognition using Python and deep learning techniques.

## Key Features
- Real-time face detection and recognition using webcam
- Custom CNN architecture for improved accuracy
- Easy-to-use data collection interface
- Automatic face detection and cropping
- Real-time FPS counter and performance metrics
- Support for multiple face recognition
- Batch training with validation split
- Model checkpointing for best weights

## System Requirements

### Hardware
- Webcam (built-in or external)
- Minimum 4GB RAM (8GB recommended)
- CPU with SSE2 instruction set support

### Software
- Python 3.7 or higher
- Operating System: Windows 10/11, macOS, or Linux

### Dependencies
- OpenCV (cv2)
- TensorFlow 2.x
- Keras
- NumPy
- PIL (Python Imaging Library)
- dlib
- imutils
- h5py
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Real-Time-Face-Recognition-Using-CNN.git
cd Real-Time-Face-Recognition-Using-CNN
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Data Collection
```bash
python src/data_collection/face_dataset.py
```
This script will:
- Activate your webcam
- Detect faces in real-time
- Capture 70 images of your face from different angles
- Save the processed images in the `dataset` directory
- Display real-time FPS and image capture count

**Tips for better data collection:**
- Ensure good lighting conditions
- Move your face slowly to capture different angles
- Maintain a distance of 0.5-1 meter from the camera
- Try different facial expressions

### 2. Model Training
```bash
python src/training/face_training.py
```
This will:
- Process the collected face images
- Create and train the CNN model
- Save the best model weights to `models/trained_model.h5`
- Display training progress and validation metrics

### 3. Face Recognition
```bash
python src/recognition/face_recognition.py
```
Features:
- Real-time face detection and recognition
- FPS counter for performance monitoring
- Confidence score display
- Multiple face detection support

## Configuration

Adjust parameters in `config.py` to customize the system:

### Data Collection
- `MIN_FACE_SIZE`: Minimum face size to detect
- `SCALE_FACTOR`: Scale factor for face detection
- `MIN_NEIGHBORS`: Minimum neighbors for face detection

### Training
- `IMAGE_SIZE`: Input image dimensions
- `BATCH_SIZE`: Training batch size
- `EPOCHS`: Number of training epochs
- `MODEL_PATH`: Path to save trained model

### Recognition
- `FACE_CASCADE_PATH`: Path to face cascade classifier
- `DATASET_PATH`: Path to face dataset

## Troubleshooting

1. **Low FPS during recognition:**
   - Reduce `MIN_FACE_SIZE` in config.py
   - Ensure no other CPU-intensive programs are running

2. **Poor recognition accuracy:**
   - Collect more training data with varied lighting/angles
   - Increase `EPOCHS` in config.py
   - Adjust `SCALE_FACTOR` and `MIN_NEIGHBORS`

3. **Face not detected:**
   - Check lighting conditions
   - Adjust `MIN_FACE_SIZE` and `SCALE_FACTOR`
   - Ensure face is clearly visible to camera

## Project Structure
```
Real-Time-Face-Recognition-Using-CNN/
├── src/                           # Source code directory
│   ├── data_collection/           # Data collection scripts
│   │   └── face_dataset.py        # Face image capture script
│   ├── training/                  # Model training scripts
│   │   ├── face_training.py       # Main training script
│   │   └── model.py              # CNN model architecture
│   └── recognition/               # Recognition scripts
│       └── face_recognition.py    # Real-time recognition
├── models/                        # Trained model storage
│   └── trained_model.h5           # Trained weights file
├── dataset/                       # Face image dataset
├── requirements.txt               # Project dependencies
├── config.py                      # Configuration parameters
└── README.md                      # Project documentation
```

## Contributing
Contributions are welcome! Please feel free to submit issues and enhancement requests.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenCV team for the face detection cascades
- TensorFlow and Keras teams for the deep learning framework
- The open-source community for various dependencies



