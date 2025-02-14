"""Configuration settings for the Face Recognition system."""

# Data Collection Settings
DATASET_PATH = 'dataset'
NUM_SAMPLES = 70
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Model Settings
MODEL_PATH = 'models/trained_model.h5'
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Face Detection Settings
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

# Training Settings
RANDOM_STATE = 0

# Recognition Settings
CONFIDENCE_THRESHOLD = 0.6

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'