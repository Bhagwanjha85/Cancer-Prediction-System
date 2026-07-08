import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Subdirectories for datasets
ORAL_DIR = os.path.join(DATASET_DIR, "oral")
SKIN_DIR = os.path.join(DATASET_DIR, "skin")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Set to a reasonable number for training
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# Allowed image extensions for security
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_CONTENT_LENGTH_MB = 10.0  

# List of architectures to train and compare
MODEL_ARCHITECTURES = [
    "ConvNeXt"
]

# Classification configuration
# Class mappings: 0: Oral_Cancer, 1: Oral_Normal, 2: Oral_Ulcer, 3: Skin_Cancer, 4: Skin_Normal
CLASS_NAMES = ["Oral_Cancer", "Oral_Normal", "Oral_Ulcer", "Skin_Cancer", "Skin_Normal"]

# Separate model classification configurations
CLASS_NAMES_ORAL = ["Oral_Cancer", "Oral_Normal", "Oral_Ulcer"]
CLASS_NAMES_SKIN = ["Skin_Cancer", "Skin_Normal"]

# Separate model files
MODEL_ORAL_PATH = os.path.join(MODELS_DIR, "best_model_oral.pth")
MODEL_SKIN_PATH = os.path.join(MODELS_DIR, "best_model_skin.pth")

WEIGHTS_ORAL_PATH = os.path.join(MODELS_DIR, "weights_oral.pth")
WEIGHTS_SKIN_PATH = os.path.join(MODELS_DIR, "weights_skin.pth")

LABEL_ENCODER_ORAL_PATH = os.path.join(MODELS_DIR, "label_encoder_oral.pkl")
LABEL_ENCODER_SKIN_PATH = os.path.join(MODELS_DIR, "label_encoder_skin.pkl")

HISTORY_ORAL_PATH = os.path.join(MODELS_DIR, "history_oral.pkl")
HISTORY_SKIN_PATH = os.path.join(MODELS_DIR, "history_skin.pkl")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "icons"), exist_ok=True)
