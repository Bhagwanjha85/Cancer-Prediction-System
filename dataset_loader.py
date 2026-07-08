import os
import glob
import numpy as np
import cv2
import logging
from typing import Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_image(label_name: str) -> np.ndarray:
    """
    Generates a synthetic image simulating skin/oral conditions.
    """
    img = np.zeros((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.uint8)
    
    if "Oral" in label_name:
        img[:, :] = [100, 50, 200]  # BGR
        cv2.ellipse(img, (112, 112), (80, 50), 0, 0, 360, (120, 80, 220), -1)
    else:
        img[:, :] = [130, 180, 220]  # BGR
    
    if "Cancer" in label_name:
        center = (np.random.randint(80, 140), np.random.randint(80, 140))
        radius = np.random.randint(15, 35)
        pts = []
        for angle in range(0, 360, 45):
            r = radius + np.random.randint(-8, 8)
            x = int(center[0] + r * np.cos(np.radians(angle)))
            y = int(center[1] + r * np.sin(np.radians(angle)))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = (np.random.randint(10, 50), np.random.randint(10, 50), np.random.randint(20, 70))
        cv2.fillPoly(img, [pts], color)
        
        for _ in range(3):
            scen = (center[0] + np.random.randint(-20, 20), center[1] + np.random.randint(-20, 20))
            srad = np.random.randint(3, 8)
            cv2.circle(img, scen, srad, (color[0]-5, color[1]-5, color[2]-5), -1)
    else:
        if np.random.rand() > 0.5:
            cv2.circle(img, (112, 112), 8, (40, 60, 80), -1)
            
    return img

def ensure_dataset_exists():
    """
    Checks if the dataset exists. If not, generates mock datasets so the code can run.
    Also handles splitting and copying of uploaded ulcer images from dataset/ulcer into oral/Ulcer.
    """
    splits = ["train", "validation", "test"]
    categories = {"oral": ["Cancer", "Normal", "Ulcer"], "skin": ["Cancer", "Normal"]}
    
    # Check if there are uploaded ulcer images in dataset/ulcer/ and split them
    ulcer_src_dir = os.path.join(config.DATASET_DIR, "ulcer")
    if os.path.exists(ulcer_src_dir):
        uploaded_ulcers = [
            os.path.join(ulcer_src_dir, f) for f in os.listdir(ulcer_src_dir)
            if os.path.splitext(f.lower())[1] in config.ALLOWED_EXTENSIONS
        ]
        if len(uploaded_ulcers) > 0:
            logger.info(f"Found {len(uploaded_ulcers)} uploaded ulcer images. Splitting and organizing...")
            import random
            import shutil
            random.seed(42)
            random.shuffle(uploaded_ulcers)
            
            total = len(uploaded_ulcers)
            train_idx = int(total * 0.70)
            val_idx = int(total * 0.85)
            
            ulcer_splits = {
                "train": uploaded_ulcers[:train_idx],
                "validation": uploaded_ulcers[train_idx:val_idx],
                "test": uploaded_ulcers[val_idx:]
            }
            
            for split_name, split_files in ulcer_splits.items():
                dest_dir = os.path.join(config.DATASET_DIR, "oral", split_name, "Ulcer")
                os.makedirs(dest_dir, exist_ok=True)
                # Only copy if folder is empty or contains fewer files than expected
                if len(os.listdir(dest_dir)) == 0:
                    for f in split_files:
                        shutil.copy2(f, dest_dir)
                    logger.info(f"Copied {len(split_files)} ulcer images to oral/{split_name}/Ulcer")
    
    dataset_created = False
    for cat_name, subcats in categories.items():
        for split in splits:
            for subcat in subcats:
                folder_path = os.path.join(config.DATASET_DIR, cat_name, split, subcat)
                if os.path.exists(folder_path):
                    existing_files = os.listdir(folder_path)
                    valid_images = [f for f in existing_files if os.path.splitext(f.lower())[1] in config.ALLOWED_EXTENSIONS]
                else:
                    valid_images = []
                if not os.path.exists(folder_path) or len(valid_images) == 0:
                    os.makedirs(folder_path, exist_ok=True)
                    logger.info(f"Generating mock images for: {cat_name}/{split}/{subcat}")
                    num_images = 15 if split == "train" else 5
                    label_name = f"{cat_name.capitalize()}_{subcat}"
                    for i in range(num_images):
                        img = generate_mock_image(label_name)
                        img_name = f"mock_{i}.jpg"
                        cv2.imwrite(os.path.join(folder_path, img_name), img)
                    dataset_created = True
                    
    if dataset_created:
        logger.info("Mock dataset successfully created at: " + config.DATASET_DIR)

def load_filepaths_and_labels(split: str, cancer_type: str = "all") -> Tuple[List[str], List[int]]:
    """
    Loads all image file paths and their respective integer labels (binary/multi-class for specific types).
    """
    ensure_dataset_exists()
    
    filepaths = []
    labels = []
    
    cancer_type_lower = cancer_type.lower()
    if cancer_type_lower == "oral":
        mapping = {
            ("oral", "Cancer"): 0,
            ("oral", "Normal"): 1,
            ("oral", "Ulcer"): 2
        }
    elif cancer_type_lower == "skin":
        mapping = {
            ("skin", "Cancer"): 0,
            ("skin", "Normal"): 1
        }
    else:
        mapping = {
            ("oral", "Cancer"): 0,
            ("oral", "Normal"): 1,
            ("oral", "Ulcer"): 2,
            ("skin", "Cancer"): 3,
            ("skin", "Normal"): 4
        }
    
    for (cat, subcat), label_idx in mapping.items():
        folder_path = os.path.join(config.DATASET_DIR, cat, split, subcat)
        search_pattern = os.path.join(folder_path, "*")
        files = glob.glob(search_pattern)
        
        valid_files = [f for f in files if os.path.splitext(f.lower())[1] in config.ALLOWED_EXTENSIONS]
        
        filepaths.extend(valid_files)
        labels.extend([label_idx] * len(valid_files))
        
    return filepaths, labels

class PyTorchCancerDataset(Dataset):
    """
    Custom PyTorch Dataset for loading cancer screening images.
    """
    def __init__(self, filepaths: List[str], labels: List[int], transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.filepaths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to a plain black image of size IMAGE_SIZE if the file is corrupted
            image = Image.new("RGB", config.IMAGE_SIZE, (0, 0, 0))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloader(split: str, augment: bool = False, batch_size: int = None, cancer_type: str = "all") -> DataLoader:
    """
    Creates a PyTorch DataLoader for the given split and cancer type.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    paths, labels = load_filepaths_and_labels(split, cancer_type)
    
    if not paths:
        raise ValueError(f"No images found for split: {split} and type: {cancer_type}")
        
    # Define transforms
    if augment:
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            # Standard normalization for pre-trained ImageNet backbones
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    dataset = PyTorchCancerDataset(paths, labels, transform=transform)
    
    shuffle = (split == "train")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 0 is safest for Windows multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader

if __name__ == "__main__":
    ensure_dataset_exists()
    train_paths, train_labels = load_filepaths_and_labels("train")
    print(f"Loaded {len(train_paths)} training paths.")
    loader = get_dataloader("train", augment=True, batch_size=4)
    images, targets = next(iter(loader))
    print(f"Images batch shape: {images.shape}, Labels: {targets}")
