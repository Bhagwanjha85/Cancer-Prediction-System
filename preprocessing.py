import cv2
import numpy as np
import torch
from typing import Tuple
import config

class ImagePreprocessor:
    """
    Handles image preprocessing including resizing, normalization, noise removal,
    and conversions to PyTorch format for inference.
    """
    
    @staticmethod
    def remove_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        Removes noise from the image using OpenCV techniques.
        """
        if method == "gaussian":
            return cv2.GaussianBlur(image, (3, 3), 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "median":
            return cv2.medianBlur(image, 3)
        return image

    @staticmethod
    def adjust_contrast_brightness(image: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        """
        Adjusts brightness (beta) and contrast (alpha).
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @classmethod
    def preprocess_for_inference(cls, image_path_or_bytes: str or bytes, remove_noise: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Loads, resizes, normalizes, and applies noise removal on an image for PyTorch inference.
        Returns:
            processed_tensor: Batch-ready PyTorch FloatTensor (1, 3, 224, 224) standardized.
            original_resized: Original image resized for visualization (224, 224, 3) in BGR.
        """
        if isinstance(image_path_or_bytes, bytes):
            nparr = np.frombuffer(image_path_or_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_path_or_bytes)
            
        if img is None:
            raise ValueError("Could not decode image.")
            
        original_resized = cv2.resize(img, config.IMAGE_SIZE)
        
        if remove_noise:
            # Apply noise removal on the resized image to avoid high computational cost on large images
            original_resized = cls.remove_noise(original_resized, method="bilateral")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Standardize for pre-trained models
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_standardized = (img_normalized - mean) / std
        
        # Transpose HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        img_chw = np.transpose(img_standardized, (2, 0, 1))
        
        # Add batch dimension and convert to PyTorch Tensor
        processed_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        
        return processed_tensor, original_resized
