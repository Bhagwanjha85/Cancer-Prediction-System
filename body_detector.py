import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from PIL import Image
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BodyDetector")

class BodyDetector:
    """
    Checks if an uploaded image is a human body part/medical image in PyTorch,
    and rejects objects, animals, vehicles, buildings, and selfies.
    """
    
    def __init__(self):
        logger.info("Initializing PyTorch MobileNetV2 for human body validation...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained MobileNetV2 with default ImageNet weights
        self.weights = models.MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()
        
        # Categories list
        self.categories = self.weights.meta["categories"]
        
        # Setup pre-processing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def is_human_body(self, image_np: np.ndarray) -> bool:
        """
        Quick check if the image has skin/mucosal colors and is not a strictly non-human object.
        """
        try:
            # Convert NumPy BGR to HSV
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            total_pixels = image_np.shape[0] * image_np.shape[1]
            
            # Skin and mucosal tones (H in [0, 30] or [160, 180], S >= 15, V >= 30)
            skin_mask = ((h <= 30) | (h >= 160)) & (s >= 15) & (v >= 30)
            skin_pct = np.sum(skin_mask) / total_pixels * 100
            
            logger.info(f"Human body validation skin tone coverage: {skin_pct:.2f}%")
            if skin_pct < 10.0:
                logger.warning("Rejected: low skin tone coverage.")
                return False
                
            # Class-based check for objects
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            top3_prob, top3_catid = torch.topk(probabilities, 3)
            decoded = [(self.categories[idx.item()].lower(), prob.item()) for idx, prob in zip(top3_catid, top3_prob)]
            
            strict_non_human = {
                "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "deer", 
                "rabbit", "mouse", "chicken", "pig", "monkey", "gorilla", "panda", "fox", "wolf", "camel", "kangaroo", 
                "koala", "penguin", "shark", "whale", "dolphin", "turtle", "crocodile", "dinosaur", "squirrel", 
                "car", "truck", "airplane", "boat", "bicycle", "motorcycle", "train", "bus", "cab", "wagon", "ship",
                "laptop", "keyboard", "computer", "phone", "television", "microwave", "toaster", "refrigerator", 
                "vacuum", "printer", "modem", "screen", "monitor",
                "building", "house", "mountain", "valley", "cliff", "alp", "seashore", "bridge", "monument", "castle", 
                "temple", "church", "barn", "greenhouse", "volcano",
                "desk", "bed", "sofa", "table", "chair",
                "guitar", "piano", "drum", "violin", "trumpet", "flute", "ball", "racket", "bat", "glove", 
                "hammer", "screwdriver", "wrench", "pliers", "saw", "axe", "shovel", "rake", "scissors"
            }
            
            for class_name, prob in decoded:
                if prob > 0.35:
                    class_words = class_name.replace(",", "").replace("-", " ").split(" ")
                    for word in class_words:
                        if word in strict_non_human:
                            logger.warning(f"Strict object rejection: '{class_name}' with confidence {prob:.4f}")
                            return False
                            
            return True
            
        except Exception as e:
            logger.error(f"Error in is_human_body check: {str(e)}")
            return True
            
    def validate_tissue_type(self, image_np: np.ndarray, expected_type: str) -> Tuple[bool, str]:
        """
        Validates if the image matches close-up skin tissue or oral cavity. Rejects selfies/portraits.
        """
        try:
            expected_type = expected_type.lower()
            if "oral" in expected_type:
                expected_key = "oral"
            elif "skin" in expected_type:
                expected_key = "skin"
            else:
                return True, ""
                
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            top5_names = [self.categories[idx.item()].lower() for idx in top5_catid]
            top5_probs = [prob.item() for prob in top5_prob]
            
            logger.info(f"Tissue verification Top 5: {list(zip(top5_names, top5_probs))}")
            
            # Selfie indicators (clothing, eyewear, background setups)
            selfie_indicators = {
                "t-shirt", "jersey", "suit", "coat", "necktie", "cardigan", "sweatshirt", "cloak", 
                "trench coat", "gown", "academic gown", "kimono", "pajamas", "pajama", "swimming trunks", 
                "sunglasses", "sunglass", "glasses", "spectacles", "eyeglasses", "groom", "bridegroom",
                "wig", "desk", "monitor", "screen", "keyboard", "office", "room", "sofa", "studio"
            }
            
            for idx in range(3):
                name = top5_names[idx]
                prob = top5_probs[idx]
                if prob > 0.15 and any(ind in name for ind in selfie_indicators):
                    logger.warning(f"Selfie indicator matched: '{name}' ({prob:.4f})")
                    return False, "Please upload a relevant image. Selfies and full portraits are not allowed."
            
            # Oral check
            if expected_key == "oral":
                oral_keywords = {"lipstick", "mouth", "tongue", "tooth", "teeth", "lip", "lips"}
                has_mouth = any(word in "".join(top5_names) for word in oral_keywords)
                
                hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                total_pixels = image_np.shape[0] * image_np.shape[1]
                red_mask = ((h <= 8) | (h >= 155)) & (s >= 40) & (v >= 30)
                red_pct = np.sum(red_mask) / total_pixels * 100
                
                if not has_mouth and red_pct < 4.0:
                    logger.warning(f"Oral check failed: no mouth keywords and red mucosal pct is {red_pct:.2f}%")
                    return False, "The uploaded image does not appear to be of the oral cavity (mouth). Please upload a valid oral image for this test."
            
            # Skin check
            if expected_key == "skin":
                # Check for skin tones percentage
                hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                total_pixels = image_np.shape[0] * image_np.shape[1]
                skin_mask = (h >= 4) & (h <= 26) & (s >= 15) & (s <= 160) & (v >= 35)
                skin_pct = np.sum(skin_mask) / total_pixels * 100
                
                has_mouth_close_up = any(word in top5_names[0] for word in ["mouth", "tongue", "tooth", "teeth"])
                if has_mouth_close_up:
                    return False, "The uploaded image appears to be an oral cavity image. Please select the Oral Cavity test or upload a skin image."
                    
                if skin_pct < 8.0:
                    logger.warning(f"Skin check failed: skin pct is {skin_pct:.2f}%")
                    return False, "The uploaded image does not appear to be skin tissue. Please upload a valid skin image."
                    
            return True, ""
            
        except Exception as e:
            logger.error(f"Error in tissue validation: {str(e)}")
            return True, ""

if __name__ == "__main__":
    detector = BodyDetector()
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    print("Dummy black image validation:", detector.is_human_body(dummy_img))
