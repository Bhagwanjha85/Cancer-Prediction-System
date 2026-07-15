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
            # Resize large images to a maximum dimension of 600 for performance and consistency
            h, w = image_np.shape[:2]
            max_dim = 600
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)))
                
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
            
            top10_prob, top10_catid = torch.topk(probabilities, 10)
            decoded = [(self.categories[idx.item()].lower(), prob.item()) for idx, prob in zip(top10_catid, top10_prob)]
            
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
            
            non_human_score = 0.0
            for class_name, prob in decoded:
                class_words = class_name.replace(",", "").replace("-", " ").split(" ")
                is_non_human = any(word in strict_non_human for word in class_words)
                if is_non_human:
                    non_human_score += prob
                    if prob > 0.15:
                        logger.warning(f"Strict object rejection: '{class_name}' with confidence {prob:.4f}")
                        return False
            
            if non_human_score > 0.25:
                logger.warning(f"Cumulative non-human rejection: score {non_human_score:.4f} exceeds 0.25 threshold")
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
                
            # Resize large images to a maximum dimension of 600 for performance and consistency
            h, w = image_np.shape[:2]
            max_dim = 600
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)))
                
            # 1. Haar Cascade Face, Profile, & Eye Detection (to reject face selfies/portraits)
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            
            if len(faces) > 0 or len(profiles) > 0 or len(eyes) > 1:
                logger.warning(f"Selfie rejected by Haar Cascade: {len(faces)} frontal faces, {len(profiles)} profile faces, {len(eyes)} eyes detected.")
                return False, "Please upload a relevant image. Selfies and full portraits are not allowed."
                
            # 2. ImageNet Class-based validation
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            top10_prob, top10_catid = torch.topk(probabilities, 10)
            top10_names = [self.categories[idx.item()].lower() for idx in top10_catid]
            top10_probs = [prob.item() for prob in top10_prob]
            
            logger.info(f"Tissue verification Top 10: {list(zip(top10_names, top10_probs))}")
            
            # Selfie & Scene indicators (clothing, eyewear, background setups, baby/infant objects)
            selfie_indicators = {
                "t-shirt", "jersey", "suit", "coat", "necktie", "cardigan", "sweatshirt", "cloak", 
                "trench coat", "gown", "academic gown", "kimono", "pajamas", "pajama", "swimming trunks", 
                "sunglasses", "sunglass", "glasses", "spectacles", "eyeglasses", "groom", "bridegroom",
                "wig", "desk", "monitor", "screen", "keyboard", "office", "room", "sofa", "studio",
                "diaper", "cradle", "crib", "high chair", "baby buggy", "plaything", "toy", "mannequin", 
                "jeans", "jean", "sock", "shoe", "sandal", "boot", "backpack", "umbrella", "seat belt",
                "people", "person", "child", "boy", "girl", "baby", "infant", "toddler", "teenager", 
                "adult", "man", "woman"
            }
            
            for idx in range(3):
                name = top10_names[idx]
                prob = top10_probs[idx]
                if prob > 0.15 and any(ind in name for ind in selfie_indicators):
                    logger.warning(f"Selfie/Portrait indicator matched: '{name}' ({prob:.4f})")
                    return False, "Please upload a relevant image. Selfies and full portraits are not allowed."
            
            # Define keywords for oral vs skin tissue checks
            oral_keywords = {
                "lipstick", "mouth", "tongue", "tooth", "teeth", "lip", "lips", "face",
                "chimpanzee", "orangutan", "gorilla", "macaque", "monkey", "proboscis",
                "stinkhorn", "pomegranate", "strawberry", "band aid", "diaper", "nipple"
            }
            
            skin_keywords = {
                "bubble", "loupe", "jellyfish", "petri dish", "eggnog", "espresso",
                "ant", "tick", "slug", "snail", "spider", "nematode", "black widow",
                "lacewing", "wok", "frying pan", "mortar", "soup bowl", "flesh", "mole",
                "freckle", "macula", "acne", "scar", "dermatitis", "melanoma", "skin"
            }
            
            oral_score = 0.0
            skin_score = 0.0
            for name, prob in zip(top10_names, top10_probs):
                for word in oral_keywords:
                    if word in name:
                        oral_score += prob
                for word in skin_keywords:
                    if word in name:
                        skin_score += prob
            
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            total_pixels = image_np.shape[0] * image_np.shape[1]
            
            # Red mucosal check
            red_mask = ((h <= 8) | (h >= 155)) & (s >= 40) & (v >= 30)
            red_pct = np.sum(red_mask) / total_pixels * 100
            
            # Skin tone check
            skin_mask = (h >= 4) & (h <= 26) & (s >= 15) & (s <= 160) & (v >= 35)
            skin_pct = np.sum(skin_mask) / total_pixels * 100
            
            logger.info(f"Tissue check - Oral Score: {oral_score:.4f}, Skin Score: {skin_score:.4f}, Red Pct: {red_pct:.2f}%, Skin Pct: {skin_pct:.2f}%")
            
            # Oral check
            if expected_key == "oral":
                # Reject if skin score is higher and significant
                if skin_score > oral_score and skin_score > 0.15:
                    logger.warning(f"Oral check rejected: skin score {skin_score:.4f} is higher than oral score {oral_score:.4f}")
                    return False, "The uploaded image appears to be skin tissue. Please select the Skin Lesion test or upload an oral image."
                    
                has_mouth = (oral_score > 0.0) or any(word in "".join(top10_names[:3]) for word in ["lipstick", "mouth", "tongue", "tooth", "teeth", "lip", "lips"])
                if not has_mouth and red_pct < 4.0:
                    logger.warning(f"Oral check failed: no mouth keywords and red mucosal pct is {red_pct:.2f}%")
                    return False, "The uploaded image does not appear to be of the oral cavity (mouth). Please upload a valid oral image for this test."
            
            # Skin check
            if expected_key == "skin":
                # Reject if oral score is higher and significant
                if oral_score > skin_score and oral_score > 0.15:
                    logger.warning(f"Skin check rejected: oral score {oral_score:.4f} is higher than skin score {skin_score:.4f}")
                    return False, "The uploaded image appears to be an oral cavity image. Please select the Oral Cavity test or upload a skin image."
                
                # Check for mouth keywords in top predictions
                has_mouth_close_up = any(word in top10_names[0] for word in ["mouth", "tongue", "tooth", "teeth"])
                if has_mouth_close_up:
                    return False, "The uploaded image appears to be an oral cavity image. Please select the Oral Cavity test or upload a skin image."
                
                # Require either skin-related keywords or sufficient skin color coverage
                has_skin = (skin_score > 0.0)
                if not has_skin and skin_pct < 20.0:
                    logger.warning(f"Skin check failed: no skin keywords and skin pct is {skin_pct:.2f}%")
                    return False, "The uploaded image does not appear to be skin tissue. Please upload a valid skin image."
                    
            return True, ""
            
        except Exception as e:
            logger.error(f"Error in tissue validation: {str(e)}")
            return True, ""

if __name__ == "__main__":
    detector = BodyDetector()
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    print("Dummy black image validation:", detector.is_human_body(dummy_img))
