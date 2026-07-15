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

# Large comprehensive set of strictly forbidden non-medical/non-tissue classes
FORBIDDEN_CLASSES = {
    # Animals
    "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "deer", 
    "rabbit", "mouse", "chicken", "pig", "monkey", "gorilla", "panda", "fox", "wolf", "camel", "kangaroo", 
    "koala", "penguin", "shark", "whale", "dolphin", "turtle", "crocodile", "dinosaur", "squirrel", 
    "snake", "frog", "toad", "lizard", "insect", "butterfly", "bee", "spider", "ant", "tick", "crab", "lobster",
    # Vehicles & Machinery
    "car", "truck", "airplane", "boat", "bicycle", "motorcycle", "train", "bus", "cab", "wagon", "ship",
    "tractor", "helicopter", "rocket", "engine", "wheel",
    # Electronics & Office
    "laptop", "keyboard", "computer", "phone", "television", "microwave", "toaster", "refrigerator", 
    "vacuum", "printer", "modem", "screen", "monitor", "projector", "camera", "copier",
    # Buildings & Outdoors
    "building", "house", "mountain", "valley", "cliff", "alp", "seashore", "bridge", "monument", "castle", 
    "temple", "church", "barn", "greenhouse", "volcano", "forest", "tree", "grass", "leaf", "plant", "flower",
    "lake", "river", "ocean", "sea", "beach", "sand", "rock", "stone", "cloud", "sun", "moon", "star",
    # Furniture & Household
    "desk", "bed", "sofa", "table", "chair", "cushion", "pillow", "wardrobe", "bookcase", "cabinet", "chest", 
    "clock", "lamp", "candle", "mirror", "window", "door", "wall", "floor", "carpet", "rug", "curtain", 
    "bath", "bathtub", "toilet", "sink", "soap dispenser", "towel", "blanket",
    # Musical Instruments & Sports
    "guitar", "piano", "drum", "violin", "trumpet", "flute", "ball", "racket", "bat", "glove", "surfboard",
    "skateboard", "ski", "snowboard",
    # Tools & Hardware
    "hammer", "screwdriver", "wrench", "pliers", "saw", "axe", "shovel", "rake", "scissors",
    # Clothing, Apparel & Accessories (Selfie / Full Body)
    "t-shirt", "jersey", "suit", "coat", "necktie", "cardigan", "sweatshirt", "cloak", "gown", "academic gown", 
    "kimono", "pajamas", "pajama", "swimming trunks", "sunglasses", "sunglass", "glasses", "spectacles", 
    "eyeglasses", "groom", "bridegroom", "bride", "wig", "diaper", "mannequin", "jeans", "jean", "sock", 
    "shoe", "sandal", "boot", "backpack", "umbrella", "seat belt", "apron", "poncho", "miniskirt", "skirt", 
    "dress", "brassiere", "maillot", "bikini", "uniform", "velvet", "wool", "hat", "cap", "bonnet", "helmet",
    "purse", "wallet", "bag", "handbag", "bow tie", "bow-tie", "hair slide", "hair spray", "face powder",
    "eyeglass", "beanie", "sombrero", "turban", "feather boa", "stole", "stethoscope", "collar",
    # Baby / Child items (Cradles, High chairs, Toys)
    "cradle", "crib", "high chair", "baby buggy", "plaything", "toy", "teddy", "doll", "pacifier",
    # Food & Kitchen (Except allowed ones)
    "banana", "apple", "orange", "lemon", "pineapple", "strawberry", "pomegranate", "mushroom", "pizza", 
    "burger", "bread", "sandwich", "cookie", "cake", "chocolate", "meat", "salad", "wine", "beer", "juice", 
    "milk", "cup", "glass", "plate", "bowl", "fork", "spoon", "knife", "bottle", "can", "pot", "pan",
    # General Person Indicators
    "people", "person", "child", "boy", "girl", "baby", "infant", "toddler", "teenager", 
    "adult", "man", "woman", "doctor", "physician", "clinician", "dentist"
}

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
            
            # Reject if any of the top 3 predictions are forbidden non-medical classes
            for name, prob in decoded[:3]:
                if prob > 0.05:
                    class_words = name.replace(",", "").replace("-", " ").split(" ")
                    for word in class_words:
                        if word in FORBIDDEN_CLASSES:
                            logger.warning(f"Rejected: forbidden class detected in top predictions: '{name}' ({prob:.4f})")
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
            face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Use high sensitivity: scaleFactor=1.05, minNeighbors=2
            faces_def = face_cascade_default.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
            faces_alt = face_cascade_alt.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
            faces_alt2 = face_cascade_alt2.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
            profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
            
            if len(faces_def) > 0 or len(faces_alt) > 0 or len(faces_alt2) > 0 or len(profiles) > 0 or len(eyes) > 1:
                logger.warning(f"Selfie rejected by Haar Cascade: frontal(def={len(faces_def)}, alt={len(faces_alt)}, alt2={len(faces_alt2)}), profile={len(profiles)}, eyes={len(eyes)}")
                return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
                
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
            
            # Reject if any of the top 3 predictions are forbidden non-medical classes (with low confidence threshold)
            for name, prob in zip(top10_names[:3], top10_probs[:3]):
                if prob > 0.05:
                    class_words = name.replace(",", "").replace("-", " ").split(" ")
                    for word in class_words:
                        if word in FORBIDDEN_CLASSES:
                            logger.warning(f"Tissue check rejected forbidden class: '{name}' ({prob:.4f})")
                            return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
            
            # 3. Top-1 ImageNet Whitelist check to reject non-human/non-medical objects and animals
            top1_name = top10_names[0]
            top1_prob = top10_probs[0]
            
            if expected_key == "skin":
                skin_allowed_keywords = {
                    "mole", "tick", "nematode", "band aid", "bandage", "bubble", "petri dish", 
                    "loupe", "jellyfish", "eggnog", "espresso", "soup bowl", "mortar", "cappuccino",
                    "flesh", "skin", "freckle", "spot", "acne", "scar", "dermatitis", "melanoma",
                    "macula", "patch", "lesion", "rash", "dermis", "epidermis", "slug", "snail",
                    "leech", "spider", "ant", "black widow", "insect", "pin", "needle",
                    "nail", "screw", "hook", "corkscrew", "band-aid"
                }
                is_allowed = any(kw in top1_name for kw in skin_allowed_keywords)
                if not is_allowed and top1_prob > 0.15:
                    logger.warning(f"Skin check rejected non-skin object: '{top1_name}' with confidence {top1_prob:.4f}")
                    return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"

            elif expected_key == "oral":
                oral_allowed_keywords = {
                    "mouth", "tongue", "tooth", "teeth", "lip", "lips", "lipstick", "cheek", "dentist",
                    "chimpanzee", "orangutan", "gorilla", "macaque", "monkey", "proboscis",
                    "stinkhorn", "pomegranate", "strawberry", "band aid", "diaper", "nipple"
                }
                is_allowed = any(kw in top1_name for kw in oral_allowed_keywords)
                if not is_allowed and top1_prob > 0.15:
                    logger.warning(f"Oral check rejected non-oral object: '{top1_name}' with confidence {top1_prob:.4f}")
                    return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
            
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
                # Must contain mouth keywords in top 5 AND NOT contain any forbidden classes in top 5
                has_mouth = any(word in "".join(top10_names[:5]) for word in ["lipstick", "mouth", "tongue", "tooth", "teeth", "lip", "lips", "cheek", "dentist"])
                
                # Check for any forbidden classes in top 5
                has_forbidden = False
                for name, prob in zip(top10_names[:5], top10_probs[:5]):
                    if prob > 0.05:
                        class_words = name.replace(",", "").replace("-", " ").split(" ")
                        if any(word in FORBIDDEN_CLASSES for word in class_words):
                            has_forbidden = True
                            
                if not has_mouth or has_forbidden:
                    logger.warning("Oral check failed: no mouth keywords or has forbidden classes.")
                    return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
            
            # Skin check
            if expected_key == "skin":
                # Reject if oral score is higher and significant
                if oral_score > skin_score and oral_score > 0.15:
                    logger.warning(f"Skin check rejected: oral score {oral_score:.4f} is higher than skin score {skin_score:.4f}")
                    return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
                
                # Check for mouth keywords in top predictions
                has_mouth_close_up = any(word in top10_names[0] for word in ["mouth", "tongue", "tooth", "teeth"])
                if has_mouth_close_up:
                    return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
                
                # Must NOT have any forbidden classes in top 5
                has_forbidden = False
                for name, prob in zip(top10_names[:5], top10_probs[:5]):
                    if prob > 0.05:
                        class_words = name.replace(",", "").replace("-", " ").split(" ")
                        if any(word in FORBIDDEN_CLASSES for word in class_words):
                            has_forbidden = True
                            
                if has_forbidden:
                    logger.warning("Skin check failed: contains forbidden classes.")
                    return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
                
                # Require either skin-related keywords or sufficient skin color coverage
                has_skin = (skin_score > 0.0)
                if not has_skin and skin_pct < 20.0:
                    logger.warning(f"Skin check failed: no skin keywords and skin pct is {skin_pct:.2f}%")
                    return False, "Your uploaded image is not relevent for this project. please upload relevent image and try again"
                    
            return True, ""
            
        except Exception as e:
            logger.error(f"Error in tissue validation: {str(e)}")
            return True, ""

if __name__ == "__main__":
    detector = BodyDetector()
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    print("Dummy black image validation:", detector.is_human_body(dummy_img))
