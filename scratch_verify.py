import os
import cv2
import numpy as np
import glob
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights).to(device)
    model.eval()
    categories = weights.meta["categories"]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define keywords for oral vs skin
    oral_keywords = {
        "lipstick", "mouth", "tongue", "tooth", "teeth", "lip", "lips", "face",
        "chimpanzee", "orangutan", "gorilla", "macaque", "monkey", "proboscis",
        "stinkhorn", "pomegranate", "strawberry", "band aid", "diaper", "nipple"
    }
    
    skin_keywords = {
        "bubble", "loupe", "jellyfish", "petri dish", "eggnog", "espresso",
        "ant", "tick", "slug", "snail", "spider", "nematode", "black widow",
        "lacewing", "wok", "frying pan", "mortar", "soup bowl"
    }
    
    def classify_image(img_path):
        img = cv2.imread(img_path)
        if img is None:
            return "unknown"
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
        top10_prob, top10_catid = torch.topk(probs, 10)
        
        oral_score = 0.0
        skin_score = 0.0
        
        # Check if any top prediction matches keywords
        for i in range(10):
            class_name = categories[top10_catid[i].item()].lower()
            prob = top10_prob[i].item()
            
            # Match oral keywords
            for word in oral_keywords:
                if word in class_name:
                    oral_score += prob
            # Match skin keywords
            for word in skin_keywords:
                if word in class_name:
                    skin_score += prob
                    
        # Fallback to HSV color ratio if score is tie or zero
        if oral_score == 0 and skin_score == 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            total_pixels = img.shape[0] * img.shape[1]
            
            red_mask = ((h <= 7) | (h >= 160)) & (s >= 50) & (v >= 40)
            red_pct = np.sum(red_mask) / total_pixels * 100
            
            skin_mask = (h >= 8) & (h <= 24) & (s >= 20) & (s <= 150) & (v >= 50)
            skin_pct = np.sum(skin_mask) / total_pixels * 100
            
            if red_pct > skin_pct:
                return "oral"
            else:
                return "skin"
                
        if oral_score >= skin_score:
            return "oral"
        else:
            return "skin"

    print("Evaluating ImageNet Keyword Classifier:")
    
    oral_files = glob.glob("dataset/oral/train/**/*.jpg", recursive=True) + \
                 glob.glob("dataset/oral/train/**/*.jpeg", recursive=True) + \
                 glob.glob("dataset/oral/train/**/*.png", recursive=True)
    
    correct_oral = 0
    total_oral = 0
    for f in oral_files[:100]:
        total_oral += 1
        if classify_image(f) == "oral":
            correct_oral += 1
            
    skin_files = glob.glob("dataset/skin/train/**/*.jpg", recursive=True) + \
                 glob.glob("dataset/skin/train/**/*.jpeg", recursive=True) + \
                 glob.glob("dataset/skin/train/**/*.png", recursive=True)
                 
    correct_skin = 0
    total_skin = 0
    for f in skin_files[:100]:
        total_skin += 1
        if classify_image(f) == "skin":
            correct_skin += 1
            
    print(f"Oral Accuracy: {correct_oral}/{total_oral} ({correct_oral/total_oral*100:.1f}%)")
    print(f"Skin Accuracy: {correct_skin}/{total_skin} ({correct_skin/total_skin*100:.1f}%)")
    print(f"Total: {correct_oral + correct_skin}/{total_oral + total_skin} ({(correct_oral + correct_skin)/(total_oral + total_skin)*100:.1f}%)")

if __name__ == "__main__":
    main()
