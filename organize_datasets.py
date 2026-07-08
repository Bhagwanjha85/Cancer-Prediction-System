import os
import glob
import shutil
import random

def main():
    print("Starting dataset organization and splitting...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    # 1. Target folders
    target_splits = ["train", "validation", "test"]
    target_cats = {
        "oral": ["Cancer", "Normal"],
        "skin": ["Cancer", "Normal"]
    }
    
    # Clear existing files in target folders (to remove mock images)
    for cat_name, subcats in target_cats.items():
        for split in target_splits:
            for subcat in subcats:
                folder = os.path.join(dataset_dir, cat_name, split, subcat)
                if os.path.exists(folder):
                    print(f"Clearing folder: {folder}")
                    for f in glob.glob(os.path.join(folder, "*")):
                        if os.path.isfile(f):
                            os.remove(f)
                else:
                    os.makedirs(folder, exist_ok=True)
                    
    # 2. Gather oral files
    oral_cancer_src = os.path.join(dataset_dir, "oral", "Oral Cancer", "Oral Cancer Dataset", "CANCER")
    oral_normal_src = os.path.join(dataset_dir, "oral", "Oral Cancer", "Oral Cancer Dataset", "NON CANCER")
    
    oral_cancer_files = glob.glob(os.path.join(oral_cancer_src, "*"))
    oral_normal_files = glob.glob(os.path.join(oral_normal_src, "*"))
    
    print(f"Gathered Oral Cancer: {len(oral_cancer_files)} files")
    print(f"Gathered Oral Normal: {len(oral_normal_files)} files")
    
    # 3. Gather skin files
    # Cancerous classes: melanoma, basal cell carcinoma, actinic keratosis, squamous cell carcinoma
    # Normal/Benign classes: nevus, dermatofibroma, pigmented benign keratosis, seborrheic keratosis, vascular lesion
    skin_train_dir = os.path.join(dataset_dir, "skin", "Skin cancer ISIC The International Skin Imaging Collaboration", "Train")
    skin_test_dir = os.path.join(dataset_dir, "skin", "Skin cancer ISIC The International Skin Imaging Collaboration", "Test")
    
    skin_cancer_files = []
    skin_normal_files = []
    
    skin_cancer_subdirs = ["melanoma", "basal cell carcinoma", "actinic keratosis", "squamous cell carcinoma"]
    skin_normal_subdirs = ["nevus", "dermatofibroma", "pigmented benign keratosis", "seborrheic keratosis", "vascular lesion"]
    
    # Gather from Train
    if os.path.exists(skin_train_dir):
        for subd in os.listdir(skin_train_dir):
            subd_path = os.path.join(skin_train_dir, subd)
            if os.path.isdir(subd_path):
                files = glob.glob(os.path.join(subd_path, "*"))
                if subd.lower() in [s.lower() for s in skin_cancer_subdirs]:
                    skin_cancer_files.extend(files)
                elif subd.lower() in [s.lower() for s in skin_normal_subdirs]:
                    skin_normal_files.extend(files)
                    
    # Gather from Test
    if os.path.exists(skin_test_dir):
        for subd in os.listdir(skin_test_dir):
            subd_path = os.path.join(skin_test_dir, subd)
            if os.path.isdir(subd_path):
                files = glob.glob(os.path.join(subd_path, "*"))
                if subd.lower() in [s.lower() for s in skin_cancer_subdirs]:
                    skin_cancer_files.extend(files)
                elif subd.lower() in [s.lower() for s in skin_normal_subdirs]:
                    skin_normal_files.extend(files)
                    
    print(f"Gathered Skin Cancer: {len(skin_cancer_files)} files")
    print(f"Gathered Skin Normal: {len(skin_normal_files)} files")
    
    # 4. Split and Copy Helper
    def split_and_copy(files, cat_name, subcat_name):
        random.seed(42)
        random.shuffle(files)
        
        total = len(files)
        if total == 0:
            print(f"Warning: No files found for {cat_name}/{subcat_name}")
            return
            
        train_idx = int(total * 0.70)
        val_idx = int(total * 0.85)
        
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]
        
        splits = {
            "train": train_files,
            "validation": val_files,
            "test": test_files
        }
        
        for split_name, split_files in splits.items():
            dest_dir = os.path.join(dataset_dir, cat_name, split_name, subcat_name)
            os.makedirs(dest_dir, exist_ok=True)
            for f in split_files:
                if os.path.isfile(f):
                    shutil.copy2(f, dest_dir)
                    
        print(f"Copied {cat_name}/{subcat_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
    # Copy Oral
    split_and_copy(oral_cancer_files, "oral", "Cancer")
    split_and_copy(oral_normal_files, "oral", "Normal")
    
    # Copy Skin
    split_and_copy(skin_cancer_files, "skin", "Cancer")
    split_and_copy(skin_normal_files, "skin", "Normal")
    
    print("Dataset organization and splitting completed successfully!")

if __name__ == "__main__":
    main()
