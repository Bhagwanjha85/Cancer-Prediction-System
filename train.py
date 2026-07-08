import os
import time
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Tuple

import config
import dataset_loader
import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PyTorchModelTraining")

class PyTorchCancerClassifier(nn.Module):
    """
    Unified PyTorch model wrapping timm pre-trained backbones
    with custom CNN-RNN (GRU) head for deep sequence-based spatial feature modeling.
    Supports a non-RNN legacy fallback head for backward compatibility with simple models.
    """
    def __init__(self, arch: str, num_classes: int = 4, use_rnn: bool = True):
        super().__init__()
        self.use_rnn = use_rnn
        
        # Map to timm registry names
        timm_mapping = {
            "MobileNetV3": "mobilenetv3_large_100",
            "EfficientNetB0": "efficientnet_b0",
            "EfficientNetB3": "efficientnet_b3",
            "EfficientNetV2": "efficientnetv2_rw_s",
            "ConvNeXt": "convnext_tiny",
            "ResNet50": "resnet50",
            "DenseNet121": "densenet121",
            "Xception": "xception",
            "VGG16": "vgg16"
        }
        
        timm_name = timm_mapping.get(arch, "resnet50")
        logger.info(f"Initializing {arch} backbone...")
        
        # Load backbone as a feature extractor (num_classes=0)
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        
        # Enable fine-tuning of the deeper layers for high-sensitivity medical feature extraction.
        # We freeze only the first 35% of parameters (general edge detectors) to retain pre-trained feature stability,
        # and unfreeze the rest to adapt specifically to pathological oral/skin color features.
        params = list(self.backbone.parameters())
        num_params = len(params)
        freeze_until = int(num_params * 0.35)
        for idx, param in enumerate(params):
            if idx < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True
            
        # Get actual feature dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
            dummy_features = self.backbone(dummy_input)
            in_features = dummy_features.shape[1]
        
        if self.use_rnn:
            # RNN Configuration
            self.seq_len = 4
            remainder = in_features % self.seq_len
            self.pad_size = (self.seq_len - remainder) % self.seq_len
            self.padded_features = in_features + self.pad_size
            self.feature_dim = self.padded_features // self.seq_len
            
            # RNN component (GRU layer)
            self.rnn = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            # Classifier head operating on bidirectional GRU features (64 hidden * 2 directions = 128)
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
        else:
            self.seq_len = 0
            self.pad_size = 0
            self.rnn = None
            # Standard multi-layer perceptron classifier for legacy / non-RNN models
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using CNN backbone
        features = self.backbone(x)  # Shape: (batch_size, num_features)
        batch_size = features.size(0)
        
        if not self.use_rnn or self.rnn is None:
            # Standard MLP head forward pass
            logits = self.classifier(features)
            return logits
            
        # Pad features to make divisible by seq_len
        if self.pad_size > 0:
            padding = torch.zeros(batch_size, self.pad_size, device=features.device, dtype=features.dtype)
            features = torch.cat([features, padding], dim=1)
            
        # Reshape to sequence: (batch_size, seq_len, feature_dim)
        x_seq = features.view(batch_size, self.seq_len, self.feature_dim)
        
        # Pass through RNN
        rnn_out, _ = self.rnn(x_seq)  # Shape: (batch_size, seq_len, hidden_size * 2)
        
        # Extract last step output
        last_step = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_size * 2)
        
        # Final classification
        logits = self.classifier(last_step)
        return logits

def train_one_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool
) -> Tuple[float, float]:
    """
    Trains the model for one epoch. Returns mean loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # AMP autocast
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluates the model. Returns validation loss, accuracy, y_true, and y_probs.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Compute probabilities via softmax
            probs = torch.softmax(outputs, dim=1)
            
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc, np.array(all_targets), np.array(all_probs)

def train_and_evaluate_for_type(cancer_type: str = "all"):
    """
    Trains all architectures in PyTorch, compares performance, selects the best model,
    saves the best model/weights, and saves comparison metrics for a specific cancer type.
    """
    # 1. Device Diagnostics & Setup
    gpu_info = utils.check_gpu_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running PyTorch training pipeline for '{cancer_type}' on device: {device}")
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        logger.info("FP16 Automatic Mixed Precision (AMP) enabled.")

    dataset_loader.ensure_dataset_exists()

    # Determine paths and class list
    cancer_type_lower = cancer_type.lower()
    if cancer_type_lower == "oral":
        class_names = config.CLASS_NAMES_ORAL
        model_path = config.MODEL_ORAL_PATH
        weights_path = config.WEIGHTS_ORAL_PATH
        label_encoder_path = config.LABEL_ENCODER_ORAL_PATH
        history_path = config.HISTORY_ORAL_PATH
    elif cancer_type_lower == "skin":
        class_names = config.CLASS_NAMES_SKIN
        model_path = config.MODEL_SKIN_PATH
        weights_path = config.WEIGHTS_SKIN_PATH
        label_encoder_path = config.LABEL_ENCODER_SKIN_PATH
        history_path = config.HISTORY_SKIN_PATH
    else:
        class_names = config.CLASS_NAMES
        model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
        weights_path = os.path.join(config.MODELS_DIR, "weights.pth")
        label_encoder_path = os.path.join(config.MODELS_DIR, "label_encoder.pkl")
        history_path = os.path.join(config.MODELS_DIR, "history.pkl")

    num_classes = len(class_names)

    # 2. Get DataLoaders
    logger.info(f"Loading PyTorch datasets and loaders for {cancer_type}...")
    train_loader = dataset_loader.get_dataloader("train", augment=True, cancer_type=cancer_type)
    val_loader = dataset_loader.get_dataloader("validation", augment=False, cancer_type=cancer_type)
    test_loader = dataset_loader.get_dataloader("test", augment=False, cancer_type=cancer_type)

    comparison_results = {}
    best_val_acc = 0.0
    best_model_name = None
    best_model_state = None
    best_history = None

    # 3. Train all architectures
    for arch in config.MODEL_ARCHITECTURES:
        logger.info(f"\n" + "="*50 + f"\nTraining Architecture: {arch} ({cancer_type})\n" + "="*50)
        
        # Build Model & Setup Training
        model = PyTorchCancerClassifier(arch, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
        
        # Training Metrics Cache
        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        best_val_loss = float("inf")
        patience = 4
        epochs_no_improve = 0
        best_epoch_weights = None
        
        start_time = time.time()
        
        for epoch in range(1, config.EPOCHS + 1):
            # Step decay scheduler logic (custom step)
            # Reduce learning rate by factor 0.5 every 4 epochs
            if epoch > 1 and epoch % 4 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    logger.info(f"LearningRateScheduler: Dropped learning rate to {param_group['lr']}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, use_amp
            )
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            
            # Step learning rate scheduler on validation loss
            scheduler.step(val_loss)
            
            # Update history
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            
            logger.info(
                f"Epoch {epoch}/{config.EPOCHS} - "
                f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Model Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch_weights = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"EarlyStopping: Validation loss hasn't improved for {patience} epochs. Stopping.")
                    break
                    
        training_duration = time.time() - start_time
        
        # Load best weights
        if best_epoch_weights is not None:
            model.load_state_dict(best_epoch_weights)
            
        # Re-evaluate best weights on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        logger.info(f"Finished {arch}. Best Val Acc: {val_acc:.4f}, Best Val Loss: {val_loss:.4f}, Duration: {training_duration:.1f}s")
        
        # Cache comparison data
        comparison_results[arch] = {
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "training_time_sec": training_duration,
            "epochs_run": len(history["accuracy"]),
            "history": history
        }
        
        # Track global best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_name = arch
            best_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            best_history = history

    logger.info(f"\n" + "*"*60 + f"\nBEST MODEL SELECTED FOR {cancer_type.upper()}: {best_model_name} with Val Accuracy: {best_val_acc:.4f}\n" + "*"*60)

    # 4. Perform complete test-set evaluation for the Best Model
    logger.info(f"Re-building best model ({best_model_name}) for full test set evaluation...")
    best_model = PyTorchCancerClassifier(best_model_name, num_classes=num_classes).to(device)
    best_model.load_state_dict(best_model_state)
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc, y_true, y_pred_prob = evaluate(best_model, test_loader, criterion, device)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate performance metrics
    test_precision = precision_score(y_true, y_pred, average="macro")
    test_recall = recall_score(y_true, y_pred, average="macro")
    test_f1 = f1_score(y_true, y_pred, average="macro")
    
    if num_classes == 2:
        test_auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    else:
        y_true_onehot = label_binarize(y_true, classes=list(range(num_classes)))
        test_auc = roc_auc_score(y_true_onehot, y_pred_prob, average="macro", multi_class="ovr")
    
    classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    logger.info(f"[{cancer_type}] Test Accuracy: {test_acc:.4f}")
    logger.info(f"[{cancer_type}] Test Precision: {test_precision:.4f}")
    logger.info(f"[{cancer_type}] Test Recall: {test_recall:.4f}")
    logger.info(f"[{cancer_type}] Test F1-Score: {test_f1:.4f}")
    logger.info(f"[{cancer_type}] Test AUC: {test_auc:.4f}")

    # 5. Save the best model, weights, label encoder, and history dictionary
    # Save state dict
    logger.info(f"Saving best model weights to {weights_path}...")
    torch.save(best_model_state, weights_path)
    
    # Save best model architecture name metadata alongside it
    logger.info(f"Saving best model metadata package to {model_path}...")
    meta_package = {
        "architecture": best_model_name,
        "state_dict": best_model_state
    }
    torch.save(meta_package, model_path)
    
    logger.info(f"Saving label encoder to {label_encoder_path}...")
    with open(label_encoder_path, "wb") as f:
        pickle.dump(class_names, f)
        
    logger.info(f"Saving history and comparison data to {history_path}...")
    meta_history = {
        "best_model_name": best_model_name,
        "best_history": best_history,
        "comparison": comparison_results,
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "auc": test_auc,
            "classification_report": classification_rep,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob
        }
    }
    
    with open(history_path, "wb") as f:
        pickle.dump(meta_history, f)
        
    logger.info(f"Training pipeline execution for '{cancer_type}' completed successfully!")

if __name__ == "__main__":
    logger.info("Initializing Cancer Detection separate models training sequence...")
    train_and_evaluate_for_type("oral")
    train_and_evaluate_for_type("skin")
    logger.info("All model training routines executed successfully!")
