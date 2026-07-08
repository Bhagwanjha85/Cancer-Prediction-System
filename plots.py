import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Any
import config

# Set plot style for professional look
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_theme(style="whitegrid")

def plot_training_history(history: Dict[str, List[float]]) -> plt.Figure:
    """
    Plots the training and validation loss and accuracy side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history["accuracy"]) + 1)
    
    # Accuracy plot
    ax1.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#1b285c", linewidth=2)
    if "val_accuracy" in history:
        ax1.plot(epochs, history["val_accuracy"], "s-", label="Validation Accuracy", color="#808080", linewidth=2)
    ax1.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Loss plot
    ax2.plot(epochs, history["loss"], "o-", label="Training Loss", color="#1b285c", linewidth=2)
    if "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], "s-", label="Validation Loss", color="#808080", linewidth=2)
    ax2.set_title("Training & Validation Loss", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix_sns(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> plt.Figure:
    """
    Plots a highly detailed and stylized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create text labels with raw count and percentage
    labels = np.asarray([
        f"{count}\n({percent:.1%})"
        for count, percent in zip(cm.flatten(), cm_norm.flatten())
    ]).reshape(cm.shape)
    
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names, 
        ax=ax,
        cbar=True,
        square=True,
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Class", fontsize=13, labelpad=10)
    ax.set_ylabel("True Class", fontsize=13, labelpad=10)
    plt.xticks(rotation=15, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_roc_curves(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, class_names: List[str]) -> plt.Figure:
    """
    Plots ROC curves for all classes with their respective AUC scores.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_classes = len(class_names)
    colors = ["#1b285c", "#000000", "#808080", "#cccccc", "#555555"]
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr, 
            color=colors[i % len(colors)], 
            lw=2.5, 
            label=f"{class_names[i]} (AUC = {roc_auc:.3f})"
        )
        
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_probability_graph(probabilities: np.ndarray, class_names: List[str]) -> plt.Figure:
    """
    Plots a horizontal bar chart displaying prediction probabilities for the classes.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    # Select colors for the classes (highlights the highest one)
    max_idx = np.argmax(probabilities)
    colors = ["#E9ECEF"] * len(class_names)
    colors[max_idx] = "#1b285c"
    
    y_pos = np.arange(len(class_names))
    
    bars = ax.barh(y_pos, probabilities * 100, align="center", color=colors, height=0.6, edgecolor="none")
    
    # Label modifications
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace('_', ' ') for c in class_names], fontsize=11, fontweight="bold")
    ax.invert_yaxis()  # Labels read top-to-bottom
    
    # X axis configurations
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_xlim(0, 105)
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5, 
            bar.get_y() + bar.get_height()/2.0, 
            f"{width:.1f}%", 
            ha="left", 
            va="center", 
            fontsize=10, 
            fontweight="bold",
            color="#212529"
        )
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DEE2E6')
    ax.spines['bottom'].set_color('#DEE2E6')
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    return fig
