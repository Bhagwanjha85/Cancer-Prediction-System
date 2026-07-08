import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PerformanceEvaluator")

class PerformanceEvaluator:
    """
    Loads, processes, and formats training history and evaluation metrics
    for all trained models.
    """
    
    def __init__(self, cancer_type: str = "oral"):
        self.cancer_type = cancer_type.lower()
        if self.cancer_type == "oral":
            self.history_path = config.HISTORY_ORAL_PATH
        elif self.cancer_type == "skin":
            self.history_path = config.HISTORY_SKIN_PATH
        else:
            self.history_path = os.path.join(config.MODELS_DIR, "history.pkl")
            
        self.data: Optional[Dict[str, Any]] = None
        self.load_history_data()

    def load_history_data(self):
        """
        Loads training history from the pickled file.
        """
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "rb") as f:
                    self.data = pickle.load(f)
                logger.info(f"Successfully loaded history and metrics for {self.cancer_type}.")
            except Exception as e:
                logger.error(f"Failed to load history pickle for {self.cancer_type}: {str(e)}")
                self.data = None
        else:
            logger.warning(f"No history file found for {self.cancer_type} at {self.history_path}.")
            self.data = None

    def get_best_model_name(self) -> str:
        """
        Returns the name of the best-performing architecture.
        """
        if self.data:
            return self.data.get("best_model_name", "N/A")
        return "N/A"

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame comparing the validation metrics of all architectures.
        """
        if not self.data or "comparison" not in self.data:
            return pd.DataFrame()
            
        comp = self.data["comparison"]
        records = []
        for arch, metrics in comp.items():
            records.append({
                "Architecture": arch,
                "Val Accuracy": metrics.get("val_accuracy", 0.0),
                "Val Loss": metrics.get("val_loss", 0.0),
                "Training Time (s)": round(metrics.get("training_time_sec", 0.0), 1),
                "Epochs Run": metrics.get("epochs_run", 0)
            })
            
        df = pd.DataFrame(records)
        df = df.sort_values(by="Val Accuracy", ascending=False).reset_index(drop=True)
        return df

    def get_best_model_metrics(self) -> Dict[str, Any]:
        """
        Returns classification metrics (accuracy, precision, recall, f1, auc) for the best model.
        """
        if not self.data or "test_metrics" not in self.data:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc": 0.0
            }
        metrics = self.data["test_metrics"]
        return {
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "auc": metrics.get("auc", 0.0)
        }

    def get_classification_report_df(self) -> pd.DataFrame:
        """
        Returns the classification report of the best model as a pandas DataFrame.
        """
        if not self.data or "test_metrics" not in self.data:
            return pd.DataFrame()
            
        test_metrics = self.data["test_metrics"]
        report = test_metrics.get("classification_report", {})
        
        if not report:
            return pd.DataFrame()
            
        records = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                records.append({
                    "Class": label,
                    "Precision": metrics.get("precision", 0.0),
                    "Recall": metrics.get("recall", 0.0),
                    "F1-Score": metrics.get("f1-score", 0.0),
                    "Support": metrics.get("support", 0)
                })
                
        return pd.DataFrame(records)

    def get_best_model_history(self) -> Dict[str, list]:
        """
        Returns the accuracy/loss history dictionary of the best model.
        """
        if self.data and "best_history" in self.data:
            return self.data["best_history"]
        return {}

    def get_test_predictions_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns y_true, y_pred, and class probabilities on the test set for plotting.
        """
        if not self.data or "test_metrics" not in self.data:
            empty = np.array([])
            return empty, empty, empty
            
        test_metrics = self.data["test_metrics"]
        return (
            np.array(test_metrics["y_true"]),
            np.array(test_metrics["y_pred"]),
            np.array(test_metrics["y_pred_prob"])
        )
