"""
Utility functions for the customer churn prediction project.

This module provides helper functions for data loading, saving, visualization,
model evaluation, logging, and other common tasks across the project.
"""

import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Import project configuration
from config.config import PROJECT_ROOT, OUTPUTS_DIR, LOG_LEVEL


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = LOG_LEVEL, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs only to console.
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# FILE AND PATH UTILITIES
# ============================================================================

def create_project_directories() -> None:
    """Create all necessary project directories if they don't exist."""
    from config.config import ensure_directories_exist
    ensure_directories_exist()
    logger.info("Project directories created/verified")


def get_project_path(relative_path: str) -> Path:
    """
    Get absolute path from a project-relative path.
    
    Args:
        relative_path: Path relative to project root
        
    Returns:
        Absolute Path object
    """
    return PROJECT_ROOT / relative_path


def save_object(obj: Any, filepath: Path, format_type: str = "pickle") -> None:
    """
    Save Python object to file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
        format_type: Format type ('pickle' or 'json')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format_type == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(obj, f)
        elif format_type == "json":
            with open(filepath, "w") as f:
                json.dump(obj, f, indent=2, default=str)
        logger.info(f"Object saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving object to {filepath}: {e}")
        raise


def load_object(filepath: Path, format_type: str = "pickle") -> Any:
    """
    Load Python object from file.
    
    Args:
        filepath: Path to file
        format_type: Format type ('pickle' or 'json')
        
    Returns:
        Loaded object
    """
    try:
        if format_type == "pickle":
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
        elif format_type == "json":
            with open(filepath, "r") as f:
                obj = json.load(f)
        logger.info(f"Object loaded from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {filepath}: {e}")
        raise


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Load CSV file into DataFrame.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
        
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV from {filepath}: {e}")
        raise


def save_csv(df: pd.DataFrame, filepath: Path, index: bool = False, **kwargs) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save file
        index: Whether to save index
        **kwargs: Additional arguments to pass to df.to_csv()
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(filepath, index=index, **kwargs)
        logger.info(f"DataFrame saved to {filepath} ({len(df)} rows, {len(df.columns)} columns)")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {e}")
        raise


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary information about a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data information
    """
    return {
        "shape": df.shape,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "missing_percentage": (df.isna().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 ** 2
    }


def print_data_summary(df: pd.DataFrame) -> None:
    """Print comprehensive summary of DataFrame."""
    info = get_data_info(df)
    print(f"\n{'='*60}")
    print(f"DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Shape: {info['shape']} (Rows x Columns)")
    print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
    print(f"\nMissing Values:")
    missing_data = pd.DataFrame({
        'Missing Count': info['missing_values'],
        'Missing %': info['missing_percentage']
    })
    print(missing_data[missing_data['Missing Count'] > 0])
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"{'='*60}\n")


# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: Optional[np.ndarray] = None,
                   model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate classification model with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
        model_name: Name of model for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        try:
            if y_pred_proba.ndim > 1:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC for {model_name}: {e}")
    
    # Log metrics
    logger.info(f"\n{model_name} Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, 
                                target_names=["No Churn", "Churn"])


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[Path] = None) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        figsize: Figure size
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  figsize: Tuple[int, int] = (8, 6),
                  save_path: Optional[Path] = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size
        save_path: Path to save plot
    """
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                           top_n: int = 15,
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[Path] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importances: Feature importance values
        feature_names: Feature names
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Path to save plot
    """
    indices = np.argsort(importances)[-top_n:]
    sorted_importances = importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_features)), sorted_importances, color='steelblue')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_distribution(data: pd.Series, title: str = "",
                     figsize: Tuple[int, int] = (10, 5),
                     save_path: Optional[Path] = None) -> None:
    """
    Plot distribution of a series.
    
    Args:
        data: Data to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save plot
    """
    plt.figure(figsize=figsize)
    data.hist(bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel(data.name)
    plt.ylabel('Frequency')
    plt.title(title or f'Distribution of {data.name}')
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_class_distribution(y: pd.Series) -> Dict[str, float]:
    """
    Calculate class distribution percentages.
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary with class distribution
    """
    distribution = (y.value_counts() / len(y) * 100).to_dict()
    logger.info(f"Class Distribution: {distribution}")
    return distribution


def detect_outliers(df: pd.DataFrame, column: str, method: str = "iqr",
                   threshold: float = 1.5) -> pd.Index:
    """
    Detect outliers in a column.
    
    Args:
        df: DataFrame
        column: Column name
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for IQR (multiplier) or z-score
        
    Returns:
        Index of outlier rows
    """
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    
    elif method == "zscore":
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[np.abs(stats.zscore(df[column])) > threshold].index
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Detected {len(outliers)} outliers in {column} using {method} method")
    return outliers


# ============================================================================
# TIMESTAMP AND VERSIONING UTILITIES
# ============================================================================

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime(format_str)


def get_experiment_dir(base_dir: Path = OUTPUTS_DIR) -> Path:
    """Create and return a timestamped experiment directory."""
    exp_dir = base_dir / f"experiment_{get_timestamp()}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def save_experiment_config(config: Dict[str, Any], exp_dir: Path) -> None:
    """
    Save experiment configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        exp_dir: Experiment directory
    """
    config_path = exp_dir / "config.json"
    save_object(config, config_path, format_type="json")


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    return load_object(config_path, format_type="json")


if __name__ == "__main__":
    logger.info("Utils module loaded successfully")
