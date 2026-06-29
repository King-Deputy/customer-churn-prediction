"""
Project configuration settings for Customer Churn Prediction.

This module contains all configurable parameters for the machine learning pipeline,
including paths, model hyperparameters, preprocessing settings, and evaluation metrics.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco_customer_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "telco_processed.csv"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Output paths
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Notebooks path
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Source code path
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

# Columns to drop during preprocessing
DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]

# Numeric columns to convert
NUMERIC_COLUMNS = ["Total Charges"]

# Target variable configuration
TARGET_COLUMN = "Churn Label"
ENCODED_TARGET_COLUMN = "Encoded Churn Label"
TARGET_MAPPING = {"Yes": 1, "No": 0}

# Missing value handling strategy
# Options: 'drop', 'mean', 'median'
MISSING_VALUE_METHOD = "drop"

# Random seed for reproducibility
RANDOM_STATE = 42


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Categorical features to encode
CATEGORICAL_ENCODING_METHOD = "onehot"  # Options: 'onehot', 'label'

# Numerical features scaling
SCALING_METHOD = "standardscaler"  # Options: 'standardscaler', 'minmaxscaler'

# Columns to drop for feature separation
FEATURE_DROP_COLUMNS = [
    "Churn Label",
    "Churn Value",
    "Encoded Churn Label"
]


# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # Portion of training data to use for validation

# Model hyperparameters
MODEL_HYPERPARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs"
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced"
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
        "subsample": 0.8
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_child_weight": 5,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": 2.77,  # Class imbalance ratio
        "objective": "binary:logistic",
        "eval_metric": "auc"
    }
}


# ============================================================================
# MODEL EVALUATION CONFIGURATION
# ============================================================================

# Classification metrics to calculate
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "confusion_matrix",
    "classification_report"
]

# Threshold for classification (default: 0.5)
CLASSIFICATION_THRESHOLD = 0.5

# Cross-validation folds
CV_FOLDS = 5


# ============================================================================
# EXPLAINABILITY CONFIGURATION
# ============================================================================

# SHAP configuration
SHAP_EXPLAINER = "tree"  # Options: 'tree', 'kernel', 'linear'
SHAP_NUM_SAMPLES = 100

# Feature importance threshold (for SHAP)
FEATURE_IMPORTANCE_TOP_N = 15


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

# Streamlit app settings
STREAMLIT_PAGE_CONFIG = {
    "page_title": "Customer Churn Prediction",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model serving settings
MODEL_VERSION = "v1.0.0"
MODEL_FORMAT = "pickle"  # Options: 'pickle', 'joblib', 'onnx'


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_FILE = OUTPUTS_DIR / "project.log"


# ============================================================================
# EXPERIMENT TRACKING CONFIGURATION
# ============================================================================

# MLflow configuration (if using MLflow)
EXPERIMENT_NAME = "customer-churn-prediction"
MLFLOW_TRACKING_URI = None  # Set to track experiments
MLFLOW_BACKEND_STORE = "sqlite:///mlruns.db"


# ============================================================================
# DATA QUALITY CONFIGURATION
# ============================================================================

# Acceptable missing value percentage per column (0-100)
MAX_MISSING_PERCENTAGE = 10

# Outlier detection method
OUTLIER_DETECTION_METHOD = "iqr"  # Options: 'iqr', 'zscore'

# IQR multiplier for outlier detection
IQR_MULTIPLIER = 1.5


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories_exist():
    """Ensure all required directories exist."""
    directories = [
        DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR,
        MODELS_DIR, FIGURES_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_summary():
    """Return a summary of key configuration settings."""
    return {
        "project_root": str(PROJECT_ROOT),
        "raw_data_path": str(RAW_DATA_PATH),
        "processed_data_path": str(PROCESSED_DATA_PATH),
        "models_directory": str(MODELS_DIR),
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS
    }


if __name__ == "__main__":
    # Create directories on import if running as main
    ensure_directories_exist()
    print("Configuration loaded successfully!")
    print("\nKey Settings:")
    for key, value in get_config_summary().items():
        print(f"  {key}: {value}")
