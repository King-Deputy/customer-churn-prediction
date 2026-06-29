"""Configuration module for the customer churn prediction project."""

from config.config import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    OUTPUTS_DIR,
    # Data processing
    DROP_COLUMNS,
    TARGET_COLUMN,
    ENCODED_TARGET_COLUMN,
    TARGET_MAPPING,
    RANDOM_STATE,
    # Feature engineering
    CATEGORICAL_ENCODING_METHOD,
    SCALING_METHOD,
    # Model training
    TEST_SIZE,
    VALIDATION_SIZE,
    MODEL_HYPERPARAMS,
    # Evaluation
    EVALUATION_METRICS,
    CV_FOLDS,
    # Utility functions
    ensure_directories_exist,
    get_config_summary,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "MODELS_DIR",
    "FIGURES_DIR",
    "OUTPUTS_DIR",
    "DROP_COLUMNS",
    "TARGET_COLUMN",
    "ENCODED_TARGET_COLUMN",
    "TARGET_MAPPING",
    "RANDOM_STATE",
    "CATEGORICAL_ENCODING_METHOD",
    "SCALING_METHOD",
    "TEST_SIZE",
    "VALIDATION_SIZE",
    "MODEL_HYPERPARAMS",
    "EVALUATION_METRICS",
    "CV_FOLDS",
    "ensure_directories_exist",
    "get_config_summary",
]
