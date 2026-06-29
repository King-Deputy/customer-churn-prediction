from pathlib import Path
import pandas as pd
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Columns to drop from the dataset
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


def convert_numeric_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Convert specified columns to numeric type, handling non-numeric values.
    
    Args:
        df: Input DataFrame
        columns: List of columns to convert. If None, converts 'Total Charges'
        
    Returns:
        DataFrame with converted columns
    """
    if columns is None:
        columns = ["Total Charges"]
    
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def handle_missing_values(df: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        method: 'drop' to remove rows with missing values, 'mean'/'median' to fill numerical columns
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if method == "drop":
        df = df.dropna()
    elif method == "mean":
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    elif method == "median":
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    return df


def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Drop unnecessary columns from the DataFrame.
    
    Args:
        df: Input DataFrame
        columns_to_drop: List of columns to drop. If None, uses predefined DROP_COLUMNS
        
    Returns:
        DataFrame with dropped columns
    """
    if columns_to_drop is None:
        columns_to_drop = DROP_COLUMNS
    
    df = df.copy()
    # Only drop columns that exist in the dataframe
    cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_target(df: pd.DataFrame, target_col: str = "Churn Label", 
                  mapping: dict = None) -> pd.DataFrame:
    """
    Encode the target variable to binary format.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        mapping: Dictionary to map target values. Default is {'Yes': 1, 'No': 0}
        
    Returns:
        DataFrame with encoded target column
    """
    if mapping is None:
        mapping = {"Yes": 1, "No": 0}
    
    df = df.copy()
    if target_col in df.columns:
        df["Encoded Churn Label"] = df[target_col].map(mapping)
    return df


def separate_features_and_target(df: pd.DataFrame, 
                                 target_col: str = "Encoded Churn Label",
                                 drop_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from the DataFrame.
    
    Args:
        df: Input DataFrame
        target_col: Name of the encoded target column
        drop_cols: Additional columns to drop besides target columns
        
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    if drop_cols is None:
        drop_cols = []
    
    cols_to_drop = [target_col, "Churn Label", "Churn Value"] + drop_cols
    # Only drop columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """
    Get numerical and categorical feature columns.
    
    Args:
        X: Features DataFrame
        
    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    return numerical_features, categorical_features


def preprocess_data(df: pd.DataFrame, 
                   numeric_cols: list = None,
                   missing_method: str = "drop",
                   cols_to_drop: list = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Index, pd.Index]:
    """
    Complete preprocessing pipeline for the telco customer churn dataset.
    
    Args:
        df: Raw input DataFrame
        numeric_cols: Columns to convert to numeric. Default is ['Total Charges']
        missing_method: Method to handle missing values ('drop', 'mean', 'median')
        cols_to_drop: Columns to drop. Default uses predefined DROP_COLUMNS
        
    Returns:
        Tuple of (processed_df, X, y, numerical_features, categorical_features)
    """
    # Step 1: Convert numeric columns
    df = convert_numeric_columns(df, numeric_cols)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df, method=missing_method)
    
    # Step 3: Drop unnecessary columns
    df = drop_unnecessary_columns(df, cols_to_drop)
    
    # Step 4: Encode target variable
    df = encode_target(df)
    
    # Step 5: Separate features and target
    X, y = separate_features_and_target(df)
    
    # Step 6: Get feature types
    numerical_features, categorical_features = get_feature_types(X)
    
    return df, X, y, numerical_features, categorical_features
