from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"


def load_data() -> pd.DataFrame:
    """Load the raw IBM Telco Customer Churn dataset."""
    return pd.read_csv(DATA_PATH)
