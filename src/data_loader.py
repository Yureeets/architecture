import pandas as pd
from pathlib import Path


# Mapping of numerical labels to diagnosis names
LABEL_NAMES = {
    0: 'akiec',  # Actinic keratoses
    1: 'bcc',    # Basal cell carcinoma
    2: 'bkl',    # Benign keratosis
    3: 'df',     # Dermatofibroma
    4: 'nv',     # Melanocytic nevi
    5: 'vasc',   # Vascular lesions
    6: 'mel',    # Melanoma
}

# Повні назви діагнозів
LABEL_FULL_NAMES = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Vascular lesions',
    6: 'Melanoma',
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_hmnist_data(filename: str = 'hmnist_28_28_L.csv') -> tuple:
    """
    Loading pixel data from a CSV file.

    Parameters:
        filename: name of the file (default hmnist_28_28_L.csv)

    Returns:
        X (pd.DataFrame) — feature matrix (pixel values)
        y (pd.Series) — target variable vector (class labels)
    """
    data_path = PROJECT_ROOT / 'data' / 'raw' / filename
    if not data_path.exists():
        data_path = PROJECT_ROOT / 'archive' / filename

    if not data_path.exists():
        raise FileNotFoundError(
            f"File {filename} not found in data/raw/ or archive/"
        )

    print(f" Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop('label', axis=1)
    y = df['label']

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {y.nunique()} classes")
    return X, y


def load_metadata(filename: str = 'HAM10000_metadata.csv') -> pd.DataFrame:
    """
    Loading HAM10000 metadata.

    Parameters:
        filename: name of the metadata file

    Returns:
        pd.DataFrame with metadata
    """
    data_path = PROJECT_ROOT / 'data' / 'raw' / filename
    if not data_path.exists():
        data_path = PROJECT_ROOT / 'archive' / filename

    if not data_path.exists():
        raise FileNotFoundError(
            f"File {filename} not found in data/raw/ or archive/"
        )

    df = pd.read_csv(data_path)
    print(f" Metadata loaded: {df.shape[0]} records")
    return df


if __name__ == '__main__':
    X, y = load_hmnist_data()
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y distribution:\n{y.value_counts().sort_index()}")
