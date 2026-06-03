import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "housing.csv"

FEATURE_COLS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET_COL = "MedHouseVal"
ALL_COLS = FEATURE_COLS + [TARGET_COL]


def _bootstrap_csv() -> None:
    if DATA_PATH.exists():
        return
    from sklearn.datasets import fetch_california_housing

    dataset = fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df[TARGET_COL] = dataset.target
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)


def read_all() -> pd.DataFrame:
    _bootstrap_csv()
    return pd.read_csv(DATA_PATH)


def create_row(row: dict) -> pd.DataFrame:
    df = read_all()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def update_row(idx: int, row: dict) -> pd.DataFrame:
    df = read_all()
    for col, val in row.items():
        df.at[idx, col] = val
    df.to_csv(DATA_PATH, index=False)
    return df


def delete_row(idx: int) -> pd.DataFrame:
    df = read_all()
    df = df.drop(index=idx).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    return df
