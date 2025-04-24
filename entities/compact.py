import numpy as np
import pandas as pd


class Compact:
    def __init__(self, v_min: np.ndarray, v_max: np.ndarray, offset: float) -> None:
        delta = (v_max - v_min) * offset
        self.v_min = v_min - delta / 2
        self.v_max = v_max + delta / 2

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for i, column in enumerate(df.columns[:-1]):
            df[column] = (df[column] - self.v_min[i]) * 0.5 / (self.v_max[i] - self.v_min[i]) - 0.25

        return df

    def normalize_np(self, data: np.ndarray) -> np.ndarray:
        return (data - self.v_min) * 0.5 / (self.v_max - self.v_min) - 0.25

    def denormalize_np(self, data: np.ndarray) -> np.ndarray:
        return self.v_min + (data + 0.25) * (self.v_max - self.v_min) / 0.5

    def random(self, axis: int) -> float:
        return np.random.uniform(self.v_min[axis], self.v_max[axis])

    def to_json(self) -> dict:
        return {"v_min": self.v_min.tolist(), "v_max": self.v_max.tolist()}

    @classmethod
    def from_df(cls, df: pd.DataFrame, offset: float = 0.05) -> "Compact":
        features = df.iloc[:, :-1]
        v_min = features.min().to_numpy()
        v_max = features.max().to_numpy()
        return Compact(v_min=v_min, v_max=v_max, offset=offset)

    @classmethod
    def from_json(cls, data: dict) -> "Compact":
        v_min = np.array(data["v_min"], dtype=np.float64)
        v_max = np.array(data["v_max"], dtype=np.float64)
        return Compact(v_min=v_min, v_max=v_max, offset=0)
