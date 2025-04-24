import json
import os
import tempfile
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import patches, pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from entities.cells_extractor import CellsExtractor
from entities.compact import Compact
from entities.train_params import TrainParams
from models.perceptron import Perceptron


class PerceptronUnaryClassifier:
    def __init__(self, dimension: int, classes: int, sizes: List[int], beta: float , device: torch.device) -> None:
        self.dimension = dimension
        self.classes = classes
        self.beta = beta
        self.device = device

        self.models = [Perceptron(dimension=dimension, sizes=[*sizes, 1], output_activation="sigmoid").to(device) for _ in range(self.classes)]
        self.compacts: List[Optional[Compact]] = [None for _ in range(self.classes)]
        self.optimizers: List[Optional[torch.optim]] = [None for _ in range(self.classes)]

    def init_compacts(self, train_df: pd.DataFrame) -> None:
        for label in train_df["label"].unique():
            df = train_df[train_df["label"] == label]
            self.compacts[label] = Compact.from_df(df=df)
            self.optimizers[label] = None

    def fit(self, train_df: pd.DataFrame, train_params: TrainParams):
        label2df = {}

        for label, compact in enumerate(self.compacts):
            df = train_df[train_df["label"] == label]
            label2df[label] = compact.normalize_df(df)

        for label, df in label2df.items():
            if not self.optimizers[label]:
                self.optimizers[label] = Adam(self.models[label].parameters(), lr=train_params.learning_rate)

            self.__train_model(model=self.models[label], df=df, train_params=train_params, optimizer=self.optimizers[label])

    def predict(self, df: pd.DataFrame) -> np.array:
        predictions = []

        with torch.no_grad():
            for label, (model, compact) in enumerate(zip(self.models, self.compacts)):
                data = df.values
                outside = np.all((data < compact.v_min) | (data > compact.v_max), axis=1)

                x = torch.tensor(compact.normalize_np(data), dtype=torch.float32, device=self.device)
                output = model(x).squeeze(dim=-1)
                output[torch.tensor(outside).to(self.device)] = 0
                predictions.append(output)

        predictions = torch.stack(predictions)
        classes = torch.argmax(predictions, dim=0)
        return np.array([label if predictions[label, i] >= self.beta else -1 for i, label in enumerate(classes.cpu().numpy())])

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        x_true, y_true = df.iloc[:, :-1], df.iloc[:, -1]
        y_pred = self.predict(x_true)

        mask = y_pred >= 0
        y_pred[~mask] = 0

        return {
            "refuse": np.mean(~mask),

            "accuracy_full": accuracy_score(y_true, y_pred),
            "f1_micro_full": f1_score(y_true, y_pred, average="micro"),
            "f1_macro_full": f1_score(y_true, y_pred, average="macro"),

            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "precision_micro": precision_score(y_true[mask], y_pred[mask], average="micro"),
            "precision_macro": precision_score(y_true[mask], y_pred[mask], average="macro"),
            "recall_micro": recall_score(y_true[mask], y_pred[mask], average="micro"),
            "recall_macro": recall_score(y_true[mask], y_pred[mask], average="macro"),
            "f1_micro": f1_score(y_true[mask], y_pred[mask], average="micro"),
            "f1_macro": f1_score(y_true[mask], y_pred[mask], average="macro")
        }

    def plot(self, train_df: pd.DataFrame, test_df: pd.DataFrame, path: str, v_min: float = -1.5, v_max: float = 1.5, n: int = 500) -> None:
        if self.dimension != 2:
            return

        rows, columns = 1, self.classes + 2
        figure = plt.figure(figsize=(columns * 8, rows * 8))
        gs = GridSpec(rows, columns)

        x = np.linspace(-0.5, 0.5, n)
        y = np.linspace(-0.5, 0.5, n)
        data = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

        label2colormap = {
            0: ListedColormap(['#ffffff', '#f0f4fa', '#e1e8f5', '#d1ddf1', '#c2d2ec', '#b3c6e7', '#a4bbe2', '#94b0de', '#85a4d9', '#7699d4']),
            1: ListedColormap(['#ffffff', '#fbefef', '#f7e0e0', '#f4d0d0', '#f0c1c1', '#ecb1b1', '#e8a2a2', '#e59292', '#e18383', '#dd7373']),
            2: ListedColormap(['#ffffff', '#f2fbef', '#e5f7e0', '#d8f4d0', '#cbf0c1', '#bdecb1', '#b0e8a2', '#a3e592', '#96e183', '#89dd73']),
            3: ListedColormap(['#ffffff', '#fbeffb', '#f7e0f7', '#f4d0f3', '#f0c1ef', '#ecb1eb', '#e8a2e7', '#e592e3', '#e183df', '#dd73db'])
        }

        label2color = {
            0: "#7699d4",
            1: "#dd7373",
            2: "#89dd73",
            3: "#dd73db"
        }

        for label, (model, compact) in enumerate(zip(self.models, self.compacts)):
            z = model.predict(torch.FloatTensor(data).to(self.device)).squeeze(dim=-1).cpu().numpy()
            z[z < self.beta] = 0
            label_df = compact.normalize_df(train_df[train_df["label"] == label])

            ax = figure.add_subplot(gs[0, label])
            ax.pcolor(x, y, z.reshape((n, n)).T, vmin=0, vmax=1, cmap=label2colormap[label])
            ax.scatter(label_df["x"], label_df["y"], color=label2color[label], s=20, edgecolors="#ffffff", linewidths=0.5)

            cells_extractor = CellsExtractor(perceptron=model)
            layer2cells = cells_extractor.extract(x_min=-0.5, y_min=-0.5, x_max=0.5, y_max=0.5, x_axis=0, y_axis=1)

            for key, cell in layer2cells[-1].items():
                ax.add_patch(patches.Polygon(xy=np.array(cell), closed=True, edgecolor="#555555", alpha=0.8, fill=None, linewidth=0.2))

            ax.set_title(f'Model for class "{label}": [{z.min():.4f}, {z.max():.4f}], {len(label_df)} points')

        x = np.linspace(v_min, v_max, n)
        y = np.linspace(v_min, v_max, n)
        data = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        z = np.zeros((n * n, self.classes))

        for label, (model, compact) in enumerate(zip(self.models, self.compacts)):
            inside = np.all((data >= compact.v_min) & (data <= compact.v_max), axis=1)
            inside_data = compact.normalize_np(data[inside])

            output = model.predict(torch.FloatTensor(inside_data).to(self.device)).squeeze(dim=-1)
            output[output < self.beta] = 0
            z[inside, label] = output.cpu().numpy()

        zz = np.zeros((n * n))

        for label in range(self.classes):
            zz[z[:, label] > self.beta] = label + 1

        for label1 in range(self.classes):
            for label2 in range(label1 + 1, self.classes):
                lines = np.array([label1 + 1] * 4 + [label2 + 1] * 4)[np.add.outer(np.arange(n), np.arange(n)) % 8]
                mask = (z[:, label1] > self.beta) & (z[:, label2] > self.beta)
                zz[mask] = lines.reshape(n * n)[mask]

        for label1 in range(self.classes):
            for label2 in range(label1 + 1, self.classes):
                for label3 in range(label2 + 1, self.classes):
                    lines = np.array([label1 + 1] * 4 + [label2 + 1] * 4 + [label3 + 1] * 4)[np.add.outer(np.arange(n), np.arange(n)) % 12]
                    mask = (z[:, label1] > self.beta) & (z[:, label2] > self.beta) & (z[:, label3] > self.beta)
                    zz[mask] = np.rot90(lines).reshape(n * n)[mask]

        for i in range(2):
            ax = figure.add_subplot(gs[0, self.classes + i])
            ax.set_title("Classifier")
            ax.pcolor(x, y, zz.reshape((n, n)).T, vmin=-1, vmax=self.classes, cmap=ListedColormap(["#000000", "#ffffff"] + [label2color[label] for label in range(self.classes)]))

            if i == 1:
                ax.scatter(test_df["x"], test_df["y"], color=test_df["label"].map(label2color), s=20, edgecolors="#ffffff", linewidths=0.5)

            for label, compact in enumerate(self.compacts):
                x_min, y_min = compact.v_min
                x_max, y_max = compact.v_max
                ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color=label2color[label]))

        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)

    def save(self, path: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "dimension": self.dimension,
                "classes": self.classes,
                "beta": self.beta,
                "models": self.__save_models(tmp_dir=tmp_dir)
            }

            with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f)

            with zipfile.ZipFile(path, "w") as f:
                for file in os.listdir(tmp_dir):
                    f.write(os.path.join(tmp_dir, file), os.path.relpath(os.path.join(tmp_dir, file), tmp_dir))

    def load(self, path: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(path, "r") as f:
                f.extractall(tmp_dir)

            with open(os.path.join(tmp_dir, "config.json")) as f:
                config = json.load(f)

            self.dimension = config["dimension"]
            self.classes = config["classes"]
            self.beta = config["beta"]
            self.__load_models(models_config=config["models"], tmp_dir=tmp_dir)

    def fill_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        dfs = []

        for label, (model, compact) in enumerate(zip(self.models, self.compacts)):
            label_df = df[df["label"] == label]
            filled_df = self.__fill_nans(model=model, compact=compact, df=label_df)
            dfs.append(filled_df)

        return pd.concat(dfs)

    def __save_models(self, tmp_dir: str) -> List[dict]:
        models = []

        for label, (model, compact) in enumerate(zip(self.models, self.compacts)):
            model.save(os.path.join(tmp_dir, f"model{label}.pth"))
            models.append({
                "label": label,
                "sizes": model.sizes,
                "activation": model.activation,
                "output_activation": model.output_activation,
                "compact": compact.to_json(),
                "path": f"model{label}.pth"
            })

        return models

    def __load_models(self, models_config: List[dict], tmp_dir: str) -> None:
        self.models = []
        self.optimizers = []
        self.compacts = []

        for label, model_config in enumerate(models_config):
            model = Perceptron(
                dimension=self.dimension,
                sizes=model_config["sizes"],
                activation=model_config["activation"],
                output_activation=model_config["output_activation"]
            ).to(self.device)
            model.load(os.path.join(tmp_dir, model_config["path"]))

            self.models.append(model)
            self.compacts.append(Compact.from_json(model_config["compact"]))
            self.optimizers.append(None)

    def __train_model(self, model: Perceptron, df: pd.DataFrame, train_params: TrainParams, optimizer: torch.optim):
        x_real = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32, device=self.device)
        y_real = torch.ones((x_real.shape[0], 1), dtype=torch.float32, device=self.device)
        dataloader = DataLoader(TensorDataset(x_real, y_real), batch_size=train_params.batch_size, shuffle=True)
        criterion = BCELoss()

        for epoch in range(train_params.epochs):
            total_loss = 0

            for x, y in dataloader:
                x_bg = torch.rand(x.shape, device=self.device) - 0.5
                y_bg = torch.zeros(y.shape, dtype=torch.float32, device=self.device)

                x_batch = torch.vstack([x, x_bg])
                y_batch = torch.vstack([y, y_bg])

                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # print(f"Epoch {epoch}: {total_loss / len(dataloader)}")

    def __fill_nans(self, model: Perceptron, compact: Compact, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        df_filled = df.copy()
        feature_columns = df.columns[:-1]
        c2i = {column_name: i for i, column_name in enumerate(feature_columns)}

        for _ in range(50):
            mask_nan_rows = df_filled[feature_columns].isna().any(axis=1)
            if not mask_nan_rows.any():
                break

            random_fill = df_filled.loc[mask_nan_rows, feature_columns].apply(
                lambda column: column.map(lambda value: compact.random(axis=c2i[column.name]) if pd.isna(value) else value)
            )

            x = torch.tensor(compact.normalize_np(random_fill.values), dtype=torch.float32, device=self.device)
            probs = model.predict(x).squeeze(dim=-1).cpu().numpy()
            p = np.random.rand(len(probs))
            accept_mask = (probs >= threshold) & (p < probs)

            accepted_indices = random_fill.index[accept_mask]
            df_filled.loc[accepted_indices, feature_columns] = random_fill.loc[accepted_indices]

        df_filled = df_filled.dropna()
        return df_filled
