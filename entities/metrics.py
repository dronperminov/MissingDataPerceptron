import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


class Metrics:
    def __init__(self) -> None:
        self.splits = {}

    def add(self, split: str, metric: str) -> None:
        if split not in self.splits:
            self.splits[split] = {}

        self.splits[split][metric] = defaultdict(list)

    def set_value(self, step: int, split: str, metric: str, value: float) -> None:
        if split in self.splits and metric in self.splits[split]:
            self.splits[split][metric][step].append(value)

    def set_values(self, step: int, split: str, metrics: Dict[str, float]) -> None:
        for metric, value in metrics.items():
            self.set_value(step=step, split=split, metric=metric, value=value)

    def get_metric(self, step: int, split: str, metric: str) -> str:
        values = self.splits[split][metric][step]
        return f"{np.mean(values):.4f}±{np.std(values):.4f}"

    def get_header(self, split_name: str, metric_names: List[str]) -> str:
        cells = " | ".join([f"{name:^20}" for name in metric_names])
        return f"| {split_name:>20} | {cells} |"

    def get_metrics(self, step: int, split: str, metric_names: List[str]) -> str:
        cells = " | ".join([f"{self.get_metric(split=split, metric=metric_name, step=step):^20}" for metric_name in metric_names])
        return f"| {split:>20} | {cells} |"

    def plot(self, path: str, split_names: List[str], metric_names: List[str], period: int, color: str = "#7699d4") -> None:
        rows, columns = len(split_names) + 1, len(metric_names)
        figure = plt.figure(figsize=(columns * 16, rows * 8))
        gs = GridSpec(rows, columns)
        axes = []

        for i in range(rows):
            axes.append([])

            for j, metric_name in enumerate(metric_names):
                ax = figure.add_subplot(gs[i, j])
                ax.set_title(metric_name)
                axes[i].append(ax)

        for i, split_name in enumerate(split_names):
            for j, metric_name in enumerate(metric_names):
                step2values = self.splits[split_name][metric_name]
                steps = sorted([step for step in step2values if step % period == 0])

                if not steps:
                    continue

                values = [step2values[step] for step in steps]

                ax = axes[i][j]
                bp = ax.boxplot(values, positions=range(len(steps)), widths=0.6, label=split_name)

                for median in bp["medians"]:
                    median.set_color(color)

                ax.plot(range(len(steps)), [np.mean(step_values) for step_values in values], color=color)
                ax.set_xticks(ticks=range(len(steps)), labels=steps)
                ax.set_xlabel("Epochs")
                ax.set_ylabel(metric_name)
                ax.set_ylim(0, 1)

                for x, step_values in enumerate(values):
                    mean = np.mean(step_values)
                    std = np.std(step_values)
                    ax.text(x, mean, f'{mean:.3f}±{std:.2f}', ha='center', va='bottom', fontsize=7, color=color)

                ax.legend(loc="best")

        for i, metric_name in enumerate(metric_names):
            steps = sorted([step for step in self.splits[split_names[0]][metric_name] if step % period == 0])

            if not steps:
                continue

            ax = axes[len(split_names)][i]
            w, x = 0.8 / len(split_names), np.arange(len(steps))

            for j, split_name in enumerate(split_names):
                step2values = self.splits[split_name][metric_name]
                values = [np.mean(step2values[step]) for step in steps]
                ax.bar(x - 0.4 + j * w, values, w, label=split_name)

            ax.set_xticks(x, steps)
            ax.legend(loc="best")

        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.splits, f)
