from dataclasses import dataclass


@dataclass
class TrainParams:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
