from typing import List

import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, dimension: int, sizes: List[int], activation: str = "abs", output_activation: str = "sigmoid") -> None:
        super(Perceptron, self).__init__()

        self.dimension = dimension
        self.sizes = sizes
        self.activation = activation
        self.output_activation = output_activation

        self.layers = nn.ModuleList()
        self.__init_layers(sizes=sizes)

    def __init_layers(self, sizes: List[int]) -> None:
        dimension = self.dimension

        for size in sizes:
            self.layers.append(nn.Linear(in_features=dimension, out_features=size, bias=True))
            dimension = size

    def __activate(self, x: torch.tensor, activation: str) -> torch.tensor:
        if activation == "sigmoid":
            x = torch.sigmoid(x)
        elif activation == "softmax":
            x = torch.softmax(x, dim=1)
        elif activation == "abs":
            x = torch.abs(x)
        elif activation == "relu":
            x = torch.relu(x)
        elif activation == "leaky-relu":
            x = torch.max(x, 0.01 * x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.__activate(x=layer(x), activation=self.activation)

        output = self.layers[-1](x)
        return self.__activate(x=output, activation=self.output_activation)

    def predict(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            return self.forward(x)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def get_shapes(self, x0: float, y0: float, width: float, height: float, font_size: float = 11, max_radius: float = 0.36,
                   max_gap: float = 0.5, min_gap: float = 0.1, padding: float = 0.1) -> List[dict]:
        coordinates = self.__get_best_coordinates(width=width, height=height, max_radius=max_radius, max_gap=max_gap, min_gap=min_gap, padding=padding)
        shapes = []

        for index, layer in enumerate(self.layers):
            for i in range(layer.out_features):
                for j in range(layer.in_features + 1):
                    value = layer.weight[i, j].item() if j < layer.in_features else layer.bias[i].item() * 5

                    shapes.append({
                        "shape": "line",
                        "x1": round(x0 + coordinates[index][i]["x"], 2),
                        "y1": round(y0 + coordinates[index][i]["y"], 2),
                        "x2": round(x0 + coordinates[index - 1][j]["x"], 2),
                        "y2": round(y0 + coordinates[index - 1][j]["y"], 2),
                        "stroke": "#dd7373" if value > 0.01 else "#7699d4" if value < -0.01 else "#000000",
                        "thickness": round(max(0.025, min(10, abs(value))), 2)
                    })

        for index in range(-1, len(self.layers)):
            bias = int(index < len(self.layers) - 1)
            size = self.dimension if index < 0 else self.layers[index].out_features

            for i in range(size + bias):
                x = round(x0 + coordinates[index][i]["x"] - coordinates[index][i]["radius"], 2)
                y = round(y0 + coordinates[index][i]["y"] - coordinates[index][i]["radius"], 2)
                d = 2 * coordinates[index][i]["radius"]
                text = "1" if i == size else self.__get_neuron_name(index, i)

                shapes.append({"shape": "ellipse", "x": x, "y": y, "d": d, "fill": "#ffffff", "stroke": "#000000", "thickness": 0.25})
                shapes.append({"shape": "textbox", "x": x, "y": y, "w": d, "h": d, "text": text, "size": font_size, "align": "center"})

        return shapes

    def __get_best_coordinates(self, width: float, height: float, max_radius: float, max_gap: float, min_gap: float, padding: float) -> dict:
        layers = len(self.layers) + 1
        neurons = max([self.dimension, *self.sizes]) + 1
        rotate = (width < height) == (neurons < layers)

        if rotate:
            width, height = height, width

        radiuses = []
        gaps_x = []
        sum_radius = 0

        for i in range(layers):
            size = (self.dimension if i == 0 else self.sizes[i - 1]) + (1 if i < layers - 1 else 0)
            radius = max_radius
            gap = min((width - size * 2 * radius) / max(size - 1, 1), max_gap)

            if gap < min_gap:
                gap = min_gap
                radius = (width - gap * (size - 1)) / (2 * size)

            radiuses.append(radius)
            gaps_x.append(gap)
            sum_radius += 2 * radius

        gap_y = (height - sum_radius) / (layers - 1)
        scale = 1

        if gap_y < min_gap:
            gap_y = min_gap
            scale = (height - gap_y * (layers - 1)) / sum_radius

        coordinates = {}
        y = padding

        for i in range(layers):
            need_one = i < layers - 1
            size = (self.dimension if i == 0 else self.sizes[i - 1]) + (1 if need_one else 0)
            radius = radiuses[i] * scale
            x0 = padding + radius + (width - size * 2 * radius - (size - 1) * gaps_x[i]) / 2

            coordinates[i - 1] = []

            for j in range(size):
                xi = x0 + j * (2 * radius + gaps_x[i])
                yi = y + radius

                coordinates[i - 1].append({"x": yi if rotate else xi, "y": xi if rotate else yi, "radius": radius})

            y += 2 * radius + gap_y

        return coordinates

    def __get_neuron_name(self, layer: int, neuron: int) -> str:
        layer_name = "xABCDEFGHIJKLMNOPQRSTUVWXYZ"[layer + 1]
        return f"{layer_name}{self.__number_to_index(neuron + 1)}"

    def __number_to_index(self, number: int) -> str:
        if number == 0:
            return "₀"

        digits_str = "₀₁₂₃₄₅₆₇₈₉"
        digits = []

        while number > 0:
            digits.append(digits_str[number % 10])
            number = number // 10

        return "".join(reversed(digits))
