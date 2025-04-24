from dataclasses import dataclass
from typing import List, Tuple

from models.perceptron import Perceptron


@dataclass
class Cell:
    points: List[Tuple[float, float]]
    signs: List[int]


@dataclass
class Line:
    a: float
    b: float
    c: float
    sign: int

    def __call__(self, x: float, y: float) -> float:
        return self.a * x + self.b * y + self.c

    def add(self, line: "Line", weight: float) -> None:
        self.a += weight * line.a
        self.b += weight * line.b
        self.c += weight * line.c


class CellsExtractor:
    def __init__(self, perceptron: Perceptron):
        self.perceptron = perceptron

    def extract(self, x_min: float, x_max: float, y_min: float, y_max: float, x_axis: int, y_axis: int) -> List[dict]:
        polygon = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
        lines = self.__get_first_layer_lines(x_axis=x_axis, y_axis=y_axis)
        tree = self.__split_polygons_by_lines(polygon=polygon, lines=lines, key=[])

        layer2lines = [[] for _ in self.perceptron.layers]
        layer2polygons = [{} for _ in self.perceptron.layers]
        layer2lines[0] = lines

        self.__extract_polygons(index=1, layer2lines=layer2lines, tree=tree, x_axis=x_axis, y_axis=y_axis, layer2polygons=layer2polygons)
        return layer2polygons

    def __extract_polygons(self, index: int, layer2lines: List[List[Line]], tree: List[dict], x_axis: int, y_axis: int, layer2polygons: List[dict]) -> None:
        for node in tree:
            layer2polygons[index - 1]["".join(node["key"])] = node["polygon"]

        if index == len(self.perceptron.layers):
            return

        for node in tree:
            for i, sign in enumerate(node["signs"]):
                layer2lines[index - 1][i].sign = sign

            layer2lines[index] = self.__get_layer_lines(index=index, prev_lines=layer2lines[index - 1])
            layer_tree = self.__split_polygons_by_lines(polygon=node["polygon"], lines=layer2lines[index], key=node["key"])
            self.__extract_polygons(index=index + 1, layer2lines=layer2lines, tree=layer_tree, x_axis=x_axis, y_axis=y_axis, layer2polygons=layer2polygons)

    def __get_first_layer_lines(self, x_axis: int, y_axis: int) -> List[Line]:
        layer = self.perceptron.layers[0]
        lines = []

        for i in range(layer.out_features):
            lines.append(Line(a=layer.weight[i, x_axis].item(), b=layer.weight[i, y_axis].item(), c=layer.bias[i].item(), sign=0))

        return lines

    def __get_layer_lines(self, index: int, prev_lines: List[Line]) -> List[Line]:
        layer = self.perceptron.layers[index]
        lines = []

        for i in range(layer.out_features):
            line = Line(a=0, b=0, c=layer.bias[i].item(), sign=0)

            for j in range(layer.in_features):
                activation = 1

                if self.perceptron.activation == "abs":
                    activation = 1 if prev_lines[j].sign >= 0 else -1
                elif self.perceptron.activation == "relu":
                    activation = 1 if prev_lines[j].sign >= 0 else 0
                elif self.perceptron.activation == "leaky-relu":
                    activation = 1 if prev_lines[j].sign >= 0 else 0.01

                line.add(prev_lines[j], weight=layer.weight[i, j].item() * activation)

            lines.append(line)

        return lines

    def __split_polygons_by_lines(self, polygon: List[Tuple[float, float]], lines: List[Line], key: List[int]) -> List[dict]:
        tree = [{"signs": [], "key": key, "polygon": polygon}]

        for line in lines:
            new_tree = []

            for node in tree:
                lower, upper = self.__split_polygon(polygon=node["polygon"], line=line)

                if len(upper) > 2:
                    new_tree.append({"signs": node["signs"] + [1], "key": node["key"] + ["+"], "polygon": upper})

                if len(lower) > 2:
                    new_tree.append({"signs": node["signs"] + [-1], "key": node["key"] + ["-"], "polygon": lower})

            tree = new_tree

        return tree

    def __split_polygon(self, polygon: List[Tuple[float, float]], line: Line) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        polygon1, polygon2 = [], []

        for i, (x1, y1) in enumerate(polygon):
            x2, y2 = polygon[(i + 1) % len(polygon)]
            sign1, sign2 = line(x1, y1), line(x2, y2)

            if sign1 <= 0:
                polygon1.append(polygon[i])

            if sign1 >= 0:
                polygon2.append(polygon[i])

            if sign1 * sign2 >= 0:
                continue

            t = sign1 / (sign1 - sign2)
            p = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            polygon1.append(p)
            polygon2.append(p)

        return polygon1, polygon2
