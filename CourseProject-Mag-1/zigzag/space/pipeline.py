import os
import re
import typing

import numpy
import joblib
import cvtda.logging
import cvtda.dumping
import matplotlib.axes
import sklearn.decomposition
from matplotlib.ticker import MaxNLocator

import zigzag.utils
from zigzag.space.filter import denoise
from zigzag.space.filter import density_filter
from zigzag.space.betti import get_space_betti


def _validate_plan(plan: typing.List[typing.Tuple[str, str, int]]):
    assert len(set([os.path.dirname(item[0]) for item in plan])) == 1, "Multiple in_file in plan"
    assert len(set([item[2] for item in plan])) == 1, "Multiple n_components in plan"


def process(in_file: str, out_file: str, n_components: int):
    if os.path.exists(out_file):
        with open(out_file, "r") as file:
            pca, z2, z3, _ = file.read().split("\n")
        return (
            re.search(r"\d+(?:\.\d+)?%", pca).group(),
            list(map(int, z2[5:-1].split(","))),
            list(map(int, z3[5:-1].split(","))),
        )

    with zigzag.utils.UniversalDumper("./"):
        data = cvtda.dumping.dumper().get_dump(in_file)

    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(data)
    var = pca.explained_variance_ratio_.sum()

    if len(X) > 15000:
        X = density_filter(X, 300, 0.5, n_jobs=1)
    X = denoise(X, k=20, iterations=3, n_jobs=1)
    results = get_space_betti(X)

    report = f"PCA to {n_components}D ({var:.1%} variance). \n"
    for coeff, betti in results.items():
        report += f"Z{coeff}: {betti}\n"

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as file:
        file.write(report)
    return process(in_file, out_file, n_components)


def process_bulk(params: typing.List[typing.Tuple[str, str, int]], n_jobs: int = -1):
    return joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(process)(*params) for params in cvtda.logging.logger().pbar(params)
    )


def plot(plan: typing.List[typing.Tuple[str, str, int]], ax: matplotlib.axes.Axes):
    _validate_plan(plan)

    bars = []
    for _, z2_bettis, z3_bettis in process_bulk(plan):
        while len(bars) < len(z2_bettis):
            bars.append([])
        for betti, bar in zip(z2_bettis, bars):
            bar.append(betti)

    width, offset = 1 / (len(bars) + 1), len(bars) // 2 - 0.5
    for dim, bar in enumerate(bars):
        x = numpy.arange(len(bar)) + width * (dim - offset)
        match dim:
            case 0:
                label = "Связные компоненты (разм. 0)"
            case 1:
                label = "Промежутки (разм. 1)"
            case __:
                label = f"Пустоты (разм. {dim})"
        ax.bar(x, bar, label=label, width=width)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Номер слоя")
    ax.set_ylabel("Число устойчивых признаков")


def make_report(plan: typing.List[typing.Tuple[str, str, int]]) -> dict:
    _validate_plan(plan)

    item = {}
    for i, (variance, z2_bettis, z3_bettis) in enumerate(process_bulk(plan)):
        z2_bettis = f"({', '.join(map(str, z2_bettis))})"
        z3_bettis = f"({', '.join(map(str, z3_bettis))})"
        item[f"layer_{i + 1}"] = variance + "\n" + z2_bettis + "\n" + z3_bettis
    return item
