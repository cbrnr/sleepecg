# © SleepECG developers
#
# License: BSD (3-clause)

"""Plot results of runtime and detection quality benchmarks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

if len(sys.argv) != 2:
    print("Usage: python plot_benchmark_results.py <results>.csv")
    exit()

results_filepath = Path(sys.argv[1])
benchmark, db_slug, *_ = results_filepath.stem.split("__")
plot_filepath = results_filepath.with_suffix(".svg")
results = pd.read_csv(results_filepath).sort_values("detector")

if benchmark == "runtime":
    fs = results["fs"][0]
    results["signal_len"] = results["num_samples"] / (results["fs"] * 3600)
    results = results.groupby(["detector", "signal_len"], as_index=False).agg(
        mean_runtime=("runtime", "mean"),
        std_runtime=("runtime", "std"),
        n=("runtime", "count"),
    )
    results["error"] = results["std_runtime"] / np.sqrt(results["n"])

    fig = px.line(
        results,
        x="signal_len",
        y="mean_runtime",
        error_y="error",
        color="detector",
        log_y=True,
        labels={
            "signal_len": "signal length in hours",
            "mean_runtime": "mean runtime in s",
        },
        title=f"Mean detector runtime for {db_slug.upper()} (fs={fs}Hz)",
        width=800,
        height=600,
        render_mode="svg",
    )
    fig.update_yaxes(rangemode="tozero")
    fig.write_image(plot_filepath)

elif benchmark == "metrics":
    results["recall"] = results["TP"] / (results["TP"] + results["FN"])
    results["precision"] = results["TP"] / (results["TP"] + results["FP"])
    results["f1"] = 2 / (results["recall"] ** -1 + results["precision"] ** -1)
    fig = (
        px.box(
            results.melt(id_vars=["detector"], value_vars=["recall", "precision", "f1"]),
            color="detector",
            y="value",
            labels={"value": ""},
            facet_col="variable",
            width=1000,
            height=600,
            title=f"Metrics for {db_slug.upper()}",
        )
        .update_xaxes(range=[-0.4, 0.4])
        .update_yaxes(range=[0, 1.01])
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.write_image(plot_filepath, scale=1.5)

elif benchmark == "rri_similarity":
    fig = px.box(
        results,
        y="pearsonr",
        color="detector",
        title=f"Pearson correlation coefficient for RRI timeseries from {db_slug.upper()}",
    ).update_yaxes(range=[-1.01, 1.01])
    fig.write_image(plot_filepath, scale=1.5)

else:
    raise ValueError(f"No plotting strategy defined for {results_filepath}.")
