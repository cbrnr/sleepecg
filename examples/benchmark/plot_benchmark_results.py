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

    # order by runtime for longest signal, slowest algorithm first
    maxlen = results["signal_len"].max()
    order = (
        results.query(f"signal_len == {maxlen}")
        .groupby("detector")["mean_runtime"]
        .mean()
        .apply(lambda x: 1 / x)  # reverse order
        .to_dict()
    )
    results = results.sort_values(by=["detector", "signal_len"], key=lambda x: x.map(order))

    # each detector should have the same color in each benchmark
    colors = [px.colors.qualitative.Plotly[i] for i in [0, 1, 7, 5, 8, 4, 2, 3, 9, 6]]

    fig = (
        px.line(
            results,
            x="signal_len",
            y="mean_runtime",
            markers=True,
            color="detector",
            color_discrete_sequence=colors,
            log_y=True,
            labels={
                "signal_len": "Signal length (hours)",
                "mean_runtime": "Mean runtime (s)",
            },
            title=f"Mean detector runtime for {db_slug.upper()}",
            width=1000,
            template="plotly_white",
        )
        .update_yaxes(rangemode="tozero")
        .update_layout(legend_title="")
    )
    fig.write_image(plot_filepath)

elif benchmark == "metrics":
    results["precision"] = results["TP"] / (results["TP"] + results["FP"])
    results["recall"] = results["TP"] / (results["TP"] + results["FN"])
    results["f1"] = 2 / (results["recall"] ** -1 + results["precision"] ** -1)
    fig = (
        px.box(
            results.melt(id_vars=["detector"], value_vars=["precision", "recall", "f1"]),
            color="detector",
            y="value",
            labels={"value": ""},
            facet_col="variable",
            title=f"Metrics for {db_slug.upper()}",
            width=1000,
            template="plotly_white",
        )
        .update_xaxes(range=[-0.4, 0.4])
        .update_yaxes(range=[-0.01, 1.01], tick0=0, dtick=0.1)
        .for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        .update_layout(legend_title="")
    )
    fig.write_image(plot_filepath)

elif benchmark == "rri_similarity":
    fig = (
        px.box(
            results,
            y="pearsonr",
            color="detector",
            labels={"pearsonr": "Correlation coefficient"},
            title=f"Correlation coefficient for RRI timeseries for {db_slug.upper()}",
            width=1000,
            template="plotly_white",
        )
        .update_yaxes(range=[-1.01, 1.01], tick0=-1, dtick=0.25)
        .update_layout(legend_title="")
    )
    fig.write_image(plot_filepath)

else:
    raise ValueError(f"No plotting strategy defined for {results_filepath}.")
