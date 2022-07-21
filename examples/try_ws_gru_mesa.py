# %%
from sleepecg import load_classifier, plot_hypnogram, read_slpdb, stage

# The model was built using tensorflow 2.7, running on higher versions might create warnings
# but should not influence the results.
clf = load_classifier("ws-gru-mesa", "SleepECG")

# %% Load record
# `ws-gru-mesa` performs poorly for most SLPDB records. It does however work well for slp03.
rec = next(read_slpdb("slp03"))

# %% Predict stages and plot hypnogram
stages_pred = stage(clf, rec, return_mode="prob")

plot_hypnogram(
    rec,
    stages_pred,
    stages_mode=clf.stages_mode,
    merge_annotations=True,
)
