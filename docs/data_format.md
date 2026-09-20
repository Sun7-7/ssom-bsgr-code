# Input format

## PDW record

A CSV with one row per intercepted pulse. The columns used by the pipeline are:

| Column | Meaning | Unit |
|---|---|---|
| `TOA`  | time of arrival | ns, non-decreasing |
| `RF`   | carrier frequency | MHz |
| `PW`   | pulse width | µs |
| `DOA`  | direction of arrival | degrees, in (−90, 90] |
| `PA`   | pulse amplitude | dB |
| `Label`| true emitter identity | integer, used only for scoring |

`TOA` is read but not used as a feature: within an observation window absolute
arrival time does not identify an emitter, and the paper reports the TOA
perturbation as a no-op control. The remaining four descriptors enter the model.

The label column name can be changed with `--label-column`. Labels are never
passed to a fit call; they are used only after the partition is produced, to
score it.

## Windows

The paper evaluates **10000-pulse windows**. `run_main.py` slices the record from
`--start` and can step through several consecutive windows with `--count` and
`--start-step`; `run_ablation.py` uses `--windows` consecutive windows of the same
length.

Every window is processed independently: feature standardization, the GMM grid,
the SSOM fit and the refinement all run from scratch, so no statistic from one
window enters another.

## Missing or spurious observations

The robustness study perturbs the descriptors before feature construction and
does not require a different input format: missing pulses are dropped from the
window, and spurious pulses are appended as extra rows with `Label = -1`, which
the scoring treats as an extra class and the reported metrics handle through the
rejection convention in `metrics.evaluate_clustering_with_rejection`.

## Output

`run_main.py --out DIR` writes

* `window{start}_seed{seed}.npz` with `y`, `pred`, `bmu` and `initial`;
* `summary.csv` and `summary.json` with the per-window metrics and their mean.

`run_ablation.py --out DIR` writes the per-window and averaged component-ablation
tables.
