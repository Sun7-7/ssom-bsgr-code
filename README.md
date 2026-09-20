# SSOM-BSGR — reference implementation

Code accompanying *Neurodynamic Spiking Self-Organizing Maps with Bidirectional
Structural Granularity Refinement for Radar Pulse Deinterleaving*.

It contains the proposed method and the ablation entry points. It runs on a PDW
record with the column layout described in [`docs/data_format.md`](docs/data_format.md).

## What is in this repository

```
src/
  spiking_som.py   spiking encoder + adaptive self-organizing map (frozen projection)
  data_utils.py    descriptor preprocessing and feature helpers
  refinement.py    bidirectional structural granularity refinement (merge / split / gate)
  pipeline.py      the full pipeline for one window: features -> GMM grid -> SSOM -> BSGR
  metrics.py       clustering metrics (ACC / ARI / NMI / F1) with Hungarian matching
  energy.py        arithmetic energy proxy used by the cost accounting
scripts/
  run_main.py      run the method on one or more windows of a record
  run_ablation.py  component ablation: SSOM, GMM, support, merge, split, gate
  smoke_test.py    end-to-end check on synthetic data; needs no external dataset
```

## What is deliberately not in this repository

* **The eight comparison baselines.** Their implementations and per-window
  results are withheld while the manuscript is under review.
* **Hyperparameter search code and ranges.**
* **The radar records.** They are not redistributed here. The numbers reported in
  the paper were produced on a withheld corpus.
* **Per-run result artifacts** (predictions and summary tables).

The method code is complete: everything needed to run SSOM-BSGR on a PDW record
is in `src/`, and the frozen thresholds of the refinement stage are the ones the
paper reports.

## Install and run

```bash
pip install -r requirements.txt

python scripts/run_main.py --pdw path/to/PDW.csv --window 10000 --start 10000 \
    --count 6 --seed 42 --out predictions/

python scripts/run_ablation.py --pdw path/to/PDW.csv --window 10000 --windows 6 \
    --out ablation/

python scripts/smoke_test.py     # end-to-end check, no dataset required
```

`run_main.py` prints ACC, ARI, NMI, F1 and the absolute emitter-count error per
window and writes the predicted labels as `.npz`. CUDA is used when available and
the code falls back to CPU otherwise.

## Method summary

1. **Features.** RF, PW, PA and the sine/cosine pair of DOA are standardized. TOA
   is excluded: absolute arrival time does not identify an emitter within a
   window.
2. **Initial hypotheses.** A grid of tied-covariance Gaussian mixtures over
   K = 4..20; K is the first point where the BIC gain per sample drops below 8% of
   the early median gain. Choosing K this way keeps labels and downstream metrics
   out of the model-order decision.
3. **Representation.** The descriptors in the order (RF, PW, PA, DOA) drive a
   frozen spiking encoder and an adaptive 30 x 30 self-organizing map. Each pulse
   is assigned its best-matching unit.
4. **Refinement.** Two corrections are proposed for the current partition — merge
   component pairs that are close in descriptor space *and* overlap on the map, or
   split a component better described by two Gaussians — and a gate selects which
   to apply. The map overlap is what distinguishes this from distance-only
   refinement: fragments of one emitter tend to occupy the same map region even
   when their descriptor means have drifted apart.

## Caveat on the K-selection rule

The 8% gain ratio is frozen from the source records. On a record whose BIC gain
profile differs substantially it can over-segment; the smoke test shows this on a
synthetic record with widely separated emitters. Re-calibrating the ratio on a new
corpus is reasonable, but any change means the pipeline is no longer the one the
paper evaluates.

## License

MIT, see [`LICENSE`](LICENSE).
