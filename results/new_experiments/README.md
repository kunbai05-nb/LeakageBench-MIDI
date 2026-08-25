# Detector and mitigation experiments

`data/` contains source values, `figures/` contains vector and PDF figures, and
`tables/` contains LaTeX tables. These files cover detector validation,
deployment on seven MIDI collections, imperfect-inference robustness, graph
completion, and sensitivity analyses.

Rebuild the cross-dataset table and optional overview figure with:

```bash
pip install -e '.[figures]'
python scripts/plot_cross_dataset_deployment.py \
  results/new_experiments/data/cross_dataset_detector_deployment.csv \
  results/new_experiments
```
