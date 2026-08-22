# Structural detector

The detector was fit on the SHS development split and frozen before ASAP
evaluation. It uses 47 MIDI-structure features, a score threshold of 0.9910136,
three-signal mutual support, and a maximum component size of 20.

| ASAP result | Precision | Recall |
|---|---:|---:|
| Direct edges | 99.73% | 44.07% |
| Guarded components | 99.61% | 72.45% |

`shs_structural_detector.npz` is a portable tree representation of the fitted
gradient-boosting model. `scripts/reproduce_detector_statistics.py --verify`
recomputes the point estimates and 10,000-sample bootstrap intervals.
