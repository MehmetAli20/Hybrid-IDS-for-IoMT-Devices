# IAT Shortcut Analysis

In CIC IoMT 2024 the `IAT` column is named after inter-arrival time, but its values behave like a recording-time stamp: every attack capture sits in its own narrow `IAT` band. A classifier can reach about 99% accuracy on the official test set by reading this one column, without learning anything about the traffic itself.

For that reason the main model drops `IAT` by default (`--drop_features IAT` in `src/main.py` and `src/baseline.py`). The results that use `IAT` are kept here as a shortcut analysis, not as model performance.

All results are on the official test split (1,614,182 rows). Training rows are deduplicated, as in the main pipeline.

## 1. Each attack sits in its own IAT band

Median `IAT` per flood class, sorted by train median:

| Class | Train median | Test median |
|---|---:|---:|
| MQTT-DoS-Connect_Flood | 84,662,360 | 84,662,384 |
| MQTT-DoS-Publish_Flood | 84,666,488 | 84,666,504 |
| MQTT-DDoS-Publish_Flood | 84,666,968 | 84,666,984 |
| MQTT-DDoS-Connect_Flood | 84,667,032 | 84,667,048 |
| DoS-TCP | 84,670,848 | 84,670,992 |
| DoS-SYN | 84,674,624 | 84,674,744 |
| DoS-UDP | 84,675,328 | 84,679,168 |
| DoS-ICMP | 84,679,320 | 84,679,608 |
| DDoS-TCP | 84,696,008 | 84,696,136 |
| DDoS-SYN | 84,696,344 | 84,696,448 |
| DDoS-UDP | 84,696,672 | 84,696,896 |
| DDoS-ICMP | 84,697,016 | 84,697,192 |

- Inside each flood class the 5–95% range is between 16 and about 3,800 units wide, next to values around 84.7 million, and the ranges of different classes do not overlap.
- In every flood class the test 5th percentile is at or above the train 95th percentile: test rows come after training rows, as if the value were read off a clock.
- The four TCP/IP DoS attacks and the four DDoS attacks appear in the same order (TCP → SYN → UDP → ICMP), with DDoS about 20,000 units later. This matches a recording schedule, not how the attacks behave.
- Benign, Recon, Spoofing and MQTT-Malformed do not form a single band; their values are mostly 0 or near 84.7 or 169 million.

`src/iat_analysis.py` writes the full table for all 19 classes, with 5th and 95th percentiles, to `iat_by_class.csv` and plots the flood classes in `iat_by_class.png`.

## 2. One column is almost enough

Random Forest (100 trees, at most 50,000 training rows per class), accuracy / macro-F1:

| Classes | All 45 features | `IAT` only | All except `IAT` |
|---|---|---|---|
| 2 | 0.9987 / 0.9859 | 0.9906 / 0.9102 | 0.9971 / 0.9684 |
| 6 | 0.9977 / 0.9341 | 0.9939 / 0.8701 | 0.6830 / 0.7627 |
| 19 | 0.9938 / 0.8952 | 0.8324 / 0.7018 | 0.6742 / 0.6602 |

- In the 6- and 19-class forests, `IAT` has the largest feature importance (23% and 27%; the next feature has 7% and 6%). In the 19-class task, `IAT` alone beats the other 44 features together.
- Without `IAT`, almost all 6-class errors are DoS ↔ DDoS: 34% of DDoS windows are predicted as DoS and 33% of DoS windows as DDoS. These two families differ mainly in the number of attacking hosts, which the per-window features do not describe. With `IAT`, the model separates them by recording time instead.
- Attack vs. benign mostly holds up without `IAT`: macro-F1 goes from 0.986 to 0.968, and `IAT` ranks only fourth in the 2-class forest (9% of the importance). 97.7% of the test rows are attacks, so macro-F1 is the more telling number here.

## 3. The deep model uses IAT only when the scaling exposes it

Autoencoder + BiLSTM + Attention, 6 classes, accuracy / macro-F1. These four runs predate the subtype-balanced sampling described in the README, so their numbers differ slightly from the README's main table; the comparison between the columns is what matters here.

| Scaling | With `IAT` | Without `IAT` |
|---|---|---|
| signed log1p + standardization (default) | 0.6763 / 0.7573 | 0.6586 / 0.7483 |
| quantile → normal | 0.9955 / 0.8980 | 0.6120 / 0.7171 |

Without `IAT`, quantile scaling does slightly worse than the default (0.6120 vs 0.6586 accuracy), so its whole gain came from `IAT`. The reason is how the two scalers treat the bands: signed log1p maps 84,670,848 (DoS-TCP) and 84,696,008 (DDoS-TCP) to 18.2543 and 18.2546, too close for the network to use, while quantile scaling replaces each value with its rank, so the bands spread apart and the network picks them up. Trees split on thresholds and ignore scaling, so Random Forest finds the shortcut either way.

## Reproduce

```bash
# Random Forest: all features / IAT only / no IAT, plus the IAT table
python src/iat_analysis.py --class_config 6

# Deep model with IAT (the shortcut) and without it (the default)
python src/main.py --class_config 6 --scaling quantile --drop_features
python src/main.py --class_config 6 --scaling quantile
```

`--drop_features` with no value keeps every feature, including `IAT`. Each deep-model run takes about 20–25 minutes on an RTX 3070 Laptop GPU.
