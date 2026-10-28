# Hybrid IDS for IoMT Devices

A deep learning–based **Intrusion Detection System (IDS)** for **Internet of Medical Things (IoMT)** devices, built on the [CIC IoMT 2024 dataset](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html). The system detects and classifies network attacks against connected medical devices — a domain where a missed intrusion can directly endanger patient safety.

Developed as my Computer Engineering **senior thesis project** at Karadeniz Technical University.

## Highlights

- Evaluated on the dataset's **official test files** (1.6 million traffic windows), which are never used for training or model selection.
- Found a **shortcut feature**: `IAT` follows the recording time of each capture rather than the traffic itself. A Random Forest using `IAT` alone reaches 99.4% accuracy on 6 classes, so `IAT` is excluded by default. Details: [IAT_SHORTCUT.md](IAT_SHORTCUT.md).
- Without the shortcut: **99.8% accuracy** for attack vs. benign, and **99.5% (6 classes) / 99.2% (19 classes)** at attack-type level. Only DoS vs. DDoS stays unreliable (66–70%), because the per-window features don't describe how many hosts send the traffic; it is reported separately.
- Includes a Random Forest baseline on the same pipeline, a reproducible IAT analysis and a prediction script.

## Why IoMT Security?

Connected medical devices (patient monitors, infusion pumps, wearables) are increasingly targeted by network attacks, yet they are too resource-constrained to run heavy endpoint security. A network-level IDS that recognizes malicious traffic patterns is one of the few practical defenses. This project explores how well a hybrid deep learning pipeline can separate benign IoMT traffic from 18 different attack subtypes.

## Hybrid Architecture

The "hybrid" design combines two complementary deep learning components:

1. **Autoencoder feature extraction** — a deep autoencoder (512 → 256 → 128 → 32, mirrored decoder) is first trained to reconstruct the input features. Its encoder then becomes the first block of the classifier and is fine-tuned together with it (`--ae_mode finetune`; `frozen` and `none` are also available).
2. **Bidirectional LSTM + Attention classifier** — the 32-dimensional code is projected to 512 units and reshaped into a sequence of 8 steps × 64 features. A BiLSTM (128 units per direction, 64 for the binary task) processes the sequence, **4-head additive attention** weights its time steps (softmax over time), and a dense head (256 → 128) ends in a softmax output.

```
CSV traffic windows (CIC IoMT 2024, 45 features)
        │
        ▼
Drop IAT · remove duplicate training rows · NaN/Inf → 0
        │
        ▼
Stratified 85/15 train/validation split · max 50,000 rows per class, shared evenly across its attack subtypes
        │
        ▼
Signed log1p + standardization (fit on training rows only)
        │
        ▼
Autoencoder encoder (512 → 256 → 128 → 32), pretrained, then fine-tuned
        │
        ▼
Dense 512 → reshape 8 × 64 → BiLSTM → 4-head attention
        │
        ▼
Dense 256 → 128 → Softmax
        │
        ▼
Evaluation on the official test files
```

**Training details:** square-root class weights for imbalanced classes (`--balance weights`; SMOTE is optional), Adam (learning rate 3e-4, `clipnorm=1.0`), `ReduceLROnPlateau`, and early stopping and checkpointing on **validation macro-F1** — the best epoch's weights are restored before testing.

## Dataset

- **Source:** CIC IoMT 2024 — Canadian Institute for Cybersecurity, University of New Brunswick
- **Content:** benign IoMT traffic + **18 attack subtypes** across 5 attack families:

| Family | Subtypes |
|---|---|
| DDoS (TCP/IP) | ICMP, SYN, TCP, UDP floods |
| DoS (TCP/IP) | ICMP, SYN, TCP, UDP floods |
| MQTT attacks | DDoS Connect/Publish Flood, DoS Connect/Publish Flood, Malformed Data |
| Reconnaissance | OS Scan, Ping Sweep, Port Scan, Vulnerability Scan |
| Spoofing | ARP Spoofing |

- **Split:** the dataset's own split — 51 training files and 21 test files. A stratified 15% validation set is taken from the training files; the test files are used only for the final evaluation. Exact duplicate rows (2.65 million) are removed from the training data.
- Dataset files are **not** committed to this repo (size). See [README_DATA.md](README_DATA.md) for the expected `data/train/` and `data/test/` layout and download instructions.

## Classification Modes

The pipeline supports three granularities via `--class_config`:

| `--class_config` | Classes | Use case |
|---|---|---|
| `2`  | Attack vs. Benign | fastest, deployment-style binary alarm |
| `6`  | Benign, DDoS, DoS, MQTT, Recon, Spoofing | attack-family triage |
| `19` | benign + all 18 subtypes | fine-grained forensics |

## Results

Official test files (1,614,182 windows), `IAT` excluded. Results are reported on two levels:

- **Attack type** (main result): the DoS and DDoS versions of the same attack count as one type. 6 classes become 5 types (Benign, DoS/DDoS, MQTT, Recon, Spoofing) and 19 classes become 13.
- **Exact class**: every class separately, so mixing up DoS and DDoS counts as an error.

The model still predicts DoS and DDoS separately. The **DoS vs. DDoS** column shows how often it picks the right one among windows whose attack type is already correct.

| Classes | Model | Attack type<br>accuracy / macro-F1 | Exact class<br>accuracy / macro-F1 | DoS vs. DDoS |
|---|---|---|---|---|
| 2 | AE + BiLSTM + Attention | 0.9981 / 0.9793 | 0.9981 / 0.9793 | – |
| 2 | Random Forest | 0.9971 / 0.9684 | 0.9971 / 0.9684 | – |
| 6 | AE + BiLSTM + Attention | 0.9950 / 0.8594 | 0.7067 / 0.7681 | 0.686 |
| 6 | Random Forest | 0.9957 / 0.8636 | 0.6830 / 0.7627 | 0.659 |
| 19 | AE + BiLSTM + Attention | 0.9919 / 0.7881 | 0.7049 / 0.6653 | 0.700 |
| 19 | Random Forest | 0.9939 / 0.8104 | 0.6742 / 0.6602 | 0.666 |

- **Why two levels:** the features describe packet rates, flags and sizes within a window, not how many hosts send the traffic, so DoS vs. DDoS stays close to a guess. On the 6-class test set, answering "DDoS" for every flood would be right 72% of the time, more than either model. Counting this in the main accuracy would hide how well the attack types themselves are detected; reporting it separately keeps the limitation visible.
- **Attack vs. benign:** the deep model misses 2,782 of 1.58 million attack windows and raises 354 false alarms (0.9% of benign windows); the Random Forest misses 2,378 and raises 2,278 (6.1%). Each class's training budget is shared evenly across its attack subtypes: with plain random capping the attack sample held only a few dozen VulScan and Ping Sweep rows, and the binary model missed almost every Ping Sweep and Spoofing window. `detection_by_attack_type.png` shows the rate per subtype; the weakest are VulScan (57%), Ping Sweep (76%) and Spoofing (77%). 97.7% of the test windows are attacks, so macro-F1 is the more telling number.
- **Attack types:** with 6 classes, floods (F1 ≈ 1.00), MQTT (0.99), Benign (0.92–0.93) and Recon (0.92–0.94) are recognized well. **Spoofing** is the weakest type (F1 0.46–0.47); most of its false alarms are benign windows. With 19 classes, the Recon subtypes (OS scan vs. port scan, vulnerability scan) and Spoofing pull macro-F1 down while accuracy stays above 99%.
- **With `IAT`** (and quantile scaling for the deep model), exact 6-class accuracy jumps to 99.5–99.8% because `IAT` separates DoS from DDoS by recording time. [IAT_SHORTCUT.md](IAT_SHORTCUT.md) explains why that number is not used.

### Run outputs

Every `main.py` run writes its artifacts to `outputs/<run_name>/`. The training plots and `training_log.csv` are refreshed after every epoch, so a run can be watched while it trains.

| When | File | What it shows |
|---|---|---|
| Before training | `config.json` | all arguments of the run |
| | `class_distribution.png` / `.csv` | rows per class in the training files, in the training set actually used, and in the test set |
| | `autoencoder_history.png` | autoencoder reconstruction loss per epoch |
| Every epoch | `training_history.png` | loss and accuracy (training vs. validation), validation macro-F1 with the best epoch, learning rate |
| | `val_f1_per_class.png` | validation F1 of every class over the epochs |
| | `training_log.csv` | the same numbers as a table |
| After testing | `test_metrics.json`, `test_report.txt` | accuracy, macro / weighted F1, per-class precision, recall and F1, attack-type metrics |
| | `confusion_matrix.csv` / `.png`, `confusion_matrix_attack_type.csv` / `.png` | counts and the row-normalized matrix, at exact-class and attack-type level |
| | `per_class_metrics.png` | precision, recall and F1 per class, plus F1 per attack type |
| | `detection_by_attack_type.png` / `.csv` | share of each of the 18 attack subtypes flagged as an attack, and the false-alarm rate on benign traffic — shows blind spots that overall accuracy hides |
| | `confidence_histogram.png` | how confident the model is on correct vs. wrong predictions |
| | `roc_pr_curves.png` | one-vs-rest ROC and precision-recall curves (one small panel per class above 8 classes) |
| | `model.h5`, `best_weights.h5`, `preprocess.joblib` | the trained model with its scaler and label encoder (used by `predict.py`) |

`baseline.py` writes the same evaluation files plus `class_distribution.png` and `feature_importances.json` / `.png`; `iat_analysis.py` adds `iat_by_class.csv` / `.png`.

## Project Structure

```
├── src/
│   ├── data_loader.py   # loading, caching, dedup, train/val split, scaling, class weights / SMOTE
│   ├── model.py         # autoencoder, BiLSTM + attention model, training with macro-F1 checkpointing and live plots
│   ├── main.py          # CLI: train + evaluate the deep model
│   ├── baseline.py      # CLI: Random Forest on the same pipeline
│   ├── iat_analysis.py  # CLI: IAT shortcut analysis
│   ├── evaluation.py    # metrics (exact and attack-type level), reports and all plots
│   └── predict.py       # CLI: predict a new CSV with a trained run
├── data/                # CIC IoMT 2024 CSVs + cache (not committed — see README_DATA.md)
├── outputs/             # one folder per run (not committed)
├── IAT_SHORTCUT.md      # why IAT is excluded
├── README_DATA.md       # dataset layout & download instructions
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

```bash
git clone https://github.com/MehmetAli20/Hybrid-IDS-for-IoMT-Devices.git
cd Hybrid-IDS-for-IoMT-Devices
pip install -r requirements.txt

# download CIC IoMT 2024 CSVs into data/train and data/test (see README_DATA.md)

# train & evaluate the deep model — pick classification granularity (2, 6, or 19)
python src/main.py --class_config 6

# Random Forest baseline on the same pipeline
python src/baseline.py --class_config 6

# IAT shortcut analysis (Random Forest with all features / IAT only / without IAT)
python src/iat_analysis.py --class_config 6

# predict a new CSV with a trained run (adds prediction, attack_type and confidence columns)
python src/predict.py --run_dir outputs/<run_name> --csv path/to/file.csv
```

`src/main.py` arguments:

| Argument | Default | Description |
|---|---|---|
| `--class_config` | `2` | Classification granularity: 2, 6, or 19 classes |
| `--epochs` | `50` | Maximum training epochs (early stopping on validation macro-F1) |
| `--batch_size` | `512` | Training batch size |
| `--max_per_class` | `50000` | Maximum training rows per class |
| `--balance` | `weights` | `weights` (square-root class weights), `smote` or `none` |
| `--scaling` | `log` | `log` (signed log1p + standardization), `quantile` or `minmax` |
| `--ae_mode` | `finetune` | Autoencoder encoder: `finetune`, `frozen` or `none` |
| `--encoding_dim` | `32` | Autoencoder bottleneck size |
| `--ae_epochs` | `30` | Autoencoder pretraining epochs |
| `--drop_features` | `IAT` | Features to drop; pass the flag with no value to keep all 45 |
| `--seed` | `42` | Random seed |
| `--run_name` | timestamp | Output folder name under `outputs/` |

`src/baseline.py` accepts the same `--class_config`, `--max_per_class`, `--drop_features`, `--seed` and `--run_name` arguments, plus `--n_estimators`. Tested with Python 3.7 and TensorFlow 2.10 (GPU) on Windows; a 6-class run takes about 20–25 minutes on an RTX 3070 Laptop GPU.

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · imbalanced-learn (SMOTE) · NumPy · Pandas · Matplotlib

## Possible Improvements

- Flow- or host-level features (for example the number of source addresses) to separate DoS from DDoS
- Higher Spoofing precision, for example per-class decision thresholds tuned on the validation set
- Real-time inference on live packet captures (e.g., via CICFlowMeter feature extraction)
- Model compression (quantization/pruning) for edge deployment near IoMT gateways

## License

MIT — see [LICENSE](LICENSE).
