# Quishing Robustness

Code and benchmark for the study "Evaluating the Robustness of Machine-Learning Quishing Detectors to Real-World QR-Code Distortions."

This repository reproduces two image-based QR-code phishing ("quishing") detectors; a classical gradient-boosted model (XGBoost) and a compact convolutional neural network (CNN). And it evaluates how their detection performance degrades when QR codes are subjected to realistic capture distortions (blur, rotation, salt-and-pepper noise, and JPEG compression). It also implements and evaluates a training-time data-augmentation defense.

## What the code does

1. Loads a public dataset of 9,987 labelled 69×69 QR-code images.
2. Creates a stratified 80/20 train/test split (fixed seed = 42).
3. Trains the classical and CNN detectors on clean images.
4. Applies parameterized distortions to the held-out test set, re-binarizing each distorted image (as a scanner would), and measures accuracy and AUC.
5. Trains an augmentation-based defense and compares it against the baseline.
6. Generates the figures used in the paper.

## Dataset

This repository does **not** redistribute the dataset. Download it (released under CC-BY-4.0) from the original authors:

> Trad, F., Chehab, A. *Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis.* arXiv:2505.03451.
> https://github.com/fouadtrad/Detecting-Quishing-Attacks-with-Machine-Learning-Techniques-Through-QR-Code-Analysis

Unzip `QuishingDataset.zip` and place the two files

```
qr_codes_29.pickle
qr_codes_29_labels.pickle
```

into a folder named `data/` in this repository (or edit `DATA_DIR` at the top of `quishing_robustness.py`).

## Requirements

```
pip install -r requirements.txt
```

## Running

```
python quishing_robustness.py
```

This prints the clean-image baselines and the degradation tables for each distortion, trains the augmentation defense, and saves three figures:

- `fig_distortion_examples.png`
- `fig_severity_sweep.png`
- `fig_distortion_types.png`

All experiments run on CPU; no GPU is required.

## Reproducibility

A fixed random seed (42) is used throughout (data split, model training, noise, and augmentation), so results are deterministic across runs on the same environment.

## License

The code in this repository is released for research use. The QR-code dataset is the property of its original authors and is licensed CC-BY-4.0; please cite their work if you use it.
