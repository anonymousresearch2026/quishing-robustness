# Quishing Robustness

Code and benchmark for the study "Evaluating the Robustness of Machine-Learning Quishing Detectors to Real-World QR-Code Distortions."

This repository reproduces two image-based QR-code phishing ("quishing") detectors, a classical gradient-boosted model (XGBoost) and a compact convolutional neural network (CNN), and evaluates how their performance degrades when QR codes undergo realistic capture distortions (blur, rotation, salt-and-pepper noise, JPEG compression). It also evaluates a training-time data-augmentation defense, the effect of decision-threshold recalibration, and a real QR localization and rectification front-end placed ahead of the classifier.

## What the code does

The consolidated pipeline used for the published results is in `quishing_robustness_v2.py`, which runs all experiments in one pass:

1. Loads a public dataset of 9,987 labelled 69x69 QR-code images.
2. Creates a stratified 80/20 train/test split (fixed split seed = 42).
3. Trains the classical and CNN detectors on clean images.
4. Applies parameterized distortions to the held-out test set, re-binarizing each distorted image (as a scanner would), and reports accuracy at the fixed 0.5 threshold, accuracy at the best (recalibrated) threshold, and AUC.
5. Trains and evaluates an augmentation-based defense.
6. Trains the CNN across 5 random seeds and reports mean +/- standard deviation (the CNN is not deterministic on CPU).
7. Evaluates a QR localization and rectification front-end (OpenCV): each code is rendered as a realistic image, distorted, then detected and rectified before classification, reporting detection rate and AUC with and without the front-end.
8. Reports bootstrap 95% confidence intervals on key AUC comparisons.

## Dataset

This repository does not redistribute the dataset. Download it (released under CC-BY-4.0) from the original authors: Trad, F., Chehab, A. "Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis." arXiv:2505.03451 — https://github.com/fouadtrad/Detecting-Quishing-Attacks-with-Machine-Learning-Techniques-Through-QR-Code-Analysis

Place the two files (`qr_codes_29.pickle` and `qr_codes_29_labels.pickle`) into a folder named `data/` at the repository root, or edit the `DATA_DIR` variable at the top of the script.

## Requirements

Install dependencies with: `pip install -r requirements.txt`

The rectification experiment additionally requires `opencv-python`.

## Running

Run the full pipeline with: `python quishing_robustness_v2.py`

This prints, in order: (A) classical results, (B) augmentation results, (C) multi-seed CNN mean +/- SD, (D) rectification and decode front-end results, and (E) bootstrap confidence intervals. All experiments run on CPU; no GPU is required. The full run takes roughly 30 minutes.

## Reproducibility

The data split, classical model, augmentation, and noise are deterministic (fixed seed = 42) and reproduce exactly across runs. The CNN is not deterministic on CPU, even with a fixed seed; we therefore train it across 5 seeds and report mean +/- standard deviation rather than a single run. Individual CNN values vary modestly from run to run, but the reported pattern is stable.

## License

The code in this repository is released for research use. The QR-code dataset is the property of its original authors and is licensed CC-BY-4.0; please cite their work if you use it.
