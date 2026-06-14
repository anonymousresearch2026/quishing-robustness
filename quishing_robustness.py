"""
quishing_robustness.py
=======================
Reproducible pipeline for the study:
"Evaluating the Robustness of Machine-Learning Quishing Detectors to
 Real-World QR-Code Distortions"

What this script does
---------------------
1. Loads the public QR-code dataset of Trad & Chehab (9,987 labelled 69x69 images).
2. Creates a stratified 80/20 train/test split (fixed seed).
3. Trains two detectors on clean images:
     - a classical gradient-boosted model (XGBoost), and
     - a compact convolutional neural network (CNN).
4. Applies parameterized distortions to the held-out TEST set
   (Gaussian blur, rotation, salt-and-pepper noise, JPEG compression),
   re-binarizing each distorted image as a scanner would, and measures
   accuracy and AUC degradation.
5. Trains an augmentation-based defense (blur+rotation) and compares it.
6. Produces the figures used in the paper.

Dataset
-------
This script does NOT redistribute the dataset. Download it (CC-BY-4.0) from:
  https://github.com/fouadtrad/Detecting-Quishing-Attacks-with-Machine-Learning-Techniques-Through-QR-Code-Analysis
Unzip it and set DATA_DIR below to the folder containing:
  qr_codes_29.pickle  and  qr_codes_29_labels.pickle

Requirements
------------
  numpy, scipy, scikit-learn, xgboost, tensorflow, pillow, matplotlib
(See requirements.txt.)

Usage
-----
  python quishing_robustness.py

Reproducibility note on noise
-----------------------------
Salt-and-pepper noise is stochastic. To reproduce the paper exactly:
  * the noise TABLE uses a whole-array draw with seed=42 (sp_noise);
  * the two FIGURES that include a noise panel use a per-image draw with
    seed=0 (sp_noise_perimage), matching how those figures were generated.
The resulting difference (e.g., AUC 0.887 vs 0.895 at p=0.20) is within the
random variation of the noise process and does not affect any conclusion.
"""

import io
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = Path("data")          # <-- EDIT: folder with the two .pickle files
SEED = 42
IMG = 69                         # images are 69 x 69
BLUR_SIGMAS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
ROT_ANGLES = [0, 2, 5, 10, 15, 25]
NOISE_PS = [0.0, 0.02, 0.05, 0.10, 0.20]
JPEG_QS = [100, 75, 50, 25, 10]

np.random.seed(SEED)


# ----------------------------------------------------------------------
# Data loading and splitting
# ----------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    with open(data_dir / "qr_codes_29.pickle", "rb") as f:
        qr = np.array(pickle.load(f))
    with open(data_dir / "qr_codes_29_labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f)).astype(int)
    print(f"Loaded {qr.shape[0]} images of shape {qr.shape[1:]}; "
          f"phishing={int((labels==1).sum())}, benign={int((labels==0).sum())}")
    return qr, labels


def make_split(qr, labels):
    X = qr.reshape(len(qr), -1).astype(np.float32)   # flatten to 4761-dim vectors
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.20, stratify=labels, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Distortions. Deterministic ones (blur, rotation, JPEG) act on a single
# image; salt-and-pepper noise is stochastic and is applied to the whole
# batch with a fixed seed (matching the experiments). All are re-binarized.
# ----------------------------------------------------------------------
def binarize(a, thresh=0.5):
    return (a >= thresh).astype(np.float32)

def do_blur(img, sigma):
    return img.copy() if sigma == 0 else binarize(gaussian_filter(img, sigma))

def do_rot(img, angle):
    if angle == 0:
        return img.copy()
    return binarize(rotate(img, angle, reshape=False, order=1, cval=0.0))

def do_jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype("uint8")).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return binarize(np.asarray(Image.open(buf).convert("L")) / 255.0)

def sp_noise(images, p, seed=42):
    """Salt-and-pepper noise over the whole array with one RNG draw (seed=42).
    Reproduces the noise TABLE in the paper. Already binary -> no thresholding."""
    rng = np.random.default_rng(seed)
    out = images.copy()
    mask = rng.random(out.shape) < p
    flip = rng.random(out.shape) < 0.5
    out[mask & flip] = 1.0
    out[mask & ~flip] = 0.0
    return out.astype(np.float32)

def sp_noise_perimage(img, p, seed=0):
    """Per-image salt-and-pepper noise (seed=0). Reproduces the noise used in
    the example and bar-chart FIGURES."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    mask = rng.random(img.shape) < p
    flip = rng.random(img.shape) < 0.5
    out[mask & flip] = 1.0
    out[mask & ~flip] = 0.0
    return out.astype(np.float32)

def batch(fn, images, *args):
    return np.stack([fn(im, *args) for im in images])


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------
def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def build_cnn():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.random.set_seed(SEED)
    model = models.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_cnn(X_train, y_train):
    cnn = build_cnn()
    cnn.fit(X_train.reshape(-1, IMG, IMG, 1).astype("float32"),
            y_train, epochs=5, batch_size=64, verbose=1)
    return cnn


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def eval_tree(clf, Xi, y):
    Xf = Xi.reshape(len(Xi), -1)
    return (accuracy_score(y, clf.predict(Xf)),
            roc_auc_score(y, clf.predict_proba(Xf)[:, 1]))

def eval_cnn(cnn, Xi, y):
    proba = cnn.predict(Xi.reshape(-1, IMG, IMG, 1).astype("float32"), verbose=0).ravel()
    return (accuracy_score(y, (proba >= 0.5).astype(int)), roc_auc_score(y, proba))

def sweep(name, model, eval_fn, X_test_img, y_test, distort_fn, values):
    """Sweep a DETERMINISTIC distortion (blur, rotation, JPEG) over severities."""
    print(f"\n--- {name} ---")
    print(f"{'severity':>10} | {'accuracy':>8} | {'AUC':>7}")
    rows = []
    for v in values:
        Xd = batch(distort_fn, X_test_img, v)
        a, u = eval_fn(model, Xd, y_test)
        rows.append((v, a, u))
        print(f"{v:>10} | {a:>8.4f} | {u:>7.4f}")
    return rows

def sweep_noise(model, eval_fn, X_test_img, y_test):
    """Noise sweep (whole-array, seed=42) -> reproduces the noise table."""
    print("\n--- Salt-and-pepper noise (seed=42) ---")
    print(f"{'p':>10} | {'accuracy':>8} | {'AUC':>7}")
    rows = []
    for p in NOISE_PS:
        Xd = sp_noise(X_test_img, p, seed=42)
        a, u = eval_fn(model, Xd, y_test)
        rows.append((p, a, u))
        print(f"{p:>10} | {a:>8.4f} | {u:>7.4f}")
    return rows


# ----------------------------------------------------------------------
# Augmentation defense (RNG-draw order matches the experiments)
# ----------------------------------------------------------------------
def augment_training(X_train, y_train):
    rng = np.random.default_rng(SEED)
    imgs = X_train.reshape(-1, IMG, IMG)
    aug = []
    for im in imgs:
        out = im
        if rng.random() < 0.7:
            out = gaussian_filter(out, sigma=rng.uniform(0.5, 2.0))
        if rng.random() < 0.7:
            out = rotate(out, rng.uniform(-15, 15), reshape=False, order=1, cval=0.0)
        aug.append(binarize(out))
    aug = np.stack(aug).reshape(len(aug), -1)
    X_aug = np.vstack([X_train, aug])
    y_aug = np.concatenate([y_train, y_train])
    return X_aug, y_aug


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def figure_examples(X_test_img):
    ex = X_test_img[0]
    items = {
        "Clean": ex,
        "Blur (s=1.5)": do_blur(ex, 1.5),
        "Rotation (10deg)": do_rot(ex, 10),
        "Noise (p=0.1)": sp_noise_perimage(ex, 0.1, seed=0),
        "JPEG (q=10)": do_jpeg(ex, 10),
    }
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    for ax, (name, img) in zip(axes, items.items()):
        ax.imshow(img, cmap="gray_r"); ax.set_title(name, fontsize=10); ax.axis("off")
    plt.tight_layout(); plt.savefig("fig_distortion_examples.png", dpi=200, bbox_inches="tight")

def figure_severity(clf, clf_aug, cnn, X_test_img, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, vals, distort, xlabel, title in [
        (ax1, BLUR_SIGMAS, do_blur, "Blur strength sigma (pixels)", "(a) Gaussian blur"),
        (ax2, ROT_ANGLES, do_rot, "Rotation angle (degrees)", "(b) Rotation"),
    ]:
        c = [eval_tree(clf, batch(distort, X_test_img, v), y_test)[1] for v in vals]
        a = [eval_tree(clf_aug, batch(distort, X_test_img, v), y_test)[1] for v in vals]
        n = [eval_cnn(cnn, batch(distort, X_test_img, v), y_test)[1] for v in vals]
        ax.plot(vals, c, "o-", label="Classical (XGBoost)")
        ax.plot(vals, n, "s-", label="CNN")
        ax.plot(vals, a, "^--", label="Classical + augmentation")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax.set_xlabel(xlabel); ax.set_ylabel("AUC"); ax.set_title(title)
        ax.set_ylim(0.35, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_severity_sweep.png", dpi=200, bbox_inches="tight")

def figure_types(clf, cnn, X_test_img, y_test):
    conds = {
        "Clean": X_test_img,
        "Blur s=1.5": batch(do_blur, X_test_img, 1.5),
        "Rotation 5deg": batch(do_rot, X_test_img, 5),
        "Noise p=0.2": batch(sp_noise_perimage, X_test_img, 0.2),  # per-image seed=0
        "JPEG q=10": batch(do_jpeg, X_test_img, 10),
    }
    labels_c = list(conds.keys())
    tree_auc = [eval_tree(clf, v, y_test)[1] for v in conds.values()]
    cnn_auc = [eval_cnn(cnn, v, y_test)[1] for v in conds.values()]
    x = np.arange(len(labels_c)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.bar(x - w/2, tree_auc, w, label="Classical (XGBoost)")
    ax.bar(x + w/2, cnn_auc, w, label="CNN")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels_c, rotation=15)
    ax.set_ylabel("AUC"); ax.set_ylim(0, 1)
    ax.set_title("Detection AUC by distortion type")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("fig_distortion_types.png", dpi=200, bbox_inches="tight")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    qr, labels = load_data()
    X_train, X_test, y_train, y_test = make_split(qr, labels)
    X_test_img = X_test.reshape(-1, IMG, IMG)

    # Baselines
    clf = train_xgboost(X_train, y_train)
    acc, auc = eval_tree(clf, X_test, y_test)
    print(f"\nClassical clean: accuracy={acc:.4f}, AUC={auc:.4f}")
    cnn = train_cnn(X_train, y_train)
    acc_c, auc_c = eval_cnn(cnn, X_test, y_test)
    print(f"CNN clean: accuracy={acc_c:.4f}, AUC={auc_c:.4f}")

    # Robustness sweeps (classical)
    sweep("Blur (classical)", clf, eval_tree, X_test_img, y_test, do_blur, BLUR_SIGMAS)
    sweep("Rotation (classical)", clf, eval_tree, X_test_img, y_test, do_rot, ROT_ANGLES)
    sweep_noise(clf, eval_tree, X_test_img, y_test)
    sweep("JPEG (classical)", clf, eval_tree, X_test_img, y_test, do_jpeg, JPEG_QS)

    # Robustness sweeps (CNN) for the damaging distortions
    sweep("Blur (CNN)", cnn, eval_cnn, X_test_img, y_test, do_blur, BLUR_SIGMAS)
    sweep("Rotation (CNN)", cnn, eval_cnn, X_test_img, y_test, do_rot, ROT_ANGLES)

    # Augmentation defense
    X_aug, y_aug = augment_training(X_train, y_train)
    print("\nAugmented training size:", X_aug.shape[0])
    clf_aug = train_xgboost(X_aug, y_aug)
    acc_a, auc_a = eval_tree(clf_aug, X_test, y_test)
    print(f"Augmented clean: accuracy={acc_a:.4f}, AUC={auc_a:.4f}")
    sweep("Blur (augmented)", clf_aug, eval_tree, X_test_img, y_test, do_blur, BLUR_SIGMAS)
    sweep("Rotation (augmented)", clf_aug, eval_tree, X_test_img, y_test, do_rot, ROT_ANGLES)

    # Figures
    figure_examples(X_test_img)
    figure_severity(clf, clf_aug, cnn, X_test_img, y_test)
    figure_types(clf, cnn, X_test_img, y_test)
    print("\nFigures saved: fig_distortion_examples.png, fig_severity_sweep.png, fig_distortion_types.png")


if __name__ == "__main__":
    main()
