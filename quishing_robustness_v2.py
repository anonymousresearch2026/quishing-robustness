"""
quishing_robustness_v2.py
=========================
Full reproducible pipeline for the REVISED study (addresses reviewer comments):
"Evaluating the Robustness of Machine-Learning Quishing Detectors to
 Real-World QR-Code Distortions"

This single script runs every experiment in the revised paper and prints one
consolidated summary at the end:

  (A) Classical (XGBoost) baseline + blur/rotation/noise/JPEG sweeps,
      reported as accuracy@0.5, accuracy@best-threshold (recalibrated), and AUC.
  (B) Augmentation-trained classical detector, same sweeps.
  (C) CNN over 5 random seeds -> mean +/- SD for acc@0.5, acc@best, AUC
      (blur, rotation, noise, JPEG).  [addresses: multi-seed, calibration]
  (D) Rectification / decode front-end: render each code as a realistic image,
      distort it, run OpenCV's QR detector to locate + rectify, then classify.
      Reports detection rate and AUC with vs. without the front-end.
      [addresses: threat-model / real-scanner comment]
  (E) Bootstrap 95% CIs on key AUC comparisons (classical vs CNN-ensemble).
      [addresses: confidence-interval comment]

Dataset (CC-BY-4.0), not redistributed here:
  https://github.com/fouadtrad/Detecting-Quishing-Attacks-with-Machine-Learning-Techniques-Through-QR-Code-Analysis
Set DATA_DIR to the folder containing qr_codes_29.pickle and qr_codes_29_labels.pickle.

Requirements: numpy, scipy, scikit-learn, xgboost, tensorflow, pillow,
              matplotlib, opencv-python
"""

import io
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = Path("data")          # <-- EDIT to the folder with the two .pickle files
SPLIT_SEED = 42
IMG = 69
CNN_SEEDS = [0, 1, 2, 3, 4]      # 5 independent CNN trainings
BLUR_SIGMAS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
ROT_ANGLES = [0, 2, 5, 10, 15, 25]
N_RECTIFY = 600                  # sample size for the (slower) rectification test
N_BOOTSTRAP = 1000               # bootstrap resamples for CIs

np.random.seed(SPLIT_SEED)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_split():
    with open(DATA_DIR / "qr_codes_29.pickle", "rb") as f:
        qr = np.array(pickle.load(f))
    with open(DATA_DIR / "qr_codes_29_labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f)).astype(int)
    X = qr.reshape(len(qr), -1).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(
        X, labels, test_size=0.20, stratify=labels, random_state=SPLIT_SEED)
    print(f"Loaded {len(qr)} images; phishing={(labels==1).sum()}, benign={(labels==0).sum()}")
    print(f"Train {len(Xtr)} / Test {len(Xte)}")
    return Xtr, Xte, ytr, yte


# ----------------------------------------------------------------------
# Distortions (all re-binarized at 0.5, scanner-style)
# ----------------------------------------------------------------------
def binz(a):
    return (a >= 0.5).astype(np.float32)

def do_blur(im, s):
    return im.copy() if s == 0 else binz(gaussian_filter(im, s))

def do_rot(im, a):
    return im.copy() if a == 0 else binz(rotate(im, a, reshape=False, order=1, cval=0.0))

def do_jpeg(im, q):
    buf = io.BytesIO()
    Image.fromarray((im * 255).astype("uint8")).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return binz(np.asarray(Image.open(buf).convert("L")) / 255.0)

def sp_noise(imgs, p, seed=42):
    rng = np.random.default_rng(seed)
    out = imgs.copy()
    m = rng.random(out.shape) < p
    f = rng.random(out.shape) < 0.5
    out[m & f] = 1.0
    out[m & ~f] = 0.0
    return out.astype(np.float32)

def bat(fn, imgs, v):
    return np.stack([fn(i, v) for i in imgs])


# ----------------------------------------------------------------------
# Metrics: accuracy at fixed 0.5 threshold, accuracy at best threshold, AUC
# ----------------------------------------------------------------------
def three_metrics(y, proba):
    af = accuracy_score(y, (proba >= 0.5).astype(int))
    ab = max(accuracy_score(y, (proba >= t).astype(int)) for t in np.linspace(0, 1, 201))
    return af, ab, roc_auc_score(y, proba)

def tree_proba(clf, imgs):
    return clf.predict_proba(imgs.reshape(len(imgs), -1))[:, 1]


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def train_xgb(Xtr, ytr):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                         subsample=0.9, eval_metric="logloss",
                         random_state=SPLIT_SEED, n_jobs=-1).fit(Xtr, ytr)

def augment_training(Xtr, ytr):
    rng = np.random.default_rng(SPLIT_SEED)
    aug = []
    for im in Xtr.reshape(-1, IMG, IMG):
        o = im
        if rng.random() < 0.7:
            o = gaussian_filter(o, sigma=rng.uniform(0.5, 2.0))
        if rng.random() < 0.7:
            o = rotate(o, rng.uniform(-15, 15), reshape=False, order=1, cval=0.0)
        aug.append(binz(o))
    Xa = np.vstack([Xtr, np.stack(aug).reshape(len(aug), -1)])
    ya = np.concatenate([ytr, ytr])
    return Xa, ya

def make_cnn(seed):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.random.set_seed(seed)
    m = models.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def cnn_proba(m, imgs):
    return m.predict(imgs.reshape(-1, IMG, IMG, 1).astype("float32"), verbose=0).ravel()


# ----------------------------------------------------------------------
# (D) Rectification front-end via OpenCV
# ----------------------------------------------------------------------
def render(grid, module_px=10, quiet=4):
    """Upscale a 69x69 {0,1} grid (1=light, 0=dark) into a realistic dark-on-white
    QR image with a white quiet zone."""
    big = np.kron(grid, np.ones((module_px, module_px)))
    big = np.pad(big, quiet * module_px, constant_values=1.0)
    return (big * 255).astype(np.uint8)

def distort_render(img, kind, v):
    if kind == "blur":
        return img if v == 0 else gaussian_filter(img, v * (img.shape[0] / IMG))
    if kind == "rot":
        return img if v == 0 else rotate(img, v, reshape=False, order=1,
                                         cval=255, mode="constant")
    return img

def rectify_to_grid(det, img):
    """Detect + perspective-rectify the QR in img; return upright 69x69 {0,1} grid
    or None if the detector cannot locate it."""
    import cv2
    try:
        ok, pts = det.detect(img.astype(np.uint8))
        if not ok or pts is None:
            return None
        s = IMG * 10
        dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts.reshape(4, 2).astype(np.float32), dst)
        warp = cv2.warpPerspective(img.astype(np.uint8), M, (s, s))
        small = cv2.resize(warp, (IMG, IMG), interpolation=cv2.INTER_AREA) / 255.0
        return binz(small)
    except Exception:
        return None


# ----------------------------------------------------------------------
# (E) Bootstrap CI on AUC difference (CNN ensemble vs classical)
# ----------------------------------------------------------------------
def bootstrap_auc_diff(y, p_cnn, p_clf, B=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs, a_cnn, a_clf = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ac = roc_auc_score(yb, p_cnn[idx]); al = roc_auc_score(yb, p_clf[idx])
        a_cnn.append(ac); a_clf.append(al); diffs.append(ac - al)
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return (np.mean(a_cnn), q(a_cnn)), (np.mean(a_clf), q(a_clf)), (np.mean(diffs), q(diffs))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    Xtr, Xte, ytr, yte = load_split()
    Xte_img = Xte.reshape(-1, IMG, IMG)

    # ---------- (A) classical ----------
    print("\n[A] Training classical (XGBoost) ...")
    clf = train_xgb(Xtr, ytr)
    print("\n=== (A) CLASSICAL: acc@0.5 | acc@best | AUC ===")
    print("BLUR:")
    for s in BLUR_SIGMAS:
        print(f"  s={s:>3}: %.3f | %.3f | %.3f" % three_metrics(yte, tree_proba(clf, bat(do_blur, Xte_img, s))))
    print("ROTATION:")
    for a in ROT_ANGLES:
        print(f"  {a:>3}d: %.3f | %.3f | %.3f" % three_metrics(yte, tree_proba(clf, bat(do_rot, Xte_img, a))))
    print("NOISE p=0.2:  %.3f | %.3f | %.3f" % three_metrics(yte, tree_proba(clf, sp_noise(Xte_img, 0.2, 42))))
    print("JPEG  q=10 :  %.3f | %.3f | %.3f" % three_metrics(yte, tree_proba(clf, bat(do_jpeg, Xte_img, 10))))

    # ---------- (B) augmented ----------
    print("\n[B] Training augmented classical ...")
    Xa, ya = augment_training(Xtr, ytr)
    clf_aug = train_xgb(Xa, ya)
    print("\n=== (B) AUGMENTED: acc@0.5 | acc@best | AUC ===")
    print("BLUR:")
    for s in BLUR_SIGMAS:
        print(f"  s={s:>3}: %.3f | %.3f | %.3f" % three_metrics(yte, tree_proba(clf_aug, bat(do_blur, Xte_img, s))))
    print("ROTATION:")
    for a in ROT_ANGLES:
        print(f"  {a:>3}d: %.3f | %.3f | %.3f" % three_metrics(yte, tree_proba(clf_aug, bat(do_rot, Xte_img, a))))

    # ---------- (C) multi-seed CNN ----------
    print("\n[C] Training %d CNNs ..." % len(CNN_SEEDS))
    conds = {f"blur{s}": bat(do_blur, Xte_img, s) for s in BLUR_SIGMAS}
    conds.update({f"rot{a}": bat(do_rot, Xte_img, a) for a in ROT_ANGLES})
    conds["noise0.2"] = sp_noise(Xte_img, 0.2, 42)
    conds["jpeg10"] = bat(do_jpeg, Xte_img, 10)
    per_seed = {k: [] for k in conds}
    cnn_probs_clean = []        # keep ensemble pieces for CIs
    cnn_models = []
    for sd in CNN_SEEDS:
        print(f"  seed {sd} ...")
        m = make_cnn(sd)
        m.fit(Xtr.reshape(-1, IMG, IMG, 1).astype("float32"), ytr,
              epochs=5, batch_size=64, verbose=0)
        cnn_models.append(m)
        for k, v in conds.items():
            per_seed[k].append(three_metrics(yte, cnn_proba(m, v)))
    print("\n=== (C) CNN over %d seeds, mean +/- SD: acc@0.5 | acc@best | AUC ===" % len(CNN_SEEDS))
    for k in conds:
        a = np.array(per_seed[k]); mu, sd = a.mean(0), a.std(0)
        print(f"  {k:>9}: {mu[0]:.3f}+/-{sd[0]:.3f} | {mu[1]:.3f}+/-{sd[1]:.3f} | {mu[2]:.3f}+/-{sd[2]:.3f}")

    # CNN ensemble probability (mean across seeds) for the bootstrap comparison
    def cnn_ensemble_proba(imgs):
        return np.mean([cnn_proba(m, imgs) for m in cnn_models], axis=0)

    # ---------- (D) rectification ----------
    print("\n[D] Rectification / decode front-end (sample of %d) ..." % N_RECTIFY)
    import cv2
    det = cv2.QRCodeDetector()
    idx = np.arange(min(N_RECTIFY, len(Xte_img)))
    print("\n=== (D) detect%% | AUC rectified | AUC no-rect (baseline) ===")
    for kind, vals in [("blur", BLUR_SIGMAS), ("rot", ROT_ANGLES)]:
        print(f"{kind.upper()}:")
        for v in vals:
            rect_grids, ys, base = [], [], []
            found = 0
            for i in idx:
                img = distort_render(render(Xte_img[i]).astype(np.float32), kind, v)
                g = rectify_to_grid(det, img)
                if kind == "blur":
                    bg = Xte_img[i] if v == 0 else binz(gaussian_filter(Xte_img[i], v))
                else:
                    bg = Xte_img[i] if v == 0 else binz(rotate(Xte_img[i], v, reshape=False, order=1, cval=0.0))
                base.append(bg.ravel())
                if g is not None:
                    rect_grids.append(g.ravel()); ys.append(yte[i]); found += 1
            auc_r = (roc_auc_score(ys, clf.predict_proba(np.array(rect_grids))[:, 1])
                     if found > 30 and len(set(ys)) == 2 else float("nan"))
            auc_b = roc_auc_score(yte[idx], clf.predict_proba(np.array(base))[:, 1])
            print(f"  {kind} {v:>4}: {found/len(idx):.0%} | {auc_r:.3f} | {auc_b:.3f}")

    # ---------- (E) bootstrap CIs ----------
    print("\n[E] Bootstrap 95%% CIs on AUC (CNN ensemble vs classical), B=%d ..." % N_BOOTSTRAP)
    key = {"clean": Xte_img, "blur1.5": bat(do_blur, Xte_img, 1.5),
           "rot2": bat(do_rot, Xte_img, 2), "rot5": bat(do_rot, Xte_img, 5)}
    print("\n=== (E) condition | CNN AUC [95% CI] | classical AUC [95% CI] | diff [95% CI] ===")
    for name, imgs in key.items():
        pc = cnn_ensemble_proba(imgs); pl = tree_proba(clf, imgs)
        (mc, ci_c), (ml, ci_l), (md, ci_d) = bootstrap_auc_diff(yte, pc, pl)
        print(f"  {name:>8}: {mc:.3f} [{ci_c[0]:.3f},{ci_c[1]:.3f}] | "
              f"{ml:.3f} [{ci_l[0]:.3f},{ci_l[1]:.3f}] | "
              f"{md:+.3f} [{ci_d[0]:+.3f},{ci_d[1]:+.3f}]")

    print("\nDONE. All experiments complete.")


if __name__ == "__main__":
    main()
