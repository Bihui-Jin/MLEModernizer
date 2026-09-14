# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.5444901728004163

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the `tensorflow_addons` dependency that is causing the protobuf `GetPrototype` crash and keep the rest of the pipeline intact. Then I fix the missing pretrained model paths by robustly locating `model_0.h5/model_1.h5/model_2.h5` under `/kaggle/input` (or, if they truly don’t exist, fall back to a simple deterministic baseline that still produces a valid submission). Finally, I make prediction/submission generation deterministic and aligned to `sample_submission.csv` order so the output `submission.csv` is always valid with the required columns and format.'
- What this solution (achieved 0.24507) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by avoiding mixed `keras`/`tensorflow.keras` imports and using only `tf.keras` utilities, which is the common trigger for this error in Kaggle TF environments. I also make model loading more robust by clearing the TF session before loading and enabling safe GPU memory growth to prevent random runtime OOMs. To move your score up toward the target (since 0.245 is far below 0.544), I add a minimal, metric-aligned calibration: per-class thresholds chosen from the training label prevalences (keeps your existing model/ensemble logic intact but reduces empty/over-prediction). Submission generation remains deterministic and strictly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which is the most reliable minimal workaround in Kaggle TF environments. I also make the TF import more defensive (clearer failure mode) while keeping your existing ensemble loading, thresholding, and submission formatting logic unchanged. Finally, I keep determinism settings but move them to the correct place (before TF import) so they actually take effect without triggering the crash. This should unblock execution and allow your pretrained models (if present) to load and score closer to your target.'
- What this solution (achieved 0.24507) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow, which is the most reliable workaround in Kaggle TF images. We also make the TF import more defensive by setting `TF_CPP_MIN_LOG_LEVEL` and keeping determinism/seed settings before TF loads (score-neutral, stability-only). Finally, we keep your existing ensemble + per-class thresholding logic intact, but add a tiny safety fix to ensure `models[i].predict` outputs are always treated as numpy arrays with the expected shape to avoid silent type/shape issues that can degrade predictions.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash by switching the protobuf implementation environment variables to be *forced* (not `setdefault`) and set them before any TensorFlow-related imports, which is the root cause of the `MessageFactory.GetPrototype` error in Kaggle TF images. I also add a small compatibility fallback to import `load_img/img_to_array` via `tensorflow.keras.utils` when `tensorflow.keras.preprocessing.image` triggers issues in some TF builds (score-neutral). The rest of your pipeline (model loading, ensemble averaging, per-class thresholds, submission alignment/format) remain unchanged to preserve core logic while allowing the pretrained models to load and improve the score toward your target. The script still always write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing a compatible protobuf version *before* importing TensorFlow (Kaggle TF images commonly break with newer protobuf). This is a runtime-only fix and does not change your model, thresholds, or prediction logic. I also add a small defensive check so the loaded model outputs are reshaped to `(1, 6)` if needed, preventing silent shape issues that can degrade predictions. The rest of your ensemble averaging, per-class thresholding, and submission alignment remain intact, and the script always write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so the safest way to move toward it without changing your model architecture/training is to fix the biggest metric mismatch: Plant Pathology 2021 is evaluated with mean F1 over 6 labels, and good performance typically requires (1) per-class thresholds tuned to optimize F1 and (2) limiting the number of predicted labels per image to avoid precision collapse. I keep your ensemble averaging and inference pipeline intact, but replace the heuristic prevalence-based thresholds with thresholds chosen by a quick out-of-fold (single split) F1 sweep using your already-available `train.csv` labels and model predictions on a small validation subset. Then I apply a minimal post-process: cap predictions to the top-`k` probabilities when too many labels fire, and still default to `healthy` only when nothing fires. These are small, metric-aligned changes that usually produce a large score lift while preserving your core logic.'
- What this solution (achieved 0.24507) has done: 'Your score (0.245) is far below the target (0.544), so we should improve only the metric-aligned post-processing while keeping your ensemble/model inference unchanged. The biggest issue is that the competition’s mean F1 is computed on a *space-delimited multi-label string*, and common strong baselines enforce (a) a guaranteed prediction of at least one label and (b) a small cap on number of predicted labels, but your current cap of 2 and “healthy fallback” can hurt recall on multi-disease images. I keep your threshold tuning, but (1) tune thresholds on a larger deterministic validation subset (still fast enough), (2) add a tiny rule to *not* force “healthy” if any other label is predicted, and (3) slightly relax the cap to 3 with a “margin” rule so true multi-label cases aren’t clipped as often—these are minimal changes that usually lift F1 substantially without changing model logic. Submission writing/order stays identical and deterministic.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.544), so the most likely reason is that the metric-aligned post-processing is still mismatched to Kaggle’s mean-F1 implementation for this competition. I keep your ensemble inference exactly as-is, but tune thresholds against a validation split using the exact label-string F1 (image-wise F1 on space-delimited labels, then mean), instead of mean per-class F1, which often calibrates poorly here. I also add one minimal rule that is standard for this dataset: if nothing passes thresholds, predict the single highest-probability label (rather than forcing “healthy”), which typically boosts recall and improves mean F1 without changing the model. Submission ordering/format stays identical and deterministic.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the smallest high-impact change is to make the threshold tuning match the *competition’s exact macro-F1 over 6 labels* rather than mean per-image F1; this is a post-processing calibration change that preserves your ensemble/model inference core logic. I compute out-of-fold probabilities once on a deterministic validation split, then grid-search per-class thresholds to maximize macro-F1 (class-wise F1 averaged), which aligns directly with the leaderboard metric. To avoid degrading “healthy” cases, I also add one tiny, standard rule: if no class passes thresholds, predict `healthy` if it is the top probability, otherwise predict the top-1 class (still always at least one label). Everything else (model loading, averaging, cap logic, submission order/format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'We keep your ensemble inference and label post-processing intact, but fix the most likely reason for the low score: the threshold tuning currently optimizes **macro-F1 on binary columns**, while this competition scores **sample-wise F1 averaged over images** from the space-delimited label strings. I change only the tuning objective to match Kaggle’s mean F1 exactly (using the same cap/top-k and “no empty prediction” rules you already use), which is a minimal, metric-aligned calibration change. To keep runtime under the 600s limit, I also avoid recomputing `label_classes.index()` inside loops by using a precomputed dict (score-neutral, speeds tuning). Everything else (model loading, preprocessing, ensemble averaging, submission ordering/format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we keep your ensemble inference intact and focus on the smallest metric-aligned post-processing change likely to lift mean F1: tune per-class thresholds on a validation split using the *exact* competition-style mean sample-wise F1 computed from space-delimited label strings (instead of binarized approximations). To avoid overfitting while staying minimal, we use a deterministic single split with a slightly larger validation size and a lightweight coordinate-descent threshold search, reusing cached validation probabilities so runtime stays reasonable. We also keep your existing cap/top-k + “no empty prediction” rules unchanged, but make the tuning objective evaluate those same rules end-to-end. Submission formatting/order remains exactly aligned to `sample_submission.csv` and still writes `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we keep your ensemble inference and existing cap/“healthy removal” rules intact and make only metric-aligned post-processing changes that typically yield a large F1 lift. The biggest likely issue is that thresholds are tuned on a *random* validation subset and with per-class coordinate search that can be noisy; we switch to a deterministic **stratified** split (by a simple label signature) and tune thresholds to maximize the **exact mean sample-wise F1** you already implemented, reducing variance and improving generalization. We also reduce overfitting by using a slightly larger validation set (still bounded for runtime) and a two-stage grid (coarse then fine) around the best thresholds, without changing model logic. Everything else (model loading, image preprocessing, ensemble averaging, submission alignment/format) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the smallest high-impact change is to make the threshold tuning objective match the competition’s **mean per-label (macro) F1** rather than mean sample-wise F1, while keeping your ensemble inference and post-processing rules (cap, healthy-removal, non-empty prediction) intact. I replace the tuning metric with macro-F1 computed from the same thresholded predictions you submit, and I use a deterministic stratified validation split as you already do to keep tuning stable. To reduce overfitting/variance without changing core logic, I increase the validation size moderately (still within the 600s budget) and keep the same two-stage coordinate search. Submission writing/order stays unchanged and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

random.seed(SEED)
np.random.seed(SEED)

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(str(pb_ver).split(".")[0])
        if major >= 4:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception:
        pass


_ensure_protobuf_compat()



## === cell 1
import gc
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except Exception:
    from tensorflow.keras.utils import load_img, img_to_array

tf.keras.utils.set_random_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## === cell 2
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
test_dir = os.path.join(DATA_ROOT, "test_images")
train_dir = os.path.join(DATA_ROOT, "train_images")
train_csv_path = os.path.join(DATA_ROOT, "train.csv")

img_size = (256, 256)

assert os.path.isdir(test_dir), f"Missing test_dir: {test_dir}"
assert os.path.isdir(train_dir), f"Missing train_dir: {train_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"



## === cell 3
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sample_sub.head()




## === cell 4
def load_images(test_path, image):
    """load image from given path"""
    img = load_img(os.path.join(test_path, image))
    img = img.resize(img_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img




## === cell 5
label_classes = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]




## === cell 6
def _find_model_path(filename: str):
    """Search /kaggle/input recursively for a given model file."""
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None


model_paths = {f"model_{i}.h5": _find_model_path(f"model_{i}.h5") for i in range(3)}
model_paths



## === cell 7
train_df = pd.read_csv(train_csv_path)


def _make_multihot(labels_series, classes):
    out = np.zeros((len(labels_series), len(classes)), dtype=np.float32)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for r, s in enumerate(labels_series.fillna("").astype(str).values):
        toks = [t for t in s.split(" ") if t]
        for t in toks:
            if t in class_to_idx:
                out[r, class_to_idx[t]] = 1.0
    return out


y_train = _make_multihot(train_df["labels"], label_classes)



## === cell 8
models = []
missing = [k for k, v in model_paths.items() if v is None]

if len(missing) == 0:
    tf.keras.backend.clear_session()
    for i in range(3):
        p = model_paths[f"model_{i}.h5"]
        models.append(load_model(p, compile=False))
else:
    models = None
    print("WARNING: Missing model files:", missing)
    print(
        "Will create a valid submission using a deterministic baseline (all 'healthy')."
    )




## === cell 9
def get_label(prediction_prob, thresh=0.3, per_class_thresh=None):
    """get label for a class that satisfies given threshold"""
    prediction_prob = prediction_prob[0].astype(np.float32)

    if per_class_thresh is None:
        mask = prediction_prob >= float(thresh)
    else:
        per_class_thresh = np.asarray(per_class_thresh, dtype=np.float32)
        if per_class_thresh.shape[0] != prediction_prob.shape[0]:
            mask = prediction_prob >= float(thresh)
        else:
            mask = prediction_prob >= per_class_thresh

    labels = [label_classes[i] for i, m in enumerate(mask) if bool(m)]
    return " ".join(labels)




## === cell 10
def _ensure_pred_shape(p, n_classes=6):
    """Defensive: ensure predictions are (1, n_classes)."""
    p = np.asarray(p, dtype=np.float32)
    if p.ndim == 1 and p.shape[0] == n_classes:
        p = p.reshape(1, n_classes)
    if p.ndim == 2 and p.shape[0] != 1 and p.shape[1] == n_classes:
        p = p[:1]
    return p


def _labels_from_probs_with_cap(prob_row, per_class_thresh, top_k_cap=3, margin=0.03):
    """
    Keep existing cap logic (3 + margin) to avoid big semantic changes.
    """
    prob_row = np.asarray(prob_row, dtype=np.float32).reshape(-1)
    thr = np.asarray(per_class_thresh, dtype=np.float32).reshape(-1)

    mask = prob_row >= thr
    idx = np.where(mask)[0]

    if idx.size == 0:
        return ""

    idx = idx[np.argsort(prob_row[idx])[::-1]]

    if idx.size > top_k_cap:
        kept = idx[:top_k_cap]
        last_p = float(prob_row[kept[-1]])
        extra = idx[top_k_cap:]
        extra = extra[prob_row[extra] >= (last_p - float(margin))]
        idx = np.concatenate([kept, extra]).astype(int)

    return " ".join([label_classes[i] for i in idx])


def _predict_probs_for_indices(models, df, img_dir, idx_sel, batch_gc=64):
    y_prob = np.zeros((len(idx_sel), len(label_classes)), dtype=np.float32)
    for j, ridx in enumerate(idx_sel):
        image = df.iloc[ridx]["image"]
        img = load_images(img_dir, image)
        p0 = _ensure_pred_shape(
            models[0].predict(img, verbose=0), n_classes=len(label_classes)
        )
        p1 = _ensure_pred_shape(
            models[1].predict(img, verbose=0), n_classes=len(label_classes)
        )
        p2 = _ensure_pred_shape(
            models[2].predict(img, verbose=0), n_classes=len(label_classes)
        )
        y_prob[j] = ((p0 + p1 + p2) / 3.0)[0]
        del img, p0, p1, p2
        if (j + 1) % batch_gc == 0:
            gc.collect()
    return y_prob


def _label_signature(s: str) -> str:
    toks = [t for t in str(s).split() if t]
    toks = sorted(toks)
    return " ".join(toks) if toks else ""


def _stratified_split_indices_by_signature(df, val_size, seed=SEED):
    """
    Deterministic stratified split by label signature (kept).
    """
    rng = np.random.RandomState(seed)

    sig = df["labels"].fillna("").astype(str).map(_label_signature).values
    groups = {}
    for i, g in enumerate(sig):
        groups.setdefault(g, []).append(i)

    for g, lst in groups.items():
        lst = np.array(lst, dtype=int)
        rng.shuffle(lst)
        groups[g] = lst.tolist()

    n = len(df)
    val_size = int(min(max(1, val_size), n))
    val_idx = []
    for g, lst in groups.items():
        take = int(round(len(lst) * (val_size / n)))
        if take > 0:
            val_idx.extend(lst[:take])

    val_idx = list(dict.fromkeys(val_idx))  # preserve order, unique
    if len(val_idx) > val_size:
        val_idx = val_idx[:val_size]
    elif len(val_idx) < val_size:
        remaining = []
        chosen = set(val_idx)
        all_idx = np.arange(n, dtype=int)
        rng.shuffle(all_idx)
        for i in all_idx.tolist():
            if i not in chosen:
                remaining.append(i)
                if len(remaining) >= (val_size - len(val_idx)):
                    break
        val_idx.extend(remaining)

    return np.array(val_idx, dtype=int)


def _pred_strings_from_probs(y_prob, thr_vec):
    """
    Apply the exact same post-processing rules used at test time,
    so threshold tuning matches submission semantics.
    """
    out = np.empty((y_prob.shape[0],), dtype=object)
    for i in range(y_prob.shape[0]):
        s = _labels_from_probs_with_cap(y_prob[i], thr_vec, top_k_cap=3, margin=0.03)

        if s.strip() != "":
            toks = s.split()
            if "healthy" in toks and len(toks) > 1:
                toks = [t for t in toks if t != "healthy"]
            s = " ".join(toks)

        if s.strip() == "":
            top_idx = int(np.argmax(y_prob[i]))
            s = label_classes[top_idx]

        out[i] = s
    return out


def _strings_to_multihot(str_arr, classes):
    out = np.zeros((len(str_arr), len(classes)), dtype=np.int8)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, s in enumerate(np.asarray(str_arr, dtype=object)):
        toks = [t for t in str(s).split() if t]
        for t in toks:
            j = class_to_idx.get(t, None)
            if j is not None:
                out[i, j] = 1
    return out


def _macro_f1(y_true_bin, y_pred_bin, eps=1e-12):
    """
    Change (score-improving, minimal): use macro-F1 over the 6 labels,
    which matches the competition metric better than mean sample-wise F1 here.
    """
    y_true_bin = np.asarray(y_true_bin, dtype=np.int8)
    y_pred_bin = np.asarray(y_pred_bin, dtype=np.int8)

    f1s = []
    for c in range(y_true_bin.shape[1]):
        yt = y_true_bin[:, c]
        yp = y_pred_bin[:, c]
        tp = float(np.sum((yt == 1) & (yp == 1)))
        fp = float(np.sum((yt == 0) & (yp == 1)))
        fn = float(np.sum((yt == 1) & (yp == 0)))
        denom = 2.0 * tp + fp + fn
        f1 = (2.0 * tp) / (denom + eps) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def _tune_thresholds_on_val_macro_f1(models, df, img_dir, n_val=4096):
    """
    Change (score-improving, minimal): tune thresholds to maximize macro-F1
    on a stratified validation split, using the exact same post-processing
    rules as submission (cap/healthy-removal/top-1 fallback).
    """
    grid_coarse = np.array(
        [
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20,
            0.23,
            0.25,
            0.28,
            0.30,
            0.33,
            0.35,
            0.38,
            0.40,
            0.45,
            0.50,
        ],
        dtype=np.float32,
    )

    idx_val = _stratified_split_indices_by_signature(df, val_size=n_val, seed=SEED)
    y_true_str = df.iloc[idx_val]["labels"].fillna("").astype(str).values
    y_true_bin = _strings_to_multihot(y_true_str, label_classes)

    y_prob = _predict_probs_for_indices(models, df, img_dir, idx_val, batch_gc=64)

    thr = np.full((len(label_classes),), 0.2, dtype=np.float32)

    base_pred_str = _pred_strings_from_probs(y_prob, thr)
    base_pred_bin = _strings_to_multihot(base_pred_str, label_classes)
    base = _macro_f1(y_true_bin, base_pred_bin)

    for c in range(len(label_classes)):
        best_t = float(thr[c])
        best_s = base
        for t in grid_coarse:
            cand = thr.copy()
            cand[c] = float(t)
            pred_str = _pred_strings_from_probs(y_prob, cand)
            pred_bin = _strings_to_multihot(pred_str, label_classes)
            s = _macro_f1(y_true_bin, pred_bin)
            if s > best_s:
                best_s = s
                best_t = float(t)
        thr[c] = best_t
        base = best_s

    for c in range(len(label_classes)):
        center = float(thr[c])
        fine = np.clip(
            np.linspace(center - 0.06, center + 0.06, 13), 0.01, 0.95
        ).astype(np.float32)
        best_t = float(thr[c])
        best_s = base
        for t in fine:
            cand = thr.copy()
            cand[c] = float(t)
            pred_str = _pred_strings_from_probs(y_prob, cand)
            pred_bin = _strings_to_multihot(pred_str, label_classes)
            s = _macro_f1(y_true_bin, pred_bin)
            if s > best_s:
                best_s = s
                best_t = float(t)
        thr[c] = best_t
        base = best_s

    return thr.astype(np.float32)


if models is None:
    thr_per_class = np.array([0.2] * len(label_classes), dtype=np.float32)
else:
    thr_per_class = _tune_thresholds_on_val_macro_f1(
        models, train_df, train_dir, n_val=4096
    )

thr_per_class




## === cell 11
def predict(test_path, threshold):
    """predict on test set, using given threshold (kept for compatibility)"""
    if "image" in sample_sub.columns and len(sample_sub) > 0:
        image_list = sample_sub["image"].tolist()
    else:
        image_list = sorted(os.listdir(test_path))

    images, labels = [], []

    if models is None:
        for image in image_list:
            images.append(image)
            labels.append("healthy")
        return images, labels

    for image in image_list:
        img = load_images(test_path, image)

        p0 = _ensure_pred_shape(
            models[0].predict(img, verbose=0), n_classes=len(label_classes)
        )
        p1 = _ensure_pred_shape(
            models[1].predict(img, verbose=0), n_classes=len(label_classes)
        )
        p2 = _ensure_pred_shape(
            models[2].predict(img, verbose=0), n_classes=len(label_classes)
        )

        pred_prob = (p0 + p1 + p2) / 3.0
        prob_row = pred_prob[0]

        preds = _labels_from_probs_with_cap(
            prob_row, thr_per_class, top_k_cap=3, margin=0.03
        )

        if preds.strip() != "":
            toks = preds.split()
            if "healthy" in toks and len(toks) > 1:
                toks = [t for t in toks if t != "healthy"]
            preds = " ".join(toks)

        if preds.strip() == "":
            top_idx = int(np.argmax(prob_row))
            preds = label_classes[top_idx]

        images.append(image)
        labels.append(preds)

        del img, pred_prob, p0, p1, p2
    gc.collect()
    return images, labels




## === cell 12
image_ids, labels = predict(test_dir, threshold=0.2)



## === cell 13
submission_file = pd.DataFrame({"image": image_ids, "labels": labels})

submission_file = submission_file[["image", "labels"]]

if "image" in sample_sub.columns and len(sample_sub) == len(submission_file):
    submission_file = (
        submission_file.set_index("image").reindex(sample_sub["image"]).reset_index()
    )

submission_file.to_csv("submission.csv", index=False)
submission_file.head()



## === cell 14
assert os.path.exists("submission.csv"), "submission.csv was not created"
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"], f"Bad columns: {chk.columns.tolist()}"
assert len(chk) == len(
    sample_sub
), f"Row count mismatch: got {len(chk)} expected {len(sample_sub)}"
chk.tail()
