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

0.7933702677747018

# 6. Current score

0.28763

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.272) has done: 'I remove the failing `kaggle_datasets` import that’s triggering the protobuf `GetPrototype` crash, since it’s not used anywhere in your pipeline. Then I fix the missing model file issue by loading the pretrained `.h5` if it exists, otherwise falling back to a small TF/Keras model so the notebook always runs end-to-end and produces predictions (this preserves your overall “load model → predict → threshold → write submission” logic). Finally, I fix the submission-length mismatch by keeping a single, consistently ordered list of test filenames and using it both for the dataset and for the `image` column, ensuring `submission.csv` is valid and aligned.'
- What this solution (achieved 0.272) has done: 'I fix the immediate runtime crash that happens on `import tensorflow` by forcing TensorFlow to use the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` incompatibility in many Kaggle images). Then I correct a logic bug in your label mapping (`healthy` was keyed as `6` instead of `5`) which was preventing the default class from ever matching the model’s 6 outputs and hurting score. Finally, I keep your existing “load model → predict → threshold → write submission.csv” flow intact, but make the label loop robust to exactly 6 outputs and ensure the submission uses the sample-submission image order for perfect alignment.'
- What this solution (achieved 0.272) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by forcing TensorFlow/Keras to use the legacy pure-Python protobuf *before any TensorFlow-related import* and by removing the direct `tensorflow.keras.backend` import that triggers the failing protobuf path. Then I add a safe fallback that uses `sample_submission.csv` directly if images aren’t present in the visible filesystem (common with hidden test sets), so the notebook still writes a correctly shaped `submission.csv`. Finally, I keep your existing “load model → predict → thresholds → write submission” logic intact, but make the fallback model compile-less and deterministic so it runs end-to-end without changing the intended semantics beyond what’s necessary to produce a valid submission and improve from the current broken state.'
- What this solution (achieved 0.272) has done: 'I fix the immediate TensorFlow/protobuf crash by avoiding the brittle “force pure-Python protobuf” env vars and by removing the `tensorflow.keras.backend as K` dependency that triggers the failing protobuf path in this environment. To preserve your core flow (“build/load model → predict → threshold → write submission”), I re-implement `FixedDropout` using only TensorFlow ops so it no longer needs `K.shape`. Finally, I make the code robust to Kaggle path variants by falling back to `../input/plant-pathology-2021-fgvc8/...` if the shorter path isn’t present, ensuring the notebook runs end-to-end and always produces a valid `submission.csv`.'
- What this solution (achieved 0.272) has done: 'We fix the runtime crash that happens at `import tensorflow` by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` protobuf incompatibility in this Kaggle image. Then we keep your exact prediction + thresholding + submission-writing logic intact, only ensuring paths are robust and `have_images` is computed correctly so inference actually runs when images exist. This should both unblock execution and substantially improve score versus the current fallback “all healthy” behavior (0.272) by enabling your pretrained model (when present) to generate real predictions. Finally, we ensure a valid `submission.csv` with correct columns and row alignment is always produced.'
- What this solution (achieved 0.272) has done: 'I fix the root cause of the crash by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow (your current `"cpp"` setting triggers the missing `_message` import). Then I make the `have_images` logic robust so it doesn’t wrongly disable inference when a few files are missing, and I ensure we always produce exactly one prediction per row in `sample_submission.csv` (padding/truncating if needed) to avoid the “arrays must be same length” submission error. These changes keep your core flow intact (load model → predict on tf.data → threshold to space-delimited labels → write `submission.csv`) while enabling real predictions instead of crashing or producing misaligned outputs. Finally, the script always write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.34001) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by avoiding TensorFlow entirely (it isn’t required here) and switching to a simple, deterministic multi-label baseline using only NumPy/Pandas. This preserves the core “read sample_submission → generate labels → write submission.csv” flow while ensuring it runs end-to-end in the Kaggle environment and writes a valid CSV with the required columns and row alignment. To improve score from the current “mostly healthy/fallback” behavior, I fit per-class prevalence on `train.csv` and output the top-k labels per image based on those learned priors (a minimal, legitimate calibration step without changing any model architecture/training loop because no usable model is available). The output remain space-delimited labels exactly as required.'
- What this solution (achieved 0.29728) has done: 'Your current TF-free baseline predicts the same labels for every test image, which caps Mean F1 very low; the smallest legitimate improvement toward your target is to keep the same “train.csv → compute label stats → generate space-delimited strings → write submission.csv” logic but make predictions vary per image using information available at inference time. Since images are present in the filesystem, we can add a lightweight, deterministic heuristic that uses each test image’s mean/variance (computed with PIL) to decide whether to emit `healthy` versus a fixed disease-set, while keeping your prevalence-based label selection and submission formatting intact. This should increase score substantially versus constant-label output without introducing new ML libraries, training loops, or architecture changes. If images are missing, it fall back to your current prior-only behavior and still write a valid submission.'
- What this solution (achieved 0.226) has done: 'The timeout is dominated by per-image PIL decoding for all test images plus an additional calibration loop that decodes up to 1200 train images; both are pure CPU/I/O and run serially. I keep the exact same heuristic logic and thresholds, but speed it up by (1) eliminating repeated `os.path.exists` scans via a single pass, (2) parallelizing image-stat extraction with a deterministic thread pool (PIL releases the GIL during decode/resize, so this helps a lot), and (3) removing avoidable Python overhead in label counting by using vectorized string operations while preserving the same semantics (count each class once per image). All paths, outputs, and decision rules remain identical, with only negligible floating-point differences possible due to operation ordering.'
- What this solution (achieved 0.21661) has done: 'Your current TF-free heuristic is far below the target (0.226 vs 0.793), so the smallest meaningful move toward the target is to keep the same overall “priors + cheap image stats + thresholding → space-delimited labels → submission.csv” logic but tune the decision rule to better match Mean F1 on multi-label data. Concretely, we (1) select a small set of most-common disease labels (instead of only top-1) to increase recall, (2) calibrate per-label emission thresholds using the provided training set by searching a few candidate cutoffs on a held-out calibration subset (still no model/training loop), and (3) keep the existing fast grayscale-stat extraction and submission alignment unchanged. These are minimal, deterministic changes that should materially increase F1 versus single-label output while staying within the 600s budget.'
- What this solution (achieved 0.2892) has done: 'Your current score (0.21661) is far below the target (0.79337), so we should improve F1 with the smallest change that preserves your TF‑free “priors + cheap image stats → threshold → space-delimited labels” core logic. The main issue is that using a fixed TOPK=3 disease set for every “unhealthy-like” image destroys precision; Mean F1 usually improves when we predict fewer labels per image and only add extra labels when we are confident. I keep your same features (grayscale mean/std) and same calibration loop, but extend calibration to choose (a) the best healthy/unhealthy thresholds and (b) the best per-image number of disease labels (TOPK_DISEASE) plus an optional rule to include `complex`, all by maximizing mean sample F1 on the same sampled train images you already decode. This keeps runtime within budget (only uses the already-loaded calibration subset), keeps inference deterministic, and should move the score materially upward toward the target.'
- What this solution (achieved 0.28763) has done: 'We keep your TF-free “priors + grayscale mean/std heuristic + small calibration grid search → space-delimited labels” pipeline intact, but fix the biggest scoring limiter: for non-healthy images you currently emit the same top-k diseases almost always, which hurts precision and Mean F1. The minimal improvement is to calibrate not only the healthy thresholds/topk/complex, but also per-label inclusion rules using the same sampled train images you already decode—i.e., learn which diseases correlate with simple brightness/contrast signals and only emit those when the signal supports them. We do this with a tiny deterministic grid over a few extra scalar features (dark-pixel fraction and edge strength) computed from the same downscaled grayscale thumbnail, then use per-label thresholds learned on the calibration subset. This keeps runtime within the same order (we reuse the same decode pass) and should move the score upward toward your target without changing the overall approach.'

# 9. Code solution

## === cell 0
import os, re, math, random, pathlib
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)




## === cell 1
def decode_image(filename, label=None, image_size=(512, 512)):
    raise RuntimeError("decode_image is unused in this TF-free fallback pipeline.")




## === cell 2
BATCH_SIZE = 32



## === cell 3
candidate_roots = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvcvc8/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
]
data_root = None
for r in candidate_roots:
    if os.path.exists(r):
        data_root = r
        break
if data_root is None:
    data_root = "../input/plant-pathology-2021-fgvc8"

source = os.path.join(data_root, "test_images")
sample_path = os.path.join(data_root, "sample_submission.csv")
train_path = os.path.join(data_root, "train.csv")

sample_sub = pd.read_csv(sample_path)
image_files = sample_sub["image"].astype(str).tolist()
IMAGE_PATHS = [os.path.join(source, f) for f in image_files]

_exists_mask = np.fromiter(
    (os.path.exists(p) for p in IMAGE_PATHS), dtype=np.bool_, count=len(IMAGE_PATHS)
)
missing = [f for f, ex in zip(image_files, _exists_mask.tolist()) if not ex]
if len(missing) > 0:
    print(
        f"WARNING: Missing {len(missing)} test images in filesystem. Example: {missing[:3]}"
    )
else:
    print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", image_files[:3])



## === cell 4
IMAGE_PATHS[:5]



## === cell 5
existing_paths = [p for p, ex in zip(IMAGE_PATHS, _exists_mask.tolist()) if ex]
if len(existing_paths) == 0:
    print(
        f"WARNING: No readable images found in {source}. Will still generate a valid submission via priors."
    )
else:
    print(f"Readable test images found: {len(existing_paths)}/{len(IMAGE_PATHS)}")



## === cell 6
AUTO = None
existing_mask = _exists_mask
existing_indices = np.flatnonzero(existing_mask).tolist()
have_images = len(existing_indices) > 0
test_dataset = None



## === cell 7
print("Using TF-free prior-based + calibrated heuristic predictor (fits on train.csv).")



## === cell 8
train_df = pd.read_csv(train_path)

classes = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(classes)}


def parse_labels(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return str(s).strip().split()


labels_series = train_df["labels"].fillna("").astype(str)
counts = np.empty(len(classes), dtype=np.float64)
for i, c in enumerate(classes):
    pat = rf"(?:^|\s){re.escape(c)}(?:\s|$)"
    counts[i] = float(labels_series.str.contains(pat, regex=True).sum())

prevalence = counts / max(1.0, float(len(train_df)))
prev_series = pd.Series(prevalence, index=classes).sort_values(ascending=False)

print("Class prevalence (descending):")
print(prev_series)



## === cell 9
disease_classes = [c for c in classes if c != "healthy"]
disease_prev = prev_series.loc[disease_classes].sort_values(ascending=False)

DEFAULT_TOPK_DISEASE = 2
base_labels = disease_prev.index[:DEFAULT_TOPK_DISEASE].tolist()

print("Base labels used for disease predictions (DEFAULT_TOPK_DISEASE):", base_labels)



## === cell 10
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def image_stats_gray(path, max_side=256):
    """
    Deterministic, cheap grayscale features from a downscaled thumbnail:
    - mean brightness (0..255)
    - std (contrast)
    - dark_frac: fraction of pixels < 80  (captures lesions/necrosis shadows)
    - edge_strength: mean abs gradient (simple texture proxy)
    These extra two scalars are a minimal extension that helps avoid emitting the same diseases for all "unhealthy" images.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            im = im.resize(
                (int(round(w / scale)), int(round(h / scale))), resample=Image.BILINEAR
            )
        arr = np.asarray(im, dtype=np.float32)

        m = float(arr.mean())
        s = float(arr.std())
        dark_frac = float((arr < 80.0).mean())

        dx = np.abs(arr[:, 1:] - arr[:, :-1]).mean() if arr.shape[1] > 1 else 0.0
        dy = np.abs(arr[1:, :] - arr[:-1, :]).mean() if arr.shape[0] > 1 else 0.0
        edge_strength = float(0.5 * (dx + dy))

        return m, s, dark_frac, edge_strength


means = np.full(len(IMAGE_PATHS), np.nan, dtype=np.float32)
stds = np.full(len(IMAGE_PATHS), np.nan, dtype=np.float32)
dark_fracs = np.full(len(IMAGE_PATHS), np.nan, dtype=np.float32)
edge_strengths = np.full(len(IMAGE_PATHS), np.nan, dtype=np.float32)

if have_images:
    max_workers = min(8, (os.cpu_count() or 4))

    def _worker(i):
        p = IMAGE_PATHS[i]
        try:
            m, s, d, e = image_stats_gray(p, max_side=256)
            return i, m, s, d, e
        except Exception:
            return i, np.nan, np.nan, np.nan, np.nan

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, m, s, d, e in ex.map(_worker, existing_indices, chunksize=32):
            means[i] = m
            stds[i] = s
            dark_fracs[i] = d
            edge_strengths[i] = e

valid_means = means[np.isfinite(means)]
valid_stds = stds[np.isfinite(stds)]

train_img_root = os.path.join(data_root, "train_images")
train_paths = [
    os.path.join(train_img_root, f) for f in train_df["image"].astype(str).tolist()
]
have_train_images = any(os.path.exists(p) for p in train_paths)

mean_thr, std_thr = None, None
train_mean_thr, train_std_thr = None, None

cal_means = None
cal_stds = None
cal_dark = None
cal_edge = None
cal_is_healthy = None
cal_labels_list = None

if have_train_images:
    N_CAL = 1500
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(train_paths))
    rng.shuffle(idx)
    idx = idx[: min(N_CAL, len(train_paths))]

    sampled_df = train_df.iloc[idx].copy()
    sampled_labels = sampled_df["labels"].fillna("").astype(str).tolist()
    sampled_paths = [train_paths[j] for j in idx.tolist()]

    train_means = []
    train_stds = []
    train_dark = []
    train_edge = []
    train_is_healthy = []
    train_labs = []

    max_workers = min(8, (os.cpu_count() or 4))

    def _cal_worker(args):
        p, lab = args
        if not os.path.exists(p):
            return None
        try:
            m, s, d, e = image_stats_gray(p, max_side=256)
            labs = set(parse_labels(lab))
            is_h = 1 if ("healthy" in labs and len(labs) == 1) else 0
            return m, s, d, e, is_h, labs
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for out in ex.map(
            _cal_worker, zip(sampled_paths, sampled_labels), chunksize=32
        ):
            if out is None:
                continue
            m, s, d, e, is_h, labs = out
            train_means.append(m)
            train_stds.append(s)
            train_dark.append(d)
            train_edge.append(e)
            train_is_healthy.append(is_h)
            train_labs.append(labs)

    cal_means = np.asarray(train_means, dtype=np.float32)
    cal_stds = np.asarray(train_stds, dtype=np.float32)
    cal_dark = np.asarray(train_dark, dtype=np.float32)
    cal_edge = np.asarray(train_edge, dtype=np.float32)
    cal_is_healthy = np.asarray(train_is_healthy, dtype=np.int32)
    cal_labels_list = train_labs

    if cal_means.size > 50 and cal_stds.size > 50:
        hm = cal_means[cal_is_healthy == 1]
        hs = cal_stds[cal_is_healthy == 1]
        dm = cal_means[cal_is_healthy == 0]
        ds = cal_stds[cal_is_healthy == 0]

        if hm.size > 10 and hs.size > 10 and dm.size > 10 and ds.size > 10:
            train_mean_thr = float(np.quantile(hm, 0.35))
            train_std_thr = float(np.quantile(hs, 0.65))

if train_mean_thr is not None and train_std_thr is not None:
    mean_thr, std_thr = train_mean_thr, train_std_thr
elif have_images and valid_means.size > 0 and valid_stds.size > 0:
    mean_thr = float(np.quantile(valid_means, 0.60))
    std_thr = float(np.quantile(valid_stds, 0.40))

print(
    "Heuristic thresholds (initial):",
    {
        "mean_thr": mean_thr,
        "std_thr": std_thr,
        "calibrated_on_train": (train_mean_thr is not None),
    },
)




## === cell 11
def f1_set(y_true_set, y_pred_set):
    if y_true_set is None:
        y_true_set = set()
    if y_pred_set is None:
        y_pred_set = set()
    inter = len(y_true_set & y_pred_set)
    denom = len(y_true_set) + len(y_pred_set)
    return (2.0 * inter / denom) if denom > 0 else 0.0


def predict_labels_from_stats(
    m, s, d, e, mean_thr_, std_thr_, topk_disease_, add_complex_mode_
):
    """
    Core logic preserved: decide healthy-like by mean/std thresholds,
    else output a small disease set (+ optional complex).
    We only pass through the extra cheap stats (d,e) for later per-label gating.
    """
    is_healthy_like = (m >= mean_thr_) and (s <= std_thr_)
    if is_healthy_like:
        return {"healthy"}
    labs = set(disease_prev.index[:topk_disease_].tolist())
    if add_complex_mode_ != 0:
        if add_complex_mode_ == 1:
            cond = (m < (mean_thr_ - 8.0)) or (s > (std_thr_ + 10.0))
        else:
            cond = (m < (mean_thr_ - 4.0)) or (s > (std_thr_ + 6.0))
        if cond:
            labs.add("complex")
    labs.discard("healthy")
    return labs


best_params = None

label_gates = None  # dict: label -> (feature_name, threshold, direction)
feature_names = ["d", "e"]  # dark_frac, edge_strength

if cal_means is not None and cal_stds is not None and cal_is_healthy is not None:
    mean_cands = np.quantile(
        cal_means[np.isfinite(cal_means)], [0.30, 0.35, 0.40, 0.45, 0.50]
    ).tolist()
    std_cands = np.quantile(
        cal_stds[np.isfinite(cal_stds)], [0.50, 0.55, 0.60, 0.65, 0.70]
    ).tolist()

    topk_cands = [1, 2, 3]
    complex_mode_cands = [0, 1, 2]

    d_cands = np.quantile(
        cal_dark[np.isfinite(cal_dark)], [0.10, 0.15, 0.20, 0.25]
    ).tolist()
    e_cands = np.quantile(
        cal_edge[np.isfinite(cal_edge)], [0.35, 0.45, 0.55, 0.65]
    ).tolist()

    def apply_gates(base_pred_set, dval, eval_, gates):
        if gates is None:
            return base_pred_set
        out = set(base_pred_set)
        if "healthy" in out:
            return out
        for lab, (fname, thr, direction) in gates.items():
            if lab not in out:
                continue
            x = dval if fname == "d" else eval_
            keep = (x >= thr) if direction == "ge" else (x <= thr)
            if not keep:
                out.discard(lab)
        if len(out) == 0:
            out = {disease_prev.index[0]}
        return out

    best_mean_f1 = -1.0
    for mt in mean_cands:
        for st in std_cands:
            for tk in topk_cands:
                for cm in complex_mode_cands:
                    base_preds = []
                    for m, s, d, e in zip(cal_means, cal_stds, cal_dark, cal_edge):
                        base_preds.append(
                            predict_labels_from_stats(
                                float(m),
                                float(s),
                                float(d),
                                float(e),
                                float(mt),
                                float(st),
                                int(tk),
                                int(cm),
                            )
                        )

                    gates = {}
                    for lab in disease_classes:
                        if lab == "healthy":
                            continue
                        present = np.array(
                            [1 if (lab in p) else 0 for p in base_preds], dtype=np.int32
                        )
                        if present.sum() < 50:
                            continue

                        y_true = np.array(
                            [1 if (lab in y) else 0 for y in cal_labels_list],
                            dtype=np.int32,
                        )

                        best_gate = None
                        best_gate_score = None

                        for fname, cands in [("d", d_cands), ("e", e_cands)]:
                            x = cal_dark if fname == "d" else cal_edge
                            for thr in cands:
                                for direction in ("ge", "le"):
                                    keep = (
                                        (x >= thr) if direction == "ge" else (x <= thr)
                                    )
                                    new_present = present & keep.astype(np.int32)
                                    tp = int(((y_true == 1) & (new_present == 1)).sum())
                                    fp = int(((y_true == 0) & (new_present == 1)).sum())
                                    fn = int(((y_true == 1) & (new_present == 0)).sum())
                                    denom = 2 * tp + fp + fn
                                    f1 = (2 * tp / denom) if denom > 0 else 0.0
                                    if best_gate_score is None or f1 > best_gate_score:
                                        best_gate_score = f1
                                        best_gate = (fname, float(thr), direction)

                        if best_gate is not None:
                            gates[lab] = best_gate

                    f1s = []
                    for m, s, d, e, yset, bp in zip(
                        cal_means,
                        cal_stds,
                        cal_dark,
                        cal_edge,
                        cal_labels_list,
                        base_preds,
                    ):
                        pred = apply_gates(bp, float(d), float(e), gates)
                        f1s.append(f1_set(yset, pred))
                    score = float(np.mean(f1s)) if len(f1s) else 0.0

                    if score > best_mean_f1:
                        best_mean_f1 = score
                        best_params = (float(mt), float(st), int(tk), int(cm), score)
                        label_gates = gates

    if best_params is not None:
        mean_thr, std_thr, TOPK_DISEASE, complex_mode, score = best_params
        base_labels = disease_prev.index[:TOPK_DISEASE].tolist()
        print(
            "Calibrated parameters (mean per-sample F1 on sampled train):",
            {
                "mean_thr": mean_thr,
                "std_thr": std_thr,
                "TOPK_DISEASE": TOPK_DISEASE,
                "complex_mode": complex_mode,
                "mean_f1": score,
                "num_label_gates": (0 if label_gates is None else len(label_gates)),
            },
        )
else:
    print("No train-image calibration data available; using initial thresholds.")
    TOPK_DISEASE = DEFAULT_TOPK_DISEASE
    complex_mode = 1
    label_gates = None


def apply_label_gates(pred_set, dval, eval_, gates):
    if gates is None:
        return pred_set
    out = set(pred_set)
    if "healthy" in out:
        return out
    for lab, (fname, thr, direction) in gates.items():
        if lab not in out:
            continue
        x = dval if fname == "d" else eval_
        keep = (x >= thr) if direction == "ge" else (x <= thr)
        if not keep:
            out.discard(lab)
    if len(out) == 0:
        out = {disease_prev.index[0]}
    return out




## === cell 12
complex_common = float(prev_series.get("complex", 0.0)) >= 0.10

pred_string = []
for i, img_name in enumerate(image_files):
    if (
        have_images
        and np.isfinite(means[i])
        and np.isfinite(stds[i])
        and np.isfinite(dark_fracs[i])
        and np.isfinite(edge_strengths[i])
        and mean_thr is not None
        and std_thr is not None
    ):
        labs = predict_labels_from_stats(
            float(means[i]),
            float(stds[i]),
            float(dark_fracs[i]),
            float(edge_strengths[i]),
            float(mean_thr),
            float(std_thr),
            int(TOPK_DISEASE),
            int(complex_mode if complex_common else 0),
        )
        labs = apply_label_gates(
            labs, float(dark_fracs[i]), float(edge_strengths[i]), label_gates
        )
        if ("healthy" in labs) and (len(labs) > 1):
            labs = {"healthy"}  # safety; should not happen
        pred_string.append(" ".join(sorted(labs)) if len(labs) else "healthy")
    else:
        fallback = disease_prev.index[:DEFAULT_TOPK_DISEASE].tolist()
        pred_string.append(" ".join(fallback) if len(fallback) > 0 else "healthy")

print("Num predictions:", len(pred_string))
print("First 5 predictions:", pred_string[:5])



## === cell 13
pred_string[:10]



## === cell 14
if len(pred_string) < len(image_files):
    pred_string = pred_string + ["healthy"] * (len(image_files) - len(pred_string))
elif len(pred_string) > len(image_files):
    pred_string = pred_string[: len(image_files)]

pred_string = [
    ("healthy" if (s is None or str(s).strip() == "") else str(s).strip())
    for s in pred_string
]



## === cell 15
df = pd.DataFrame({"image": image_files, "labels": pred_string})
assert (
    len(df) == len(image_files) == len(pred_string)
), "Submission lengths do not match."
assert list(df.columns) == [
    "image",
    "labels",
], "Submission columns do not match required format."

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
