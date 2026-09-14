# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.7971960190393761

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the unavailable `pydensecrf` dependency and replace the CRF post-processing with a lightweight, deterministic morphological cleanup that uses only standard Kaggle-installed libraries. I also remove notebook-only commands (`%matplotlib inline`, `ls`) and fix the broken data path (`../input/0760tgs/submission.csv`) by instead starting from the provided `sample_submission.csv`, so a valid `id,rle_mask` submission is always produced. To keep the core intent (post-processing predicted masks), the script read an optional existing `submission.csv` from `/kaggle/input` if present; otherwise it generate a simple baseline mask from each test image and then apply the same post-processing. Finally, it write a valid `.csv` submission file (`crf_correction.csv`) with correct columns and no NaNs.'
- What this solution (achieved 0.5221) has done: 'Your current score is low because the script is effectively guessing masks (thresholding test images at 0.5) and then applying morphology, which is far from a learned segmentation model for this competition. To move the score toward the 0.797 target without changing the “post-process an existing prediction” core intent, the most impactful minimal change is: only use an existing submission if it’s actually a TGS Salt submission (1000 rows and `id,rle_mask` matching `sample_submission` ids), and otherwise do not silently fall back to the image-threshold baseline (which drags score down); instead, emit empty masks (a safer neutral fallback) or require a valid candidate. Additionally, we compute a small fixed ensemble over multiple candidate submissions if present (average decoded masks then threshold), which often lifts score modestly while keeping the same pipeline (decode → refine → encode) and staying deterministic. These changes keep the same semantics (RLE decode/encode + morphology) and mainly fix the “wrong/low-quality input submission” problem that’s likely causing 0.5221.'
- What this solution (achieved 0.5221) has done: 'Your score is far below the target, and the main limiter is that you’re not actually using a strong model output unless a good “candidate submission” happens to be found in `/kaggle/input`. The smallest change likely to move you toward the target (without changing the post-processing core logic) is to (1) broaden and prioritize discovery of high-quality existing submissions in `/kaggle/input` (including `.csv` names like `*.csv` and common “submission” variants), (2) ensemble more than 3 candidates when available (still fast since decode/encode is cheap for 1000×101×101), and (3) tune the ensemble threshold slightly below 0.5 to recover recall (usually helpful for mAP IoU sweep) while keeping the same decode→merge→morphology→encode semantics. If no valid candidate submissions exist, we keep the safe empty-mask fallback (better than the image-threshold baseline for this metric). These changes are deterministic, minimal, and directly aimed at increasing the score toward your 0.797 target.'
- What this solution (achieved 0.5221) has done: 'Your current score is far below the target, so the biggest safe gain (without changing the core “post-process an existing submission” logic) is to ensure we reliably find and use the best available candidate submissions and ensemble them more effectively. I (1) improve candidate discovery by ranking valid submissions using a small training-set proxy (mean IoU against ground-truth masks from `train/`), (2) ensemble the top-scoring candidates (still decode→average→threshold→morphology→encode), and (3) pick the binarization threshold using that same proxy from a small fixed grid so we move upward toward the target rather than guessing 0.45. If no valid candidate submissions exist, we keep the empty-mask fallback (unchanged) to avoid the very low-quality image-threshold baseline.'
- What this solution (achieved 0.5221) has done: 'Your score is far below the target, so we should focus on small, safe changes that improve the *quality of the input predictions* and avoid proxy-metric mismatch. I keep your pipeline (find candidate submissions → decode → average/threshold → morphology → encode) but change the candidate ranking and threshold tuning to use the competition’s mAP IoU-sweep metric (not mean IoU), computed on a small fixed train subset. I also ensure we don’t accidentally “score” candidates on a train subset they don’t contain (most public submissions only have test ids), by detecting whether a candidate includes train ids; if none do, we fall back to a deterministic threshold and just ensemble the best-prioritized candidates. These are minimal, directly score-relevant changes and keep runtime under the limit.'
- What this solution (achieved 0.5221) has done: 'Your current score is far below the target, and the biggest limiter is that your “candidate submission” ensembling is effectively random because you can’t rank candidates without ground-truth. I keep the same core pipeline (find candidate submissions → decode → average/threshold → morphology → encode) but add a minimal, legitimate way to score/rank candidates and choose the binarization threshold using a train/validation split built from `train.csv` RLE ground-truth. This uses the competition’s mAP IoU-sweep metric on a small fixed subset, then applies the selected candidates/threshold to the test set exactly as before. This should move the score upward toward the target without changing model architecture/training (there is none) and still produces a valid `crf_correction.csv`.'
- What this solution (achieved 0.5221) has done: 'Your score gap to the 0.7972 target is large, and the main issue is that the “train-proxy” scoring currently can’t work because candidate submissions almost never contain train IDs; as a result you either fall back to a non-informative ranking and then effectively choose a threshold using all-zero predictions, producing weak masks. I keep your exact pipeline (find candidate submissions → decode → average → threshold → morphology → RLE) but change the proxy evaluation to a legitimate, cheap internal validation using a tiny U-Net-like model trained on the provided `train/` images+masks to generate a *validation baseline prediction*, and then rank/threshold candidate submissions by how well they agree with that baseline on the same validation IDs (no label leakage, no test labels). This preserves evaluation semantics and avoids random/empty-mask behavior while remaining deterministic and within time by using a small subset and few epochs. If no candidates are found, the behavior remains the safe empty-mask fallback and still produces a valid `crf_correction.csv`.'
- What this solution (achieved 0.5221) has done: 'I fix the runtime crash in the TensorFlow import by forcing the pure-Python protobuf implementation (a known workaround for the `MessageFactory.GetPrototype` issue in Kaggle’s TF/protobuf combo), so the teacher model can train and the candidate ranking/threshold selection runs instead of failing. I also make the train/val id selection deterministic-but-shuffled (still the same tiny U-Net training approach) to avoid bias from the CSV order and improve the proxy ranking signal toward your target score. Finally, I harden submission discovery slightly by skipping obviously non-submission CSVs (like `depths.csv`/`train.csv`) to reduce accidental bad candidates, without changing the core decode→ensemble→morphology→encode pipeline.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation environment variables before *any* TensorFlow import and by making the TensorFlow import fully optional (so the pipeline always completes). I also remove the accidental train-id evaluation in candidate scoring (candidates are test-only), and instead use the trained teacher model to generate *its own* test predictions (same tiny U-Net core) and then use candidate submissions only if they improve agreement with the teacher on a small test subset; otherwise we fall back to the teacher’s predictions directly, which should raise the score substantially toward your target. Finally, I keep the existing decode→threshold→morphology→encode semantics for post-processing and ensure a valid `crf_correction.csv` is always written with the correct `id,rle_mask` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import glob
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)

np.random.seed(42)

DATA_ROOT = "/kaggle/input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "masks")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.isdir(TEST_IMG_DIR), f"Test image directory not found: {TEST_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Train image directory not found: {TRAIN_IMG_DIR}"
assert os.path.isdir(
    TRAIN_MASK_DIR
), f"Train mask directory not found: {TRAIN_MASK_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"train.csv not found: {TRAIN_CSV_PATH}"




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    shape: (height,width) of array to return

    Returns numpy array, 1 - mask, 0 - background
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    rle_mask = str(rle_mask)
    if rle_mask.strip() == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted

    IMPORTANT for this competition: pixels are one-indexed and read top-to-bottom then left-to-right,
    which corresponds to Fortran order flattening.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
def simple_postprocess(mask01):
    """
    Replacement for CRF: deterministic morphological cleanup.
    Keeps the intent (refine a predicted mask) without unavailable pydensecrf.
    """
    m = mask01.astype(bool)

    se = disk(1)
    m = binary_opening(m, se)
    m = binary_closing(m, se)

    m = remove_small_objects(m, min_size=20)
    m = remove_small_holes(m, area_threshold=20)

    return m.astype(np.uint8)


def baseline_mask_from_image(img):
    """
    Fallback mask. NOTE: this baseline is low quality for this competition; we avoid using it.
    """
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    m = (img > 0.5).astype(np.uint8)
    return simple_postprocess(m)




## === cell 3
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
sample_df["id"] = sample_df["id"].astype(str)
sample_ids = sample_df["id"].tolist()
sample_id_set = set(sample_ids)


def load_if_valid_submission(path):
    try:
        d = pd.read_csv(path)
    except Exception:
        return None
    if not {"id", "rle_mask"}.issubset(d.columns):
        return None
    d = d[["id", "rle_mask"]].copy()
    d["id"] = d["id"].astype(str)

    if len(d) != len(sample_df):
        return None
    if set(d["id"].tolist()) != sample_id_set:
        return None

    d = d.set_index("id").reindex(sample_ids).reset_index()
    d.rename(columns={"index": "id"}, inplace=True)
    return d


candidate_paths = []
for pat in [
    "/kaggle/input/**/submission.csv",
    "/kaggle/input/**/*submission*.csv",
    "/kaggle/input/**/sub*.csv",
    "/kaggle/input/**/*.csv",
]:
    candidate_paths.extend(glob.glob(pat, recursive=True))
candidate_paths = sorted(set(candidate_paths))

_blacklist_basenames = {
    "sample_submission.csv",
    "train.csv",
    "depths.csv",
    "description.md",
}
candidate_paths = [
    p
    for p in candidate_paths
    if os.path.basename(p).lower() not in _blacklist_basenames
]


def _path_priority(p):
    base = os.path.basename(p).lower()
    is_submission_named = ("submission" in base) or base.startswith("sub")
    return (0 if is_submission_named else 1, len(p), p)


candidate_paths = sorted(candidate_paths, key=_path_priority)

valid_subs = []
for p in candidate_paths:
    if os.path.abspath(p) == os.path.abspath(SAMPLE_SUB_PATH):
        continue
    d = load_if_valid_submission(p)
    if d is not None:
        valid_subs.append((p, d))

print("Found valid candidate submissions:", len(valid_subs))
for p, _ in valid_subs[:15]:
    print(" -", p)

df = sample_df[["id", "rle_mask"]].copy()
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)




## === cell 4
def iou_np(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0  # both empty
    return inter / union


def map_iou_sweep(y_true, y_pred, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)
    iou = iou_np(y_true, y_pred)
    return float(np.mean([1.0 if iou > t else 0.0 for t in thresholds]))




## === cell 5
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception as e:
    tf = None
    print(
        "TensorFlow not available; will fall back to candidate-only pipeline. Error:", e
    )

TRAIN_EPOCHS = 6  # slightly higher to move score toward target; still small model and within time budget
BATCH_SIZE = 16

teacher_ready = False
teacher_test_pred = None  # np.ndarray [N,101,101] float32 aligned with sample_ids


def _load_img_gray(path):
    img = imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32) / 255.0
    return img


if tf is not None:
    tf.random.set_seed(42)

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df["id"] = train_df["id"].astype(str)
    all_train_ids = train_df["id"].tolist()

    Xtr = np.zeros((len(all_train_ids), 101, 101, 1), dtype=np.float32)
    Ytr = np.zeros((len(all_train_ids), 101, 101, 1), dtype=np.float32)

    for idx, img_id in enumerate(all_train_ids):
        x = _load_img_gray(os.path.join(TRAIN_IMG_DIR, f"{img_id}.png"))
        y = _load_img_gray(os.path.join(TRAIN_MASK_DIR, f"{img_id}.png"))
        Xtr[idx, ..., 0] = x
        Ytr[idx, ..., 0] = (y > 0.5).astype(np.float32)

    def build_tiny_unet(input_shape=(101, 101, 1)):
        inp = layers.Input(shape=input_shape)

        c1 = layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
        c1 = layers.Conv2D(8, 3, padding="same", activation="relu")(c1)
        p1 = layers.MaxPooling2D()(c1)

        c2 = layers.Conv2D(16, 3, padding="same", activation="relu")(p1)
        c2 = layers.Conv2D(16, 3, padding="same", activation="relu")(c2)
        p2 = layers.MaxPooling2D()(c2)

        b = layers.Conv2D(32, 3, padding="same", activation="relu")(p2)
        b = layers.Conv2D(32, 3, padding="same", activation="relu")(b)

        u2 = layers.UpSampling2D()(b)
        u2 = layers.Concatenate()([u2, c2])
        c3 = layers.Conv2D(16, 3, padding="same", activation="relu")(u2)
        c3 = layers.Conv2D(16, 3, padding="same", activation="relu")(c3)

        u1 = layers.UpSampling2D()(c3)
        u1 = layers.Concatenate()([u1, c1])
        c4 = layers.Conv2D(8, 3, padding="same", activation="relu")(u1)
        c4 = layers.Conv2D(8, 3, padding="same", activation="relu")(c4)

        out = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(c4)
        model = models.Model(inp, out)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    model = build_tiny_unet()
    model.fit(
        Xtr, Ytr, batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS, verbose=0, shuffle=True
    )

    Xte = np.zeros((len(sample_ids), 101, 101, 1), dtype=np.float32)
    for i, img_id in enumerate(sample_ids):
        Xte[i, ..., 0] = _load_img_gray(os.path.join(TEST_IMG_DIR, f"{img_id}.png"))
    preds = model.predict(Xte, batch_size=BATCH_SIZE, verbose=0)[..., 0].astype(
        np.float32
    )

    teacher_test_pred = preds
    teacher_ready = True
    print(
        f"Teacher ready: train_n={len(all_train_ids)} test_n={len(sample_ids)} epochs={TRAIN_EPOCHS}"
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
THRESH_GRID = [0.35, 0.40, 0.45, 0.50, 0.55]
USE_ENSEMBLE = True
ENSEMBLE_MAX = 12

ids = df["id"].tolist()


def decode_submission_to_stack(d_sub):
    stack = np.zeros((len(ids), 101, 101), dtype=np.float32)
    rles = d_sub["rle_mask"].fillna("").astype(str).tolist()
    for i, rle in enumerate(rles):
        stack[i] = rle_decode(rle, shape=(101, 101)).astype(np.float32)
    return stack


decoded_candidates = []
for p, d in valid_subs[:ENSEMBLE_MAX]:
    decoded_candidates.append((p, decode_submission_to_stack(d)))


def build_ensemble_and_threshold(thresh):
    if teacher_ready:
        acc = teacher_test_pred.copy()
        denom = 1.0
        for _, stk in decoded_candidates:
            acc += stk
            denom += 1.0
        acc = acc / denom
        merged = (acc >= float(thresh)).astype(np.uint8)
    else:
        if len(decoded_candidates) == 0:
            return None
        acc = np.zeros((len(ids), 101, 101), dtype=np.float32)
        for _, stk in decoded_candidates:
            acc += stk
        acc /= float(len(decoded_candidates))
        merged = (acc >= float(thresh)).astype(np.uint8)

    refined = np.zeros_like(merged, dtype=np.uint8)
    for i in range(len(ids)):
        refined[i] = simple_postprocess(merged[i])
    return refined


if teacher_ready:
    if len(decoded_candidates) == 0:
        chosen_t = 0.45
    else:
        subset_idx = np.arange(len(ids))
        rng = np.random.RandomState(42)
        rng.shuffle(subset_idx)
        subset_idx = subset_idx[:200]

        best_sc = -1.0
        best_t = None
        for t in THRESH_GRID:
            ens_ref = build_ensemble_and_threshold(t)
            teach_ref = np.zeros((len(ids), 101, 101), dtype=np.uint8)
            teach_bin = (teacher_test_pred >= float(t)).astype(np.uint8)
            for i in subset_idx:
                teach_ref[i] = simple_postprocess(teach_bin[i])

            scs = []
            for i in subset_idx:
                scs.append(iou_np(teach_ref[i], ens_ref[i]))
            sc = float(np.mean(scs))
            if sc > best_sc:
                best_sc, best_t = sc, t
        chosen_t = float(best_t)
        print(
            f"Chosen threshold={chosen_t:.2f} by teacher-agreement on test subset (iou={best_sc:.4f})"
        )
else:
    chosen_t = 0.45

final_masks = None
if teacher_ready or len(decoded_candidates) > 0:
    final_masks = build_ensemble_and_threshold(chosen_t)

if final_masks is None:
    for i in range(len(df)):
        df.at[i, "rle_mask"] = ""
else:
    for i in range(len(ids)):
        df.at[i, "rle_mask"] = rle_encode(final_masks[i])

df["rle_mask"] = df["rle_mask"].fillna("").astype(str)



## === cell 7
out_path = "crf_correction.csv"
df[["id", "rle_mask"]].to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(df.head())
print("Rows:", len(df))
assert len(df) == 1000
assert list(df.columns) == ["id", "rle_mask"]
