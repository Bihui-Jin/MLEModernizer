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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8675437886331567

# 6. Current score

0.06219

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/compatibility issues that prevent TensorFlow/Keras from importing cleanly (the protobuf `MessageFactory.GetPrototype` error) by removing the unused `tensorflow.set_random_seed` import and using `tf.random.set_seed` instead. I also make the path setup robust to the Kaggle dataset folder structure and remove the hard-coded `../input/densenetmulti` directory listing so the notebook doesn’t crash before defining `INPUT_FOLDER`. Finally, I ensure `Sequential` is imported reliably via `tensorflow.keras`, fix deprecated `np.int`, and make sure `submission.csv` is always written with the required `id_code,diagnosis` columns.'
- What this solution (achieved -0.015) has done: 'I fix the TensorFlow/protobuf import crash by avoiding TensorFlow/Keras entirely (it’s not needed if the external weights file is missing anyway) and switching to a pure-OpenCV baseline that runs in this Kaggle environment. I also remove the hard dependency on the unattached `densenetmulti` dataset (currently causing a `FileNotFoundError`) and ensure predictions are generated deterministically for every test image. Finally, I keep your existing image preprocessing functions (crop/color/resize) and produce a valid `submission.csv` with exactly `id_code,diagnosis` columns, which move the score up from 0.0 (no valid working model) toward a reasonable baseline.'
- What this solution (achieved 0.03102) has done: 'Your current baseline is scoring poorly mainly because the handcrafted score is only very weakly aligned with the ordinal DR labels, and the “match the class histogram via quantiles” calibration does not optimize the quadratic weighted kappa objective. To move the score up toward the target while preserving your core approach (handcrafted continuous score → thresholds → integer labels), I keep your feature extraction exactly the same but replace the quantile thresholds with thresholds optimized on a small validation split using QWK. This is a minimal change that directly targets the competition metric and usually yields a large jump from near-random kappa without introducing new models or training loops. I also ensure determinism (fixed split/seed) and keep the same submission format/path.'
- What this solution (achieved 0.03859) has done: 'Your current pipeline is logically valid but the single handcrafted severity score is only weakly correlated with the ordinal labels, so even well-tuned thresholds can’t recover much kappa. To move the score substantially upward toward the target while preserving the same core approach (handcrafted continuous scores → thresholding into 0–4), I keep your preprocessing and base score intact and add a few additional inexpensive, deterministic handcrafted signals (red-channel intensity, saturation, vessel/lesion-like morphology, and edge density at a second scale), then tune thresholds on a validation split exactly as you already do. This is still the same training approach (no ML model, no new loops beyond threshold tuning), but it aligns the continuous score better with DR severity which typically yields a large kappa jump from near-random. I also keep the same I/O paths and ensure the submission format remains unchanged.'
- What this solution (achieved 0.05381) has done: 'Your current approach is bottlenecked by weak correlation between the handcrafted score and the ordinal labels, so the smallest meaningful way to move toward the target is to keep your same “handcrafted score → tuned thresholds → label” pipeline but make the score more DR-specific using a couple of standard, cheap lesion/vessel proxies. Concretely, I (1) mask out the black background so global means/stds aren’t dominated by non-retina pixels, and (2) add two additional signals computed on the green channel: a CLAHE-enhanced top-hat (bright lesions) and an additional black-hat at a second scale (vessels/hemorrhage-like), then keep the same threshold tuning procedure. This preserves your core logic and runtime profile (still simple OpenCV features + grid threshold tuning) while typically giving a large QWK gain vs. background-sensitive global stats. The submission writing and paths remain unchanged.'
- What this solution (achieved 0.04125) has done: 'Your current gap to the target is large (0.05381 vs 0.8675), so the most direct minimal improvement that preserves your core pipeline is to keep the same handcrafted continuous score + threshold tuning, but make the score correlate better with DR severity by adding one additional standard lesion proxy: a CLAHE-enhanced “exudate fraction” (count of very bright top-hat responses within the retina mask). Then, without changing the tuning approach, I tune thresholds on out-of-fold predictions (5 folds) instead of a single 80/20 split to reduce overfitting of thresholds to one validation subset, which typically increases public QWK while keeping the same logic. I also ensure the train/test preprocessing is identical and deterministic, and keep the same submission writing and paths.'
- What this solution (achieved 0.06093) has done: 'Your current score is far below the target, so we should increase QWK while keeping your same handcrafted-score → tuned-thresholds pipeline intact. The largest low-risk gain is to tune thresholds more robustly: instead of tuning once on the full training scores (which can overfit thresholds), we compute out-of-fold (OOF) continuous scores (same scores you already compute) and then optimize thresholds on OOF predictions, plus we add a tiny local coordinate-descent refinement after the grid search. We also make folds stratified by diagnosis to stabilize class balance per fold, which usually improves QWK calibration without changing the model/feature logic. Finally, we keep I/O and submission schema identical and ensure determinism.'
- What this solution (achieved 0.07749) has done: 'Your current score (0.06093) is far below the target (0.8675), so we should increase QWK while preserving your exact “handcrafted continuous score → tuned thresholds → integer labels” pipeline. The largest minimal win here is to make the OOF threshold tuning better match Kaggle’s metric by tuning on *rounded* predictions derived from a monotonic mapping, rather than only moving raw thresholds on the original score scale; we do this by adding a simple 1D monotonic calibration (isotonic regression) on OOF scores before threshold search (still the same semantics: continuous score → thresholds). This keeps the same feature extraction and label mapping, but typically boosts ordinal agreement substantially because it corrects non-linear score-to-severity distortion. We keep determinism, keep runtime within limits, and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.07749) has done: 'Your current score is far below the target, so we should increase QWK while keeping your exact “handcrafted continuous score → isotonic calibration → tuned thresholds → integer labels” pipeline intact. The biggest minimal fix is to make the isotonic regression correct: your current PAV implementation deletes `x_s` values and then uses them as bin boundaries, which breaks the mapping and can seriously hurt calibration/QWK. I replace it with a standard, deterministic PAV that returns proper stepwise-constant bins (with correct boundary handling), and keep everything else (features, scoring, threshold search) the same. This should move the score meaningfully upward without changing core modeling semantics or adding new dependencies.'
- What this solution (achieved 0.07749) has done: 'I keep your handcrafted feature extraction and the overall “continuous score → monotonic calibration → thresholds → 0–4 labels” pipeline unchanged, and only fix the isotonic calibration step so it is mathematically correct and stable. Your current PAV implementation merges blocks by repeatedly deleting array entries, which can subtly mis-handle block boundaries/means and hurt QWK; replacing it with a standard, deterministic PAV using block lists preserves the same semantics but yields a better monotonic mapping. With a correct mapping, the same OOF threshold tuning you already do be optimizing on a more faithful calibrated scale, which should move QWK upward toward the target. All I/O paths and the submission format remain identical, and the script still writes `submission.csv`.'
- What this solution (achieved 0.07749) has done: 'Your score gap to the target is still very large, so we should push QWK upward while keeping your exact pipeline (handcrafted continuous score → monotonic calibration → thresholds → 0–4 labels). The biggest minimal, metric-aligned fix is that the isotonic regression must be fit on *out-of-fold (OOF) calibrated predictions*; your current code inadvertently fits isotonic on in-sample scores (no true OOF), which tends to overfit and can degrade public QWK. I keep your feature extraction and threshold tuning unchanged, but change the OOF generation to be genuine: for each fold, fit isotonic on the other folds and predict the held-out fold, then tune thresholds on those OOF-calibrated values. Finally, I fit the final isotonic mapping on the full train set for test-time calibration, so train/test use consistent calibration.'
- What this solution (achieved 0.06219) has done: 'We keep your exact handcrafted feature extraction and the same “continuous score → monotonic calibration → tuned thresholds → integer labels” pipeline, but fix a key mismatch: your isotonic regressor is trained to output values in [0,4], yet your thresholds are optimized on that scale while `score_to_label` treats inputs as continuous; this can create overly “compressed” predictions and weak separation. The minimal, metric-aligned change is to fit isotonic to predict the *probability of being above each class boundary* (ordinal cumulative link style) while still remaining a 1D monotonic mapping of your same score, then convert that to an expected class score in [0,4] and tune thresholds exactly as you already do. This keeps the core logic intact (same features, same threshold tuning/search loops), but typically increases QWK substantially because the calibrated value better preserves ordinal spacing. Submission writing, paths, determinism, and runtime constraints remain unchanged.'
- What this solution (achieved 0.06219) has done: 'Your current pipeline is valid but the metric gap is huge, so the smallest score-moving change is to make the OOF calibration *actually* consistent and stable: right now the fold train/val split is created via `setdiff1d` without guaranteeing disjointness/uniqueness and can silently mis-handle indices, which hurts the ordinal isotonic fit and thus QWK. I keep your exact feature extraction and the same “continuous score → ordinal isotonic (cumulative) → thresholds → labels” semantics, but (1) build proper stratified folds with disjoint indices, (2) generate `tr_idx` via a boolean mask (not `setdiff1d`), and (3) clip/clean any non-finite scores before calibration/tuning so a few bad images can’t collapse the monotonic mapping. These are minimal, directly metric-aligned robustness fixes that should move QWK upward toward the target without changing the core approach or adding dependencies, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.06219) has done: 'Your current score (0.06219) is far below the target (0.8675), so we should increase QWK while keeping your handcrafted-score → ordinal isotonic calibration → tuned thresholds pipeline intact. The biggest minimal, metric-aligned issue is that your `fit_isotonic_1d` currently uses each block’s **rightmost x** as the step boundary, which creates incorrect mappings when there are duplicate/flat x regions; this can severely distort the ordinal probability curves. I replace it with a standard PAV implementation that tracks **block start/end indices** and builds boundaries at midpoints between blocks (stable with ties), while keeping the same semantics and callers. I also add a tiny safety sort+tie handling in the transform to ensure monotonic output for equal x, without changing any feature extraction, threshold tuning loops, or submission format.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import cv2

np.random.seed(42)

IMG_DIM = 256

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"

COMP_FOLDER = os.path.join(BASE_INPUT, "aptos2019-blindness-detection")
if os.path.exists(COMP_FOLDER):
    INPUT_FOLDER = COMP_FOLDER + "/"
else:
    INPUT_FOLDER = BASE_INPUT + "/"

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images") + "/"
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images") + "/"

print("BASE_INPUT:", BASE_INPUT)
print("INPUT_FOLDER:", INPUT_FOLDER)
print("Has test_images:", os.path.exists(TEST_IMAGES_DIR), TEST_IMAGES_DIR)
print("Has train_images:", os.path.exists(TRAIN_IMAGES_DIR), TRAIN_IMAGES_DIR)

TRAIN_CSV = os.path.join(INPUT_FOLDER, "train.csv")
TEST_CSV = os.path.join(INPUT_FOLDER, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_FOLDER, "sample_submission.csv")

if not os.path.exists(TRAIN_CSV):
    alt = os.path.join(BASE_INPUT, "aptos2019-blindness-detection", "train.csv")
    if os.path.exists(alt):
        TRAIN_CSV = alt
if not os.path.exists(TEST_CSV):
    alt = os.path.join(BASE_INPUT, "aptos2019-blindness-detection", "test.csv")
    if os.path.exists(alt):
        TEST_CSV = alt
if not os.path.exists(SAMPLE_SUB):
    alt = os.path.join(
        BASE_INPUT, "aptos2019-blindness-detection", "sample_submission.csv"
    )
    if os.path.exists(alt):
        SAMPLE_SUB = alt

print("TRAIN_CSV:", TRAIN_CSV, os.path.exists(TRAIN_CSV))
print("TEST_CSV:", TEST_CSV, os.path.exists(TEST_CSV))
print("SAMPLE_SUB:", SAMPLE_SUB, os.path.exists(SAMPLE_SUB))



## === cell 1
test_df = pd.read_csv(TEST_CSV)
test_df["id_code"] = test_df["id_code"].astype(str).apply(lambda x: x + ".png")
test_df.head()




## === cell 2
def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    thresh = 5
    rowMaxes = gray.max(axis=1)

    top = 0
    while top < len(rowMaxes) and rowMaxes[top] < thresh:
        top += 1

    bottom = len(rowMaxes) - 1
    while bottom >= 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    if bottom <= top:
        return bgr

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1

    right = len(middleRow) - 1
    while right >= 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left

    if height < 100 or width < 100 or right <= left:
        return bgr

    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    modified = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return modified


def processImageBgrToRgb(bgr):
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)
    return modified




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str).apply(lambda x: x + ".png")

print("train rows:", len(train_df), "test rows:", len(test_df))
print(train_df.head())


def severity_score_from_bgr(bgr):
    """
    Keep the same core approach (handcrafted continuous score -> tuned thresholds).

    Minimal score improvement aimed at QWK:
    - Keep your existing retina masking + morphology signals.
    - Add ONE extra DR-specific signal: "exudate fraction" estimated as the fraction
      of very bright top-hat responses (after CLAHE on green) within the retina mask.
    """
    rgb = processImageBgrToRgb(bgr)

    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray_u8 > 10).astype(np.uint8)
    mask_mean = float(mask.mean())  # fraction of pixels considered retina

    g_u8 = gray_u8
    g = g_u8.astype(np.float32) / 255.0

    if mask_mean > 0.01:
        g_masked = g[mask.astype(bool)]
        mean_intensity = float(g_masked.mean())
        std_intensity = float(g_masked.std())
    else:
        mean_intensity = float(g.mean())
        std_intensity = float(g.std())

    edges = cv2.Canny(g_u8, 30, 90)
    if mask_mean > 0.01:
        edge_density = float(edges[mask.astype(bool)].mean() / 255.0)
    else:
        edge_density = float(edges.mean() / 255.0)

    edges2 = cv2.Canny(g_u8, 10, 60)
    if mask_mean > 0.01:
        edge_density2 = float(edges2[mask.astype(bool)].mean() / 255.0)
    else:
        edge_density2 = float(edges2.mean() / 255.0)

    base_score = (
        (0.9 * edge_density) + (0.6 * std_intensity) + (0.2 * (1.0 - mean_intensity))
    )

    r = rgb[:, :, 0].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    if mask_mean > 0.01:
        m = mask.astype(bool)
        r_mean = float(r[m].mean())
        r_std = float(r[m].std())
        s_mean = float(s[m].mean())
        s_std = float(s[m].std())
    else:
        r_mean = float(r.mean())
        r_std = float(r.std())
        s_mean = float(s.mean())
        s_std = float(s.std())

    green = rgb[:, :, 1]

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat1 = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq = clahe.apply(green)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(green_eq, cv2.MORPH_TOPHAT, k2)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat2 = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k3)

    def masked_mean_std(img_u8, mask_u8):
        if mask_mean > 0.01:
            vals = img_u8[mask_u8.astype(bool)].astype(np.float32) / 255.0
            return float(vals.mean()), float(vals.std())
        vals = img_u8.astype(np.float32) / 255.0
        return float(vals.mean()), float(vals.std())

    bh1_mean, bh1_std = masked_mean_std(blackhat1, mask)
    th_mean, th_std = masked_mean_std(tophat, mask)
    bh2_mean, bh2_std = masked_mean_std(blackhat2, mask)

    if mask_mean > 0.01:
        th_vals = tophat[mask.astype(bool)]
    else:
        th_vals = tophat.reshape(-1)
    if th_vals.size > 0:
        thr_tail = float(np.percentile(th_vals.astype(np.float32), 98.5))
        exudate_frac = float((th_vals > thr_tail).mean())
    else:
        exudate_frac = 0.0

    score = (
        1.00 * base_score
        + 0.25 * (1.0 - r_mean)
        + 0.20 * r_std
        + 0.15 * s_mean
        + 0.10 * s_std
        + 0.30 * bh1_mean
        + 0.12 * bh1_std
        + 0.25 * edge_density2
        + 0.45 * th_mean
        + 0.18 * th_std
        + 0.18 * bh2_mean
        + 0.08 * bh2_std
        + 0.55 * exudate_frac
        - 0.10 * max(0.0, 0.10 - mask_mean)
    )
    return float(score)


def compute_scores_for_df(df, images_dir, max_items=None):
    scores = np.zeros(len(df), dtype=np.float32)
    n = len(df) if max_items is None else min(len(df), max_items)
    for i in range(n):
        fn = df.iloc[i].id_code
        bgr = cv2.imread(os.path.join(images_dir, fn))
        if bgr is None:
            scores[i] = 0.0
        else:
            try:
                scores[i] = severity_score_from_bgr(bgr)
            except Exception:
                scores[i] = 0.0
        if (i + 1) % 500 == 0:
            gc.collect()
            print("processed", i + 1, "images")
    if n < len(df):
        scores = scores[:n]
    return scores


def score_to_label(scores, thr):
    return np.digitize(scores, thr, right=False).astype(int)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / float((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


train_scores = compute_scores_for_df(train_df, TRAIN_IMAGES_DIR)
y_train = train_df["diagnosis"].astype(int).values

train_scores = np.asarray(train_scores, dtype=np.float32)
finite_mask = np.isfinite(train_scores)
if not finite_mask.all():
    med = float(np.median(train_scores[finite_mask])) if finite_mask.any() else 0.0
    train_scores[~finite_mask] = med

print("Computed train_scores:", train_scores.shape, "y_train:", y_train.shape)




## === cell 4
def enforce_increasing(thr, eps=1e-6):
    thr = np.array(thr, dtype=np.float64)
    for k in range(1, len(thr)):
        if thr[k] <= thr[k - 1] + eps:
            thr[k] = thr[k - 1] + eps
    return thr


def tune_thresholds_grid(s_val, y_val, init_thr, n_steps=31, shrink=0.50):
    thr = enforce_increasing(init_thr)
    best = quadratic_weighted_kappa(y_val, score_to_label(s_val, thr), n_classes=5)

    s_std = float(np.std(s_val)) + 1e-6
    base_span = shrink * s_std

    for _ in range(2):  # keep runtime low
        for t in range(4):
            center = thr[t]
            candidates = np.linspace(center - base_span, center + base_span, n_steps)
            best_local_thr = thr[t]
            best_local = best
            for c in candidates:
                cand = thr.copy()
                cand[t] = c
                cand = enforce_increasing(cand)
                kappa = quadratic_weighted_kappa(
                    y_val, score_to_label(s_val, cand), n_classes=5
                )
                if kappa > best_local:
                    best_local = kappa
                    best_local_thr = cand[t]
                    thr = cand
                    best = kappa
            thr[t] = best_local_thr
        base_span *= 0.5
    return enforce_increasing(thr), float(best)


def make_stratified_folds(y, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=np.int64)

    folds = [[] for _ in range(n_splits)]
    for c in range(5):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        parts = np.array_split(idx_c, n_splits)
        for k in range(n_splits):
            folds[k].append(parts[k])

    folds = [
        np.concatenate(f) if len(f) else np.array([], dtype=np.int64) for f in folds
    ]
    for k in range(n_splits):
        rng.shuffle(folds[k])

    all_idx = np.concatenate(folds) if n_splits > 0 else np.array([], dtype=np.int64)
    assert len(all_idx) == len(y)
    assert len(np.unique(all_idx)) == len(y)

    return [f.astype(np.int64) for f in folds]


def refine_thresholds_coordinate(s_val, y_val, thr_init, iters=20):
    thr = enforce_increasing(thr_init)
    best = quadratic_weighted_kappa(y_val, score_to_label(s_val, thr), n_classes=5)

    s_std = float(np.std(s_val)) + 1e-6
    step = 0.05 * s_std

    for _ in range(iters):
        improved = False
        for t in range(4):
            for delta in (-step, step):
                cand = thr.copy()
                cand[t] += delta
                cand = enforce_increasing(cand)
                kappa = quadratic_weighted_kappa(
                    y_val, score_to_label(s_val, cand), n_classes=5
                )
                if kappa > best:
                    thr = cand
                    best = kappa
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return enforce_increasing(thr), float(best)


def fit_isotonic_1d(x, y, y_min=0.0, y_max=1.0):
    """
    Change (metric-relevant, minimal): make PAV isotonic regression mathematically correct
    and stable with duplicate x values by constructing step boundaries between blocks.
    This improves ordinal calibration quality (and thus QWK) without changing the overall
    pipeline (still 1D monotonic calibration).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    x_s = x[order]
    y_s = y[order]
    n = len(x_s)

    starts = []
    ends = []
    sum_y = []
    cnt = []

    for i in range(n):
        starts.append(i)
        ends.append(i)
        sum_y.append(float(y_s[i]))
        cnt.append(1.0)

        while len(sum_y) >= 2:
            avg_prev = sum_y[-2] / cnt[-2]
            avg_last = sum_y[-1] / cnt[-1]
            if avg_prev <= avg_last + 1e-12:
                break
            ends[-2] = ends[-1]
            sum_y[-2] += sum_y[-1]
            cnt[-2] += cnt[-1]
            starts.pop()
            ends.pop()
            sum_y.pop()
            cnt.pop()

    block_avg = np.clip(
        (np.array(sum_y, dtype=np.float64) / np.array(cnt, dtype=np.float64)),
        y_min,
        y_max,
    ).astype(np.float32)

    m = len(block_avg)
    right_bounds = np.empty(m, dtype=np.float64)
    for b in range(m):
        if b == m - 1:
            right_bounds[b] = np.inf
        else:
            xb = x_s[ends[b]]
            xn = x_s[starts[b + 1]]
            right_bounds[b] = 0.5 * (xb + xn)

    def transform(x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        idx = np.searchsorted(right_bounds, x_new, side="right")
        idx = np.clip(idx, 0, m - 1)
        return block_avg[idx]

    return transform


def fit_ordinal_isotonic_expected_score(x, y, n_classes=5):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    iso_ge = []
    for k in range(1, n_classes):
        target = (y >= k).astype(np.float32)
        iso_ge.append(fit_isotonic_1d(x, target, y_min=0.0, y_max=1.0))

    def transform(x_new):
        x_new = np.asarray(x_new, dtype=np.float32)
        probs = np.zeros((len(x_new), n_classes - 1), dtype=np.float32)
        for i, fn in enumerate(iso_ge):
            probs[:, i] = fn(x_new).astype(np.float32)
        for i in range(1, n_classes - 1):
            probs[:, i] = np.minimum(probs[:, i - 1], probs[:, i])
        expected = probs.sum(axis=1)  # in [0,4]
        return expected.astype(np.float32)

    return transform


def tune_thresholds_oof(scores, y, n_splits=5, seed=42):
    folds = make_stratified_folds(y, n_splits=n_splits, seed=seed)
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)

    finite_mask = np.isfinite(scores)
    if not finite_mask.all():
        med = float(np.median(scores[finite_mask])) if finite_mask.any() else 0.0
        scores = scores.copy()
        scores[~finite_mask] = med

    oof_cal = np.zeros(len(y), dtype=np.float32)
    oof_mask = np.zeros(len(y), dtype=np.uint8)

    for fi, val_idx in enumerate(folds):
        tr_mask = np.ones(len(y), dtype=bool)
        tr_mask[val_idx] = False
        tr_idx = np.where(tr_mask)[0]

        iso_fold = fit_ordinal_isotonic_expected_score(
            scores[tr_idx], y[tr_idx], n_classes=5
        )
        oof_cal[val_idx] = iso_fold(scores[val_idx]).astype(np.float32)
        oof_mask[val_idx] = 1

        print("calibrated fold", fi + 1, "/", len(folds), "val size", len(val_idx))

    assert int(oof_mask.sum()) == len(y)

    hist = np.bincount(y, minlength=5).astype(np.float64)
    cum = np.cumsum(hist) / max(hist.sum(), 1.0)
    quantiles = [cum[0], cum[1], cum[2], cum[3]]
    init_thr = np.quantile(oof_cal, quantiles).astype(np.float32)
    for k in range(1, len(init_thr)):
        if init_thr[k] <= init_thr[k - 1]:
            init_thr[k] = init_thr[k - 1] + 1e-6

    thr_grid, oof_kappa_grid = tune_thresholds_grid(
        oof_cal, y, init_thr, n_steps=31, shrink=0.50
    )
    thr_refined, oof_kappa_refined = refine_thresholds_coordinate(
        oof_cal, y, thr_grid, iters=20
    )

    iso_full = fit_ordinal_isotonic_expected_score(scores, y, n_classes=5)

    return (
        iso_full,
        thr_refined,
        float(oof_kappa_refined),
        init_thr,
        thr_grid,
        float(oof_kappa_grid),
        oof_cal,
    )


iso_fn, thresholds, val_kappa, init_thr, thr_grid, val_kappa_grid, oof_cal = (
    tune_thresholds_oof(train_scores, y_train, n_splits=5, seed=42)
)

print("Init thresholds (calibrated OOF expected-score):", init_thr.tolist())
print(
    "Grid thresholds (calibrated OOF expected-score):",
    thr_grid.astype(np.float32).tolist(),
)
print("OOF QWK (grid):", float(val_kappa_grid))
print(
    "Refined thresholds (calibrated OOF expected-score):",
    thresholds.astype(np.float32).tolist(),
)
print("OOF QWK (refined):", float(val_kappa))

train_cal_full = iso_fn(train_scores)
train_pred_full = score_to_label(train_cal_full, thresholds)
print(
    "Train predicted distribution (full-cal):",
    np.bincount(train_pred_full, minlength=5).tolist(),
)



## === cell 5
test_scores = compute_scores_for_df(test_df, TEST_IMAGES_DIR)

test_scores = np.asarray(test_scores, dtype=np.float32)
finite_mask = np.isfinite(test_scores)
if not finite_mask.all():
    med = float(np.median(test_scores[finite_mask])) if finite_mask.any() else 0.0
    test_scores = test_scores.copy()
    test_scores[~finite_mask] = med

test_cal = iso_fn(test_scores)
y_pred_list = score_to_label(test_cal, thresholds)

print("Test predicted distribution:", np.bincount(y_pred_list, minlength=5).tolist())
print("First 10 preds:", y_pred_list[:10].tolist())



## === cell 6
sub_df = pd.read_csv(TEST_CSV)
sub_df["diagnosis"] = y_pred_list.astype(int)
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub_df.head())
print("submission.csv rows:", len(sub_df))
assert list(sub_df.columns) == ["id_code", "diagnosis"]
assert len(sub_df) == len(test_df)
