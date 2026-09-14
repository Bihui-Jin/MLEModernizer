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
Classify and localize common thoracic lung diseases and critical findings.

For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "14 1 0 0 1 1" (14 is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).

## Metric
PASCAL VOC 2010 [mean Average Precision (mAP)](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf) at IoU > 0.4.

## Submission Format
Images in the test set may contain more than one object. For each object in a given test image, you must predict a class ID, `confidence` score, and bounding box in format `xmin ymin xmax ymax`. If you predict that there are NO objects in a given image, you should predict `14 1.0 0 0 1 1`, where `14` is the class ID for "No finding", 1.0 is the confidence, and `0 0 1 1` is a one-pixel bounding box.

The submission file should contain a header and have the following format:

```
ID,TARGET
004f33259ee4aef671c2b95d54e4be68,14 1 0 0 1 1
004f33259ee4aef671c2b95d54e4be69,11 0.5 100 100 200 200 13 0.7 10 10 20 20
etc.
```

## Dataset
The dataset comprises postero-anterior (PA) CXR scans in DICOM format.

All images were labeled for the presence of 14 critical radiographic findings as listed below:

```
0 - Aortic enlargement
1 - Atelectasis
2 - Calcification
3 - Cardiomegaly
4 - Consolidation
5 - ILD
6 - Infiltration
7 - Lung Opacity
8 - Nodule/Mass
9 - Other lesion
10 - Pleural effusion
11 - Pleural thickening
12 - Pneumothorax
13 - Pulmonary fibrosis
```

The "No finding" observation (`14`) was intended to capture the absence of all findings above.

### Files
- **train.csv** - the train set metadata, with one row for each object, including a class and a bounding box. Some images in both test and train have multiple objects.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_id` - unique image identifier
- `class_name` - the name of the class of detected object (or "No finding")
- `class_id` - the ID of the class of detected object
- `rad_id` - the ID of the radiologist that made the observation
- `x_min` - minimum X coordinate of the object's bounding box
- `y_min` - minimum Y coordinate of the object's bounding box
- `x_max` - maximum X coordinate of the object's bounding box
- `y_max` - maximum Y coordinate of the object's bounding box

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (1501 lines)
            sample_submission.csv.zip (30.9 kB)
            test.zip (12.7 GB)
            train.csv (61172 lines)
            train.csv.zip (1.7 MB)
            train.zip (114.6 GB)
            test/
                00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                ... and 1498 other files
                test/
            train/
                000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                ... and 13498 other files
                train/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
        input/
            description.md (132 lines)
            sample_submission.csv (1501 lines)
            sample_submission.csv.zip (30.9 kB)
            test.zip (12.7 GB)
            train.csv (61172 lines)
            train.csv.zip (1.7 MB)
            train.zip (114.6 GB)
            test/
                00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                ... and 1498 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
            train/
                000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                ... and 13498 other files
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
        working/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
```

-> data/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> data/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> data/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> data/vinbigdata-chest-xray-abnormalities-detection/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> input/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> input/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> (stopped after 10 files for performance)

# 5. Target score

0.2288081773496749

# 6. Current score

0.04836

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails because it depends on two external Kaggle datasets (`2classtest-pred-file` and `vinbigdata-post-processing`) that are not available in your environment, so the CSV reads crash before a submission can be written. I make the smallest change that preserves the intended “2-class gating + detector predictions” logic by (1) loading the official `sample_submission.csv` as the base detection predictions when the postprocessed detector file is missing, and (2) defaulting the 2-class probabilities to “no finding” when the 2-class prediction file is missing. I also make the merge/indexing robust (avoid `.loc[i]` on non-range index) and ensure the output file is a valid `submission.csv` with the required columns. This yield a valid submission end-to-end; score likely be low without the missing external predictions, but it run reliably and produce the correct format.'
- What this solution (achieved 0.0655) has done: 'Your current score is low because, in this environment, you fall back to the all-“No finding” sample submission (no detector predictions), so the 2-class gating thresholds don’t matter. To move the score toward the target with minimal changes while preserving the same “2-class gating + detector predictions” core logic, I (1) synthesize a simple, legitimate detector-like prediction from `train.csv` priors (class frequency + mean box per class) and use that as the fallback when the external detector file is missing, and (2) make the gating actually use the provided `target` as “No finding probability” directly (instead of converting to `1-target`) while keeping the same thresholding mechanism. This should substantially increase mAP from the near-baseline “all normal” submission, without changing your overall approach (still: base detector predictions + optional no-finding gating). The script still writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.0568) has done: 'Your current score (0.0655) is far below the target (0.2288), so we should legitimately increase mAP while keeping your “2-class gating + detector predictions” structure intact. The biggest weakness is the fallback detector: repeating a single “average box” for a few classes on every image produces many false positives; instead, we keep the same fallback idea but (a) add a realistic “No finding” probability prior derived from `train.csv` and (b) make the fallback detector emit only *one* common class box per image (lower FP pressure) unless the gating says “No finding”. Finally, we tune the gating thresholds slightly toward a balanced mix (not all normal, not all abnormal) using the same thresholding mechanism, which should move the score closer to the target without changing the core pipeline.'
- What this solution (achieved 0.04806) has done: 'Your current score is far below the target, so we should legitimately increase mAP while preserving your exact “detector predictions + 2-class no-finding gating” structure. The main weakness is the fallback detector: emitting the same abnormal box for every image creates many false positives; instead, we still use `train.csv` priors but make the fallback detector output either (a) “No finding” only, or (b) a small number of plausible abnormal predictions, controlled by the learned abnormal prior from `train.csv`. We also make the gating thresholds consistent with that prior by setting `low_threshold`/`high_threshold` from the estimated `p_no_finding`, so the system doesn’t over-predict normal or abnormal across the whole test set. These are minimal changes: same files, same merge, same thresholding loop, same output format—just a better fallback `pred_det_df` and data-driven thresholds.'
- What this solution (achieved 0.05245) has done: 'Your current score is far below the target, so we should legitimately increase mAP while preserving your exact “fallback detector predictions + 2-class no-finding gating” structure. The main issue is the fallback detector emits the same box(es) for many images, creating heavy false positives that mAP punishes; the smallest safe fix is to make the fallback detector *image-conditional* using only `train.csv` metadata: sample a small number of predicted classes per image from the learned per-image object-count distribution, and draw boxes from class-specific box distributions (instead of a single mean box). We keep your gating loop and thresholds mechanism intact, but we also ensure the appended “No finding” token is never duplicated if already present in the base prediction string. These changes keep the same pipeline/semantics (detector string + no-finding gating) while reducing systematic FP patterns and should move score toward your target.'
- What this solution (achieved 0.04868) has done: 'Your current score is far below the target, so we should increase mAP with minimal, “same-pipeline” changes. The biggest issue is that the fallback detector generates noisy boxes for many images, creating heavy false positives; we keep the same fallback-from-train.csv approach but make it more conservative by (a) scaling down the number of predicted boxes per image and (b) adding a simple, train-prior-informed confidence threshold so low-confidence synthetic detections are dropped. We also make the “No finding” gating less aggressive by narrowing the forced-NORMAL region (raise `high_threshold` slightly and raise `low_threshold` slightly) so we don’t wipe detector predictions too often. These changes preserve your exact core logic (base PredictionString + 2-class no-finding gating + same output schema) while reducing systematic false positives and improving precision, which should move the score toward 0.2288.'
- What this solution (achieved 0.0475) has done: 'Your score is far below the target, and the main limiting factor is still the synthetic fallback detector producing too many false positives in a metric (mAP@IoU>0.4) that heavily penalizes them. I keep your exact pipeline (fallback detector → merge 2-class no-finding prob → low/high threshold gating) but make the fallback detector more image-conditional and conservative by (1) sampling per-image detection counts from the *full* train count distribution (not clipped to 3) and (2) lowering synthetic confidences + raising the keep-threshold so low-quality synthetic boxes are dropped. This should reduce systematic FP spam while keeping some true-positive chances, moving mAP upward toward your target without changing your overall approach or adding any new dependencies. I also make the “replace with NORMAL” region slightly less aggressive (data-driven but smaller offset) so we don’t wipe out remaining detections too often.'
- What this solution (achieved 0.04771) has done: 'Your current gap to target is large (0.0475 → 0.2288), and the main bottleneck is the synthetic fallback detector producing many false positives across all images. I keep the same overall pipeline (fallback detector → merge 2-class no-finding probability → low/high threshold gating) but make the fallback detector *more conservative and more realistic* by (1) using the train-set **per-image object count distribution including zeros** (so many images naturally get no detections), and (2) sampling boxes from **real train rows** (not Gaussian around means), which reduces implausible boxes that hurt IoU/mAP. I also make the fallback confidence distribution class-frequency-based but slightly higher for the kept detections, while still filtering by a keep-threshold, to improve precision without spamming predictions. These are minimal changes localized to the fallback block and keep your architecture/loop/threshold gating semantics intact.'
- What this solution (achieved 0.0477) has done: 'Your score is far below the target, so we should cautiously increase mAP by reducing systematic false positives while keeping your exact “fallback detector predictions + 2-class no-finding gating” pipeline intact. The smallest high-impact change is to make the synthetic fallback detector more realistic by conditioning the number of predicted boxes on image-level metadata derived from train (including a strong mass at zero), and by sampling whole (class, box) rows from real train annotations rather than mixing separate class/box distributions. I also make the synthetic confidence distribution slightly more conservative and data-driven (per-class mean confidence with noise) and ensure that when we append a “No finding” token we don’t accidentally keep any existing “14 …” tokens. All paths, thresholds mechanism, and submission schema remain the same; this only changes the fallback detector block that is currently the main bottleneck.'
- What this solution (achieved 0.04745) has done: 'Your score is far below the target, so we should increase mAP while keeping your exact “fallback detector predictions + 2-class no-finding gating” pipeline intact. The biggest issue is that the synthetic fallback detector still emits too many low-quality false positives; mAP@0.4 punishes this heavily, so we make the fallback more conservative and more IoU-plausible without changing the overall structure. Concretely, we (1) deduplicate/trim synthetic detections per image (keep top-K by confidence), (2) filter out very small/degenerate boxes using size/area priors learned from train boxes, and (3) calibrate the “No finding” appended token confidence to be competitive but not overpower everything, while keeping your same low/high threshold gating logic.'
- What this solution (achieved 0.04742) has done: 'I keep your exact pipeline (synthetic fallback detector → merge 2-class no-finding probability → low/high threshold gating) and only make the fallback detector less systematically wrong. Concretely, I (1) build class-specific *per-image presence probabilities* from `train.csv` and use them to decide whether to emit a detection for each class, which dramatically reduces false positives compared to always sampling a few boxes, and (2) sample at most 1–2 detections per image from real train boxes (same as you already do) but only for classes that are plausible under those probabilities. This should increase mAP materially from ~0.047 toward your target while staying within your “same logic” constraints (still no real model; still prior-based detector + gating). Thresholding logic and submission formatting stay the same.'
- What this solution (achieved 0.04781) has done: 'Your current gap to the target is large (0.04742 → 0.22881), and the main bottleneck is that the fallback detector is still too “uninformed” per-image, so it creates many false positives/poor-IoU boxes that mAP@0.4 punishes heavily. I keep your exact pipeline (synthetic detector → merge 2-class no-finding prob → low/high threshold gating) but make the synthetic detector *image-conditional using only train.csv*, by learning a per-image abnormality prior from train metadata and using it to decide whether to emit any detections at all for a given test image. When we do emit detections, we sample 1–2 detections from real train boxes with class probabilities conditioned on “abnormal”, and we keep your existing size/area filtering and top-2 cap to stay conservative. Finally, I keep your gating thresholds structure but slightly reduce conflicts by setting the “add no-finding token” confidence below many detector confidences (so it doesn’t swamp true positives), which typically helps mAP.'
- What this solution (achieved 0.04835) has done: 'Your current score (0.04781) is far below the target (0.22881), so we need a legitimate mAP increase while keeping the same “fallback detector predictions + 2-class no-finding gating” structure. The smallest high-impact fix is to make the synthetic fallback detector less noisy and more “detector-like” by (1) emitting detections only for a subset of images based on a train-derived abnormal prior, (2) sampling 1–2 detections from the most likely classes (top-K by frequency) to reduce false positives, and (3) setting confidence values in a more competitive range (still conservative) while keeping your existing filtering and top-2 cap. I keep your thresholding mechanism intact but make it data-driven and slightly less “replace with NORMAL” heavy by tightening the forced-normal region around the train prior. This should move the score upward toward your target without changing files/paths, core gating logic, or output format.'
- What this solution (achieved 0.0493) has done: 'Your score is far below the target, so the smallest legitimate way to move mAP upward (without changing your overall “fallback detector + 2-class no-finding gating” pipeline) is to make the fallback detector less false-positive heavy while increasing the chance of having at least one plausible true-positive per abnormal image. I keep your gating structure and thresholds, but improve the synthetic detector by sampling the predicted class from the full train class distribution (not just top-6), making the number of detections per image match the train per-image object-count distribution (including many zeros), and sampling confidence from the real train confidences for that class (more detector-like calibration). I also avoid emitting NORMAL inside the detector fallback (so gating remains the single source of “No finding” decisions), which reduces conflicting behavior and makes the gating thresholds actually control the FP/TP tradeoff. These changes are localized to the fallback block and should increase mAP toward your target while preserving the same semantics and producing the same submission format.'
- What this solution (achieved 0.04836) has done: 'Your current score is far below the target, so we should increase mAP with the smallest change that reduces false positives while keeping your exact “synthetic detector → merge 2-class no-finding prob → low/high threshold gating” pipeline intact. The largest issue is that the fallback detector still emits too many boxes on images that are likely normal; we can fix that by making the synthetic detector *conditional on the train-derived no-finding prior* (i.e., only emit detections with probability ≈ p_any_abnormal) while leaving the gating logic unchanged. Concretely, we (1) add an “emit any detections?” Bernoulli step using a stable per-image RNG, (2) slightly tighten detector confidence filtering (keep-threshold) to reduce low-quality boxes, and (3) keep everything else (sampling from real train boxes, top-2 cap, thresholds, formatting) the same so runtime and semantics remain stable.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(0)

DATA_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission at {SAMPLE_SUB_PATH}"
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub.columns = [c.strip() for c in sample_sub.columns]

if "PredictionString" not in sample_sub.columns and "TARGET" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"TARGET": "PredictionString"})
sample_sub = sample_sub[["image_id", "PredictionString"]].copy()
sample_sub.head()




## === cell 1
pred_2class_path = "../input/2classtest-pred-file/2-cls test pred.csv"

train_path = os.path.join(DATA_DIR, "train.csv")
assert os.path.exists(train_path), f"Missing train.csv at {train_path}"
train_df = pd.read_csv(train_path)

img_has_abnormal = train_df.groupby("image_id")["class_id"].apply(
    lambda s: (s != 14).any()
)
p_any_abnormal = float(img_has_abnormal.mean())
p_no_finding_prior = float(1.0 - p_any_abnormal)

high_threshold = float(np.clip(p_no_finding_prior + 0.10, 0.70, 0.95))
low_threshold = float(np.clip(p_no_finding_prior - 0.06, 0.05, 0.55))

if os.path.exists(pred_2class_path):
    pred_2class = pd.read_csv(pred_2class_path)
else:
    pred_2class = sample_sub[["image_id"]].copy()
    pred_2class["target"] = p_no_finding_prior

print(
    f"Train prior p_no_finding={p_no_finding_prior:.4f}; thresholds low={low_threshold:.3f}, high={high_threshold:.3f}"
)
pred_2class.head()




## === cell 2
NORMAL = "14 1 0 0 1 1"

pred_det_path = "../input/vinbigdata-post-processing/submission_postprocessed.csv"

if os.path.exists(pred_det_path):
    pred_det_df = pd.read_csv(pred_det_path)
    if (
        "PredictionString" not in pred_det_df.columns
        and "TARGET" in pred_det_df.columns
    ):
        pred_det_df = pred_det_df.rename(columns={"TARGET": "PredictionString"})
    if "PredictionString" not in pred_det_df.columns:
        raise ValueError(
            f"pred_det_df must contain PredictionString (or TARGET). Got: {pred_det_df.columns.tolist()}"
        )
    pred_det_df = pred_det_df[["image_id", "PredictionString"]].copy()
else:
    box_df = train_df[train_df["class_id"] != 14].copy()

    if len(box_df) == 0:
        pred_det_df = sample_sub[["image_id"]].copy()
        pred_det_df["PredictionString"] = ""  # let gating decide NORMAL
    else:
        box_df["w"] = (box_df["x_max"] - box_df["x_min"]).astype(float).clip(lower=0.0)
        box_df["h"] = (box_df["y_max"] - box_df["y_min"]).astype(float).clip(lower=0.0)
        box_df["area"] = (box_df["w"] * box_df["h"]).astype(float)

        w_min = float(np.nanquantile(box_df["w"].to_numpy(), 0.03))
        h_min = float(np.nanquantile(box_df["h"].to_numpy(), 0.03))
        area_min = float(np.nanquantile(box_df["area"].to_numpy(), 0.03))
        w_min = max(2.0, w_min)
        h_min = max(2.0, h_min)
        area_min = max(10.0, area_min)

        box_df["class_id"] = box_df["class_id"].astype(int)

        class_counts = box_df["class_id"].value_counts().sort_index()
        classes_all = class_counts.index.to_numpy(dtype=int)
        cls_prob_all = (
            class_counts.to_numpy(dtype=float) / float(class_counts.sum())
        ).astype(float)

        by_class_boxes = {}
        by_class_conf = {}

        freq = (class_counts / float(class_counts.sum())).to_dict()
        for cid, g in box_df.groupby("class_id"):
            cid = int(cid)
            by_class_boxes[cid] = g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(
                dtype=float
            )
            mu = float(np.clip(0.18 + 0.90 * freq.get(cid, 0.0), 0.10, 0.75))
            sigma = float(np.clip(0.10 - 0.05 * freq.get(cid, 0.0), 0.04, 0.10))
            by_class_conf[cid] = (mu, sigma)

        per_img_counts = box_df.groupby("image_id").size().astype(int)
        count_values = per_img_counts.to_numpy(dtype=int)
        max_count = int(np.clip(np.max(count_values), 1, 10))
        pmf = np.zeros(max_count + 1, dtype=float)
        pmf[0] = float(np.clip(p_no_finding_prior, 0.05, 0.95))
        clipped = np.clip(count_values, 1, max_count)
        for k in clipped:
            pmf[int(k)] += 1.0
        if pmf[1:].sum() > 0:
            pmf[1:] /= pmf[1:].sum()
            pmf[1:] *= 1.0 - pmf[0]
        pmf = pmf / pmf.sum()

        conf_keep_threshold = 0.18

        def _stable_u01(s: str) -> float:
            h = 2166136261
            for ch in s:
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
            return h / 2**32

        def _rng_from_image_id(iid: str) -> np.random.RandomState:
            seed = int(_stable_u01(iid) * (2**32 - 1)) & 0xFFFFFFFF
            return np.random.RandomState(seed)

        def _sample_real_box(rs: np.random.RandomState, cid: int):
            arr = by_class_boxes.get(int(cid), None)
            if arr is None or len(arr) == 0:
                arr_all = box_df[["x_min", "y_min", "x_max", "y_max"]].to_numpy(
                    dtype=float
                )
                vals = arr_all[int(rs.randint(0, len(arr_all)))]
            else:
                vals = arr[int(rs.randint(0, len(arr)))]
            xmin, ymin, xmax, ymax = [float(v) for v in vals]
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = max(xmin + 1.0, xmax)
            ymax = max(ymin + 1.0, ymax)
            return xmin, ymin, xmax, ymax

        def _sample_conf(rs: np.random.RandomState, cid: int) -> float:
            mu, sigma = by_class_conf.get(int(cid), (0.18, 0.08))
            c = float(rs.normal(mu, sigma))
            return float(np.clip(c, 0.02, 0.95))

        def _format_dets(dets):
            parts = []
            for cid, conf, xmin, ymin, xmax, ymax in dets:
                parts.append(
                    f"{int(cid)} {float(conf):.6f} {float(xmin):.2f} {float(ymin):.2f} {float(xmax):.2f} {float(ymax):.2f}"
                )
            return " ".join(parts).strip()

        image_ids = sample_sub["image_id"].astype(str).tolist()
        preds = []

        for iid in image_ids:
            rs = _rng_from_image_id(iid)

            if rs.rand() > p_any_abnormal:
                preds.append("")  # let gating decide NORMAL
                continue

            n_det = int(rs.choice(np.arange(len(pmf)), p=pmf))
            if n_det <= 0:
                preds.append("")  # let gating decide NORMAL
                continue

            n_det = int(min(n_det, 2))

            if n_det == 1:
                chosen = [int(rs.choice(classes_all, p=cls_prob_all))]
            else:
                if len(classes_all) >= 2:
                    chosen = rs.choice(
                        classes_all, size=2, replace=False, p=cls_prob_all
                    ).tolist()
                    chosen = [int(c) for c in chosen]
                else:
                    chosen = [int(classes_all[0]), int(classes_all[0])]

            dets = []
            for cid in chosen:
                conf = _sample_conf(rs, int(cid))
                if conf < conf_keep_threshold:
                    continue

                xmin, ymin, xmax, ymax = _sample_real_box(rs, int(cid))
                w = xmax - xmin
                h = ymax - ymin
                if (w < w_min) or (h < h_min) or (w * h < area_min):
                    continue

                dets.append(
                    (
                        int(cid),
                        float(conf),
                        float(xmin),
                        float(ymin),
                        float(xmax),
                        float(ymax),
                    )
                )

            dets.sort(key=lambda x: x[1], reverse=True)
            dets = dets[:2]  # preserve your previous top-2 cap

            preds.append(_format_dets(dets))

        pred_det_df = pd.DataFrame({"image_id": image_ids, "PredictionString": preds})

pred_det_df["PredictionString"] = pred_det_df["PredictionString"].fillna("").astype(str)
pred_det_df = pred_det_df[["image_id", "PredictionString"]].copy()

n_normal_before = int((pred_det_df["PredictionString"].str.strip() == "").sum())

merged_df = pd.merge(pred_det_df, pred_2class, on="image_id", how="left")

if "target" in merged_df.columns:
    merged_df["no_finding_prob"] = merged_df["target"].astype(float)
elif "class0" in merged_df.columns:
    merged_df["no_finding_prob"] = merged_df["class0"].astype(float)
else:
    merged_df["no_finding_prob"] = 0.0

merged_df["no_finding_prob"] = (
    merged_df["no_finding_prob"].fillna(p_no_finding_prior).astype(float).clip(0.0, 1.0)
)


def _strip_no_finding_tokens(pred_str: str) -> str:
    toks = str(pred_str).strip().split()
    kept = []
    for j in range(0, len(toks), 6):
        chunk = toks[j : j + 6]
        if len(chunk) < 6:
            continue
        if chunk[0] == "14":
            continue
        kept.extend(chunk)
    return " ".join(kept).strip()


c0, c1, c2 = 0, 0, 0
pred_strings = merged_df["PredictionString"].tolist()
nf_vals = merged_df["no_finding_prob"].to_numpy()

for i, p_nf in enumerate(nf_vals):
    if p_nf < low_threshold:
        pred_strings[i] = _strip_no_finding_tokens(pred_strings[i])
        if pred_strings[i] == "":
            pred_strings[i] = NORMAL
        c0 += 1
    elif low_threshold <= p_nf and p_nf < high_threshold:
        pred_strings[i] = _strip_no_finding_tokens(pred_strings[i])
        if pred_strings[i] == "":
            pred_strings[i] = NORMAL
        else:
            nf_conf = float(np.clip(0.55 * p_nf, 0.12, 0.45))
            pred_strings[i] = (pred_strings[i] + f" 14 {nf_conf:.6f} 0 0 1 1").strip()
        c1 += 1
    else:
        pred_strings[i] = NORMAL
        c2 += 1

merged_df["PredictionString"] = pred_strings

n_normal_after = int((merged_df["PredictionString"] == NORMAL).sum())
print(
    f"n_empty(detector): {n_normal_before} (empty) -> n_normal(after gating): {n_normal_after} with threshold {low_threshold} & {high_threshold}"
)
print(f"Keep {c0} Add {c1} Replace {c2}")

submission_filepath = "submission.csv"
submission_df = merged_df[["image_id", "PredictionString"]].copy()
submission_df.to_csv(submission_filepath, index=False)

print(f"Saved to {submission_filepath}")
print(submission_df.head())
print(f"Rows: {len(submission_df)} (expected {len(sample_sub)})")
assert submission_filepath.endswith(".csv")
assert len(submission_df) == len(sample_sub)
assert submission_df["image_id"].isna().sum() == 0
assert submission_df["PredictionString"].isna().sum() == 0
