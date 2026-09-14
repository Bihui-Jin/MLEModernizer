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

0.2277679860930401

# 6. Current score

0.0475

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0238) has done: 'Your notebook fails because it depends on several external “../input/*/submission.csv” files that do not exist in this Kaggle environment, so nothing is defined downstream and the pipeline never writes a valid CSV. To make it run end-to-end with minimal logic changes, I keep the same ensembling/post-processing structure but generate the required per-class probability columns and an initial `PredictionString` directly from the provided competition `sample_submission.csv` (fallback-safe, no missing files). I also make the loops robust to the real test size (1500 rows) and ensure every row has a non-empty `PredictionString`, defaulting to `14 1 0 0 1 1` as required. This produce a valid `submission.csv` in the working directory and should score above zero (though likely below strong detector-based baselines), moving toward the target compared to “Not yielded”.'
- What this solution (achieved 1e-05) has done: 'Your current low score comes mainly from always emitting “No finding” with a 1×1 box, which yields almost no true positives under mAP@0.4. To move toward the target with minimal core-logic changes, I keep your dataframe/merging/post-processing structure but replace the “always no finding” pseudo-probabilities with simple, legitimate priors learned from `train.csv` (class frequency + typical box sizes) and generate a small number of plausible boxes per image. I also fix the submission column names to exactly match the competition’s sample (`image_id,PredictionString`) while leaving your downstream string-normalization steps intact. This increase recall (and thus mAP) without introducing any new model/architecture or changing the overall pipeline shape.'
- What this solution (achieved 1e-05) has done: 'Your current score is far below the target because the post-filter in cell 6 deletes several classes entirely (0, 7, 13) and also strips “No finding”, collapsing many rows into weak/empty predictions; we keep your same pipeline but stop removing those classes so recall (and thus mAP) can rise toward the target. Next, we reduce the “No finding” mass slightly so more images emit actual findings (still using your learned priors/median box sizes), which should improve score without changing the overall approach. Finally, we make the string merge in cell 7 safe by avoiding duplicate identical prediction blocks, so we don’t spam repeated boxes that can hurt precision. These are minimal, local edits that preserve your core logic (priors → boxes → string post-processing) while moving performance upward toward 0.2278.'
- What this solution (achieved 0.00099) has done: 'Your current score is far below the target because the pipeline generates the same centered “median-size” boxes for every image and always outputs multiple findings, which produces many false positives and very low mAP. To move the score upward with minimal logic change, I keep your exact prior→top-k→string post-processing structure, but make the generated boxes image-specific using the *training distribution of box centers* (per class) instead of always centering at (512,512). I also add a very small, deterministic per-image offset (based on `image_id` hash) so boxes aren’t identical across all images, which tends to improve matching chances without changing the overall approach. Finally, I reduce `top_k` from 3 to 2 to slightly improve precision (less clutter per image) while keeping your same confidence/prior semantics.'
- What this solution (achieved 0.00099) has done: 'Your current score is far below the target, so we should improve mAP by reducing obvious false positives while keeping your same “priors → top-k boxes → string post-processing” pipeline intact. The smallest high-impact change is to add a confidence floor (don’t emit very low-confidence classes) and to make “No finding” more likely unless at least one predicted finding is reasonably confident; this typically boosts precision and mAP for naive box generators. I keep your class priors/median sizes/center medians and deterministic jitter exactly as-is, but change `make_initial_prediction_string` to filter by a threshold and to fall back to `14 1 0 0 1 1` when nothing passes. I also slightly reduce `finding_mass` so fewer images spam findings, which should move the score upward toward your target without changing the core logic.'
- What this solution (achieved 0.00089) has done: 'Your current gap to the target is large (0.00099 vs 0.2278), and the biggest limiter in your existing pipeline is that it emits the same “median center/size” box templates for every image/class, so IoU matching stays near-random and mAP remains near zero. With minimal core-logic change (still: train priors → top-k classes → generate boxes → same string post-processing), I make boxes more image-specific by sampling class-specific box centers from the *training center distribution* (median + IQR) and using a deterministic per-image offset scaled by that IQR, instead of scaling jitter by box width/height. I also make the default more conservative by slightly lowering `finding_mass` and raising `conf_thresh` so you emit fewer low-quality false positives (this tends to improve mAP more than recall in such a naive generator). All downstream merge/normalization cells remain intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your current score is far below the target, so we should increase mAP by reducing systematic false positives and making predictions less duplicated, while keeping your same “priors → top-k boxes → string post-processing” structure. The smallest high-impact fixes are: (1) stop concatenating identical prediction strings in cell 7 (currently doubles every box, hurting precision), (2) make the “No finding” override less aggressive so it doesn’t wipe out predicted findings, and (3) slightly raise the confidence threshold and reduce top_k to emit fewer low-quality boxes per image. These changes keep your learned priors/box templates intact and only adjust filtering and merging so the output better matches the metric. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your current score (0.0475) is far below the target (0.2278), so we should cautiously increase mAP by improving recall a bit while keeping your same “priors → top-k classes → template boxes → string post-processing” pipeline. The smallest impactful lever here is to emit a second box when the second-best class is reasonably confident, while keeping a stricter threshold for that second box to avoid flooding false positives. I also make the “No finding” override slightly less aggressive (it currently almost never triggers, but making it consistent with your new two-box behavior helps stability). Everything else (priors, box generation from train medians/IQR, and downstream formatting) remains unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your current score (0.0475) is far below the target (0.2278), so we should increase mAP by improving recall a bit without changing your core “priors → top-k classes → template boxes → string post-processing” pipeline. The smallest lever is to allow one more predicted box when it’s reasonably confident, while also softening the second/third-box thresholds so the extra box actually appears (your current priors often sit below 0.20). I keep your box generation, jitter, and all downstream merging/normalization intact, only adjusting `make_initial_prediction_string` thresholds and `top_k` to emit up to 3 findings per image in a controlled way. This should move the score upward toward the target while remaining stable and still producing a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your current score is far below the target, so we should gently improve mAP by increasing the chance that at least one predicted box overlaps a true box, while keeping your same “priors → top-k classes → template boxes → string post-processing” pipeline intact. The smallest high-impact adjustment is to emit *multiple candidate boxes per predicted class* (still using your learned medians/IQR and deterministic hash jitter), because a single template box per class per image has very low IoU-match probability. To avoid flooding false positives, we keep the same thresholds and top-k logic, but split each class confidence across 2 box variants with small deterministic offsets. Everything else (priors, merging, normalization, submission writing) remains unchanged.'
- What this solution (achieved 0.0475) has done: 'We keep your exact “priors → top‑k classes → template boxes with deterministic jitter → string post-processing” pipeline, but make a small adjustment to increase recall without flooding false positives: emit a third box variant only for the top-1 class (and only if that class is confident enough). This increases the chance at least one predicted box overlaps the true box at IoU>0.4, which is the main bottleneck for template-box approaches, while keeping the number of extra boxes controlled. We also ensure the new variant’s confidence is small (split from the same class confidence), so precision doesn’t collapse. Everything else (priors, thresholds, merging, formatting, and submission writing) stays the same.'
- What this solution (achieved 0.0475) has done: 'Your current score (0.0475) is far below the target (0.2278), so we should increase mAP mainly by improving localization/recall without changing your pipeline shape (priors → top‑k classes → template boxes w/ deterministic jitter → string post-processing). The most leverage with minimal disruption is to make the per-class box *size* less “median-like” (often too big/too small) by using a conservative per-class trimmed-quantile size (slightly smaller than median tends to help IoU>0.4) and to add one additional box variant for rank‑1 and rank‑2 classes (controlled, low extra confidence) to raise the chance of IoU overlap. I also make the final “No finding” override slightly less likely (it currently almost never triggers anyway) and keep all downstream formatting/merge logic intact so submission semantics stay identical. These are local edits that keep your overall approach and runtime similar, but should move recall/mAP upward toward the target band.'
- What this solution (achieved 0.0475) has done: 'We keep your priors→top‑k→template boxes→string post-processing pipeline intact, but make a small, targeted change to improve localization chance (IoU>0.4) without flooding false positives: emit one additional **scale variant** (slightly smaller box) for the top-1 class only, splitting the same class confidence across variants. This increases the probability that at least one predicted box per image overlaps a true box at sufficient IoU, which is the main bottleneck of template-box approaches and should move mAP upward from 0.0475 toward your target. We also make the “No finding” override use the already-present mass more consistently by lowering the trigger slightly (still conservative), so empty/low-confidence cases don’t spam findings. All file paths, columns, and the final `submission.csv` writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import zlib

np.random.seed(42)

BASE = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE, "train.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = (
        "/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
    )

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if "ID" in sample_sub.columns and "image_id" not in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"ID": "image_id"})
if "TARGET" in sample_sub.columns and "PredictionString" not in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"TARGET": "PredictionString"})

assert (
    "image_id" in sample_sub.columns
), f"sample_submission missing image_id/ID column: {sample_sub.columns}"
if "PredictionString" not in sample_sub.columns:
    sample_sub["PredictionString"] = ""

sample_sub.head()



## === cell 1
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_pos = train_df[train_df["class_id"] != 14].copy()

class_counts = train_pos["class_id"].value_counts().sort_index()
total_pos = class_counts.sum()
class_prior = (class_counts / total_pos).reindex(range(14), fill_value=0.0)

train_pos["w"] = (train_pos["x_max"] - train_pos["x_min"]).clip(lower=1.0)
train_pos["h"] = (train_pos["y_max"] - train_pos["y_min"]).clip(lower=1.0)


def _trimmed_quantile(s: pd.Series, q: float) -> float:
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    s2 = s[(s >= lo) & (s <= hi)]
    if len(s2) == 0:
        s2 = s
    return float(s2.quantile(q))


wh_median = train_pos.groupby("class_id")[["w", "h"]].median().reindex(range(14))

wh_q = (
    train_pos.groupby("class_id")
    .agg(
        w_q=("w", lambda x: _trimmed_quantile(x, 0.45)),
        h_q=("h", lambda x: _trimmed_quantile(x, 0.45)),
    )
    .reindex(range(14))
)

global_w = float(train_pos["w"].median()) if len(train_pos) else 100.0
global_h = float(train_pos["h"].median()) if len(train_pos) else 100.0

wh_median["w"] = wh_median["w"].fillna(global_w)
wh_median["h"] = wh_median["h"].fillna(global_h)

wh_q["w_q"] = wh_q["w_q"].fillna(wh_median["w"]).fillna(global_w)
wh_q["h_q"] = wh_q["h_q"].fillna(wh_median["h"]).fillna(global_h)

wh_use = pd.DataFrame(index=wh_median.index)
wh_use["w"] = np.clip(
    wh_q["w_q"].values, 0.70 * wh_median["w"].values, 1.05 * wh_median["w"].values
)
wh_use["h"] = np.clip(
    wh_q["h_q"].values, 0.70 * wh_median["h"].values, 1.05 * wh_median["h"].values
)

train_pos["cx"] = ((train_pos["x_min"] + train_pos["x_max"]) / 2.0).clip(0, 1024.0)
train_pos["cy"] = ((train_pos["y_min"] + train_pos["y_max"]) / 2.0).clip(0, 1024.0)
center_median = train_pos.groupby("class_id")[["cx", "cy"]].median().reindex(range(14))
center_median["cx"] = center_median["cx"].fillna(512.0)
center_median["cy"] = center_median["cy"].fillna(512.0)


def _iqr(s: pd.Series) -> float:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return float(q3 - q1)


center_iqr = (
    train_pos.groupby("class_id")
    .agg(cx_iqr=("cx", _iqr), cy_iqr=("cy", _iqr))
    .reindex(range(14))
)
global_cx_iqr = float(_iqr(train_pos["cx"])) if len(train_pos) else 150.0
global_cy_iqr = float(_iqr(train_pos["cy"])) if len(train_pos) else 150.0
center_iqr["cx_iqr"] = center_iqr["cx_iqr"].fillna(global_cx_iqr).clip(lower=50.0)
center_iqr["cy_iqr"] = center_iqr["cy_iqr"].fillna(global_cy_iqr).clip(lower=50.0)

IMG_W = 1024.0
IMG_H = 1024.0

(class_prior.head(), wh_use.head(), center_median.head(), center_iqr.head())



## === cell 2
df = sample_sub[["image_id"]].copy()

for k in range(15):
    df[str(k)] = 0.0

finding_mass = 0.50
nofinding_mass = 1.0 - finding_mass

prior_vals = class_prior.values
prior_sum = prior_vals.sum()
if prior_sum <= 0:
    prior_norm = np.ones(14) / 14.0
else:
    prior_norm = prior_vals / prior_sum

for k in range(14):
    df[str(k)] = float(finding_mass * prior_norm[k])

df["14"] = float(nofinding_mass)

df1 = df.copy()
df_densenet = df.copy()

cols = [str(k) for k in range(15)]
df[cols] = df[cols] * 0.25 + df1[cols] * 0.5 + df_densenet[cols] * 0.25

df.head()



## === cell 3
df_heart_cnn = df.copy()

df_heart_cnn["0"] = max(float(df_heart_cnn["0"].iloc[0]), 0.02)
df_heart_cnn["3"] = max(float(df_heart_cnn["3"].iloc[0]), 0.02)
df_heart_cnn["14"] = float(df["14"].iloc[0])

df_heart_cnn.head()




## === cell 4
def _clip_box(xmin, ymin, xmax, ymax, img_w=IMG_W, img_h=IMG_H):
    xmin = float(np.clip(xmin, 0, img_w - 2))
    ymin = float(np.clip(ymin, 0, img_h - 2))
    xmax = float(np.clip(xmax, xmin + 1, img_w - 1))
    ymax = float(np.clip(ymax, ymin + 1, img_h - 1))
    return xmin, ymin, xmax, ymax


def _stable_u01_from_image_id(image_id: str, salt: str) -> float:
    x = zlib.crc32((str(image_id) + "|" + salt).encode("utf-8")) & 0xFFFFFFFF
    return x / 0xFFFFFFFF


def make_initial_prediction_string(
    row,
    top_k=3,
    conf_thresh=0.12,
    conf_thresh_2=0.16,
    conf_thresh_3=0.20,
):
    probs = [(int(k), float(row[str(k)])) for k in range(14)]
    probs.sort(key=lambda x: x[1], reverse=True)

    parts = []
    image_id = row["image_id"]

    for rank, (cls, conf) in enumerate(probs[:top_k], start=1):
        if rank == 1:
            th = conf_thresh
        elif rank == 2:
            th = conf_thresh_2
        else:
            th = conf_thresh_3

        if conf < th:
            continue

        w = float(wh_use.loc[cls, "w"])
        h = float(wh_use.loc[cls, "h"])

        cx0 = float(center_median.loc[cls, "cx"])
        cy0 = float(center_median.loc[cls, "cy"])

        cx_iqr = float(center_iqr.loc[cls, "cx_iqr"])
        cy_iqr = float(center_iqr.loc[cls, "cy_iqr"])

        jx = (_stable_u01_from_image_id(image_id, f"cxoff{cls}") - 0.5) * 0.55 * cx_iqr
        jy = (_stable_u01_from_image_id(image_id, f"cyoff{cls}") - 0.5) * 0.55 * cy_iqr
        cx_base = cx0 + jx
        cy_base = cy0 + jy

        sx = (_stable_u01_from_image_id(image_id, f"cxoff2{cls}") - 0.5) * 0.35 * cx_iqr
        sy = (_stable_u01_from_image_id(image_id, f"cyoff2{cls}") - 0.5) * 0.35 * cy_iqr

        tx = (_stable_u01_from_image_id(image_id, f"cxoff3{cls}") - 0.5) * 0.22 * cx_iqr
        ty = (_stable_u01_from_image_id(image_id, f"cyoff3{cls}") - 0.5) * 0.22 * cy_iqr

        if rank <= 2 and conf >= (0.16 if rank == 1 else 0.18):
            ux = (
                (_stable_u01_from_image_id(image_id, f"cxoff4{cls}") - 0.5)
                * 0.15
                * cx_iqr
            )
            uy = (
                (_stable_u01_from_image_id(image_id, f"cyoff4{cls}") - 0.5)
                * 0.15
                * cy_iqr
            )
            if rank == 1:
                conf1 = max(min(conf * 0.40, 1.0), 0.0)
                conf2 = max(min(conf * 0.24, 1.0), 0.0)
                conf3 = max(min(conf * 0.16, 1.0), 0.0)
                conf4 = max(min(conf * 0.10, 1.0), 0.0)
                conf5 = max(min(conf * 0.10, 1.0), 0.0)  # new scale-variant share
                variants = [
                    (cx_base, cy_base, conf1, 1.00),
                    (cx_base + sx, cy_base + sy, conf2, 1.00),
                    (cx_base + tx, cy_base + ty, conf3, 1.00),
                    (cx_base + ux, cy_base + uy, conf4, 1.00),
                    (cx_base, cy_base, conf5, 0.86),  # slightly smaller box
                ]
            else:
                conf1 = max(min(conf * 0.46, 1.0), 0.0)
                conf2 = max(min(conf * 0.28, 1.0), 0.0)
                conf3 = max(min(conf * 0.16, 1.0), 0.0)
                conf4 = max(min(conf * 0.10, 1.0), 0.0)
                variants = [
                    (cx_base, cy_base, conf1, 1.00),
                    (cx_base + sx, cy_base + sy, conf2, 1.00),
                    (cx_base + tx, cy_base + ty, conf3, 1.00),
                    (cx_base + ux, cy_base + uy, conf4, 1.00),
                ]
        elif rank == 1 and conf >= 0.18:
            conf1 = max(min(conf * 0.46, 1.0), 0.0)
            conf2 = max(min(conf * 0.28, 1.0), 0.0)
            conf3 = max(min(conf * 0.16, 1.0), 0.0)
            conf4 = max(min(conf * 0.10, 1.0), 0.0)  # new scale-variant share
            variants = [
                (cx_base, cy_base, conf1, 1.00),
                (cx_base + sx, cy_base + sy, conf2, 1.00),
                (cx_base + tx, cy_base + ty, conf3, 1.00),
                (cx_base, cy_base, conf4, 0.86),  # slightly smaller box
            ]
        else:
            conf1 = max(min(conf * 0.60, 1.0), 0.0)
            conf2 = max(min(conf * 0.40, 1.0), 0.0)
            variants = [
                (cx_base, cy_base, conf1, 1.00),
                (cx_base + sx, cy_base + sy, conf2, 1.00),
            ]

        for cx, cy, cconf, scale in variants:
            ww = w * float(scale)
            hh = h * float(scale)
            xmin, ymin = cx - ww / 2.0, cy - hh / 2.0
            xmax, ymax = cx + ww / 2.0, cy + hh / 2.0
            xmin, ymin, xmax, ymax = _clip_box(xmin, ymin, xmax, ymax)

            parts.extend(
                [
                    str(cls),
                    f"{cconf:.6f}",
                    f"{xmin:.1f}",
                    f"{ymin:.1f}",
                    f"{xmax:.1f}",
                    f"{ymax:.1f}",
                ]
            )

    if not parts:
        return "14 1 0 0 1 1"
    return " ".join(parts)


df_pred_base = sample_sub[["image_id"]].copy()
df_pred_base["PredictionString"] = df.apply(make_initial_prediction_string, axis=1)

df2 = df_pred_base.copy()
df3 = df_pred_base.copy()

df2.head()



## === cell 5
df4 = pd.merge(df, df3, on="image_id", how="left")
df5 = pd.merge(df, df2, on="image_id", how="left")

if "PredictionString" not in df4.columns:
    df4["PredictionString"] = "14 1 0 0 1 1"
if "PredictionString" not in df5.columns:
    df5["PredictionString"] = "14 1 0 0 1 1"

(df4.shape, df5.shape, df4.columns[:5])



## === cell 6
n = df4.shape[0]
for i in range(n):
    ps = df4.loc[i, "PredictionString"]
    if not isinstance(ps, str) or len(ps.strip()) == 0:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    list2 = ps.split()
    h = ""
    for j in range(int(len(list2) / 6)):
        cls = list2[0 + 6 * j]
        h = (
            h
            + " "
            + list2[0 + 6 * j]
            + " "
            + list2[1 + 6 * j]
            + " "
            + list2[2 + 6 * j]
            + " "
            + list2[3 + 6 * j]
            + " "
            + list2[4 + 6 * j]
            + " "
            + list2[5 + 6 * j]
        )
    df4.loc[i, "PredictionString"] = h.strip()

for i in range(n):
    ps = df4.loc[i, "PredictionString"]
    if not isinstance(ps, str) or len(ps.strip()) == 0:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
    else:
        df4.loc[i, "PredictionString"] = " ".join(ps.split())

df4.loc[:3, ["image_id", "PredictionString"]]



## === cell 7
df4["PredictionString"] = df4["PredictionString"].fillna("").astype(str)
df5["PredictionString"] = df5["PredictionString"].fillna("").astype(str)

ps4 = df4["PredictionString"].str.strip()
ps5 = df5["PredictionString"].str.strip()
same_mask = ps4 == ps5
df4.loc[same_mask, "PredictionString"] = ps4[same_mask]

diff_mask = ~same_mask
ps4d = ps4[diff_mask]
ps5d = ps5[diff_mask]

ps5d_clean = ps5d.where(ps5d != "14 1 0 0 1 1", "")
ps4d_clean = ps4d.where(ps4d != "14 1 0 0 1 1", "")

df4.loc[diff_mask, "PredictionString"] = (ps4d_clean + " " + ps5d_clean).str.strip()

df4["PredictionString"] = df4["PredictionString"].apply(
    lambda s: " ".join(str(s).split()) if isinstance(s, str) else "14 1 0 0 1 1"
)
df4.loc[:3, ["image_id", "PredictionString"]]



## === cell 8
list1 = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for i in range(df4.shape[0]):
    if df4.loc[i, "PredictionString"] == "14 1 0 0 1 1":
        continue
    a = df4.loc[i, "PredictionString"]
    b = a.split()
    for j in range(int(len(a.split()) / 6)):
        for k in list1:
            if int(float(b[0 + 6 * j])) == k:
                if float(df4.loc[i, f"{k}"]) < 0.92:
                    continue
                c = b[0 + 6 * j + 1]
                b[0 + 6 * j + 1] = str(float(df4.loc[i, f"{k}"]) * 0.4 + float(c) * 0.6)
    df4.loc[i, "PredictionString"] = " ".join(b)

df4.loc[:3, ["image_id", "PredictionString"]]



## === cell 9
list1 = [0, 3]

for i in range(df4.shape[0]):
    if df4.loc[i, "PredictionString"] == "14 1 0 0 1 1":
        continue
    a = df4.loc[i, "PredictionString"]
    b = a.split()
    for j in range(int(len(a.split()) / 6)):
        for k in list1:
            if int(float(b[0 + 6 * j])) == k:
                if float(df_heart_cnn.loc[i, f"{k}"]) < 0.92:
                    continue
                c = b[0 + 6 * j + 1]
                b[0 + 6 * j + 1] = str(
                    float(df_heart_cnn.loc[i, f"{k}"]) * 0.4 + float(c) * 0.6
                )
    df4.loc[i, "PredictionString"] = " ".join(b)

df4.loc[:3, ["image_id", "PredictionString"]]




## === cell 10
def normalize_prediction_string(ps: str) -> str:
    if not isinstance(ps, str):
        return "14 1 0 0 1 1"
    ps = " ".join(ps.split()).strip()
    if ps == "":
        return "14 1 0 0 1 1"
    parts = ps.split()
    if len(parts) % 6 != 0:
        return "14 1 0 0 1 1"
    return ps


df4["PredictionString"] = df4["PredictionString"].apply(normalize_prediction_string)

for i in range(df4.shape[0]):
    if float(df4.loc[i, "14"]) > 0.995:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.head()



## === cell 11
out_path = "submission.csv"
df_final.to_csv(out_path, index=False)

assert os.path.exists(out_path), "submission.csv was not created"
sub_check = pd.read_csv(out_path)
assert list(sub_check.columns) == [
    "image_id",
    "PredictionString",
], f"Wrong columns: {sub_check.columns}"
assert len(sub_check) == len(sample_sub), "Row count mismatch vs sample_submission"
sub_check.head()



## === cell 12
df_final



## === cell 13
df_final.iloc[8, 1]
