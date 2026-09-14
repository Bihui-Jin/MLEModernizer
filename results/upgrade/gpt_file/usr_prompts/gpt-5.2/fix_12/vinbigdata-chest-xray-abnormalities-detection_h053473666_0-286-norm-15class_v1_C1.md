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

0.2904152814104442

# 6. Current score

0.02173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook is failing because it depends on external Kaggle datasets (`vin-15-cnn-predict`, `vin-15-cnn-predict-1`, `0-286-private-norm`) that are not present in this environment, so the very first `read_csv` calls crash and everything downstream is undefined. To make it run end-to-end and still produce a valid submission, I keep the same overall “merge + optional confidence recalibration” flow but fall back to the provided `sample_submission.csv` whenever those external files are missing. I also fix the submission column names to match the competition (`image_id, PredictionString`) and ensure every test image has a non-empty prediction string (defaulting to `14 1 0 0 1 1`). These changes are score-neutral (and likely below target), but they unblock execution and generate a valid `submission.csv`.'
- What this solution (achieved 0.018) has done: 'Your current score is far below the target, so we should improve it with minimal, metric-aligned changes while keeping your “use PredictionString + light confidence recalibration” core flow intact. The biggest issue is that your code always falls back to the sample submission (i.e., predicts “No finding” for essentially everything), which yields very low mAP; we can legitimately improve by generating simple, data-driven boxes from `train.csv` priors (per-class mean box, plus class frequency for confidence). We still output valid `image_id,PredictionString` for every test image, and keep the existing recalibration loop (it just become a small adjustment on top of the new priors). This stays within constraints (no new model, no new loss/training loops) and should move the score upward toward your 0.29 target.'
- What this solution (achieved 0.01829) has done: 'Your current score (0.018) is far below the target (0.2904), so we should legitimately increase mAP with the smallest changes that keep your existing “prior boxes from train.csv + light confidence recalibration” flow. The main issue is that you only predict the top-3 most frequent classes with a single mean box for every image, which yields very low recall; we can raise recall by emitting one prior prediction for every class (0–13) using per-class mean boxes and a frequency-based confidence, while still defaulting to “No finding” when needed. To avoid hurting precision too much, we keep confidences small and cap the number of emitted classes to a safe K (e.g., 8) based on frequency, which is a minimal extension of your current topK=3 logic. We also make the calibration columns match the emitted classes and keep your existing 95/5 blending unchanged.'
- What this solution (achieved 0.01815) has done: 'Your current score is far below the target, so we should increase recall in a minimal, “prior-based” way without changing your overall pipeline (sample submission → add per-class columns → fill No Finding with a fixed set of class priors → light confidence recalibration → write `submission.csv`). The main issue is that you emit the same small set of classes for every image, which caps recall; we instead emit per-image priors: the globally frequent classes plus a few extra “context” classes that commonly co-occur with them in `train.csv`. This keeps your existing semantics (still just priors + calibration), but creates more varied and more plausible multi-label predictions per image, which typically improves mAP at IoU>0.4 versus a single static list. We also ensure the emitted boxes are robust by falling back to class median boxes when means are missing.'
- What this solution (achieved 0.01819) has done: 'Your current score (0.01815) is far below the target (0.2904), so we should push mAP upward with the smallest, metric-aligned changes while keeping your existing “train.csv priors → fill No Finding → light confidence recalibration” flow intact. The biggest limiter is recall: you only generate priors for images that are exactly `No finding` in the template, and you emit a relatively small/low-confidence set; we instead generate priors for *all* test images (overwriting the template safely) and slightly strengthen the confidence scaling while keeping it bounded to avoid extreme false positives. We also make the boxes a bit more robust by using per-class *trimmed* means (reduces outlier impact) but still preserves the same “single prior box per class” core idea. Finally, we keep your existing 95/5 blending loop unchanged so evaluation semantics remain the same, and still guarantee a valid `submission.csv`.'
- What this solution (achieved 0.02402) has done: 'Your current score (0.01819) is far below the target (0.2904), so we should increase mAP mainly by improving recall with minimal, still “prior-based” changes. Keeping your exact pipeline (train.csv priors → build PredictionString for each test image → same 95/5 confidence blending → write submission.csv), I (1) use per-class **normalized** boxes (relative to image width/height) computed from train, and then (2) **scale those boxes to each test image’s actual size** by reading only DICOM headers (fast, no pixel decode). This preserves your core idea of “one prior box per class”, but fixes a major mismatch: one absolute box size does not fit all images, which hurts IoU-based mAP. I also slightly adjust confidence scaling to be a touch more conservative (to reduce false positives) while still increasing recall versus “No finding”.'
- What this solution (achieved 0.02402) has done: 'Your current score (0.024) is far below the target (0.2904), so we should improve mAP mainly by increasing recall in a metric-consistent way without changing your core “train priors → per-test-image scaled boxes → light 95/5 confidence blend → write submission.csv” pipeline. The smallest high-impact fix is to emit not just one “mean” box per class, but a small *mixture* of typical boxes per class (via per-class quantile boxes in normalized coordinates) so predictions cover more spatial modes and achieve IoU>0.4 more often. We keep your existing DICOM-header scaling (fast) and your calibration loop unchanged; we only adjust how the prior boxes are computed/selected and cap the total emitted predictions per image to stay reasonable. This should move the score upward toward the target band while remaining within the same prior-based semantics and runtime limits.'
- What this solution (achieved 0.02402) has done: 'Your current score (0.024) is far below the target (0.2904), so we should increase mAP mainly by improving IoU/recall while keeping your same “train priors → per-test-image scaled boxes → light 95/5 confidence blend → write submission.csv” pipeline intact. The minimal, highest-impact adjustment is to make the per-class “mixture of typical boxes” actually cover multiple spatial modes by emitting more robust quantile prototypes (including more extreme quantiles) and allowing one additional prototype per class, still under a strict total box cap. To avoid precision collapsing, we slightly downweight the extra prototypes’ confidences and keep your existing frequency-based confidence computation and calibration loop unchanged. Finally, we add a small safety clamp so all confidences stay in [0, 1] after blending (valid and slightly stabilizing for evaluation).'
- What this solution (achieved 0.02322) has done: 'Your current score (0.024) is far below the target (0.2904), so we should legitimately increase mAP with the smallest changes that improve IoU>0.4 while preserving your exact “train priors → scale to test DICOM sizes → 95/5 confidence blend → submission.csv” pipeline. The lowest-risk, highest-impact tweak here is to stop emitting quantile *coordinate-wise* boxes (which can create invalid/low-IoU boxes) and instead emit a small mixture of *real* prototype boxes per class by clustering training boxes in normalized space (simple k-means-like iterations implemented in pandas/numpy). This keeps the same prior-based semantics (still a few typical boxes per class, still scaled per image) but makes each prototype correspond to plausible box shapes/locations, typically improving IoU coverage and recall. I also add a tiny box-area sanity clamp and enforce x2>x1,y2>y1 in normalized space before scaling (stability, not a logic change). Everything else (chosen classes, confidence scaling, calibration blend, I/O paths, submission schema) stays the same.'
- What this solution (achieved 0.01963) has done: 'Your current score (0.02322) is far below the target (0.2904), so we should increase mAP mainly by improving recall while keeping your exact “train priors → per-test-image scaled boxes → 95/5 confidence blend → submission.csv” pipeline intact. The minimal, highest-impact gap here is that each class only emits up to 3 prototypes, and those prototypes are learned with plain L2 k-means on raw coordinates, which tends to under-cover spatial modes that matter for IoU>0.4. I (1) make the prototype generation slightly more mode-covering by increasing k from 3→5 with the same loop (still the same core logic: cluster boxes), and (2) select which prototypes to emit by favoring larger-area prototypes first (often closer to “covering” lesions at IoU 0.4), while keeping the same hard caps on total boxes and the same confidence blending. These changes are small, preserve your semantics, stay fast, and should move the score upward toward the target band without changing the overall approach.'
- What this solution (achieved 0.02173) has done: 'Your current score (0.01963) is far below the target (0.2904), so we should increase it with the smallest changes that improve IoU>0.4 while keeping your same prior-based pipeline (train-derived prototypes → scale to each test DICOM size → 95/5 confidence blend → submission.csv). The main issue is that you only emit up to 3 prototypes per class and cap total boxes at 22, which heavily limits recall; increasing those caps slightly (without changing the prototype generation method) should move mAP upward. To avoid precision collapse, we keep the same class-selection logic and confidence formula, but apply a mild extra downweight to the additional (newly allowed) prototypes only. Finally, we fix a small bug in k-means seeding (quantile of indices) that can pick poor initial centers and reduce prototype quality, while still keeping the same k-means core logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_DIR_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]


def find_first_existing_file(rel_path: str):
    for base in DATA_DIR_CANDIDATES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None


sample_path = find_first_existing_file(
    "sample_submission.csv"
) or find_first_existing_file(
    "vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv"
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input paths."
    )

sample_sub = pd.read_csv(sample_path)

expected_cols = ["image_id", "PredictionString"]
if list(sample_sub.columns) != expected_cols:
    colmap = {}
    if "ID" in sample_sub.columns:
        colmap["ID"] = "image_id"
    if "TARGET" in sample_sub.columns:
        colmap["TARGET"] = "PredictionString"
    sample_sub = sample_sub.rename(columns=colmap)
    if ("image_id" not in sample_sub.columns) or (
        "PredictionString" not in sample_sub.columns
    ):
        sample_sub = sample_sub.iloc[:, :2].copy()
        sample_sub.columns = expected_cols

NO_FINDING_STR = "14 1 0 0 1 1"
sample_sub["PredictionString"] = sample_sub["PredictionString"].fillna(NO_FINDING_STR)
sample_sub.loc[
    sample_sub["PredictionString"].astype(str).str.len() == 0, "PredictionString"
] = NO_FINDING_STR

train_path = find_first_existing_file("train.csv") or find_first_existing_file(
    "vinbigdata-chest-xray-abnormalities-detection/train.csv"
)
train_df = None
if train_path is not None and os.path.exists(train_path):
    train_df = pd.read_csv(train_path)

df = sample_sub.copy()



## === cell 1
df



## === cell 2
df3 = pd.DataFrame({"image_id": df["image_id"].values})
for k in range(15):
    df3[str(k)] = 0.0
df3.head()



## === cell 3
df4 = pd.merge(df, df3, on="image_id", how="left")
df4.head()




## === cell 4
def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


try:
    import pydicom  # type: ignore

    _HAVE_PYDICOM = True
except Exception:
    pydicom = None
    _HAVE_PYDICOM = False


def get_dicom_hw(dicom_path: str):
    if not _HAVE_PYDICOM:
        return None
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        h = _safe_int(getattr(ds, "Rows", None))
        w = _safe_int(getattr(ds, "Columns", None))
        if h is None or w is None or h <= 0 or w <= 0:
            return None
        return (h, w)
    except Exception:
        return None


TEST_DIR_CANDIDATES = [
    find_first_existing_file("test"),
    find_first_existing_file("vinbigdata-chest-xray-abnormalities-detection/test"),
]
TEST_DIR = next(
    (p for p in TEST_DIR_CANDIDATES if p is not None and os.path.isdir(p)), None
)

img_hw = {}
if TEST_DIR is not None and _HAVE_PYDICOM:
    for img_id in df4["image_id"].values.tolist():
        dp = os.path.join(TEST_DIR, f"{img_id}.dicom")
        if os.path.exists(dp):
            hw = get_dicom_hw(dp)
            if hw is not None:
                img_hw[img_id] = hw

if train_df is not None:
    req_cols = {"image_id", "class_id", "x_min", "y_min", "x_max", "y_max"}
    TRAIN_DIR_CANDIDATES = [
        find_first_existing_file("train"),
        find_first_existing_file("vinbigdata-chest-xray-abnormalities-detection/train"),
    ]
    TRAIN_DIR = next(
        (p for p in TRAIN_DIR_CANDIDATES if p is not None and os.path.isdir(p)), None
    )

    if req_cols.issubset(set(train_df.columns)):
        t = train_df.copy()
        t = t[t["class_id"].between(0, 13)].copy()

        for c in ["x_min", "y_min", "x_max", "y_max"]:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        t["class_id"] = pd.to_numeric(t["class_id"], errors="coerce")
        t = t.dropna(
            subset=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"]
        )
        t["class_id"] = t["class_id"].astype(int)

        t = t[(t["x_max"] > t["x_min"]) & (t["y_max"] > t["y_min"])]

        train_hw = {}
        if TRAIN_DIR is not None and _HAVE_PYDICOM:
            uniq_train_imgs = t["image_id"].drop_duplicates().values.tolist()
            cap = min(2500, len(uniq_train_imgs))
            for img_id in uniq_train_imgs[:cap]:
                dp = os.path.join(TRAIN_DIR, f"{img_id}.dicom")
                if os.path.exists(dp):
                    hw = get_dicom_hw(dp)
                    if hw is not None:
                        train_hw[img_id] = hw

        use_norm = (
            len(train_hw) > 200
        )  # require enough samples for stable normalization

        proto_norm = None

        def _clip01(x: float):
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return x

        if use_norm:
            tt = t[t["image_id"].isin(train_hw.keys())].copy()
            if tt.shape[0] > 0:

                def _norm_row(r):
                    h, w = train_hw.get(r["image_id"], (None, None))
                    if h is None or w is None:
                        return pd.Series(
                            [pd.NA, pd.NA, pd.NA, pd.NA],
                            index=["x1n", "y1n", "x2n", "y2n"],
                        )
                    return pd.Series(
                        [
                            float(r["x_min"]) / float(w),
                            float(r["y_min"]) / float(h),
                            float(r["x_max"]) / float(w),
                            float(r["y_max"]) / float(h),
                        ],
                        index=["x1n", "y1n", "x2n", "y2n"],
                    )

                tt[["x1n", "y1n", "x2n", "y2n"]] = tt.apply(_norm_row, axis=1)
                tt = tt.dropna(subset=["x1n", "y1n", "x2n", "y2n"]).copy()

                for c in ["x1n", "y1n", "x2n", "y2n"]:
                    tt[c] = tt[c].astype(float).clip(0.0, 1.0)
                tt = tt[(tt["x2n"] > tt["x1n"]) & (tt["y2n"] > tt["y1n"])].copy()
                tt["arean"] = (tt["x2n"] - tt["x1n"]) * (tt["y2n"] - tt["y1n"])
                tt = tt[(tt["arean"] >= 1e-4) & (tt["arean"] <= 0.95)].copy()

                def trimmed_mean_box_norm(g: pd.DataFrame, q: float = 0.05):
                    gg = g
                    for col in ["x1n", "y1n", "x2n", "y2n"]:
                        lo = gg[col].quantile(q)
                        hi = gg[col].quantile(1 - q)
                        gg = gg[(gg[col] >= lo) & (gg[col] <= hi)]
                    if gg.shape[0] == 0:
                        gg = g
                    return gg[["x1n", "y1n", "x2n", "y2n"]].mean()

                box_trim_mean_norm = (
                    tt.groupby("class_id", sort=True)
                    .apply(trimmed_mean_box_norm)
                    .reset_index()
                )
                box_trim_mean_norm = box_trim_mean_norm.set_index("class_id")[
                    ["x1n", "y1n", "x2n", "y2n"]
                ]
                box_median_norm = tt.groupby("class_id")[
                    ["x1n", "y1n", "x2n", "y2n"]
                ].median()

                def _kmeans_boxes(g: pd.DataFrame, k: int = 5, iters: int = 6):
                    X = g[["x1n", "y1n", "x2n", "y2n"]].to_numpy(dtype=np.float64)
                    n = X.shape[0]
                    if n == 0:
                        return np.zeros((0, 4), dtype=np.float64)
                    if n <= k:
                        return X.copy()

                    area = (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
                    qs = np.linspace(0.12, 0.88, k)
                    target_areas = np.quantile(area, qs)
                    seed_idx = [
                        int(np.argmin(np.abs(area - ta))) for ta in target_areas
                    ]
                    C = X[seed_idx, :].copy()

                    for _ in range(iters):
                        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
                        a = d2.argmin(axis=1)
                        for j in range(C.shape[0]):
                            m = X[a == j]
                            if m.shape[0] > 0:
                                C[j, :] = m.mean(axis=0)
                    C[:, 0] = np.clip(C[:, 0], 0.0, 0.999)
                    C[:, 1] = np.clip(C[:, 1], 0.0, 0.999)
                    C[:, 2] = np.clip(C[:, 2], 0.001, 1.0)
                    C[:, 3] = np.clip(C[:, 3], 0.001, 1.0)
                    C[:, 2] = np.maximum(C[:, 2], C[:, 0] + 1e-3)
                    C[:, 3] = np.maximum(C[:, 3], C[:, 1] + 1e-3)
                    return C

                protos = []
                for cls_id, g in tt.groupby("class_id", sort=True):
                    C = _kmeans_boxes(g, k=5, iters=6)

                    if C.shape[0] > 0:
                        areas = (C[:, 2] - C[:, 0]) * (C[:, 3] - C[:, 1])
                        order = np.argsort(-areas)  # descending area
                        C = C[order, :]

                    for pi in range(C.shape[0]):
                        protos.append(
                            {
                                "class_id": int(cls_id),
                                "p": int(pi),
                                "x1n": float(C[pi, 0]),
                                "y1n": float(C[pi, 1]),
                                "x2n": float(C[pi, 2]),
                                "y2n": float(C[pi, 3]),
                            }
                        )
                proto_norm = pd.DataFrame(protos) if len(protos) else None
            else:
                use_norm = False

        if not use_norm:

            def trimmed_mean_box_abs(g: pd.DataFrame, q: float = 0.05):
                gg = g
                for col in ["x_min", "y_min", "x_max", "y_max"]:
                    lo = gg[col].quantile(q)
                    hi = gg[col].quantile(1 - q)
                    gg = gg[(gg[col] >= lo) & (gg[col] <= hi)]
                if gg.shape[0] == 0:
                    gg = g
                return gg[["x_min", "y_min", "x_max", "y_max"]].mean()

            box_trim_mean_abs = (
                t.groupby("class_id", sort=True)
                .apply(trimmed_mean_box_abs)
                .reset_index()
            )
            box_trim_mean_abs = box_trim_mean_abs.set_index("class_id")[
                ["x_min", "y_min", "x_max", "y_max"]
            ]
            box_median_abs = t.groupby("class_id")[
                ["x_min", "y_min", "x_max", "y_max"]
            ].median()

        cls_counts = t["class_id"].value_counts().sort_index()
        total = float(cls_counts.sum()) if float(cls_counts.sum()) > 0 else 1.0
        cls_freq = (cls_counts / total).to_dict()

        img_classes = t.groupby("image_id")["class_id"].apply(
            lambda s: sorted(set(int(x) for x in s.tolist()))
        )
        co_counts = {}
        for cls_list in img_classes.tolist():
            for i in range(len(cls_list)):
                a = cls_list[i]
                for j in range(i + 1, len(cls_list)):
                    b = cls_list[j]
                    co_counts[(a, b)] = co_counts.get((a, b), 0) + 1

        co_top = {c: [] for c in range(14)}
        for (a, b), cnt in co_counts.items():
            co_top[a].append((b, cnt))
            co_top[b].append((a, cnt))
        for c in co_top:
            co_top[c] = [
                x for x, _ in sorted(co_top[c], key=lambda z: z[1], reverse=True)[:5]
            ]

        top_classes = sorted(cls_freq.items(), key=lambda x: x[1], reverse=True)
        global_top = [int(c) for c, _ in top_classes]

        baseK = min(8, len(global_top))
        ctxK = 3

        max_emit_classes = 10  # number of classes to consider per image

        max_boxes_per_class = 4
        max_total_boxes = 30

        df4["PredictionString"] = (
            df4["PredictionString"].fillna(NO_FINDING_STR).astype(str)
        )

        def get_boxes_for_class(cls_id: int, image_id: str):
            if use_norm and (image_id in img_hw):
                h, w = img_hw[image_id]
                out = []

                if proto_norm is not None and proto_norm.shape[0] > 0:
                    sub = proto_norm[proto_norm["class_id"] == int(cls_id)]
                    if sub.shape[0] > 0:
                        sub = sub.sort_values("p")
                        for _, r in sub.head(max_boxes_per_class).iterrows():
                            x1 = _clip01(float(r["x1n"])) * float(w)
                            y1 = _clip01(float(r["y1n"])) * float(h)
                            x2 = _clip01(float(r["x2n"])) * float(w)
                            y2 = _clip01(float(r["y2n"])) * float(h)
                            out.append((x1, y1, x2, y2))
                        if len(out) > 0:
                            return out

                if cls_id in box_trim_mean_norm.index:
                    r = box_trim_mean_norm.loc[cls_id]
                elif cls_id in box_median_norm.index:
                    r = box_median_norm.loc[cls_id]
                else:
                    return []
                x1 = _clip01(float(r["x1n"])) * float(w)
                y1 = _clip01(float(r["y1n"])) * float(h)
                x2 = _clip01(float(r["x2n"])) * float(w)
                y2 = _clip01(float(r["y2n"])) * float(h)
                return [(x1, y1, x2, y2)]

            if (not use_norm) and (cls_id in box_trim_mean_abs.index):
                r = box_trim_mean_abs.loc[cls_id]
                return [
                    (
                        float(r["x_min"]),
                        float(r["y_min"]),
                        float(r["x_max"]),
                        float(r["y_max"]),
                    )
                ]
            if (not use_norm) and (cls_id in box_median_abs.index):
                r = box_median_abs.loc[cls_id]
                return [
                    (
                        float(r["x_min"]),
                        float(r["y_min"]),
                        float(r["x_max"]),
                        float(r["y_max"]),
                    )
                ]
            return []

        for i in df4.index:
            image_id = df4.loc[i, "image_id"]

            chosen = []
            chosen.extend(global_top[:baseK])

            for seed_cls in chosen[: min(4, len(chosen))]:
                for ctx in co_top.get(int(seed_cls), [])[:ctxK]:
                    if ctx not in chosen:
                        chosen.append(ctx)
                    if len(chosen) >= max_emit_classes:
                        break
                if len(chosen) >= max_emit_classes:
                    break

            parts = []
            emitted = 0

            for cls in chosen[:max_emit_classes]:
                boxes = get_boxes_for_class(int(cls), image_id)
                if not boxes:
                    continue

                freq = float(cls_freq.get(int(cls), 0.0))
                base_conf = float(min(0.55, max(0.03, freq * 3.8)))

                for bi, box in enumerate(boxes[:max_boxes_per_class]):
                    if emitted >= max_total_boxes:
                        break

                    x1, y1, x2, y2 = box

                    if bi == 0:
                        conf = base_conf
                    elif bi == 1:
                        conf = base_conf * 0.85
                    elif bi == 2:
                        conf = base_conf * 0.72
                    else:
                        conf = base_conf * 0.60

                    x1i = int(max(0, round(x1)))
                    y1i = int(max(0, round(y1)))
                    x2i = int(max(x1i + 1, round(x2)))
                    y2i = int(max(y1i + 1, round(y2)))

                    parts.extend(
                        [
                            str(int(cls)),
                            f"{conf:.6f}",
                            str(x1i),
                            str(y1i),
                            str(x2i),
                            str(y2i),
                        ]
                    )
                    emitted += 1

                df4.loc[i, str(int(cls))] = float(base_conf)

                if emitted >= max_total_boxes:
                    break

            prior_ps = " ".join(parts).strip()
            if len(prior_ps) == 0:
                prior_ps = NO_FINDING_STR
            df4.loc[i, "PredictionString"] = prior_ps



## === cell 5
list1 = list(range(15))

df4["PredictionString"] = df4["PredictionString"].fillna(NO_FINDING_STR).astype(str)


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


for i in range(df4.shape[0]):
    ps = df4.loc[i, "PredictionString"]
    if ps == NO_FINDING_STR:
        continue

    b = ps.split()
    if len(b) % 6 != 0:
        continue

    for j in range(len(b) // 6):
        try:
            cls = int(float(b[0 + 6 * j]))
        except Exception:
            continue

        if cls in list1:
            try:
                c = float(b[0 + 6 * j + 1])
                calib = float(df4.loc[i, f"{cls}"])
                new_c = calib * 0.05 + c * 0.95
                b[0 + 6 * j + 1] = str(_clamp01(float(new_c)))
            except Exception:
                continue

    new_ps = " ".join(b).strip()
    if len(new_ps) == 0:
        new_ps = NO_FINDING_STR
    df4.loc[i, "PredictionString"] = new_ps

df4["PredictionString"] = df4["PredictionString"].fillna(NO_FINDING_STR)
df4.loc[df4["PredictionString"].astype(str).str.len() == 0, "PredictionString"] = (
    NO_FINDING_STR
)



## === cell 6
df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)

assert (
    df_final.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample_submission."
assert list(df_final.columns) == [
    "image_id",
    "PredictionString",
], "Submission columns must match required format."



## === cell 7
df_final.head()
