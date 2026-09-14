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

0.2595048332452013

# 6. Current score

0.01732

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails because it depends on external Kaggle datasets (e.g., `vin-15-cnn-predict`, `0284-norm`) that are not available in this environment, so all `read_csv('../input/...')` calls error and downstream variables are undefined. To make it run end-to-end and still generate a valid VinBigData submission, I replace those missing ensemble inputs with a deterministic fallback that uses the provided `sample_submission.csv` as a template and outputs the required columns and “No finding” default prediction. I keep your core intent (produce `image_id` + prediction string and write `submission.csv`) and add light safety checks for column names (`PredictionString` vs `TARGET`) so the output is always valid. This yield a low baseline score, but it unblocks execution and produces a correct `.csv` submission file.'
- What this solution (achieved 3e-05) has done: 'Your current code always predicts “No finding” for every image, which caps mAP very low; to move toward the target score we need to output some plausible positive boxes/classes for at least a subset of images using only the provided `train.csv`. I keep your overall flow (read sample submission → build `PredictionString` → write `submission.csv`) but replace the constant default with a tiny, deterministic heuristic that assigns a few common findings with typical bounding boxes. Specifically, I compute per-class “prototype” boxes (median normalized coordinates) and class frequencies from `train.csv`, then for every test image emit the top-K classes with those prototype boxes and a conservative confidence; this should raise recall and mAP versus all-negative while staying simple and fast. I also keep your “No finding” fallback if anything goes wrong and ensure the submission schema stays `image_id,PredictionString`.'
- What this solution (achieved 0.00042) has done: 'Your current heuristic is producing very low mAP mainly because it uses a fixed dummy image size (W0/H0=3000) and very low confidences, so boxes are often badly scaled and scored down. To move toward the target with minimal logic change, I keep the same “prototype boxes from train + emit top-K classes for every test image” approach, but (1) estimate a more realistic global image size from train (max x/y), (2) use class-wise mean box sizes and add a small set of size-quantile variants per class to increase IoU chances without changing the overall method, and (3) slightly raise confidences while keeping them conservative. This should increase recall/IoU alignment and lift mAP substantially versus the current 3e-05, while still staying simple and fast and producing the same required submission format.'
- What this solution (achieved 0.0003) has done: 'Your current score (0.00042) is far below the target (0.2595), so we should improve recall/IoU in the smallest way without changing the core “prototype boxes from train → emit top-K per image” heuristic. The biggest issue is that you scale boxes with a single global (W0,H0) inferred from train bboxes, which can be very mismatched per-image; instead, we estimate each test image’s true width/height from its DICOM header (fast, no pixel decoding) and scale the same normalized prototypes per-image. To better match multiple findings while staying within the same approach, we slightly increase K and emit a small set of size variants per class (still deterministic), with a modest confidence schedule so predictions aren’t ignored. The rest of the pipeline (submission template, formatting, fallback “No finding”) stays unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.03376) has done: 'Your current score (0.0003) is far below the target (0.2595), so we need a meaningful but still minimal improvement in recall/IoU while preserving the same core heuristic: “learn class-wise prototype boxes from `train.csv` and emit top‑K predictions for every test image.” The biggest miss is that we’re always predicting only the most frequent classes, which ignores the strong per-image prior that many images are actually “No finding”; so I add a lightweight per-image “No finding” probability using only `train.csv` label frequencies and include class 14 when appropriate. To better match the metric without changing the approach, I also (a) include class-specific confidence derived from train frequency (still deterministic) and (b) emit a couple of prototype center jitters (small shifts) in addition to size variants to raise IoU chances. All I/O paths stay the same and the script still writes a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.0487) has done: 'Your current score (0.03376) is far below the target (0.2595), so we should improve recall/IoU with the smallest possible change while preserving your core heuristic: class-wise box prototypes from `train.csv` and emitting fixed top‑K predictions per test image. The biggest easy gain is to stop scaling boxes by per-image max bbox (which underestimates true image size) and instead normalize using the DICOM image `Columns/Rows` for train images as well (header-only, fast), making prototypes much more aligned to real coordinates. I also expand the predicted class set slightly (K 6→8) while keeping output bounded, and make “No finding” inclusion a bit less frequent to reduce false positives since you already always emit multiple boxes. All paths and submission schema remain identical, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.04242) has done: 'Your current heuristic is already running end-to-end but is far below the target, so the smallest path upward is to better match the competition’s expected “No finding vs findings” behavior without changing the core prototype-box approach. I (1) estimate a per-image “No finding” propensity from train by counting images with any positive labels, then use that to decide whether to output findings at all (otherwise output only the required `14 1 0 0 1 1`), and (2) when outputting findings, avoid flooding the evaluator by reducing duplicate jitters/variants and raising the best few confidences while keeping the same prototype scaling by DICOM `Rows/Columns`. This keeps your core logic (prototypes from `train.csv` + deterministic emissions) but should improve precision/recall balance and move mAP toward the target. Paths and submission schema stay identical, and it still writes `submission.csv`.'
- What this solution (achieved 0.0325) has done: 'We keep your current “prototype boxes from train.csv + per-test DICOM Rows/Columns scaling + deterministic per-image no-finding gate” core heuristic intact, but adjust two high-impact knobs that are currently suppressing recall too much for mAP@0.4. First, we reduce the probability of outputting only “No finding” (class 14) so more images emit findings, moving the score upward toward the target. Second, we slightly increase the number of predicted classes per image (K) and mildly increase the base confidence so predictions are less likely to be ignored, while keeping the same formatting and still writing a valid `submission.csv`.'
- What this solution (achieved 0.02874) has done: 'Your current score (0.0325) is far below the target (0.2595), so we should increase mAP by improving precision/recall while keeping your exact core heuristic (train-derived prototype boxes + per-test DICOM scaling + deterministic per-image no-finding gate). The biggest low-risk gain is to stop emitting lots of boxes for many classes in every “finding” image; instead we rank classes by how often they appear per-image in train (stronger prior for detection) and only emit a small, high-confidence set, which typically improves mAP for this competition’s evaluator. We also tune the no-finding gate slightly downward to increase recall and keep confidence scaling modestly higher so predictions aren’t ignored, without changing any model/learning (still pure deterministic heuristics from `train.csv`). All paths stay unchanged and the script still writes a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.02546) has done: 'Your current heuristic is still under-shooting the target, so the smallest likely lift is to (1) predict a slightly broader set of classes per image (to improve recall), (2) add one additional size-variant per class (to improve IoU>0.4 hit-rate without changing the prototype-box approach), and (3) make the “no finding only” gate a bit less aggressive so more images emit findings. I keep your exact pipeline (train-derived prototypes + DICOM Rows/Columns scaling + deterministic per-image gate + formatted `PredictionString`) and only tune these knobs. I also clamp confidences to a conservative band so we don’t flood the evaluator with many high-scoring false positives. Output remains a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.0222) has done: 'Your current score (0.02546) is far below the target (0.2595), so we should make a small, low-risk change that increases recall/IoU without changing your core heuristic (train-derived prototypes + DICOM Rows/Columns scaling + per-image no-finding gate + formatted `PredictionString`). The smallest likely gain is to (1) increase the number of ranked classes emitted per “finding” image (K) slightly and (2) add exactly one more size-variant per class (q65) while keeping the same jitters and confidence schedule, which should raise the chance that at least one predicted box overlaps a GT box at IoU>0.4. To avoid flooding with too many false positives, we keep the no-finding gate unchanged and keep confidence clamping conservative. All I/O paths are preserved and the script still writes a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.02105) has done: 'We keep your exact prototype-box heuristic, DICOM header scaling, and per-image “no finding only” gate, but make two minimal adjustments that should lift mAP toward the target by improving recall without exploding false positives. First, we expand the class set slightly (K from 8→10) while reducing duplicates by emitting fewer jitters per size (3→2) so total boxes per image stays similar but covers more classes. Second, we make the “no finding only” gate a bit less aggressive (slightly lower probability) so more images output findings, which is important given your current score is far below target. All paths, formatting, and the required `"14 1 0 0 1 1"` fallback remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.01732) has done: 'Your current score (0.02105) is far below the target (0.2595), so we should increase it with the smallest changes that boost recall/IoU without changing the core heuristic (train-derived class prototypes + DICOM header scaling + deterministic per-image no-finding gate). The biggest low-risk issue is that the per-image gate still suppresses too many positives and the confidence schedule is likely too low for later classes/variants, so I slightly reduce the “no finding only” probability and raise the base confidence a bit while keeping clamping and the same ranking logic. To improve IoU>0.4 hit-rate with minimal extra output, I add one tiny additional center jitter (vertical) but keep total jitter count small. All I/O paths and the required submission schema remain unchanged, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE_INPUT = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test")
TRAIN_DIR = os.path.join(BASE_INPUT, "train")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/input/train.csv"
if not os.path.isdir(TEST_DIR):
    TEST_DIR = "/kaggle/input/test"
if not os.path.isdir(TRAIN_DIR):
    TRAIN_DIR = "/kaggle/input/train"



## === cell 1
sub = pd.read_csv(SAMPLE_SUB_PATH)

if "PredictionString" not in sub.columns:
    if "TARGET" in sub.columns and "image_id" in sub.columns:
        sub = sub.rename(columns={"TARGET": "PredictionString"})
    elif "ID" in sub.columns and "TARGET" in sub.columns:
        sub = sub.rename(columns={"ID": "image_id", "TARGET": "PredictionString"})
    else:
        raise ValueError(
            f"Unexpected sample submission columns: {sub.columns.tolist()}"
        )

sub["PredictionString"] = "14 1 0 0 1 1"
df = sub.copy()



## === cell 2
df



## === cell 3
df3 = df[["image_id"]].copy()



## === cell 4
df4 = pd.merge(df, df3, on="image_id", how="left")



## === cell 5
_ = df4.iloc[1, 0] if len(df4) > 1 else None
_



## === cell 6
for k in range(15):
    if str(k) not in df4.columns:
        df4[str(k)] = 0.0



## === cell 7
df4["PredictionString"] = df4["PredictionString"].astype(str)




## === cell 8
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _load_dicom_sizes(dicom_dir: str, image_ids, max_read: int | None = None):
    sizes = {}
    try:
        import pydicom
    except Exception:
        return sizes

    n = 0
    for iid in image_ids:
        if max_read is not None and n >= max_read:
            break
        fp = os.path.join(dicom_dir, f"{iid}.dicom")
        if not os.path.exists(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            w = int(getattr(ds, "Columns", 0) or 0)
            h = int(getattr(ds, "Rows", 0) or 0)
            if w > 0 and h > 0:
                sizes[iid] = (float(w), float(h))
                n += 1
        except Exception:
            continue
    return sizes


def build_prototypes(train_csv_path: str, train_dir: str):
    tr = pd.read_csv(train_csv_path)

    for c in ["class_id", "x_min", "y_min", "x_max", "y_max"]:
        if c in tr.columns:
            tr[c] = pd.to_numeric(tr[c], errors="coerce")

    tr = tr.dropna(subset=["image_id", "class_id"]).copy()
    tr["class_id"] = tr["class_id"].astype(int)

    img_pos_counts = tr[tr["class_id"] != 14].groupby("image_id").size()
    all_img_ids = tr["image_id"].drop_duplicates()
    img_pos_counts = img_pos_counts.reindex(all_img_ids, fill_value=0)
    nf_rate = float((img_pos_counts == 0).mean()) if len(img_pos_counts) else 0.0

    tr_pos = (
        tr[tr["class_id"] != 14]
        .dropna(subset=["x_min", "y_min", "x_max", "y_max"])
        .copy()
    )
    if tr_pos.empty:
        return {}, [], (3000.0, 3000.0), {}, {}, nf_rate, {}

    tr_pos["class_id"] = tr_pos["class_id"].astype(int)

    unique_train_ids = tr_pos["image_id"].drop_duplicates().tolist()
    train_sizes = _load_dicom_sizes(train_dir, unique_train_ids, max_read=3500)

    g_bbox = (
        tr_pos.groupby("image_id")[["x_max", "y_max"]]
        .max()
        .rename(columns={"x_max": "W_bbox", "y_max": "H_bbox"})
    )
    g_bbox["W_dcm"] = [train_sizes.get(i, (0.0, 0.0))[0] for i in g_bbox.index]
    g_bbox["H_dcm"] = [train_sizes.get(i, (0.0, 0.0))[1] for i in g_bbox.index]

    g_bbox["W"] = g_bbox["W_dcm"].where(g_bbox["W_dcm"] > 1.0, g_bbox["W_bbox"])
    g_bbox["H"] = g_bbox["H_dcm"].where(g_bbox["H_dcm"] > 1.0, g_bbox["H_bbox"])

    tr_pos = tr_pos.merge(
        g_bbox[["W", "H"]], left_on="image_id", right_index=True, how="left"
    )
    tr_pos["W"] = tr_pos["W"].where(tr_pos["W"] > 1, 3000.0)
    tr_pos["H"] = tr_pos["H"].where(tr_pos["H"] > 1, 3000.0)

    tr_pos["nx1"] = (tr_pos["x_min"] / tr_pos["W"]).clip(0, 1)
    tr_pos["ny1"] = (tr_pos["y_min"] / tr_pos["H"]).clip(0, 1)
    tr_pos["nx2"] = (tr_pos["x_max"] / tr_pos["W"]).clip(0, 1)
    tr_pos["ny2"] = (tr_pos["y_max"] / tr_pos["H"]).clip(0, 1)

    prot = (
        tr_pos.groupby("class_id")[["nx1", "ny1", "nx2", "ny2"]]
        .median()
        .to_dict(orient="index")
    )

    tr_pos["nw"] = (tr_pos["nx2"] - tr_pos["nx1"]).clip(1e-6, 1.0)
    tr_pos["nh"] = (tr_pos["ny2"] - tr_pos["ny1"]).clip(1e-6, 1.0)

    size_stats = {}
    for cid, grp in tr_pos.groupby("class_id"):
        size_stats[int(cid)] = {
            "nw_q35": float(grp["nw"].quantile(0.35)),
            "nw_q50": float(grp["nw"].quantile(0.50)),
            "nw_q65": float(grp["nw"].quantile(0.65)),
            "nh_q35": float(grp["nh"].quantile(0.35)),
            "nh_q50": float(grp["nh"].quantile(0.50)),
            "nh_q65": float(grp["nh"].quantile(0.65)),
        }

    img_cls = tr_pos.groupby(["image_id", "class_id"]).size().reset_index(name="n")
    img_cls_counts = img_cls["class_id"].value_counts().sort_values(ascending=False)
    freq_rank = img_cls_counts.index.astype(int).tolist()

    class_counts = tr_pos["class_id"].value_counts()
    max_cnt = float(class_counts.max()) if len(class_counts) else 1.0
    class_prior = {int(k): float(v) / max_cnt for k, v in class_counts.items()}

    max_img_cls = float(img_cls_counts.max()) if len(img_cls_counts) else 1.0
    class_img_prior = {
        int(k): float(v) / max_img_cls for k, v in img_cls_counts.items()
    }

    if len(train_sizes) >= 50:
        ws = pd.Series([v[0] for v in train_sizes.values()])
        hs = pd.Series([v[1] for v in train_sizes.values()])
        Wg = float(ws.median())
        Hg = float(hs.median())
    else:
        Wg = float(tr_pos["x_max"].quantile(0.995))
        Hg = float(tr_pos["y_max"].quantile(0.995))
    if not (Wg > 10 and Hg > 10):
        Wg, Hg = 3000.0, 3000.0

    return prot, freq_rank, (Wg, Hg), size_stats, class_prior, nf_rate, class_img_prior


def format_pred_string(class_id, conf, box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    x1, x2 = sorted([_safe_float(x1), _safe_float(x2)])
    y1, y2 = sorted([_safe_float(y1), _safe_float(y2)])
    if x2 <= x1:
        x2 = x1 + 1.0
    if y2 <= y1:
        y2 = y1 + 1.0
    conf = max(0.01, min(0.99, _safe_float(conf, 0.05)))
    return f"{int(class_id)} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"


prototypes, class_rank, (W0, H0), size_stats, class_prior, nf_rate, class_img_prior = (
    build_prototypes(TRAIN_CSV_PATH, TRAIN_DIR)
)


def get_test_image_sizes(test_dir: str, image_ids):
    sizes = {}
    try:
        import pydicom  # available on Kaggle for this competition environment
    except Exception:
        return sizes

    for iid in image_ids:
        fp = os.path.join(test_dir, f"{iid}.dicom")
        if not os.path.exists(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            w = int(getattr(ds, "Columns", 0) or 0)
            h = int(getattr(ds, "Rows", 0) or 0)
            if w > 0 and h > 0:
                sizes[iid] = (float(w), float(h))
        except Exception:
            continue
    return sizes


test_ids = df4["image_id"].tolist()
test_sizes = get_test_image_sizes(TEST_DIR, test_ids)

K = 10

base_conf = 0.74

top_classes = [c for c in class_rank if c in prototypes][:K]

if len(top_classes) > 0:
    pred_strings = []
    for _img_id in test_ids:
        W_img, H_img = test_sizes.get(_img_id, (W0, H0))

        h = abs(hash(_img_id)) % 1_000_000
        u = h / 1_000_000.0

        p_no_find_only = min(0.22, max(0.02, 0.095 * nf_rate))
        if u < p_no_find_only:
            pred_strings.append("14 1 0 0 1 1")
            continue

        parts = []

        for idx, cid in enumerate(top_classes):
            b = prototypes[cid]
            cx0 = 0.5 * (b["nx1"] + b["nx2"])
            cy0 = 0.5 * (b["ny1"] + b["ny2"])

            ss = size_stats.get(int(cid), None)
            if ss is None:
                sizes = [(b["nx2"] - b["nx1"], b["ny2"] - b["ny1"])]
            else:
                sizes = [
                    (ss["nw_q50"], ss["nh_q50"]),
                    (ss["nw_q35"], ss["nh_q35"]),
                    (ss["nw_q65"], ss["nh_q65"]),
                ]

            jit = [(0.0, 0.0), (-0.015, 0.0), (0.0, -0.015)]

            for j, (nw, nh) in enumerate(sizes[:3]):
                nw = max(1e-4, min(1.0, float(nw)))
                nh = max(1e-4, min(1.0, float(nh)))

                for t, (dx, dy) in enumerate(jit):
                    cx = min(1.0, max(0.0, cx0 + dx))
                    cy = min(1.0, max(0.0, cy0 + dy))

                    nx1 = max(0.0, cx - 0.5 * nw)
                    ny1 = max(0.0, cy - 0.5 * nh)
                    nx2 = min(1.0, cx + 0.5 * nw)
                    ny2 = min(1.0, cy + 0.5 * nh)

                    prior_obj = class_prior.get(int(cid), 0.5)
                    prior_img = class_img_prior.get(int(cid), 0.5)

                    conf = (
                        base_conf
                        * (0.90**idx)
                        * (0.975**t)
                        * (0.90**j)
                        * (0.65 + 0.60 * prior_obj)
                        * (0.70 + 0.55 * prior_img)
                    )

                    x1 = nx1 * W_img
                    y1 = ny1 * H_img
                    x2 = nx2 * W_img
                    y2 = ny2 * H_img
                    parts.append(format_pred_string(cid, conf, (x1, y1, x2, y2)))

        pred_strings.append(" ".join(parts) if parts else "14 1 0 0 1 1")
    df4["PredictionString"] = pred_strings



## === cell 9
df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1")
df4.loc[df4["PredictionString"].astype(str).str.strip().eq(""), "PredictionString"] = (
    "14 1 0 0 1 1"
)



## === cell 10
df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)

print(df_final.head())
print("Wrote submission.csv with shape:", df_final.shape)



## === cell 11
df_final
