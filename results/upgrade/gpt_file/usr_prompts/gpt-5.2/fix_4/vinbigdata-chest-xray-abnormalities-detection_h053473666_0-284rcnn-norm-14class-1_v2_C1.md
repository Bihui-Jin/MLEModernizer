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

0.00042

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails because it depends on external Kaggle datasets (e.g., `vin-15-cnn-predict`, `0284-norm`) that are not available in this environment, so all `read_csv('../input/...')` calls error and downstream variables are undefined. To make it run end-to-end and still generate a valid VinBigData submission, I replace those missing ensemble inputs with a deterministic fallback that uses the provided `sample_submission.csv` as a template and outputs the required columns and “No finding” default prediction. I keep your core intent (produce `image_id` + prediction string and write `submission.csv`) and add light safety checks for column names (`PredictionString` vs `TARGET`) so the output is always valid. This yield a low baseline score, but it unblocks execution and produces a correct `.csv` submission file.'
- What this solution (achieved 3e-05) has done: 'Your current code always predicts “No finding” for every image, which caps mAP very low; to move toward the target score we need to output some plausible positive boxes/classes for at least a subset of images using only the provided `train.csv`. I keep your overall flow (read sample submission → build `PredictionString` → write `submission.csv`) but replace the constant default with a tiny, deterministic heuristic that assigns a few common findings with typical bounding boxes. Specifically, I compute per-class “prototype” boxes (median normalized coordinates) and class frequencies from `train.csv`, then for every test image emit the top-K classes with those prototype boxes and a conservative confidence; this should raise recall and mAP versus all-negative while staying simple and fast. I also keep your “No finding” fallback if anything goes wrong and ensure the submission schema stays `image_id,PredictionString`.'
- What this solution (achieved 0.00042) has done: 'Your current heuristic is producing very low mAP mainly because it uses a fixed dummy image size (W0/H0=3000) and very low confidences, so boxes are often badly scaled and scored down. To move toward the target with minimal logic change, I keep the same “prototype boxes from train + emit top-K classes for every test image” approach, but (1) estimate a more realistic global image size from train (max x/y), (2) use class-wise mean box sizes and add a small set of size-quantile variants per class to increase IoU chances without changing the overall method, and (3) slightly raise confidences while keeping them conservative. This should increase recall/IoU alignment and lift mAP substantially versus the current 3e-05, while still staying simple and fast and producing the same required submission format.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE_INPUT = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/input/train.csv"



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


def build_prototypes(train_csv_path: str):
    tr = pd.read_csv(train_csv_path)
    tr = tr[tr["class_id"] != 14].copy()

    if tr.empty:
        return {}, [], (3000.0, 3000.0), {}

    for c in ["x_min", "y_min", "x_max", "y_max", "class_id"]:
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    tr = tr.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"]).copy()
    tr["class_id"] = tr["class_id"].astype(int)

    Wg = float(tr["x_max"].quantile(0.995))
    Hg = float(tr["y_max"].quantile(0.995))
    if not (Wg > 10 and Hg > 10):
        Wg, Hg = 3000.0, 3000.0

    g = (
        tr.groupby("image_id")[["x_max", "y_max"]]
        .max()
        .rename(columns={"x_max": "W", "y_max": "H"})
    )
    tr = tr.merge(g, left_on="image_id", right_index=True, how="left")
    tr["W"] = tr["W"].where(tr["W"] > 1, 1.0)
    tr["H"] = tr["H"].where(tr["H"] > 1, 1.0)

    tr["nx1"] = (tr["x_min"] / tr["W"]).clip(0, 1)
    tr["ny1"] = (tr["y_min"] / tr["H"]).clip(0, 1)
    tr["nx2"] = (tr["x_max"] / tr["W"]).clip(0, 1)
    tr["ny2"] = (tr["y_max"] / tr["H"]).clip(0, 1)

    prot = (
        tr.groupby("class_id")[["nx1", "ny1", "nx2", "ny2"]]
        .median()
        .to_dict(orient="index")
    )

    tr["nw"] = (tr["nx2"] - tr["nx1"]).clip(1e-6, 1.0)
    tr["nh"] = (tr["ny2"] - tr["ny1"]).clip(1e-6, 1.0)

    size_stats = {}
    for cid, grp in tr.groupby("class_id"):
        size_stats[int(cid)] = {
            "nw_q30": float(grp["nw"].quantile(0.30)),
            "nw_q50": float(grp["nw"].quantile(0.50)),
            "nw_q70": float(grp["nw"].quantile(0.70)),
            "nh_q30": float(grp["nh"].quantile(0.30)),
            "nh_q50": float(grp["nh"].quantile(0.50)),
            "nh_q70": float(grp["nh"].quantile(0.70)),
        }

    freq = tr["class_id"].value_counts().index.astype(int).tolist()
    return prot, freq, (Wg, Hg), size_stats


def format_pred_string(class_id, conf, box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    x1, x2 = sorted([_safe_float(x1), _safe_float(x2)])
    y1, y2 = sorted([_safe_float(y1), _safe_float(y2)])
    if x2 <= x1:
        x2 = x1 + 1.0
    if y2 <= y1:
        y2 = y1 + 1.0
    conf = max(0.0, min(1.0, _safe_float(conf, 0.05)))
    return f"{int(class_id)} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"


prototypes, class_rank, (W0, H0), size_stats = build_prototypes(TRAIN_CSV_PATH)

K = 3  # was 2
base_conf = 0.22  # was 0.12; still conservative but avoids being ignored

top_classes = [c for c in class_rank if c in prototypes][:K]

if len(top_classes) > 0:
    pred_strings = []
    for _img_id in df4["image_id"].tolist():
        parts = []
        for idx, cid in enumerate(top_classes):
            b = prototypes[cid]

            cx = 0.5 * (b["nx1"] + b["nx2"])
            cy = 0.5 * (b["ny1"] + b["ny2"])

            ss = size_stats.get(int(cid), None)
            if ss is None:
                sizes = [(b["nx2"] - b["nx1"], b["ny2"] - b["ny1"])]
            else:
                sizes = [
                    (ss["nw_q50"], ss["nh_q50"]),
                    (ss["nw_q30"], ss["nh_q30"]),
                    (ss["nw_q70"], ss["nh_q70"]),
                ]

            sizes = sizes[:2]

            for j, (nw, nh) in enumerate(sizes):
                nw = max(1e-4, min(1.0, float(nw)))
                nh = max(1e-4, min(1.0, float(nh)))

                nx1 = max(0.0, cx - 0.5 * nw)
                ny1 = max(0.0, cy - 0.5 * nh)
                nx2 = min(1.0, cx + 0.5 * nw)
                ny2 = min(1.0, cy + 0.5 * nh)

                conf = base_conf * (0.82**idx) * (0.88**j)

                x1 = nx1 * W0
                y1 = ny1 * H0
                x2 = nx2 * W0
                y2 = ny2 * H0
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
