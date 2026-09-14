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

0.2125188729757582

# 6. Current score

0.00424

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails because it depends on two external Kaggle dataset outputs (`vinbigdata-2-class-classifier-complete-pipeline` and `vinbigdata-post-processing`) that are not available in this environment, so the CSV reads throw `FileNotFoundError`. I make the smallest change to keep the same post-processing logic, but add safe fallbacks: if those files are missing, we use the provided `sample_submission.csv` as the detection submission base and create a neutral 2-class prediction table aligned to `image_id`. This guarantees the merge works, `class0` exists, and a valid `submission.csv` is written end-to-end. The produced submission be valid format-wise; without the missing model/postprocess inputs, score improvement can’t be meaningfully targeted, but this fixes execution and output generation.'
- What this solution (achieved 0.0475) has done: 'Your current score is far below the target, and the biggest issue is that the “2-class” gating (`class0`) is being set to a constant 0.5 fallback, which mostly appends a “No finding” box to every image and hurts mAP. I keep your post-processing logic intact, but change only the fallback `class0` estimation to be data-driven from `train.csv` (per-class no-finding prior) so that fewer images get an incorrect “No finding” appended/replaced. I also make the merge robust by ensuring `class0` exists and is clipped to a valid probability range. This is a minimal change and should move the score upward toward your target without changing model architecture/training (none exists here).'
- What this solution (achieved 0.0475) has done: 'Your current score is far below the target, and the biggest lever within your existing logic is the 2-class “No finding” gating (`class0`) plus the add/replace thresholds. I keep your exact post-processing semantics (keep/add/replace with `NORMAL`) but make the fallback `class0` (when the external 2-class model file is missing) image-specific instead of a constant prior by estimating a “no-finding likelihood” from the training annotations’ average box area per image. Then I set conservative thresholds (`low_threshold`, `high_threshold`) so you only add/replace “No finding” when the fallback is truly high, which should reduce the widespread incorrect “No finding” boxes that depress mAP. These are minimal changes that preserve your pipeline structure and still write a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your score is far below the target, and with your current pipeline the main controllable lever is the “No finding” gating that appends/replaces class 14 based on `class0`. Right now your fallback `class0` is not image-specific enough and tends to add “No finding” too often, which harms mAP; I keep the same keep/add/replace logic but make `class0` a better per-image estimate using only `train.csv` statistics that are available here. Concretely, I compute a per-image “no finding likelihood” from both (a) whether the image is annotated as only class 14 and (b) how many non-14 boxes it tends to have (images that look like “busy” positives should get lower `class0`). Finally, I make the add/replace thresholds slightly more conservative so we append/replace “No finding” less frequently, which should move the score upward toward your target without changing core semantics.'
- What this solution (achieved 0.0475) has done: 'I fix the runtime error by reading `rad_id` in the fallback `train.csv` loader (it was omitted from `usecols`, causing the `KeyError`), which also unblocks creation of `pred_2class` so cell 2 can merge successfully. I also make the `rad_id`-based aggregation robust in case `rad_id` is missing or read as all-NaN, falling back to an image-level “no finding” frequency. Finally, I ensure the submission is written with the correct column name and `.csv` suffix exactly as required, without changing your keep/add/replace post-processing semantics.'
- What this solution (achieved 0.0475) has done: 'Your current score (0.0475) is far below the target (0.2125), so we should cautiously increase it by reducing the main known failure mode: incorrectly adding/replacing predictions with “No finding” (class 14) too often. I keep your exact keep/add/replace post-processing logic, but make the fallback `class0` estimation more conservative and image-aware by calibrating it from `train.csv` using only “no-finding frequency” and “non-14 box count” statistics (no new models, no new files). Concretely, I (1) lower the default prior toward the true no-finding rate but cap it tighter, (2) penalize images that historically have many findings so they rarely get class-14 appended, and (3) slightly raise the add/replace thresholds so “No finding” is only injected when the fallback probability is very high. This should move mAP upward toward your target while preserving your pipeline structure and producing a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your current score is far below the target, so the safest way to move upward (without changing your pipeline structure) is to stop injecting “No finding” into almost every image via an overly-high fallback `class0`. I keep your keep/add/replace logic exactly the same, but make the fallback `class0` more conservative and better calibrated from `train.csv` by estimating a global no-finding prior and (crucially) shrinking per-image `class0` downward so it rarely crosses your `low_threshold/high_threshold`. This should reduce incorrect class-14 additions/replacements, which typically depress mAP, while still producing a valid submission. I also ensure that if your detection base is the sample submission (i.e., no detector file), we do not append “No finding” to the existing “No finding” rows (avoids duplicate class-14 tokens).'
- What this solution (achieved 0.0475) has done: 'We keep your keep/add/replace “No finding” post-processing exactly the same, but make the fallback `class0` (used when the external 2-class model file is missing) more realistic for this competition by estimating the overall No Finding rate from `train.csv` and then setting a conservative per-image probability around that prior. This avoids the current failure mode where many images cross your `low_threshold/high_threshold` and get “14 …” appended/replaced too often, which can severely depress mAP when your detector base is weak (sample submission). The change is minimal (only how `pred_2class["class0"]` is produced in fallback), keeps I/O paths, and still writes a valid `submission.csv`. With your target much higher than current, this should move the score upward by reducing harmful “No finding” injections while keeping your existing semantics intact.'
- What this solution (achieved 0.0475) has done: 'Your current score is far below the target, so we should increase it by removing the main known failure mode in your pipeline: adding/replacing with “No finding” (class 14) too often when the 2-class file is missing and `class0` is only a weak heuristic. I keep your exact keep/add/replace logic and thresholds, but make the fallback `class0` estimate much more conservative by (1) using the global “No finding only” prior from `train.csv`, and (2) shrinking all per-image probabilities downward so almost nothing crosses your `low_threshold/high_threshold` (which avoids injecting incorrect class-14 predictions on top of your detector base). This is a minimal change (only the fallback `class0` construction), preserves evaluation semantics, and still writes a valid `submission.csv` with the required columns. If the external 2-class file exists, nothing changes.'
- What this solution (achieved 0.018) has done: 'Your score is far below the target, so we should increase it with the smallest change that reduces an obvious failure mode: when your detector base is just the `sample_submission`, any “No finding” gating cannot create real lesion boxes, so the best we can do is avoid making things worse. I keep your keep/add/replace logic and thresholds exactly the same, but change the fallback so `pred_det_df` comes from a simple, legitimate baseline built from `train.csv` (per-class average box + frequency), rather than the sample’s all-normal predictions. This preserves your overall pipeline (2-class merge + postprocess) but gives non-trivial boxes/classes on test, which should move mAP upward toward your target. If the external postprocessed detector CSV exists, nothing changes.'
- What this solution (achieved 0.004) has done: 'We keep your exact keep/add/replace “No finding” post-processing, thresholds, and output format, but make the fallback detector baseline (used when the external detector CSV is missing) much less destructive to mAP by emitting one reasonable box per image instead of the same few boxes for all images. The smallest legitimate improvement is to derive per-image boxes directly from `train.csv` by reusing the most common class and sampling an annotated box for that class (deterministically via a fixed seed), which increases variety and gives some chance of IoU>0.4 on positives. This doesn’t change model/training (none here), preserves your pipeline structure, and still finishes quickly by sampling a limited number of rows. If the external detector/postprocess files exist, behavior remains unchanged.'
- What this solution (achieved 0.01805) has done: 'Your current score (0.004) is far below the target (0.2125), and the main limiter is the fallback detector: it predicts the same single “top class” for every test image, which is extremely unlikely to overlap true findings and yields near-zero mAP. Keeping your exact pipeline (same thresholds and keep/add/replace “No finding” logic), I only change the fallback detector construction to a slightly stronger but still lightweight, train.csv-derived prior: emit a small fixed set of the most frequent classes with their average boxes and modest confidences for every image. This preserves your post-processing semantics and I/O while materially increasing the chance of matching at least some true classes/boxes and moving the score upward toward the target. If the external detector/postprocess CSV exists, nothing changes.'
- What this solution (achieved 0.00909) has done: 'Your current score is far below the target, so we should increase it with the smallest change that affects actual mAP: improve the fallback detector predictions (used when the external detector CSV is missing) without changing your overall keep/add/replace “No finding” gating logic. Right now the fallback emits the same mean box(es) for every test image, which almost never overlaps true boxes; instead, we deterministically sample realistic boxes from `train.csv` per class and attach a small set of top classes per image to add diversity and increase IoU hit chance. We keep your thresholds, NORMAL string, merge semantics, and output format unchanged; only the fallback detector construction changes, and it stays lightweight (sampling a limited pool per class) to fit the timeout. If the external `submission_postprocessed.csv` exists, behavior remains identical.'
- What this solution (achieved 0.00424) has done: 'Your current score is far below the target, and the main bottleneck is that the fallback detector (used when the external postprocessed detector CSV is missing) is effectively a weak prior that rarely matches the true per-image class/box distribution. Keeping your “2-class gate + keep/add/replace NORMAL” logic intact, I only strengthen the fallback detector in a minimal, legitimate way by making it image-specific: for each test image, deterministically sample a small set of realistic boxes/classes from `train.csv` (instead of using the same top classes/mean boxes for everyone). This increases class/box diversity and makes it more likely that some predictions overlap true findings at IoU>0.4, moving mAP upward toward your target without changing model/training (none exists). All I/O paths stay the same and the script still writes a valid `submission.csv` with `image_id,PredictionString`.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(42)

pred_2class_path = "../input/vinbigdata-2-class-classifier-complete-pipeline/results/tmp_debug/test_pred.csv"

low_threshold = 0.85
high_threshold = 0.995

if os.path.exists(pred_2class_path):
    pred_2class = pd.read_csv(pred_2class_path)
else:
    sample_path = (
        "../input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv"
    )
    if not os.path.exists(sample_path):
        sample_path = "../input/sample_submission.csv"
    sample_df = pd.read_csv(sample_path)

    train_path = "../input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
    if not os.path.exists(train_path):
        train_path = "../input/train.csv"

    default_prior = 0.10

    if os.path.exists(train_path):
        train_df = pd.read_csv(
            train_path,
            usecols=[
                "image_id",
                "class_id",
                "rad_id",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            ],
        )

        by_img_cls = train_df.groupby("image_id")["class_id"].agg(["min", "max"])
        no_finding_only = (by_img_cls["min"] == 14) & (by_img_cls["max"] == 14)
        default_prior = float(no_finding_only.mean())
        default_prior = float(np.clip(default_prior, 0.05, 0.35))

        pred_2class = sample_df[["image_id"]].copy()

        nf_freq_by_img = None
        if "rad_id" in train_df.columns and train_df["rad_id"].notna().any():
            by_img_rad = train_df.groupby(["image_id", "rad_id"])["class_id"].agg(
                ["min", "max"]
            )
            nf_by_img_rad = (
                (by_img_rad["min"] == 14) & (by_img_rad["max"] == 14)
            ).astype("float32")
            nf_freq_by_img = nf_by_img_rad.groupby("image_id").mean()  # in [0,1]

        if nf_freq_by_img is not None and len(nf_freq_by_img) > 0:
            pred_2class = pred_2class.merge(
                nf_freq_by_img.rename("nf_freq"),
                on="image_id",
                how="left",
            )
            pred_2class["nf_freq"] = (
                pred_2class["nf_freq"]
                .fillna(np.float32(default_prior))
                .astype("float32")
            )

            base = np.float32(default_prior)

            shrink = np.float32(0.00)  # keep identical behavior

            pred_2class["class0"] = (
                base + shrink * (pred_2class["nf_freq"] - base)
            ).astype("float32")

            pred_2class["class0"] = pred_2class["class0"].clip(
                0.001, np.float32(low_threshold - 1e-4)
            )
            pred_2class = pred_2class[["image_id", "class0"]]
        else:
            pred_2class["class0"] = np.float32(default_prior)
            pred_2class["class0"] = pred_2class["class0"].clip(
                0.001, np.float32(low_threshold - 1e-4)
            )
    else:
        pred_2class = sample_df[["image_id"]].copy()
        pred_2class["class0"] = np.float32(np.clip(default_prior, 0.05, 0.35))
        pred_2class["class0"] = pred_2class["class0"].clip(
            0.001, np.float32(low_threshold - 1e-4)
        )

pred_2class




## === cell 1
NORMAL = "14 1 0 0 1 1"

pred_det_path = "../input/vinbigdata-post-processing/submission_postprocessed.csv"

if os.path.exists(pred_det_path):
    pred_det_df = pd.read_csv(pred_det_path)
else:
    sample_path = (
        "../input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv"
    )
    if not os.path.exists(sample_path):
        sample_path = "../input/sample_submission.csv"
    pred_det_df = pd.read_csv(sample_path)

    train_path = "../input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
    if not os.path.exists(train_path):
        train_path = "../input/train.csv"

    if os.path.exists(train_path):
        train_df = pd.read_csv(
            train_path,
            usecols=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
        )

        det_df = train_df[train_df["class_id"] != 14].copy()
        det_df = det_df.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])
        det_df["class_id"] = det_df["class_id"].astype(int)

        if len(det_df) > 0:
            K = 5
            top_classes = (
                det_df["class_id"].value_counts().head(K).index.astype(int).tolist()
            )

            per_class_pool = {}
            max_pool = 800  # still small enough; a bit larger to increase variety
            for cls in top_classes:
                cls_rows = det_df.loc[
                    det_df["class_id"] == cls, ["x_min", "y_min", "x_max", "y_max"]
                ]
                if len(cls_rows) == 0:
                    continue
                if len(cls_rows) > max_pool:
                    cls_rows = cls_rows.sample(n=max_pool, random_state=42)
                per_class_pool[int(cls)] = cls_rows.reset_index(drop=True)

            global_pool = det_df[["class_id", "x_min", "y_min", "x_max", "y_max"]]
            if len(global_pool) > 6000:
                global_pool = global_pool.sample(n=6000, random_state=42)
            global_pool = global_pool.reset_index(drop=True)

            base_conf = 0.22
            step = 0.03

            pred_strings = []
            for img_id in pred_det_df["image_id"].astype(str).tolist():
                seed = (abs(hash(img_id)) ^ 42) % (2**32 - 1)
                rng = np.random.RandomState(seed)

                toks = []

                n_global = 2
                for r in range(K):
                    if r < n_global and len(global_pool) > 0:
                        row = global_pool.iloc[int(rng.randint(0, len(global_pool)))]
                        cls = int(row["class_id"])
                        x1, y1, x2, y2 = (
                            float(row["x_min"]),
                            float(row["y_min"]),
                            float(row["x_max"]),
                            float(row["y_max"]),
                        )
                    else:
                        cls = int(top_classes[r])
                        pool = per_class_pool.get(cls, None)
                        if pool is not None and len(pool) > 0:
                            j = int(rng.randint(0, len(pool)))
                            x1, y1, x2, y2 = pool.loc[j].astype(float).tolist()
                        else:
                            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0

                    x1 = float(max(0.0, x1))
                    y1 = float(max(0.0, y1))
                    x2 = float(max(x2, x1 + 1.0))
                    y2 = float(max(y2, y1 + 1.0))

                    conf = float(max(0.05, base_conf - step * r))
                    toks.append(
                        f"{int(cls)} {conf:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
                    )

                pred_strings.append(" ".join(toks))

            pred_det_df["PredictionString"] = pred_strings
        else:
            pred_det_df["PredictionString"] = NORMAL
    else:
        pred_det_df["PredictionString"] = NORMAL

if "PredictionString" not in pred_det_df.columns and "TARGET" in pred_det_df.columns:
    pred_det_df = pred_det_df.rename(columns={"TARGET": "PredictionString"})

n_normal_before = len(pred_det_df.query("PredictionString == @NORMAL"))

merged_df = pd.merge(pred_det_df, pred_2class, on="image_id", how="left")

if "target" in merged_df.columns:
    merged_df["class0"] = 1 - merged_df["target"]

if "class0" not in merged_df.columns:
    merged_df["class0"] = 0.5

merged_df["class0"] = merged_df["class0"].fillna(0.5).astype(float).clip(0.0, 1.0)

c0, c1, c2 = 0, 0, 0
for i in range(len(merged_df)):
    p0 = float(merged_df.loc[i, "class0"])
    if p0 < low_threshold:
        c0 += 1
    elif low_threshold <= p0 and p0 < high_threshold:
        ps = str(merged_df.loc[i, "PredictionString"])
        if ps.strip() == NORMAL:
            c0 += 1
        else:
            merged_df.loc[i, "PredictionString"] = ps + f" 14 {p0} 0 0 1 1"
            c1 += 1
    else:
        merged_df.loc[i, "PredictionString"] = NORMAL
        c2 += 1

n_normal_after = len(merged_df.query("PredictionString == @NORMAL"))
print(
    f"n_normal: {n_normal_before} -> {n_normal_after} with threshold {low_threshold} & {high_threshold}"
)
print(f"Keep {c0} Add {c1} Replace {c2}")

submission_filepath = "submission.csv"
submission_df = merged_df[["image_id", "PredictionString"]].copy()
submission_df.to_csv(submission_filepath, index=False)
print(f"Saved to {submission_filepath}")
print(submission_df.head())
