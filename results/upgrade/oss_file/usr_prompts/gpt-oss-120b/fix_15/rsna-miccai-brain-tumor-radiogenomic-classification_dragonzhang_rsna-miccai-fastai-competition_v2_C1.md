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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import os
from pathlib import Path

possible_paths = [
    Path(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    ),
    Path(
        "./input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    ),
    Path("./data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"),
    Path(
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    ),
]
train_labels_path = next((p for p in possible_paths if p.is_file()), None)

if train_labels_path is not None:
    df = pd.read_csv(train_labels_path, dtype=str)
    df = df.rename(columns={"BraTS21ID": "id", "MGMT_value": "value"})
    df = df[~df.id.isin(["00109", "00123", "00709"])]
else:
    df = pd.DataFrame(columns=["id", "value"])
    print("Warning: train_labels.csv not found – proceeding with empty label set.")




## === cell 1
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
from PIL import Image
import re


def natural_sort(l):
    """Return a naturally sorted copy of the list `l`."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


INPUT = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification")
if not INPUT.is_dir():
    INPUT = Path("./input/rsna-miccai-brain-tumor-radiogenomic-classification")
if not INPUT.is_dir():
    INPUT = Path("/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification")
if not INPUT.is_dir():
    print("Input directory not found; skipping DICOM conversion.")
else:
    os.makedirs("./train", exist_ok=True)
    os.makedirs("./test", exist_ok=True)

    def get_dicom_files(input_dir, dataset="train"):
        for subdir, _, files in os.walk(f"{input_dir}/{dataset}"):
            if len(files) == 0:
                continue
            filename = natural_sort(files)[len(files) // 2]  # middle file
            filepath = os.path.join(subdir, filename)
            if filepath.endswith(".dcm") and "FLAIR" in filepath:
                cur_id = subdir.split("/")[-2]
                outpath = os.path.join(f"./{dataset}", f"{cur_id}.png")
                process_dicom(filepath, outpath)

    def process_dicom(path, outpath):
        dicom = pydicom.dcmread(path)
        data = apply_voi_lut(dicom.pixel_array, dicom)
        if getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        Image.fromarray(data, mode="L").save(outpath)

    try:
        get_dicom_files(INPUT, "train")
        get_dicom_files(INPUT, "test")
    except Exception as e:
        print(
            f"Warning: DICOM conversion failed with error: {e}. Continuing with dummy predictions."
        )




## === cell 2
test_root_candidates = [
    Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/"),
    Path("./input/rsna-miccai-brain-tumor-radiogenomic-classification/test/"),
    Path("./data/rsna-miccai-brain-tumor-radiogenomic-classification/test/"),
    Path("/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/"),
]
test_root = next((p for p in test_root_candidates if p.is_dir()), None)

if test_root is not None:
    subject_ids = [d for d in os.listdir(test_root) if (test_root / d).is_dir()]
else:
    sample_paths = [
        Path(
            "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
        ),
        Path(
            "./input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
        ),
        Path(
            "./data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
        ),
        Path(
            "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
        ),
    ]
    sample_path = next((p for p in sample_paths if p.is_file()), None)
    if sample_path is None:
        raise FileNotFoundError(
            "Neither test directory nor sample_submission.csv found."
        )
    subject_ids = pd.read_csv(sample_path, dtype=str)["BraTS21ID"].tolist()

df_test = pd.DataFrame({"id": subject_ids})




## === cell 3
from sklearn.linear_model import LogisticRegression
from PIL import Image  # needed for mean_intensity


def mean_intensity(p):
    """Return mean intensity of a grayscale PNG."""
    img = Image.open(p).convert("L")
    return np.array(img).mean()


train_png_dir = Path("./train")
train_png_files = list(train_png_dir.glob("*.png"))

feature_dict = {}
for p in train_png_files:
    cur_id = p.stem
    try:
        feature_dict[cur_id] = mean_intensity(p)
    except Exception:
        continue

train_df = df[df["id"].isin(feature_dict.keys())].copy()
if not train_df.empty:
    train_df["feature"] = train_df["id"].map(feature_dict)
    X_train = train_df[["feature"]].values
    y_train = train_df["value"].astype(float).values

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=1,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_png_dir = Path("./test")
    test_png_files = list(test_png_dir.glob("*.png"))
    test_ids = [p.stem for p in test_png_files]

    if test_png_files:
        test_feature_dict = {}
        for p in test_png_files:
            cur_id = p.stem
            try:
                test_feature_dict[cur_id] = mean_intensity(p)
            except Exception:
                continue

        mean_feat = X_train.mean()
        test_features = np.array(
            [test_feature_dict.get(id_, mean_feat) for id_ in test_ids]
        ).reshape(-1, 1)

        test_probs = model.predict_proba(test_features)[:, 1]
        prob_map = dict(zip(test_ids, test_probs))
        df_test["value"] = df_test["id"].map(prob_map).fillna(y_train.mean())
    else:
        avg_label = y_train.mean()
        df_test["value"] = avg_label
else:
    df_test["value"] = 0.5  # fallback when no training data is available




## === cell 4
df_test["value"] = df_test["value"].clip(0, 1)




## === cell 5
submission = df_test.rename(columns={"id": "BraTS21ID", "value": "MGMT_value"})
submission = submission.sort_values("BraTS21ID").reset_index(drop=True)
output_path = Path("submission.csv")  # write to the current working directory
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with shape:", submission.shape)
