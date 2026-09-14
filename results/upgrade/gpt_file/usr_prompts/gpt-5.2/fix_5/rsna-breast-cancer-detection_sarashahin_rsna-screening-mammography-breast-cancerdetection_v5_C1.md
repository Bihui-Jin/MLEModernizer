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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

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
tensorflow_decision_forests==1.11.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.02

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03064) has done: 'I fix the runtime error caused by an incompatibility between `tensorflow_decision_forests==1.11.0` and the protobuf runtime by forcing the pure-Python protobuf implementation before importing TF/TF-DF. I also make the train/validation split deterministic (seeded) to stabilize results without changing the modeling approach. Since your current score (0.03007) is already above the target (0.02) and within the ±10% tolerance band for “toward target” isn’t achievable without unnecessary degradation, I avoid any score-changing model tweaks and focus on correctness and submission generation. Finally, I ensure prediction alignment and output a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.03064) has done: 'The crash happens before training because TF-DF 1.11.0 is incompatible with the default (C++) protobuf runtime in this environment, so we need to force the pure-Python protobuf implementation **before** importing anything that can pull in protobuf (including TensorFlow). I move that environment-variable setup into the very first cell and add a safe fallback to avoid hard-crashing if TF/TF-DF get imported too early. I also keep your deterministic patient split and the same TF-DF GradientBoostedTreesModel logic, and ensure the submission is aligned to `sample_submission.csv` and written as `submission.csv` with the required columns. No score-tuning changes are introduced since the current score is already better than the target and we want to avoid unnecessary degradation.'
- What this solution (achieved 0.03071) has done: 'I fix the TF-DF/protobuf crash by forcing the pure-Python protobuf runtime *before* any TensorFlow/TF-DF import and by restarting the interpreter once if an early protobuf import already happened. I also make the dataset creation robust to missing values by filling `age` and `implant` consistently and casting to safe dtypes, which is score-neutral but prevents runtime/type issues. The modeling approach (TF-DF GradientBoostedTreesModel, same features, same patient split, same aggregation to prediction_id) stays unchanged to avoid unnecessary score movement since you’re already above the target. Finally, I ensure the submission is aligned to `sample_submission.csv` and always written to `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    need_install = pb_ver is None or not pb_ver.startswith("3.20.")
    if need_install and os.environ.get("_PROTOBUF_PIN_DONE", "0") != "1":
        os.environ["_PROTOBUF_PIN_DONE"] = "1"
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_compatible_protobuf()

import pandas as pd
import numpy as np

DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"

train_data = pd.read_csv(f"{DATA_DIR}/train.csv")
test_data = pd.read_csv(f"{DATA_DIR}/test.csv")



## === cell 1
train_data



## === cell 2
test_data



## === cell 3
import math

ratio = 0.8
rng = np.random.default_rng(42)

patient_id = train_data["patient_id"].unique()
indexs = rng.permutation(len(patient_id))
split_idx = math.ceil(len(indexs) * ratio)

train_patients = patient_id[indexs[:split_idx]]
val_patients = patient_id[indexs[split_idx:]]

val_data = train_data.loc[train_data["patient_id"].isin(val_patients)].copy()
train_data = train_data.loc[train_data["patient_id"].isin(train_patients)].copy()

print("{} for training, {} for validation.".format(len(train_data), len(val_data)))
print(val_data["patient_id"].unique())
print(train_data["patient_id"].unique())




## === cell 4
def prep_df(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    if "age" in df.columns:
        age_med = df["age"].median()
        if pd.isna(age_med):
            age_med = 0.0
        df["age"] = df["age"].fillna(age_med).astype(np.float32)
    else:
        df["age"] = np.float32(0.0)

    if "implant" in df.columns:
        df["implant"] = df["implant"].fillna(0).astype(np.int32)
    else:
        df["implant"] = np.int32(0)

    for c in ["laterality", "view"]:
        if c in df.columns:
            df[c] = df[c].fillna("UNK").astype(str)
        else:
            df[c] = "UNK"

    if is_train:
        df["cancer"] = df["cancer"].fillna(0).astype(np.int32)

    return df


train_data_p = prep_df(train_data, is_train=True)
val_data_p = prep_df(val_data, is_train=True)
test_data_p = prep_df(test_data, is_train=False)



## === cell 5
import tensorflow_decision_forests as tfdf

batch_size = 256

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_data_p.loc[:, ["laterality", "view", "age", "implant", "cancer"]],
    label="cancer",
    batch_size=batch_size,
)
val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    val_data_p.loc[:, ["laterality", "view", "age", "implant", "cancer"]],
    label="cancer",
    batch_size=batch_size,
)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_data_p.loc[:, ["laterality", "view", "age", "implant"]],
    batch_size=batch_size,
)



## === cell 6
model = tfdf.keras.GradientBoostedTreesModel(verbose=1)
model.fit(train_ds)



## === cell 7
model.summary()



## === cell 8
try:
    tfdf.model_plotter.plot_model_in_colab(model, tree_idx=2, max_depth=10)
except Exception as e:
    print("Skipping model plot (not supported in this environment):", repr(e))



## === cell 9
decision_forests_modal = model.evaluate(val_ds)
decision_forests_modal



## === cell 10
predictions = model.predict(test_ds)
predictions



## === cell 11
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sample_submission



## === cell 12
print("prediction shape", np.asarray(predictions).shape)



## === cell 13
test_data_orig = pd.read_csv(f"{DATA_DIR}/test.csv")
prediction_ids = test_data_orig["prediction_id"].copy()

preds = np.asarray(predictions).ravel().astype(float)

submission = (
    pd.DataFrame({"prediction_id": prediction_ids, "cancer": preds})
    .groupby("prediction_id", as_index=False)["cancer"]
    .mean()
)

submission = sample_submission[["prediction_id"]].merge(
    submission, on="prediction_id", how="left"
)
submission["cancer"] = submission["cancer"].fillna(0.0).astype(float)

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 14
pd.read_csv("/kaggle/working/submission.csv")
