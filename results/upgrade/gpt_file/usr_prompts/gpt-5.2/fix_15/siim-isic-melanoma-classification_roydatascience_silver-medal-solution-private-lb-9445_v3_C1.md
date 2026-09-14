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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.942396872030668

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66764) has done: 'The notebook fails because it expects an external Kaggle dataset (`../input/pseudolabelmodels/*.csv`) that is not present in your environment, so no prediction files are loaded and everything downstream crashes. To make it run end-to-end and still produce a valid submission, I keep the same “blend/rank-average predictions then write submission.csv” structure, but add a safe fallback that trains a simple metadata-only model from the provided `train.csv` and predicts on `test.csv` when those external CSVs are missing. This preserves evaluation semantics (probability prediction for AUC) and fixes the runtime errors by guaranteeing `predictions`/`predictions_2` exist with the correct shape aligned to `sample_submission.csv`. The output is always a correctly formatted `submission.csv`.'
- What this solution (achieved 0.66769) has done: 'I fix the runtime error by ensuring the one-hot encoded features are converted to a dense matrix before reaching `HistGradientBoostingClassifier`, which does not accept sparse input. This is a minimal change that preserves your fallback model’s structure (same features, same CV loop, same classifier) while making the pipeline compatible with the installed scikit-learn version. I also add a small safety check to guarantee prediction shapes match `sample_submission` before writing. The script then run end-to-end and always produce a valid `submission.csv` with `image_name,target`.'
- What this solution (achieved 0.77462) has done: 'Your current score (0.66769) is far below the target (0.9424), so we should improve the fallback metadata model rather than touch the (missing) pseudo-label ensemble path. I keep the same core approach (5-fold StratifiedKFold + HistGradientBoostingClassifier on engineered metadata) but add two minimal, high-impact, metadata-only features: patient-level target encoding (out-of-fold) and a simple age×site interaction proxy (age_bin + site concatenation). I also add class-imbalance handling via `sample_weight` (doesn’t change the model family or training loop, but usually improves AUC substantially for this dataset). These changes keep evaluation semantics identical (predict probabilities for AUC) and still write a valid `submission.csv`.'
- What this solution (achieved 0.77568) has done: 'Your current score (0.77462) is still far below the target (0.9424), so we should improve the fallback metadata model while keeping the same overall structure (StratifiedKFold + HistGradientBoostingClassifier + OOF target encoding) and still writing `submission.csv`. The biggest low-risk lift here is to prevent leakage by computing the patient/site target-encodings **using only the training fold labels** (your current code groups on `tr.groupby(...)"target"` which accidentally uses labels from the full dataframe), which typically improves generalization AUC. In the same spirit, we keep the model/loop identical but add one more minimal metadata interaction (`sex__site`) and make the blending step robust (when pseudo preds exist, align them by `image_name` to avoid silent row-order mismatches that can hurt AUC). These are small, metric-aligned corrections that should move the score upward toward the target without changing the core approach.'
- What this solution (achieved 0.77537) has done: 'Your current score (0.77568) is far below the target (0.94240), so we should improve the fallback metadata model while keeping the same overall structure (OOF target encoding + 5-fold StratifiedKFold + HistGradientBoostingClassifier, writing `submission.csv`). The smallest high-impact change here is to add one more fold-safe (OOF) target-encoded signal: `sex` and `sex__site`, which are strong metadata predictors in this competition and don’t change the model family or training loop. I also slightly adjust the smoothing for the existing target encodings to reduce noise (a calibration/generalization tweak, not a new approach). All changes are confined to the fallback path and keep the submission format/semantics identical.'
- What this solution (achieved 0.5) has done: 'Your current score (0.77537) is far below the target (0.94240), so the smallest likely lift is to strengthen the *existing fallback metadata model* without changing its core structure (same 5-fold StratifiedKFold + HistGradientBoostingClassifier + OOF target-encoding + weighted training). I add one more fold-safe, high-signal OOF target-encoding for `diagnosis` (train-only column) and use it as a numeric feature at training time; at test time it naturally falls back to the global mean, so it’s safe and aligned with the metric. I also add a simple, stable OOF encoding for the interaction `site_agebin`, which you already create, again computed fold-safely to avoid leakage. These are minimal feature additions that typically improve AUC materially on this competition while preserving your training loop and submission semantics.'
- What this solution (achieved 0.55575) has done: 'Your current 0.5 score indicates the submission is effectively constant/near-constant, so the smallest meaningful improvement is to fix the fallback pipeline’s feature handling so the model can actually learn signal (AUC > 0.5). I keep the same core approach (5-fold StratifiedKFold + HistGradientBoostingClassifier + OOF target mean encodings + weighted training + same blending) but remove `patient_id` from the one-hot categorical set (it explodes dimensionality and tends to flatten predictions) while keeping its OOF target-mean numeric feature. I also add a tiny amount of deterministic jitter to the final probabilities (doesn’t change semantics, but breaks ties/constant predictions that can collapse AUC to 0.5). Finally, I add a safety print of prediction std to confirm we’re not outputting a constant vector again, while still writing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current score (0.55575) is far below the target (0.9424), so we should improve the *fallback metadata model* while keeping the same model family (HistGradientBoostingClassifier), CV loop, and target-encoding approach. The biggest likely issue is that the last change removed one-hot encoding of `patient_id`, which is a very strong signal for this competition; we re-introduce it in a controlled way by one-hot encoding only the top‑K most frequent patient IDs and mapping the rest to `"other"`, preserving your feature pipeline while avoiding dimensionality blow-ups. We also remove the tiny random jitter added to predictions (unnecessary for AUC and can slightly hurt stability), keeping outputs deterministic. These are minimal, directly score-relevant changes that should move AUC back up toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC strongly suggests the submission probabilities are effectively constant or misaligned (AUC collapses to 0.5 when predictions don’t vary or the order doesn’t match labels). I keep your exact fallback model family (HistGradientBoostingClassifier), CV loop, and target-encoding approach, but fix two score-critical issues: (1) remove the raw `patient_id` high-cardinality column from the feature set (you already have a strong patient target-mean plus controlled `patient_id_topk` one-hot), and (2) make pseudo-pred loading/alignment robust by always keying by `image_name` and reindexing to `sample_submission` to prevent silent row-order mismatches. These are minimal changes that should restore non-constant, correctly ordered probabilities and move AUC back up toward your target. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests the test predictions are being assigned to the wrong `image_name` order (a row-order mismatch between `test_df` and `sample_submission`), which collapses AUC to random. I make the fallback path explicitly align `test_df` to `sample_submission` by `image_name` before feature engineering/prediction, and add a hard alignment check so we never silently write a mis-ordered submission again. This is a minimal, score-critical fix that preserves your exact model family, CV loop, target-encoding logic, and blending semantics, but should move the score back upward toward your target. I also ensure any pseudo-pred rank-averaging remains keyed to `sample_submission` order (already done) and keep all I/O paths unchanged.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC indicates the submission is effectively uninformative, and given your code already prints non-constant stats, the most likely remaining score-killer is a subtle misalignment introduced when blending (numpy array assignment into a 1-col DataFrame can mis-handle index/shape) or from duplicate/unsorted `image_name` handling across intermediate frames. I make the smallest changes that enforce strict `image_name`-keyed alignment at every stage (including the final blend) and ensure predictions are 1D vectors aligned to `sample_submission` before writing. I not change the model, folds, features, loss/metric semantics, or training loop—only the alignment/assignment logic that can collapse AUC to ~0.5. This should move you back toward the previously achieved ~0.77+ and closer to the target.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC is consistent with predictions being effectively random due to an `image_name`/row-order mismatch during final assembly, even if the fallback model itself produces non-constant probabilities. I make the smallest score-critical change: enforce strict `image_name`-keyed alignment for **all** prediction vectors (pseudo-blend and fallback) right before blending and writing, and add hard checks that the final `submission.image_name` order exactly matches `sample_submission`. I also ensure the final `target` is a true 1D float array aligned by `image_name` (not by implicit integer index), which prevents silent misassignment that collapses AUC to ~0.5. No model architecture/training loop/feature logic is changed; only the alignment and final assembly are corrected to move the score back toward your prior ~0.77 and closer to the target.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC strongly suggests the written predictions are effectively random/constant relative to labels; given your fallback model should produce non-trivial variance, the most likely remaining issue is that `test.csv` is being aligned to `sample_submission` incorrectly because `sample_submission` in this environment has 4142 rows while your raw `test.csv` shows 4143 in the directory listing (a known packaging inconsistency), so some images may be silently dropped/shifted before prediction. I make the smallest change to enforce a single source of truth for test ordering: build `test_df` strictly from `sample_submission.image_name` and left-join the test metadata onto it, so every predicted row corresponds 1:1 to the submission row, with no reindexing NaNs or mismatched lengths. I also ensure any pseudo prediction series are reindexed to that exact `sample_submission` index before rank-averaging/blending (same logic, just stricter alignment), which prevents accidental off-by-one/order issues that collapse AUC to ~0.5. No model, features, CV loop, loss/metric semantics, or blending weights are changed—only the test-frame construction/alignment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input"
DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DATA_DIR = _first_existing_path(DATA_DIR_CANDIDATES)
if BASE_DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset directory under expected /kaggle/input or /kaggle/data paths."
    )

print("Using BASE_DATA_DIR =", BASE_DATA_DIR)
if os.path.exists("/kaggle/input"):
    print("Listing /kaggle/input:", os.listdir("/kaggle/input")[:20])



## === cell 1
from scipy.stats import rankdata
import glob
import warnings

warnings.filterwarnings("ignore")

LABELS = ["target"]

PSEUDO_DIR_CANDIDATES = [
    os.path.join(INPUT_DIR, "pseudolabelmodels"),
    os.path.join("/kaggle/data", "pseudolabelmodels"),
    "../input/pseudolabelmodels",  # original relative path (may not exist)
]
PSEUDO_DIR = _first_existing_path(PSEUDO_DIR_CANDIDATES)

all_files = []
if PSEUDO_DIR is not None:
    all_files = glob.glob(os.path.join(PSEUDO_DIR, "*.csv"))

print("PSEUDO_DIR:", PSEUDO_DIR)
print("Found pseudo submission files:", len(all_files))



## === cell 2
concat_sub = None
if len(all_files) > 0:
    outs = []
    for f in all_files:
        df = pd.read_csv(f)

        if "image_name" in df.columns and "target" in df.columns:
            df = df[["image_name", "target"]].copy().set_index("image_name")
        else:
            df2 = pd.read_csv(f, index_col=0)
            if df2.index.name is None:
                df2.index.name = "image_name"
            if "target" not in df2.columns:
                raise ValueError(f"Pseudo file {f} does not contain 'target' column.")
            df = df2[["target"]].copy()

        df.columns = [os.path.basename(f)]
        outs.append(df)

    concat_sub = pd.concat(outs, axis=1)
    concat_sub.reset_index(inplace=True)
    print("concat_sub shape:", concat_sub.shape)
else:
    print("No pseudo files available; will use fallback model.")



## === cell 3
m_gmean = None
if concat_sub is not None:
    rank = np.tril(concat_sub.iloc[:, 1:].corr().values, -1)
    m = (rank > 0).sum()
    m_gmean, s = 0, 0
    for n in range(min(rank.shape[0], m)):
        mx = np.unravel_index(rank.argmin(), rank.shape)
        w = (m - n) / (m + n / 10)
        a = np.clip(concat_sub.iloc[:, mx[0] + 1].to_numpy(dtype=float), 1e-12, 1.0)
        b = np.clip(concat_sub.iloc[:, mx[1] + 1].to_numpy(dtype=float), 1e-12, 1.0)
        m_gmean += w * (np.log(a) + np.log(b)) / 2
        s += w
        rank[mx] = 1
    m_gmean = np.exp(m_gmean / s).clip(0.0, 1.0)
    print("Computed m_gmean from pseudo ensemble.")
else:
    print("Skipping corr-based ensemble (no pseudo files).")



## === cell 4
predict_list = []
predict_list_2 = []

if PSEUDO_DIR is not None and len(all_files) > 0:
    all_files_sorted = sorted(all_files)
    mid = len(all_files_sorted) // 2
    group1 = all_files_sorted[:mid] if mid > 0 else all_files_sorted
    group2 = all_files_sorted[mid:] if mid > 0 else all_files_sorted

    def _load_preds_as_series(file_list):
        preds = []
        for f in file_list:
            df = pd.read_csv(f)
            if "image_name" in df.columns and "target" in df.columns:
                s = df.set_index("image_name")["target"].astype(np.float32)
            else:
                df2 = pd.read_csv(f, index_col=0)
                if df2.index.name is None:
                    df2.index.name = "image_name"
                if "target" not in df2.columns:
                    continue
                s = df2["target"].astype(np.float32)
            preds.append(s)
        return preds

    predict_list = _load_preds_as_series(group1)
    predict_list_2 = _load_preds_as_series(group2)

print("predict_list:", len(predict_list), "predict_list_2:", len(predict_list_2))




## === cell 5
def rank_average_from_series(predict_series_list, index):
    if len(predict_series_list) == 0:
        return None
    n = len(index)
    acc = np.zeros(n, dtype=np.float32)
    for s in predict_series_list:
        vals = s.reindex(index).astype(np.float32)
        fillv = float(vals.mean()) if np.isfinite(vals.mean()) else 0.5
        vals = vals.fillna(fillv).values
        acc += (rankdata(vals) / n).astype(np.float32)
    acc /= len(predict_series_list)
    return acc.astype(np.float32)


predictions = None
predictions_2 = None



## === cell 6
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

train_csv_candidates = [
    os.path.join(BASE_DATA_DIR, "train.csv"),
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
]
test_csv_candidates = [
    os.path.join(BASE_DATA_DIR, "test.csv"),
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
]
sample_sub_candidates = [
    os.path.join(BASE_DATA_DIR, "sample_submission.csv"),
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]

TRAIN_CSV = _first_existing_path(train_csv_candidates)
TEST_CSV = _first_existing_path(test_csv_candidates)
SAMPLE_SUB = _first_existing_path(sample_sub_candidates)

if TRAIN_CSV is None or TEST_CSV is None or SAMPLE_SUB is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv/sample_submission.csv in expected locations."
    )

sample_submission = pd.read_csv(SAMPLE_SUB)
raw_test_df = pd.read_csv(TEST_CSV)

if sample_submission["image_name"].duplicated().any():
    raise RuntimeError(
        "sample_submission has duplicated image_name; cannot safely align."
    )
if raw_test_df["image_name"].duplicated().any():
    raise RuntimeError("test.csv has duplicated image_name; cannot safely align.")

sample_submission["image_name"] = sample_submission["image_name"].astype(str)
raw_test_df["image_name"] = raw_test_df["image_name"].astype(str)
sample_index = sample_submission["image_name"].values

test_df = pd.DataFrame({"image_name": sample_index}).merge(
    raw_test_df, on="image_name", how="left", validate="one_to_one"
)

if test_df["image_name"].isna().any():
    raise RuntimeError("test_df has NaN image_name after merge (unexpected).")
if not np.array_equal(test_df["image_name"].values, sample_index):
    raise RuntimeError(
        "test_df is not perfectly aligned to sample_submission after merge."
    )

missing_meta = test_df.drop(columns=["image_name"]).isna().all(axis=1).sum()
print(
    "Rows with all test metadata missing after merge:",
    int(missing_meta),
    "of",
    len(test_df),
)

if len(predict_list) > 0:
    predictions = rank_average_from_series(predict_list, sample_index)
if len(predict_list_2) > 0:
    predictions_2 = rank_average_from_series(predict_list_2, sample_index)

if predictions is not None:
    print("Rank averaged predictions shape:", predictions.shape)
if predictions_2 is not None:
    print("Rank averaged predictions_2 shape:", predictions_2.shape)

if (
    (predictions is None)
    or (predictions_2 is None)
    or (predictions.shape[0] != len(sample_submission))
    or (predictions_2.shape[0] != len(sample_submission))
):
    print(
        "Using fallback metadata model to generate predictions (pseudo files missing/misaligned)."
    )

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["image_name"] = train_df["image_name"].astype(str)

    TOPK_PATIENTS = 500  # keep as-is
    top_patients = (
        train_df["patient_id"]
        .astype(str)
        .fillna("unknown")
        .replace({"": "unknown"})
        .value_counts()
        .head(TOPK_PATIENTS)
        .index
    )
    top_patients_set = set(top_patients.tolist())

    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["age_approx"] = pd.to_numeric(out["age_approx"], errors="coerce")
        out["age_missing"] = out["age_approx"].isna().astype(np.int8)

        out["age_bin"] = pd.cut(
            out["age_approx"],
            bins=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 120],
            labels=[f"b{i}" for i in range(9)],
        ).astype("object")

        for c in ["sex", "anatom_site_general_challenge"]:
            out[c] = out[c].fillna("unknown").astype(str).replace({"": "unknown"})

        out["site_agebin"] = (
            out["anatom_site_general_challenge"].astype(str)
            + "__"
            + out["age_bin"].astype(str)
        ).astype("object")

        out["sex_site"] = (
            out["sex"].astype(str)
            + "__"
            + out["anatom_site_general_challenge"].astype(str)
        ).astype("object")

        out["patient_id"] = (
            out["patient_id"].astype(str).fillna("unknown").replace({"": "unknown"})
        )
        out["patient_id_topk"] = (
            out["patient_id"]
            .where(out["patient_id"].isin(top_patients_set), "other")
            .astype("object")
        )

        if "diagnosis" in out.columns:
            out["diagnosis"] = (
                out["diagnosis"].fillna("unknown").astype(str).replace({"": "unknown"})
            )
        else:
            out["diagnosis"] = "unknown"

        return out

    train_df_fe = _add_features(train_df)
    test_df_fe = _add_features(test_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = train_df_fe["target"].astype(int).values
    global_mean = float(np.mean(y))

    def _oof_target_mean(
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        y_arr: np.ndarray,
        group_col: str,
        splitter: StratifiedKFold,
        smoothing: float = 20.0,
    ):
        oof = np.zeros(len(train_frame), dtype=np.float32)
        test_accum = np.zeros(len(test_frame), dtype=np.float32)

        for tr_idx, va_idx in splitter.split(train_frame, y_arr):
            tr = train_frame.iloc[tr_idx]
            va = train_frame.iloc[va_idx]
            y_tr = y_arr[tr_idx]

            tr_stats = pd.DataFrame({group_col: tr[group_col].values, "y": y_tr})
            stats = tr_stats.groupby(group_col)["y"].agg(["mean", "count"])
            means = stats["mean"]
            counts = stats["count"]

            smooth = ((means * counts) + (global_mean * smoothing)) / (
                counts + smoothing
            )

            oof[va_idx] = (
                va[group_col].map(smooth).fillna(global_mean).astype(np.float32).values
            )
            test_accum += (
                test_frame[group_col]
                .map(smooth)
                .fillna(global_mean)
                .astype(np.float32)
                .values
            )

        test_te = test_accum / splitter.get_n_splits()
        return oof, test_te

    site_oof, site_test = _oof_target_mean(
        train_df_fe, test_df_fe, y, "anatom_site_general_challenge", skf, smoothing=50.0
    )
    pat_oof, pat_test = _oof_target_mean(
        train_df_fe, test_df_fe, y, "patient_id", skf, smoothing=80.0
    )
    sex_oof, sex_test = _oof_target_mean(
        train_df_fe, test_df_fe, y, "sex", skf, smoothing=200.0
    )
    sexsite_oof, sexsite_test = _oof_target_mean(
        train_df_fe, test_df_fe, y, "sex_site", skf, smoothing=80.0
    )
    siteage_oof, siteage_test = _oof_target_mean(
        train_df_fe, test_df_fe, y, "site_agebin", skf, smoothing=80.0
    )
    diag_oof, diag_test = _oof_target_mean(
        train_df_fe, test_df_fe, y, "diagnosis", skf, smoothing=30.0
    )

    train_df_fe["site_target_mean"] = site_oof
    test_df_fe["site_target_mean"] = site_test
    train_df_fe["patient_target_mean"] = pat_oof
    test_df_fe["patient_target_mean"] = pat_test
    train_df_fe["sex_target_mean"] = sex_oof
    test_df_fe["sex_target_mean"] = sex_test
    train_df_fe["sexsite_target_mean"] = sexsite_oof
    test_df_fe["sexsite_target_mean"] = sexsite_test
    train_df_fe["siteage_target_mean"] = siteage_oof
    test_df_fe["siteage_target_mean"] = siteage_test
    train_df_fe["diagnosis_target_mean"] = diag_oof
    test_df_fe["diagnosis_target_mean"] = diag_test

    feature_cols = [
        "sex",
        "age_approx",
        "age_missing",
        "age_bin",
        "anatom_site_general_challenge",
        "site_agebin",
        "sex_site",
        "patient_id_topk",
        "site_target_mean",
        "patient_target_mean",
        "sex_target_mean",
        "sexsite_target_mean",
        "siteage_target_mean",
        "diagnosis_target_mean",
    ]
    X = train_df_fe[feature_cols].copy()
    X_test = test_df_fe[feature_cols].copy()

    numeric_features = [
        "age_approx",
        "age_missing",
        "site_target_mean",
        "patient_target_mean",
        "sex_target_mean",
        "sexsite_target_mean",
        "siteage_target_mean",
        "diagnosis_target_mean",
    ]

    categorical_features = [
        "sex",
        "age_bin",
        "anatom_site_general_challenge",
        "site_agebin",
        "sex_site",
        "patient_id_topk",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0 or neg == 0:
        base_w = np.ones_like(y, dtype=np.float32)
    else:
        w_pos = neg / pos
        base_w = np.where(y == 1, w_pos, 1.0).astype(np.float32)

    test_pred = np.zeros(len(test_df_fe), dtype=np.float64)

    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        w_tr = base_w[tr_idx]

        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=300,
            random_state=42,
        )
        model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        model.fit(X_tr, y_tr, clf__sample_weight=w_tr)
        test_pred += model.predict_proba(X_test)[:, 1]

    test_pred /= skf.get_n_splits()
    test_pred = np.clip(test_pred, 0.0, 1.0)

    print(
        "Fallback test_pred mean/std/min/max:",
        float(test_pred.mean()),
        float(test_pred.std()),
        float(test_pred.min()),
        float(test_pred.max()),
    )

    predictions = test_pred.astype(np.float32)
    predictions_2 = predictions.copy()

if predictions is None or len(predictions) != len(sample_submission):
    raise RuntimeError("Predictions are missing or not aligned with sample_submission.")
if predictions_2 is None or len(predictions_2) != len(sample_submission):
    raise RuntimeError(
        "Predictions_2 are missing or not aligned with sample_submission."
    )

pred1_s = pd.Series(
    np.asarray(predictions, dtype=np.float64), index=sample_index, name="target"
)
pred2_s = pd.Series(
    np.asarray(predictions_2, dtype=np.float64), index=sample_index, name="target"
)

final_pred = (0.55 * pred2_s + 0.45 * pred1_s).clip(0.0, 1.0)

submission = sample_submission[["image_name"]].copy()
submission["target"] = final_pred.reindex(
    submission["image_name"].values
).values.astype(np.float64)

if submission["image_name"].isna().any():
    raise RuntimeError("Submission has NaN image_name (unexpected).")
if submission["target"].isna().any():
    raise RuntimeError(
        "Submission has NaN target after final reindexing (alignment issue)."
    )
if not np.array_equal(submission["image_name"].values, sample_index):
    raise RuntimeError("Final submission order does not match sample_submission order.")

print(
    "Final submission target mean/std/min/max:",
    float(submission["target"].mean()),
    float(submission["target"].std()),
    float(submission["target"].min()),
    float(submission["target"].max()),
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
