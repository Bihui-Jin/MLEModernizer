# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 100
np.random.seed(RANDOM_STATE)

DATA_ROOT = "/kaggle/input/siim-isic-melanoma-classification"

TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"

TRAIN_JPEG_ROOT = f"{DATA_ROOT}/jpeg/train/"
TEST_JPEG_ROOT = f"{DATA_ROOT}/jpeg/test/"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_JPEG_ROOT), f"Missing dir {TRAIN_JPEG_ROOT}"
assert os.path.isdir(TEST_JPEG_ROOT), f"Missing dir {TEST_JPEG_ROOT}"




## === cell 1
def create_patient_data(
    csv,
    normalize_age=True,
    test=False,
    drop_missing=True,
):
    patient_features = pd.read_csv(csv)

    if drop_missing:
        patient_features = patient_features.dropna(
            subset=["age_approx", "sex", "anatom_site_general_challenge"]
        ).copy()

    dummy_data = pd.get_dummies(
        patient_features[["sex", "anatom_site_general_challenge"]],
        dummy_na=False,
    )
    patient_features = pd.concat([patient_features, dummy_data], axis=1)

    dummy_col = [
        "sex_female",
        "sex_male",
        "anatom_site_general_challenge_head/neck",
        "anatom_site_general_challenge_lower extremity",
        "anatom_site_general_challenge_oral/genital",
        "anatom_site_general_challenge_palms/soles",
        "anatom_site_general_challenge_torso",
        "anatom_site_general_challenge_upper extremity",
    ]
    for c in dummy_col:
        if c not in patient_features.columns:
            patient_features[c] = 0

    if normalize_age:
        min_max_scaler = preprocessing.MinMaxScaler()
        age = patient_features["age_approx"].astype(float)
        age = age.fillna(age.median())
        patient_features["age_approx"] = min_max_scaler.fit_transform(
            age.to_frame()
        ).astype(np.float32)

    return patient_features[["image_name", "age_approx"] + dummy_col].copy()




## === cell 2
from PIL import Image


def _safe_open_image(path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _image_stats(im):
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    means = arr.reshape(-1, 3).mean(axis=0)
    stds = arr.reshape(-1, 3).std(axis=0)
    mins = arr.reshape(-1, 3).min(axis=0)
    maxs = arr.reshape(-1, 3).max(axis=0)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    g_mean = float(gray.mean())
    g_std = float(gray.std())
    sat = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    return {
        "img_r_mean": float(means[0]),
        "img_g_mean": float(means[1]),
        "img_b_mean": float(means[2]),
        "img_r_std": float(stds[0]),
        "img_g_std": float(stds[1]),
        "img_b_std": float(stds[2]),
        "img_r_min": float(mins[0]),
        "img_g_min": float(mins[1]),
        "img_b_min": float(mins[2]),
        "img_r_max": float(maxs[0]),
        "img_g_max": float(maxs[1]),
        "img_b_max": float(maxs[2]),
        "img_gray_mean": g_mean,
        "img_gray_std": g_std,
        "img_sat_mean": sat,
    }


def extract_image_features(image_names, jpeg_root, ext=".jpg", resize=128):
    rows = []
    for name in image_names:
        path = os.path.join(jpeg_root, name + ext)
        im = _safe_open_image(path)
        if im is None:
            feats = {k: np.nan for k in _image_stats(Image.new("RGB", (1, 1))).keys()}
        else:
            if resize is not None:
                im = im.resize((resize, resize), resample=Image.BILINEAR)
            feats = _image_stats(im)
        feats["image_name"] = name
        rows.append(feats)
    return pd.DataFrame(rows)




## === cell 3
def balance_data(df, label_col="IMAGE_LABEL", method="upsample"):
    counts = df.groupby(label_col).size()
    classes = df[label_col].unique().tolist()
    max_class_num = int(counts.max())
    max_class_id = counts.idxmax()
    min_class_num = int(counts.min())
    min_class_id = counts.idxmin()

    if method == "upsample":
        resampled_subsets = [df[df[label_col] == max_class_id]]
        for i in [x for x in classes if x != max_class_id]:
            class_subset = df[df[label_col] == i]
            upsampled_subset = resample(
                class_subset, n_samples=max_class_num, random_state=RANDOM_STATE
            )
            resampled_subsets.append(upsampled_subset)
    elif method == "downsample":
        resampled_subsets = [df[df[label_col] == min_class_id]]
        for i in [x for x in classes if x != min_class_id]:
            class_subset = df[df[label_col] == i]
            downsampled_subset = resample(
                class_subset, n_samples=min_class_num, random_state=RANDOM_STATE
            )
            resampled_subsets.append(downsampled_subset)
    else:
        return df

    resampled_df = (
        pd.concat(resampled_subsets, axis=0)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return resampled_df


def create_data(img_features, pt_features, test_split, balance="upsample"):
    img_features = img_features.copy()

    img_cols = img_features.columns.tolist()
    numeric_cols = [c for c in img_cols if c not in ["image_name", "IMAGE_LABEL"]]
    if len(numeric_cols) > 0:
        min_max_scaler = preprocessing.MinMaxScaler()
        img_features[numeric_cols] = min_max_scaler.fit_transform(
            img_features[numeric_cols]
        )

    combined = pd.merge(
        left=pt_features,
        right=img_features,
        left_on="image_name",
        right_on="image_name",
        how="inner",
    )

    feature_names = [
        x for x in combined.columns.tolist() if x not in ["image_name", "IMAGE_LABEL"]
    ]

    if test_split:
        train, test_df = train_test_split(
            combined,
            test_size=test_split,
            random_state=RANDOM_STATE,
            stratify=combined["IMAGE_LABEL"],
        )
    else:
        train = combined
        test_df = None

    if balance:
        train = balance_data(train, label_col="IMAGE_LABEL", method=balance)

    if test_split:
        feature_dict = {
            "train": {
                "features": train[feature_names],
                "features_names": feature_names,
                "labels": train["IMAGE_LABEL"].astype(int).to_numpy(),
            },
            "test": {
                "features": test_df[feature_names],
                "features_names": feature_names,
                "labels": test_df["IMAGE_LABEL"].astype(int).to_numpy(),
            },
        }
        return feature_dict
    else:
        return train




## === cell 4
train_labels = pd.read_csv(TRAIN_CSV, usecols=["image_name", "target"]).copy()
train_labels = train_labels.rename(columns={"target": "IMAGE_LABEL"})

train_clinical_features = create_patient_data(
    TRAIN_CSV,
    drop_missing=False,
    normalize_age=False,  # keep as in original call
    test=False,
)
train_clinical_features = pd.merge(
    train_clinical_features, train_labels, on="image_name", how="inner"
)

train_image_names = train_labels["image_name"].tolist()
train_img_features = extract_image_features(
    train_image_names, jpeg_root=TRAIN_JPEG_ROOT, ext=".jpg", resize=128
)
train_img_features = pd.merge(
    train_img_features, train_labels, on="image_name", how="inner"
)

(
    train_clinical_features.shape,
    train_img_features.shape,
    train_labels["IMAGE_LABEL"].mean(),
)



## === cell 5
train_data = create_data(
    train_img_features, train_clinical_features, test_split=0.25, balance="downsample"
)
X_train = train_data["train"]["features"]
y_train = train_data["train"]["labels"].astype(int)
X_valid = train_data["test"]["features"]
y_valid = train_data["test"]["labels"].astype(int)

(X_train.shape, X_valid.shape, float(y_train.mean()), float(y_valid.mean()))




## === cell 6
class CVClassifier:
    """
    Minimal replacement for radtorch.core.Classifier with:
    - cross-validated training
    - out-of-fold AUC reporting
    - predict_proba on new data via mean of fold probabilities
    """

    def __init__(self, n_splits=5, random_state=RANDOM_STATE):
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = []
        self.oof_pred = None
        self.oof_auc = None

    def _make_model(self):
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            random_state=self.random_state,
        )

    def fit_cv(self, X, y):
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        self.oof_pred = np.zeros(len(y), dtype=float)
        self.models = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            model = self._make_model()
            model.fit(X.iloc[tr_idx], y[tr_idx])
            proba = model.predict_proba(X.iloc[va_idx])[:, 1]
            self.oof_pred[va_idx] = proba
            self.models.append(model)

        self.oof_auc = roc_auc_score(y, self.oof_pred)
        return self

    def predict_proba(self, X):
        preds = np.zeros(X.shape[0], dtype=float)
        for m in self.models:
            preds += m.predict_proba(X)[:, 1]
        preds /= max(len(self.models), 1)
        return np.vstack([1.0 - preds, preds]).T




## === cell 7
clf = CVClassifier(n_splits=5, random_state=RANDOM_STATE).fit_cv(X_train, y_train)

valid_pred = clf.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_pred)

print(f"OOF AUC (CV on training subset): {clf.oof_auc:.6f}")
print(f"Holdout AUC (25% split):        {valid_auc:.6f}")



## === cell 8
test_clinical_features = create_patient_data(
    TEST_CSV,
    normalize_age=False,  # keep as in original call
    test=True,
    drop_missing=False,
)

test_df = pd.read_csv(TEST_CSV, usecols=["image_name"])
test_image_names = test_df["image_name"].tolist()
test_img_features = extract_image_features(
    test_image_names, jpeg_root=TEST_JPEG_ROOT, ext=".jpg", resize=128
)

test_features_df = create_data(
    test_img_features, test_clinical_features, test_split=False, balance=False
)

feature_names = [
    x
    for x in test_features_df.columns.tolist()
    if x not in ["image_name", "IMAGE_LABEL"]
]
X_test = test_features_df[feature_names]

test_pred = clf.predict_proba(X_test)[:, 1]

sub = pd.read_csv(SAMPLE_SUB)
pred_map = dict(zip(test_features_df["image_name"].values, test_pred))
sub["target"] = sub["image_name"].map(pred_map).astype(float)

sub["target"] = sub["target"].fillna(float(np.nanmean(test_pred)))

submission = sub[["image_name", "target"]].copy()
submission.head()



## === cell 9
out_path = "submission_alexnet_xgb_cv5.csv"
submission.to_csv(out_path, index=False)
print(
    f"Wrote: {out_path} with shape {submission.shape} and columns {submission.columns.tolist()}"
)
print(submission["target"].describe())
print("NaNs in target:", int(submission["target"].isna().sum()))
