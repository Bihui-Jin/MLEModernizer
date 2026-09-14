# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9380388595462292

# 6. Current score

0.50294

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.52373) has done: 'I avoid the TensorFlow import error by wrapping it in a try‑except and falling back to a dummy model that returns zeros for every test image. The prediction loop is simplified to generate a zero matrix matching the required number of rows and columns, then the submission DataFrame is built with the correct column order and saved as `submission.csv`. This fixes the runtime crashes and ensures a valid CSV is produced.'
- What this solution (achieved 0.50294) has done: 'I fixed the TensorFlow import error by falling back to a lightweight scikit‑learn model when TensorFlow cannot be loaded. The fallback extracts simple RGB‑mean features from each image, trains a separate LogisticRegression for every target label using the provided training data, and generates probability predictions for the test set. This provides a meaningful model (instead of all‑zero outputs) and thus moves the score toward the target while keeping the original TensorFlow workflow intact for environments where it works.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression

try:
    import tensorflow as tf

    tf_available = True
except Exception as e:
    tf = None
    tf_available = False
    print("TensorFlow import failed:", e)

W = H = 338
N_CLASSES = 11
autotune = tf.data.experimental.AUTOTUNE if tf_available else None

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_image_dir = "../input/ranzcr-clip-catheter-line-classification/test"
train_image_dir = "../input/ranzcr-clip-catheter-line-classification/train"
sample_submission_path = (
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)
weight_dir = "../input/cassava2020weights"

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3 if tf_available else None,
        weight_dir + "/ranzcr_efficientb3.h5",
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5 if tf_available else None,
        weight_dir + "/ranzcr_efficientb5.h5",
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7 if tf_available else None,
        weight_dir + "/ranzcr_efficientb5.h5",
    ],
}

sample_df = pd.read_csv(sample_submission_path, nrows=0)
target_cols = list(sample_df.columns)[1:]


def load_and_preprocess(path):
    """Read an image file, decode, resize and normalise."""
    if not tf_available:
        return None
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (H, W))
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - mean) / std
    return img


def get_filenames_and_ids(image_dir):
    """Return sorted filenames and corresponding StudyInstanceUIDs."""
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
    paths = [os.path.join(image_dir, f) for f in files]
    ids = [os.path.splitext(f)[0] for f in files]  # UID is filename without extension
    return paths, ids


def extract_rgb_mean(path):
    """Simple feature: mean of each RGB channel scaled to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.array([0.0, 0.0, 0.0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img.mean(axis=(0, 1))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_paths, test_ids = get_filenames_and_ids(test_image_dir)

if tf_available:
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=autotune)
    test_ds = test_ds.batch(16).prefetch(autotune)
else:
    test_ds = None
    test_features = np.array(
        [extract_rgb_mean(p) for p in test_paths], dtype=np.float32
    )




## === cell 2
train_df = pd.read_csv("../input/ranzcr-clip-catheter-line-classification/train.csv")
train_paths, train_ids = get_filenames_and_ids(train_image_dir)

if not tf_available:
    train_features = np.array(
        [extract_rgb_mean(p) for p in train_paths], dtype=np.float32
    )

    sk_models = {}
    X = train_features
    for col in target_cols:
        y = train_df[col].values
        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=200,
            multi_class="ovr",
            n_jobs=-1,
        )
        lr.fit(X, y)
        sk_models[col] = lr




## === cell 3
class DummyModel:
    def predict_on_batch(self, batch):
        batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)
        return np.zeros((batch_size, N_CLASSES), dtype=np.float32)


def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam if tf_available else None,
):
    """Return a dummy model when TensorFlow is unavailable."""
    if not tf_available:
        return DummyModel()
    base = base_model(
        include_top=False, input_shape=(W, H, 3), pooling="avg", weights=baseline_weight
    )
    x = tf.keras.layers.Dropout(0.3)(base.output)
    x = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=x)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )
    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    return model


base_model, weight_path = model_map["efficientb5"]
model = get_model(base_model, init_weight=weight_path)




## === cell 4
if tf_available and test_ds is not None:
    preds_list = []
    for batch in test_ds:
        batch_preds = model.predict_on_batch(batch)  # already a numpy array
        preds_list.append(batch_preds[:, : len(target_cols)])
    preds = np.concatenate(preds_list, axis=0)
else:
    preds = np.column_stack(
        [sk_models[col].predict_proba(test_features)[:, 1] for col in target_cols]
    )

submission = pd.DataFrame(preds, columns=target_cols)
submission.insert(0, "StudyInstanceUID", test_ids)

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
