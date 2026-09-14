# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
        input/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
            test/
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
            train/
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> input/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> input/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> input/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> working/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception as e:
    print("Protobuf shim not applied:", e)

import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.optimizer.set_jit(True)

tf.config.threading.set_intra_op_parallelism_threads(
    tf.config.threading.get_intra_op_parallelism_threads()
)
tf.config.threading.set_inter_op_parallelism_threads(
    tf.config.threading.get_inter_op_parallelism_threads()
)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy(mixed_precision.Policy("mixed_float16"))

W = H = 338
BASE_DIR = "../input/ranzcr-clip-catheter-line-classification"
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUBMIT_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

WEIGHT_DIR = "./weights"
os.makedirs(WEIGHT_DIR, exist_ok=True)
weight_file = os.path.join(WEIGHT_DIR, "ranzcr_efficientb5.weights.h5")

sample_df = pd.read_csv(SAMPLE_SUBMIT_PATH)
uid_list = sample_df["StudyInstanceUID"].tolist()

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def preprocess_image(image_bytes, augment=False):
    """Decode, resize and normalize a JPEG image. Optional simple augmentation."""
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, (H, W))
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - mean) / std
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img


def get_model(base_cls, weight_path=None, lr=0.001):
    """Create EfficientNet model with the correct number of output classes."""
    base = base_cls(
        include_top=False,
        input_shape=(H, W, 3),
        pooling="avg",
        weights="imagenet",
    )
    x = tf.keras.layers.Dropout(0.3)(base.output)
    x = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    if weight_path and os.path.exists(weight_path):
        try:
            model.load_weights(weight_path)
            print(f"Loaded weights from {weight_path}")
        except Exception as e:
            print(f"Failed to load weights from {weight_path}: {e}")
    else:
        print("Weight file not found – using ImageNet initialization.")
    return model


base_cls = tf.keras.applications.EfficientNetB5




## === cell 1
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

train_df = pd.read_csv(TRAIN_CSV_PATH)

target_cols = [
    col for col in train_df.columns if col not in ("StudyInstanceUID", "PatientID")
]
N_CLASSES = len(target_cols)
print(f"Detected {N_CLASSES} target columns:", target_cols)

model = get_model(base_cls, weight_path=weight_file, lr=1e-4)

train_df = train_df[["StudyInstanceUID"] + target_cols]
train_df["img_path"] = TRAIN_DIR + "/" + train_df["StudyInstanceUID"] + ".jpg"
train_df = train_df[train_df["img_path"].apply(os.path.exists)]

label_array = train_df[target_cols].values.astype(np.float32)
path_array = train_df["img_path"].values

train_paths, val_paths, train_labels, val_labels = train_test_split(
    path_array, label_array, test_size=0.1, random_state=42, shuffle=True
)


def tf_data_generator(paths, labels, batch_size=32, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_image(path, label):
        image_bytes = tf.io.read_file(path)
        img = preprocess_image(image_bytes, augment=augment)
        return img, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


BATCH_SIZE = 32
train_ds = tf_data_generator(
    train_paths, train_labels, batch_size=BATCH_SIZE, augment=True
)
val_ds = tf_data_generator(val_paths, val_labels, batch_size=BATCH_SIZE, augment=False)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=weight_file,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", factor=0.5, patience=2, mode="max", min_lr=1e-6, verbose=1
)

EPOCHS = 3
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, reduce_lr_cb],
    verbose=2,
)

model.load_weights(weight_file)
print(f"Best weights loaded from {weight_file} for inference.")
print(f"Model training completed. Best weights saved to {weight_file}")




## === cell 2
submission_target_cols = [col for col in sample_df.columns if col != "StudyInstanceUID"]
col_index_map = [target_cols.index(col) for col in submission_target_cols]

existing_indices = []
existing_paths = []
for idx, uid in enumerate(uid_list):
    img_path = os.path.join(TEST_DIR, f"{uid}.jpg")
    if os.path.exists(img_path):
        existing_indices.append(idx)
        existing_paths.append(img_path)


def _load_test_image(path):
    image_bytes = tf.io.read_file(path)
    img = preprocess_image(image_bytes, augment=False)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(existing_paths)
test_ds = test_ds.map(
    _load_test_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

full_preds = model.predict(test_ds, verbose=0)  # shape (num_existing, N_CLASSES)
selected_preds = full_preds[:, col_index_map]  # keep only submission columns

num_test = len(uid_list)
num_target = len(submission_target_cols)
pred_array = np.zeros((num_test, num_target), dtype=np.float32)
pred_array[existing_indices] = selected_preds.astype(np.float32)

submission = pd.DataFrame(pred_array, columns=submission_target_cols)
submission.insert(0, "StudyInstanceUID", uid_list)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
