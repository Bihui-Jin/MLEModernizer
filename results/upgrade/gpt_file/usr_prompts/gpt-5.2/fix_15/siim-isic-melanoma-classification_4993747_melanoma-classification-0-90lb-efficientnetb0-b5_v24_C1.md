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

No external packages required in the script and installed.

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
import random
import datetime

import numpy as np
from numpy.random import shuffle
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam

pd.set_option("expand_frame_repr", False)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


BASE_DIR = "../input/siim-isic-melanoma-classification"
Labels_Dir = os.path.join(BASE_DIR, "{}.csv")
Train_Images_Dir = os.path.join(BASE_DIR, "jpeg", "train")
Test_Images_Dir = os.path.join(BASE_DIR, "jpeg", "test")

Output_Dir = r"/kaggle/working/{}.h5"

Epochs = 4
Batch_Size = 32
Early_Stop = EarlyStopping(
    monitor="val_loss", mode="auto", verbose=1, patience=1, restore_best_weights=True
)

Image_Size = (299, 299)  # (W,H) used throughout this notebook
Class_Weight = {0: 1, 1: 3.32}

Train_Over_Sampel_Count = 12_000  # for data augmentation
Valid_Over_Sampel_Count = 3_254  # kept for compatibility (unused)

print("TF version:", tf.__version__)
print("Train images dir exists:", os.path.isdir(Train_Images_Dir), Train_Images_Dir)
print("Test images dir exists:", os.path.isdir(Test_Images_Dir), Test_Images_Dir)

CPU_COUNT = os.cpu_count() or 4
MAP_PARALLEL = min(8, CPU_COUNT)

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
def data_augmentation(labels, target_length):
    labels = np.asarray(labels)
    targets = labels[:, -2].astype(np.int64, copy=False)
    pos_idx = np.flatnonzero(targets == 1)
    if pos_idx.size == 0 or target_length <= 0:
        shuffle(labels)
        return labels

    chosen = np.random.choice(pos_idx, size=target_length, replace=True)
    modes = np.random.randint(0, 7, size=target_length, dtype=np.int64)

    aug = labels[chosen].copy()
    aug[:, -1] = modes.astype(aug[:, -1].dtype, copy=False)

    out = np.concatenate([labels, aug], axis=0)
    shuffle(out)
    return out




## === cell 2
Labels = pd.read_csv(Labels_Dir.format("train"))
Labels = Labels.sample(frac=1, random_state=SEED).reset_index(drop=True)

Labels["argu_mode"] = 0

Labels["sex"] = Labels["sex"].replace({"male": 1, "female": 0})
Labels["sex"] = Labels["sex"].fillna(-1)

Positive_Indexs = Labels.index[Labels["target"] == 1].to_list()
Negative_Indexs = Labels.index[Labels["target"] == 0].to_list()

Labels_np = Labels.to_numpy()
Positive_Cases = Labels_np[Positive_Indexs]
Negative_Cases = Labels_np[Negative_Indexs]

shuffle(Positive_Cases)
shuffle(Negative_Cases)

Place = int(len(Negative_Cases) * 0.1)

Train_Labels = np.concatenate([Positive_Cases[4:], Negative_Cases[Place:]])
Validation_Labels = np.concatenate([Positive_Cases[0:4], Negative_Cases[:Place]])

Train_Labels = data_augmentation(Train_Labels, Train_Over_Sampel_Count)

shuffle(Train_Labels)
shuffle(Validation_Labels)

Train_Positive_Count = int(
    np.count_nonzero(Train_Labels[:, -2].astype(np.int64, copy=False) == 1)
)
Valid_Positive_Count = int(
    np.count_nonzero(Validation_Labels[:, -2].astype(np.int64, copy=False) == 1)
)

print(
    f"Len Validation_Data: {len(Validation_Labels)}\tLen Train Data: {len(Train_Labels)}"
)
print(
    "\tTrain positives:",
    Train_Positive_Count,
    "\tValidation positives:",
    Valid_Positive_Count,
)




## === cell 3
def _make_paths_targets_modes(labels, imgs_dir):
    names = labels[:, 0].astype(str)
    base = imgs_dir.rstrip("/") + "/"
    paths = np.char.add(np.char.add(base, names), ".jpg").astype(str)
    targets = labels[:, -2].astype(np.float32, copy=False)
    modes = labels[:, -1].astype(np.int32, copy=False)
    return paths, targets, modes


@tf.function(reduce_retracing=True)
def _tf_shift_int_constant(im, dx, dy):
    dx = tf.cast(dx, tf.int32)
    dy = tf.cast(dy, tf.int32)
    h = tf.shape(im)[0]
    w = tf.shape(im)[1]

    pad_top = tf.maximum(dy, 0)
    pad_bottom = tf.maximum(-dy, 0)
    pad_left = tf.maximum(dx, 0)
    pad_right = tf.maximum(-dx, 0)

    padded = tf.pad(
        im,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        mode="CONSTANT",
        constant_values=0.0,
    )

    start_y = tf.maximum(-dy, 0)
    start_x = tf.maximum(-dx, 0)
    return padded[start_y : start_y + h, start_x : start_x + w, :]


@tf.function(reduce_retracing=True)
def _tf_augment(img, mode):
    w = Image_Size[0]
    h = Image_Size[1]
    mode = tf.cast(mode, tf.int32)

    def mode0():
        return img

    def mode1():
        return tf.image.flip_left_right(img)

    def mode2():
        im2 = tf.image.flip_up_down(img)
        return _tf_shift_int_constant(im2, 20, 20)

    def mode3():
        im3 = tf.image.flip_left_right(tf.image.flip_up_down(img))
        return _tf_shift_int_constant(im3, 33, 33)

    def mode4():
        im4 = tf.image.rot90(img, k=3)
        return _tf_shift_int_constant(im4, 23, 25)

    def mode5():
        return tf.image.rot90(img, k=1)

    def mode6():
        im6 = img[25 : h - 25, 25 : w - 25, :]
        im6 = tf.image.resize(im6, [h, w], method="bilinear", antialias=False)
        return _tf_shift_int_constant(im6, 32, 32)

    return tf.switch_case(
        mode,
        branch_fns={
            0: mode0,
            1: mode1,
            2: mode2,
            3: mode3,
            4: mode4,
            5: mode5,
            6: mode6,
        },
        default=mode0,
    )


@tf.function(reduce_retracing=True)
def _tf_load_base(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [Image_Size[1], Image_Size[0]], method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((Image_Size[1], Image_Size[0], 3))
    return img


@tf.function(reduce_retracing=True)
def _tf_load_apply_aug(path, target, mode):
    img = _tf_load_base(path)
    img = _tf_augment(img, mode)
    img.set_shape((Image_Size[1], Image_Size[0], 3))
    return img, tf.cast(target, tf.float32)


@tf.function(reduce_retracing=True)
def _tf_load_no_aug(path):
    img = _tf_load_base(path)
    img.set_shape((Image_Size[1], Image_Size[0], 3))
    return img


Train_Paths, Train_Targets, Train_Modes = _make_paths_targets_modes(
    Train_Labels, Train_Images_Dir
)
Valid_Paths, Valid_Targets, Valid_Modes = _make_paths_targets_modes(
    Validation_Labels, Train_Images_Dir
)

options = tf.data.Options()
options.deterministic = True

train_img_cache = (
    tf.data.Dataset.from_tensor_slices(Train_Paths)
    .with_options(options)
    .map(_tf_load_no_aug, num_parallel_calls=MAP_PARALLEL, deterministic=True)
    .cache()
)
train_ds = (
    tf.data.Dataset.zip(
        (
            train_img_cache,
            tf.data.Dataset.from_tensor_slices(
                (Train_Targets, Train_Modes)
            ).with_options(options),
        )
    )
    .map(
        lambda img, tm: (_tf_augment(img, tm[1]), tf.cast(tm[0], tf.float32)),
        num_parallel_calls=MAP_PARALLEL,
        deterministic=True,
    )
    .batch(Batch_Size, drop_remainder=True)
    .repeat()
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((Valid_Paths, Valid_Targets))
    .with_options(options)
    .map(
        lambda p, t: (_tf_load_no_aug(p), tf.cast(t, tf.float32)),
        num_parallel_calls=MAP_PARALLEL,
        deterministic=True,
    )
    .cache()
    .batch(Batch_Size, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

steps_per_epoch = len(Train_Labels) // Batch_Size
validation_steps = max(1, len(Validation_Labels) // Batch_Size)

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 4
def build_lrfn(
    lr_start=0.00001,
    lr_max=0.0001,
    lr_min=0.000001,
    lr_rampup_epochs=20,
    lr_sustain_epochs=0,
    lr_exp_decay=0.8,
):
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (
                epoch - lr_rampup_epochs - lr_sustain_epochs
            ) + lr_min
        return lr

    return lrfn


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)




## === cell 5
EfficientNetB2 = tf.keras.applications.EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3)
)


def make_model(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="sigmoid"))
    return model




## === cell 6
Model = make_model(EfficientNetB2)
Model.Name = "efficentB2"

SGD_Optimizer = SGD(learning_rate=0.1)
Adam_Optimizer = Adam(learning_rate=0.1)

Model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

Model.summary()

history = Model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    epochs=Epochs,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    class_weight=Class_Weight,
    callbacks=[lr_schedule, Early_Stop],
)




## === cell 7
model_path = Output_Dir.format(
    f"{Model.Name}--{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
Model.save(model_path)
print("Saved model to:", model_path)




## === cell 8
Sample_Submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
Test_Labels = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

test_names = Test_Labels["image_name"].tolist()
test_paths = (
    np.char.add(
        np.char.add(Test_Images_Dir.rstrip("/") + "/", np.array(test_names, dtype=str)),
        ".jpg",
    )
).astype(str)

test_options = tf.data.Options()
test_options.deterministic = True

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(test_options)
    .map(_tf_load_no_aug, num_parallel_calls=MAP_PARALLEL, deterministic=True)
    .batch(256, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_preds = Model.predict(test_ds, verbose=1).reshape(-1)

My_Submission = pd.DataFrame(
    {"image_name": test_names, "target": test_preds.astype(float)}
)

My_Submission = Sample_Submission[["image_name"]].merge(
    My_Submission, on="image_name", how="left"
)
My_Submission["target"] = My_Submission["target"].fillna(0.5).astype(float)

My_Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", My_Submission.shape)
print(My_Submission.head())
print(My_Submission.tail())
