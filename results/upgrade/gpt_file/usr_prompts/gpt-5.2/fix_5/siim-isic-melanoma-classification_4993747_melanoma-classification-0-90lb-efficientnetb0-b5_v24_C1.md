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
import copy
import datetime

import numpy as np
from numpy.random import shuffle
import pandas as pd

import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
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

TB_Log_Dir = r"/kaggle/working/{}/"  # TensorBoard logs directory
Output_Dir = r"/kaggle/working/{}.h5"

Epochs = 4
Batch_Size = 32
Early_Stop = EarlyStopping(
    monitor="val_loss", mode="auto", verbose=1, patience=1, restore_best_weights=True
)

Image_Size = (299, 299)
Class_Weight = {0: 1, 1: 3.32}

Train_Over_Sampel_Count = 12_000  # for data augmentation
Valid_Over_Sampel_Count = 3_254  # kept for compatibility (unused)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Train images dir exists:", os.path.isdir(Train_Images_Dir), Train_Images_Dir)
print("Test images dir exists:", os.path.isdir(Test_Images_Dir), Test_Images_Dir)




## === cell 1
def data_augmentation(labels, target_length):
    labels = np.asarray(labels)
    targets = labels[:, -2].astype(np.int64, copy=False)
    pos_idx = np.flatnonzero(targets == 1)
    if pos_idx.size == 0 or target_length <= 0:
        shuffle(labels)
        return labels

    chosen = [pos_idx[random.randrange(pos_idx.size)] for _ in range(target_length)]
    modes = [random.randint(0, 6) for _ in range(target_length)]

    aug = labels[chosen].copy()
    aug[:, -1] = np.asarray(modes, dtype=aug[:, -1].dtype)

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
    paths = np.array([os.path.join(imgs_dir, f"{n}.jpg") for n in names], dtype=object)
    targets = labels[:, -2].astype(np.float32, copy=False)
    modes = labels[:, -1].astype(np.int32, copy=False)
    return paths, targets, modes


def _unique_base_cache_from_paths(paths, image_size):
    uniq_paths, inv = np.unique(paths, return_inverse=True)
    h, w = image_size[1], image_size[0]
    base = np.empty((len(uniq_paths), h, w, 3), dtype=np.float32)
    for i, p in enumerate(tqdm(uniq_paths, desc="Caching resized train/valid images")):
        img = cv2.imread(str(p), 1)
        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base[i] = img.astype(np.float32) / 255.0
    return uniq_paths, inv.astype(np.int32, copy=False), base


def _tf_augment(img, mode, image_size):
    w = int(image_size[0])
    h = int(image_size[1])
    mode = tf.cast(mode, tf.int32)

    def _translate(im, dx, dy):
        dx = tf.cast(dx, tf.float32)
        dy = tf.cast(dy, tf.float32)
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(im, 0),
            transforms=tf.reshape(
                tf.stack([1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0], axis=0), [1, 8]
            ),
            output_shape=tf.constant([h, w], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
            fill_value=0.0,
        )[0]

    def mode0(im):
        return im

    def mode1(im):
        return tf.image.flip_left_right(im)

    def mode2(im):
        im = tf.image.flip_up_down(im)
        return _translate(im, 20.0, 20.0)

    def mode3(im):
        im = tf.image.flip_left_right(tf.image.flip_up_down(im))
        return _translate(im, 33.0, 33.0)

    def mode4(im):
        im = tf.image.rot90(im, k=3)
        return _translate(im, 23.0, 25.0)

    def mode5(im):
        return tf.image.rot90(im, k=1)

    def mode6(im):
        im = im[25 : h - 25, 25 : w - 25, :]
        im = tf.image.resize(im, [h, w], method="bilinear", antialias=False)
        return _translate(im, 32.0, 32.0)

    return tf.switch_case(
        mode,
        branch_fns={
            0: lambda: mode0(img),
            1: lambda: mode1(img),
            2: lambda: mode2(img),
            3: lambda: mode3(img),
            4: lambda: mode4(img),
            5: lambda: mode5(img),
            6: lambda: mode6(img),
        },
        default=lambda: mode0(img),
    )


Train_Paths, Train_Targets, Train_Modes = _make_paths_targets_modes(
    Train_Labels, Train_Images_Dir
)
Valid_Paths, Valid_Targets, Valid_Modes = _make_paths_targets_modes(
    Validation_Labels, Train_Images_Dir
)

_all_paths = np.concatenate([Train_Paths, Valid_Paths], axis=0)
uniq_paths, inv_all, base_imgs = _unique_base_cache_from_paths(_all_paths, Image_Size)

inv_train = inv_all[: len(Train_Paths)]
inv_valid = inv_all[len(Train_Paths) :]

options = tf.data.Options()
options.deterministic = True

base_imgs_tf = tf.constant(base_imgs, dtype=tf.float32)


def _map_from_cache(idx, target, mode):
    img = tf.gather(base_imgs_tf, idx)
    img = _tf_augment(img, mode, Image_Size)
    img.set_shape((Image_Size[1], Image_Size[0], 3))
    target = tf.cast(target, tf.float32)
    return img, target


train_ds = (
    tf.data.Dataset.from_tensor_slices((inv_train, Train_Targets, Train_Modes))
    .with_options(options)
    .map(_map_from_cache, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    .batch(Batch_Size, drop_remainder=False)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((inv_valid, Valid_Targets, Valid_Modes))
    .with_options(options)
    .map(_map_from_cache, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    .batch(Batch_Size, drop_remainder=False)
    .cache()
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

steps_per_epoch = int(np.ceil(len(Train_Labels) / Batch_Size))
validation_steps = int(np.ceil(len(Validation_Labels) / Batch_Size))

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
def _tf_load_test(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [Image_Size[1], Image_Size[0]], method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((Image_Size[1], Image_Size[0], 3))
    return img


Sample_Submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
Test_Labels = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
test_names = Test_Labels["image_name"].tolist()
test_paths = [os.path.join(Test_Images_Dir, f"{n}.jpg") for n in test_names]

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(tf.data.Options())
    .map(_tf_load_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
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
