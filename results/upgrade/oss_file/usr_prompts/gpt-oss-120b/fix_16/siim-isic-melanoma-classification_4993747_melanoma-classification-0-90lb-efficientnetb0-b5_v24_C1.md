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

# 5. Target score

0.8663659563933698

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Implemented fixes to resolve import errors, incorrect label handling, wrong image identifier usage, and missing model definitions. Replaced the external EfficientNet package with TensorFlow’s built‑in EfficientNet models, corrected the negative index selection, used the proper image name column for loading files, and streamlined paths. Added necessary imports and ensured the script writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.3922) has done: 'The fix adds dataset caching to avoid re‑reading and augmenting images each epoch and replaces the per‑image test loop with a batched preprocessing‑and‑prediction step, which dramatically reduces I/O and TensorFlow overhead while keeping the model, augmentation, and training logic unchanged.'
- What this solution (achieved 0.3922) has done: 'I fix the import of EfficientNetB2 to avoid the protobuf error, initialise the augmentation mode column with integer zeros (so TensorFlow can cast it), and increase training epochs slightly to give the model more learning capacity. These changes resolve the runtime crashes and let the pipeline produce a proper `submission.csv`, while keeping the original model architecture and training logic intact.'
- What this solution (achieved 0.3922) has done: 'We replace the slow Python + OpenCV image loader used in the `tf_data_generator` with a fully TensorFlow‑based pipeline that decodes JPEGs, resizes, and applies the same seven augmentation modes using TensorFlow image ops. This removes the per‑sample Python overhead and lets TensorFlow read, preprocess, and batch images in parallel on the C++ backend. We also drop the large in‑memory cache for the training set (it caused heavy swapping) while keeping a small cache for validation so later epochs stay fast. The rest of the model, training loop, and prediction logic stay unchanged, preserving exact architecture, loss, and evaluation semantics.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import random, datetime
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

try:
    from tensorflow.keras.applications import EfficientNetB2
except Exception:  # pragma: no cover
    EfficientNetB2 = tf.keras.applications.EfficientNetB2

tf.config.threading.set_intra_op_parallelism_threads(tf.config.threading.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(tf.config.threading.cpu_count())

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")

np.random.seed(42)
random.seed(42)
tf.get_logger().setLevel("ERROR")

Labels_Dir = "../input/siim-isic-melanoma-classification/{}.csv"
Images_Dir = "../input/siim-isic-melanoma-classification/jpeg/train"
Test_Images_Dir = "../input/siim-isic-melanoma-classification/jpeg/test"

Image_Size = (260, 260)  # width, height
Batch_Size = 128  # larger batch reduces steps per epoch without altering model logic
Epochs = 30  # unchanged
Train_Over_Sampel_Count = 0  # will be set later after split

TB_Log_Dir = "./logs/{}"
Output_Dir = "./models/{}.h5"
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models", exist_ok=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def data_augmentation(labels, target_length):
    """
    Expand `labels` to `target_length` by duplicating positive rows
    and assigning a random augmentation mode (0‑6). If the target
    length is not larger, simply shuffle and return the original.
    """
    current_len = len(labels)
    if current_len >= target_length:
        np.random.shuffle(labels)
        return labels

    n_needed = target_length - current_len
    positive_rows = labels[labels[:, -2] == 1]  # column -2 is target

    if len(positive_rows) == 0:
        np.random.shuffle(labels)
        return labels

    idx = np.random.randint(0, len(positive_rows), size=n_needed)
    aug_rows = positive_rows[idx].copy()
    aug_rows[:, -1] = np.random.randint(0, 7, size=n_needed)  # random mode
    new_labels = np.concatenate([labels, aug_rows], axis=0)
    np.random.shuffle(new_labels)
    return new_labels




## === cell 2
Labels = pd.read_csv(Labels_Dir.format("train"))
Labels = Labels.sample(frac=1, random_state=42).reset_index(drop=True)

Labels["argu_mode"] = 0
Labels["sex"].replace({"male": 1, "female": 0}, inplace=True)

Positive_Indexs = Labels.index[Labels["target"] == 1].tolist()
Negative_Indexs = Labels.index[Labels["target"] == 0].tolist()

Labels_np = Labels.to_numpy()
Positive_Cases = Labels_np[Positive_Indexs]
Negative_Cases = Labels_np[Negative_Indexs]

np.random.shuffle(Positive_Cases)
np.random.shuffle(Negative_Cases)

Place = int(len(Negative_Cases) * 0.1)

Train_Labels = np.concatenate([Positive_Cases[4:], Negative_Cases[Place:]])
Validation_Labels = np.concatenate([Positive_Cases[:4], Negative_Cases[:Place]])

Train_Over_Sampel_Count = len(Train_Labels) * 2
Train_Labels = data_augmentation(Train_Labels, Train_Over_Sampel_Count)

np.random.shuffle(Train_Labels)
np.random.shuffle(Validation_Labels)

Train_Positive_Count = np.count_nonzero(Train_Labels[:, -2] == 1)
Valid_Positive_Count = np.count_nonzero(Validation_Labels[:, -2] == 1)

print(
    f"Train positives: {Train_Positive_Count}, Validation positives: {Valid_Positive_Count}"
)
print(f"Len Train: {len(Train_Labels)}, Len Validation: {len(Validation_Labels)}")

neg_cnt = np.count_nonzero(Train_Labels[:, -2] == 0)
pos_cnt = np.count_nonzero(Train_Labels[:, -2] == 1)
Class_Weight = {0: 1.0, 1: (neg_cnt / (pos_cnt + 1e-6))}

Early_Stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4186737677.py in <cell line: 0>()
----> 1 Labels = pd.read_csv(Labels_Dir.format("train"))
      2 Labels = Labels.sample(frac=1, random_state=42).reset_index(drop=True)
      3 
      4 Labels["argu_mode"] = 0
      5 Labels["sex"].replace({"male": 1, "female": 0}, inplace=True)

NameError: name 'Labels_Dir' is not defined

## === cell 3
def _tf_augment(img, mode, img_h, img_w):
    """Apply the same 7 augmentation modes using TF ops."""

    def _identity():
        return img

    def _hflip():
        return tf.image.flip_left_right(img)

    def _vflip_trans():
        flipped = tf.image.flip_up_down(img)
        padded = tf.image.pad_to_bounding_box(flipped, 20, 20, img_h + 20, img_w + 20)
        return tf.image.crop_to_bounding_box(padded, 20, 20, img_h, img_w)

    def _rot180_trans():
        rotated = tf.image.rot90(tf.image.rot90(img))
        padded = tf.image.pad_to_bounding_box(rotated, 33, 33, img_h + 33, img_w + 33)
        return tf.image.crop_to_bounding_box(padded, 33, 33, img_h, img_w)

    def _rot90_cw_trans():
        rotated = tf.image.rot90(img, k=1)
        padded = tf.image.pad_to_bounding_box(rotated, 25, 23, img_h + 25, img_w + 23)
        return tf.image.crop_to_bounding_box(padded, 25, 23, img_h, img_w)

    def _rot90_ccw():
        return tf.image.rot90(img, k=3)

    def _crop_resize_trans():
        cropped = tf.image.crop_to_bounding_box(img, 25, 25, img_h - 50, img_w - 50)
        resized = tf.image.resize(cropped, [img_h, img_w])
        padded = tf.image.pad_to_bounding_box(resized, 32, 32, img_h + 32, img_w + 32)
        return tf.image.crop_to_bounding_box(padded, 32, 32, img_h, img_w)

    cases = {
        0: _identity,
        1: _hflip,
        2: _vflip_trans,
        3: _rot180_trans,
        4: _rot90_cw_trans,
        5: _rot90_ccw,
        6: _crop_resize_trans,
    }
    return tf.switch_case(mode, branch_fns=cases)


def _tf_load_and_preprocess(id_str, img_size):
    """Load JPEG, decode, resize and normalise (no augmentation)."""
    img_path = tf.strings.join([Images_Dir, "/", id_str, ".jpg"])
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def tf_data_generator(labels_np, imgs_dir, image_size, batch_size, cache=True):
    ids = labels_np[:, 0].astype(str)
    targets = labels_np[:, -2].astype(np.float32)
    modes = labels_np[:, -1].astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((ids, targets, modes))

    def _load(id_str, target, mode):
        img = _tf_load_and_preprocess(id_str, image_size)
        return img, target, mode

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    def _augment(img, target, mode):
        img_h = tf.cast(image_size[1], tf.int32)
        img_w = tf.cast(image_size[0], tf.int32)
        img = _tf_augment(img, mode, img_h, img_w)
        img.set_shape((image_size[1], image_size[0], 3))
        return img, tf.cast(target, tf.float32)

    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


Train_DS = tf_data_generator(
    Train_Labels, Images_Dir, Image_Size, Batch_Size, cache=True
)
Validation_DS = tf_data_generator(
    Validation_Labels, Images_Dir, Image_Size, Batch_Size, cache=True
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1461883503.py in <cell line: 0>()
     85 
     86 Train_DS = tf_data_generator(
---> 87     Train_Labels, Images_Dir, Image_Size, Batch_Size, cache=True
     88 )
     89 Validation_DS = tf_data_generator(

NameError: name 'Train_Labels' is not defined

## === cell 4
base_efficientnet = EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(Image_Size[1], Image_Size[0], 3)
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/150474184.py in <cell line: 0>()
      1 base_efficientnet = EfficientNetB2(
----> 2     include_top=False, weights="imagenet", input_shape=(Image_Size[1], Image_Size[0], 3)
      3 )
      4 
      5 

NameError: name 'Image_Size' is not defined

## === cell 5
def build_lrfn(
    lr_start=1e-5,
    lr_max=1e-4,
    lr_min=1e-6,
    lr_rampup_epochs=5,
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
lr_schedule = LearningRateScheduler(lrfn, verbose=1)




## === cell 6
def make_model(base_model):
    model = Sequential(
        [base_model, GlobalAveragePooling2D(), Dense(1, activation="sigmoid")]
    )
    return model


Model = make_model(base_efficientnet)
Model.Name = "efficientB2"


Model.compile(
    loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"]
)

Model.summary()

Model.fit(
    Train_DS,
    epochs=Epochs,
    validation_data=Validation_DS,
    class_weight=Class_Weight,
    callbacks=[lr_schedule, Early_Stop],  # TensorBoard callback omitted
    verbose=1,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/820170189.py in <cell line: 0>()
      6 
      7 
----> 8 Model = make_model(base_efficientnet)
      9 Model.Name = "efficientB2"
     10 

NameError: name 'base_efficientnet' is not defined

## === cell 7
Model.save(
    Output_Dir.format(
        f"{Model.Name}--{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/779900174.py in <cell line: 0>()
----> 1 Model.save(
      2     Output_Dir.format(
      3         f"{Model.Name}--{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
      4     )
      5 )

NameError: name 'Model' is not defined

## === cell 8
def tf_batch_predict(image_paths, model, image_size=(260, 260), batch_sz=128):
    """Fast TensorFlow‑based prediction over a list of image file paths."""
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = path_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_sz).prefetch(tf.data.AUTOTUNE)
    preds = model.predict(ds, verbose=0)
    return preds.squeeze()


Test_Labels = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
test_image_paths = [
    os.path.join(Test_Images_Dir, f"{row['image_name']}.jpg")
    for _, row in Test_Labels.iterrows()
]

predictions = tf_batch_predict(
    test_image_paths, Model, image_size=Image_Size, batch_sz=Batch_Size
)

submission_df = pd.DataFrame(
    {"image_name": Test_Labels["image_name"], "target": predictions}
)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
print(submission_df.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2249215398.py in <cell line: 0>()
     17 
     18 Test_Labels = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
---> 19 test_image_paths = [
     20     os.path.join(Test_Images_Dir, f"{row['image_name']}.jpg")
     21     for _, row in Test_Labels.iterrows()

/tmp/ipykernel_11/2249215398.py in <listcomp>(.0)
     18 Test_Labels = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
     19 test_image_paths = [
---> 20     os.path.join(Test_Images_Dir, f"{row['image_name']}.jpg")
     21     for _, row in Test_Labels.iterrows()
     22 ]

NameError: name 'Test_Images_Dir' is not defined
