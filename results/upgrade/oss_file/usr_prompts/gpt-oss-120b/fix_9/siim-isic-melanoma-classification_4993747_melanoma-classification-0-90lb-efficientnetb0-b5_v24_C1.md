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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import random, datetime
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import EfficientNetB2

np.random.seed(42)
random.seed(42)
tf.get_logger().setLevel("ERROR")

Labels_Dir = "../input/siim-isic-melanoma-classification/{}.csv"
Images_Dir = "../input/siim-isic-melanoma-classification/jpeg/train"
Test_Images_Dir = "../input/siim-isic-melanoma-classification/jpeg/test"

Image_Size = (260, 260)  # width, height
Batch_Size = 32
Epochs = 30  # more epochs for better learning
Train_Over_Sampel_Count = 0  # will be set later after split

TB_Log_Dir = "./logs/{}"
Output_Dir = "./models/{}.h5"
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models", exist_ok=True)




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




## === cell 3
def _load_and_preprocess(id_bytes, target, mode, imgs_dir, img_size):
    if isinstance(imgs_dir, bytes):
        imgs_dir = imgs_dir.decode()
    if isinstance(img_size, bytes):
        img_size = eval(img_size.decode())  # safely reconstruct tuple

    img_id = id_bytes.decode() if isinstance(id_bytes, bytes) else str(id_bytes)
    img_path = os.path.join(imgs_dir, f"{img_id}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    img = cv2.resize(img, img_size)

    if mode == 1:
        img = cv2.flip(img, 1)
    elif mode == 2:
        img = cv2.flip(img, 0)
        matrix = np.float32([[1, 0, 20], [0, 1, 20]])
        img = cv2.warpAffine(img, matrix, img_size)
    elif mode == 3:
        img = cv2.flip(img, -1)
        matrix = np.float32([[1, 0, 33], [0, 1, 33]])
        img = cv2.warpAffine(img, matrix, img_size)
    elif mode == 4:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        matrix = np.float32([[1, 0, 23], [0, 1, 25]])
        img = cv2.warpAffine(img, matrix, img_size)
    elif mode == 5:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif mode == 6:
        img = img[25 : img_size[1] - 25, 25 : img_size[0] - 25]
        img = cv2.resize(img, (img_size[0], img_size[1]))
        matrix = np.float32([[1, 0, 32], [0, 1, 32]])
        img = cv2.warpAffine(img, matrix, img_size)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = img.astype("float32")
    return img, np.float32(target)


def tf_data_generator(labels_np, imgs_dir, image_size, batch_size):
    ids = labels_np[:, 0].astype(str)
    targets = labels_np[:, -2].astype(np.float32)
    modes = labels_np[:, -1].astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((ids, targets, modes))

    def _wrapper(id_str, target, mode):
        img, tar = tf.numpy_function(
            _load_and_preprocess,
            [id_str, target, mode, imgs_dir, image_size],
            [tf.float32, tf.float32],
        )
        img.set_shape((image_size[0], image_size[1], 3))
        tar.set_shape(())
        return img, tar

    dataset = dataset.map(_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024).cache()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


Train_DS = tf_data_generator(Train_Labels, Images_Dir, Image_Size, Batch_Size)
Validation_DS = tf_data_generator(Validation_Labels, Images_Dir, Image_Size, Batch_Size)



## === cell 4
base_efficientnet = EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3)
)




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
TB_Callback = TensorBoard(log_dir=TB_Log_Dir.format(Model.Name), histogram_freq=1)

Model.compile(
    loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"]
)

Model.summary()

Model.fit(
    Train_DS,
    epochs=Epochs,
    validation_data=Validation_DS,
    class_weight=Class_Weight,
    callbacks=[lr_schedule, Early_Stop, TB_Callback],
    verbose=1,
)



## === cell 7
Model.save(
    Output_Dir.format(
        f"{Model.Name}--{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
)




## === cell 8
def batch_predict(image_paths, model, image_size=(260, 260), batch_sz=32):
    preds = []
    num_images = len(image_paths)
    for start in range(0, num_images, batch_sz):
        batch_paths = image_paths[start : start + batch_sz]
        batch_imgs = []
        for p in batch_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            batch_imgs.append(img)
        batch_arr = np.stack(batch_imgs, axis=0)
        batch_pred = model.predict(batch_arr, verbose=0)
        preds.append(batch_pred.squeeze())
    return np.concatenate(preds, axis=0)




## === cell 9
Test_Labels = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
test_image_paths = [
    os.path.join(Test_Images_Dir, f"{row['image_name']}.jpg")
    for _, row in Test_Labels.iterrows()
]

predictions = batch_predict(
    test_image_paths, Model, image_size=Image_Size, batch_sz=Batch_Size
)

submission_df = pd.DataFrame(
    {"image_name": Test_Labels["image_name"], "target": predictions}
)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
print(submission_df.head())
