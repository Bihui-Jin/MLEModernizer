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
import random, copy, datetime
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB2

np.random.seed(42)
random.seed(42)
tf.get_logger().setLevel("ERROR")

Labels_Dir = "../input/siim-isic-melanoma-classification/{}.csv"  # expects format string with "train" or "test"
Images_Dir = "../input/siim-isic-melanoma-classification/jpeg/train"
Test_Images_Dir = "../input/siim-isic-melanoma-classification/jpeg/test"

Image_Size = (260, 260)  # EfficientNetB2 default input size
Batch_Size = 32
Epochs = 5  # modest number for quick runs
Train_Over_Sampel_Count = 0  # no extra oversampling (function will just shuffle)

TB_Log_Dir = "./logs/{}"
Output_Dir = "./models/{}.h5"
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

Class_Weight = {0: 1.0, 1: 1.0}
Early_Stop = EarlyStopping(patience=3, restore_best_weights=True, verbose=1)




## === cell 1
def data_augmentation(labels, target_length):
    """
    Efficiently expand the dataset to `target_length` by duplicating positive samples
    and assigning a random augmentation mode (0‑6). If target_length <= current size,
    the function simply shuffles and returns the original array.
    """
    current_len = len(labels)
    if current_len >= target_length:
        np.random.shuffle(labels)
        return labels

    n_needed = target_length - current_len
    positive_rows = labels[labels[:, -2] == 1]

    if len(positive_rows) == 0:
        np.random.shuffle(labels)
        return labels

    idx = np.random.randint(0, len(positive_rows), size=n_needed)
    aug_rows = positive_rows[idx].copy()
    aug_rows[:, -1] = np.random.randint(0, 7, size=n_needed)
    new_labels = np.concatenate([labels, aug_rows], axis=0)
    np.random.shuffle(new_labels)
    return new_labels




## === cell 2
Labels = pd.read_csv(Labels_Dir.format("train"))
Labels = Labels.sample(frac=1, random_state=42).reset_index(drop=True)

Labels["argu_mode"] = [None] * len(Labels)
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

Train_Labels = data_augmentation(Train_Labels, Train_Over_Sampel_Count)

np.random.shuffle(Train_Labels)
np.random.shuffle(Validation_Labels)

Train_Positive_Count = np.count_nonzero(Train_Labels[:, -2] == 1)
Valid_Positive_Count = np.count_nonzero(Validation_Labels[:, -2] == 1)

print(
    f"Train positives: {Train_Positive_Count}, Validation positives: {Valid_Positive_Count}"
)
print(f"Len Train: {len(Train_Labels)}, Len Validation: {len(Validation_Labels)}")




## === cell 3
def data_generator(labels, imgs_dir, image_size, batch_size=32):
    images, targets = [], []
    while True:
        for index in range(len(labels)):
            Id = labels[index][0]
            target = labels[index][-2]
            argu_mode = int(labels[index][-1]) if labels[index][-1] is not None else 0

            image_path = os.path.join(imgs_dir, f"{Id}.jpg")
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue  # skip missing files
            image = cv2.resize(image, image_size)

            if argu_mode == 1:
                image = cv2.flip(image, 1)
            elif argu_mode == 2:
                image = cv2.flip(image, 0)
                matrix = np.float32([[1, 0, 20], [0, 1, 20]])
                image = cv2.warpAffine(image, matrix, image_size)
            elif argu_mode == 3:
                image = cv2.flip(image, -1)
                matrix = np.float32([[1, 0, 33], [0, 1, 33]])
                image = cv2.warpAffine(image, matrix, image_size)
            elif argu_mode == 4:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                matrix = np.float32([[1, 0, 23], [0, 1, 25]])
                image = cv2.warpAffine(image, matrix, image_size)
            elif argu_mode == 5:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif argu_mode == 6:
                image = image[25 : image_size[1] - 25, 25 : image_size[0] - 25]
                image = cv2.resize(image, (image_size[0], image_size[1]))
                matrix = np.float32([[1, 0, 32], [0, 1, 32]])
                image = cv2.warpAffine(image, matrix, image_size)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            images.append(image.astype("float32"))
            targets.append(target)

            if len(images) >= batch_size:
                yield np.array(images), np.array(targets)
                images, targets = [], []




## === cell 4
Train_Gen = data_generator(Train_Labels, Images_Dir, Image_Size, Batch_Size)
Validation_Gen = data_generator(Validation_Labels, Images_Dir, Image_Size, Batch_Size)



## === cell 5
base_efficientnet = EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3)
)




## === cell 6
def build_lrfn(
    lr_start=1e-5,
    lr_max=1e-4,
    lr_min=1e-6,
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
lr_schedule = LearningRateScheduler(lrfn, verbose=1)




## === cell 7
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
    Train_Gen,
    steps_per_epoch=len(Train_Labels) // Batch_Size,
    epochs=Epochs,
    validation_data=Validation_Gen,
    validation_steps=len(Validation_Labels) // Batch_Size,
    class_weight=Class_Weight,
    callbacks=[lr_schedule, Early_Stop, TB_Callback],
    verbose=1,
)



## === cell 8
Model.save(
    Output_Dir.format(
        f"{Model.Name}--{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
)




## === cell 9
def predict(images, model, image_size=(260, 260), is_rgb=True):
    preds = []
    for img in images:
        img = cv2.resize(img, image_size)
        if not is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.reshape(img, (1, image_size[0], image_size[1], 3))
        preds.append(model.predict(img, verbose=0))
    return preds




## === cell 10
Test_Labels = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

My_Submission = {"image_name": [], "target": []}
for idx in tqdm(range(len(Test_Labels))):
    image_name = Test_Labels.loc[idx, "image_name"]
    img_path = os.path.join(Test_Images_Dir, f"{image_name}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        pred = 0.0
    else:
        pred = predict([img], Model, is_rgb=False)[0][0][0]
    My_Submission["image_name"].append(image_name)
    My_Submission["target"].append(float(pred))

submission_df = pd.DataFrame(My_Submission)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
print("First rows of submission:")
print(submission_df.head())
