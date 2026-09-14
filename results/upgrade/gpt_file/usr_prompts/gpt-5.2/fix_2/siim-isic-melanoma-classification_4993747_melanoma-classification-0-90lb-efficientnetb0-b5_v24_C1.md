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

print("TF version:", tf.__version__)
print("Train images dir exists:", os.path.isdir(Train_Images_Dir), Train_Images_Dir)
print("Test images dir exists:", os.path.isdir(Test_Images_Dir), Test_Images_Dir)




## === cell 1
def data_augmentation(labels, target_length):
    counter = 0
    while counter < target_length:
        for i in tqdm(range(len(labels)), desc="Augmenting positives"):
            if counter >= target_length:
                continue

            row = labels[i]
            target = row[-2]

            if target == 0:
                continue

            mode = random.randint(0, 6)
            row_copy = copy.deepcopy(row)
            row_copy[-1] = mode
            labels = np.concatenate([labels, [row_copy]])

            counter += 1

    shuffle(labels)
    return labels




## === cell 2
Labels = pd.read_csv(Labels_Dir.format("train"))
Labels = Labels.sample(frac=1, random_state=SEED).reset_index(drop=True)

Labels["argu_mode"] = [0 for _ in range(len(Labels))]

Labels["sex"] = Labels["sex"].replace({"male": 1, "female": 0})
Labels["sex"] = Labels["sex"].fillna(-1)

Positive_Indexs = Labels.index[Labels["target"] == 1].tolist()
Negative_Indexs = Labels.index[Labels["target"] == 0].tolist()

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

Train_Positive_Count = np.count_nonzero([row[-2] for row in Train_Labels])
Valid_Positive_Count = np.count_nonzero([row[-2] for row in Validation_Labels])

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
def data_generator(labels, imgs_dir, image_size, batch_size=32):
    images, targets = [], []

    while True:
        for index in range(len(labels)):
            image_name = labels[index][0]
            target = labels[index][-2]
            argu_mode = labels[index][-1]

            image_path = os.path.join(imgs_dir, f"{image_name}.jpg")
            image = cv2.imread(image_path, 1)

            if image is None:
                continue

            image = cv2.resize(image, image_size)

            if argu_mode == 0:
                pass
            elif argu_mode == 1:
                image = cv2.flip(image, 1)  # horizontal
            elif argu_mode == 2:
                image = cv2.flip(image, 0)  # vertical
                matrix = np.float32([[1, 0, 20], [0, 1, 20]])
                image = cv2.warpAffine(image, matrix, image_size)
            elif argu_mode == 3:
                image = cv2.flip(image, -1)  # both
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

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.float32(image)

            images.append(image)
            targets.append(target)

            if len(images) >= batch_size:
                yield np.array(images, dtype="float32"), np.array(
                    targets, dtype="float32"
                )
                images, targets = [], []




## === cell 4
Train_Gen = data_generator(Train_Labels, Train_Images_Dir, Image_Size, Batch_Size)

Validation_Gen = data_generator(
    Validation_Labels, Train_Images_Dir, Image_Size, Batch_Size
)

steps_per_epoch = int(np.ceil(len(Train_Labels) / Batch_Size))
validation_steps = int(np.ceil(len(Validation_Labels) / Batch_Size))

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 5
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



## === cell 6
EfficientNetB2 = tf.keras.applications.EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3)
)


def make_model(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="sigmoid"))
    return model




## === cell 7
Model = make_model(EfficientNetB2)
Model.Name = "efficentB2"
TB_Callback = TensorBoard(log_dir=TB_Log_Dir.format(Model.Name), histogram_freq=1)

SGD_Optimizer = SGD(learning_rate=0.1)
Adam_Optimizer = Adam(learning_rate=0.1)

Model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

Model.summary()

history = Model.fit(
    Train_Gen,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    epochs=Epochs,
    validation_data=Validation_Gen,
    validation_steps=validation_steps,
    class_weight=Class_Weight,
    callbacks=[lr_schedule, Early_Stop, TB_Callback],
)



## === cell 8
model_path = Output_Dir.format(
    f"{Model.Name}--{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
Model.save(model_path)
print("Saved model to:", model_path)




## === cell 9
def predict(images, model, image_size=(299, 299), is_rgb=True):
    predictions = []
    for image in images:
        if image is None:
            predictions.append(np.array([[0.5]], dtype=np.float32))
            continue

        image = cv2.resize(image, image_size)

        if not is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.float32(image) / 255.0
        image = np.reshape(image, (1, image_size[0], image_size[1], 3))
        predictions.append(model.predict(image, verbose=0))

    return predictions




## === cell 10
Sample_Submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
Test_Labels = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

My_Submission = {"image_name": [], "target": []}

for index in tqdm(range(len(Test_Labels)), desc="Predicting test"):
    image_name = Test_Labels.loc[index, "image_name"]
    image = cv2.imread(os.path.join(Test_Images_Dir, f"{image_name}.jpg"), 1)

    My_Submission["image_name"].append(image_name)
    My_Submission["target"].append(
        float(predict([image], Model, image_size=Image_Size, is_rgb=False)[0][0][0])
    )

My_Submission = pd.DataFrame(My_Submission)

My_Submission = Sample_Submission[["image_name"]].merge(
    My_Submission, on="image_name", how="left"
)
My_Submission["target"] = My_Submission["target"].fillna(0.5).astype(float)

My_Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", My_Submission.shape)
print(My_Submission.head())
print(My_Submission.tail())
