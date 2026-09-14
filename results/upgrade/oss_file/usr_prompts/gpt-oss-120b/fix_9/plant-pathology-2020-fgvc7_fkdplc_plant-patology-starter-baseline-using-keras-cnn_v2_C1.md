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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.82603

# 6. Current score

0.51979

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.54002) has done: 'I fixed the protobuf import error by setting the appropriate environment variable before loading TensorFlow, corrected the import of `Input` (now taken from `tensorflow.keras.layers`), and updated the import statements accordingly. These changes resolve the runtime failures, allowing the model to be built, trained, evaluated, and a proper `submission.csv` to be generated.'
- What this solution (achieved 0.51979) has done: 'Implemented fixes to resolve runtime errors and improve model performance:
- Removed direct TensorFlow import and switched to pure Keras imports to avoid protobuf issues.
- Adjusted model compilation to supply a metric for each of the four outputs.
- Initialized the CNN with more filters (32) for higher capacity.
- Switched optimizer to Adam with a lower learning rate and provided metrics per output.
- Extended training epochs and patience for better convergence.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"

train = pd.read_csv(os.path.join(PATH, "train.csv"))
test = pd.read_csv(os.path.join(PATH, "test.csv"))

target = train[["healthy", "multiple_diseases", "rust", "scab"]]
test_ids = test["image_id"]

train_len = train.shape[0]
test_len = test.shape[0]

SIZE = 224

train_images = np.empty((train_len, SIZE, SIZE, 3), dtype=np.uint8)
for i, img_name in enumerate(
    tqdm(train["image_id"].values, desc="Loading train images")
):
    img_file = img_name if img_name.lower().endswith(".jpg") else f"{img_name}.jpg"
    img_path = os.path.join(PATH, "images", img_file)
    img = Image.open(img_path).convert("RGB").resize((SIZE, SIZE))
    train_images[i] = np.array(img, dtype=np.uint8)

test_images = np.empty((test_len, SIZE, SIZE, 3), dtype=np.uint8)
for i, img_name in enumerate(tqdm(test["image_id"].values, desc="Loading test images")):
    img_file = img_name if img_name.lower().endswith(".jpg") else f"{img_name}.jpg"
    img_path = os.path.join(PATH, "images", img_file)
    img = Image.open(img_path).convert("RGB").resize((SIZE, SIZE))
    test_images[i] = np.array(img, dtype=np.uint8)

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

print(train_images.shape, test_images.shape)



## === cell 1
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    train_images,
    target.to_numpy().astype("float32"),
    test_size=0.2,
    random_state=289,
)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 2
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

rlr = ReduceLROnPlateau(patience=5, verbose=1)
es = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
mc = ModelCheckpoint("model.h5", save_best_only=True, verbose=1)

filters = 32  # increased initial filter count for higher capacity
base_model = Sequential()
base_model.add(Conv2D(filters, 3, activation="relu", input_shape=(SIZE, SIZE, 3)))
base_model.add(Conv2D(filters, 3, activation="relu"))
base_model.add(Conv2D(filters, 5, activation="relu"))
base_model.add(MaxPooling2D())
base_model.add(Dropout(0.5))
base_model.add(BatchNormalization())

filters *= 2
base_model.add(Conv2D(filters, 3, activation="relu"))
base_model.add(Conv2D(filters, 3, activation="relu"))
base_model.add(Conv2D(filters, 5, activation="relu"))
base_model.add(MaxPooling2D())
base_model.add(Dropout(0.5))
base_model.add(BatchNormalization())

filters *= 2
base_model.add(Conv2D(filters, 3, activation="relu"))
base_model.add(Conv2D(filters, 3, activation="relu"))
base_model.add(Conv2D(filters, 5, activation="relu"))
base_model.add(MaxPooling2D())
base_model.add(Dropout(0.5))
base_model.add(BatchNormalization())

filters *= 2
base_model.add(Conv2D(filters, 3, activation="relu"))
base_model.add(Conv2D(filters, 3, activation="relu"))
base_model.add(Conv2D(filters, 5, activation="relu"))
base_model.add(MaxPooling2D())
base_model.add(Dropout(0.5))
base_model.add(BatchNormalization())

base_model.add(Flatten())
base_model.add(Dense(32, activation="sigmoid"))  # slightly larger dense layer

inp = Input(shape=(SIZE, SIZE, 3))
im_model = base_model(inp)

out_1 = Dense(1, activation="sigmoid", name="out_1")(im_model)
out_2 = Dense(1, activation="sigmoid", name="out_2")(im_model)
out_3 = Dense(1, activation="sigmoid", name="out_3")(im_model)
out_4 = Dense(1, activation="sigmoid", name="out_4")(im_model)

model = Model(inputs=inp, outputs=[out_1, out_2, out_3, out_4])
model.summary()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", "accuracy", "accuracy", "accuracy"],
)

train_dict = {"out_" + str(i + 1): y_train[:, i] for i in range(4)}
val_dict = {"out_" + str(i + 1): y_val[:, i] for i in range(4)}

history = model.fit(
    x_train,
    train_dict,
    epochs=120,
    batch_size=32,
    verbose=1,
    callbacks=[rlr, es, mc],
    validation_data=(x_val, val_dict),
)



## === cell 4
from sklearn.metrics import roc_auc_score

pred_val = model.predict(x_val)
roc_sum = 0
for i in range(4):
    score = roc_auc_score(y_val[:, i], pred_val[i].ravel())
    roc_sum += score
    print(f"Class {i+1} AUC: {score:.4f}")

roc_mean = roc_sum / 4
print(f"Average ROC AUC: {roc_mean:.4f}")



## === cell 5
pred_test = model.predict(test_images)

submission = pd.DataFrame(
    {
        "image_id": test_ids,
        "healthy": pred_test[0].ravel(),
        "multiple_diseases": pred_test[1].ravel(),
        "rust": pred_test[2].ravel(),
        "scab": pred_test[3].ravel(),
    }
)

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission.head()
