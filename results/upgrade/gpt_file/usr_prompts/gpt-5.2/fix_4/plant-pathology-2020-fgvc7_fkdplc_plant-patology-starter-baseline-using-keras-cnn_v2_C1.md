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

0.47796

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58999) has done: 'You’re hitting an environment-level incompatibility between `keras==3.x` and an older `protobuf` stack in Kaggle, which triggers the `MessageFactory.GetPrototype` error before the model is created; fixing this makes `model` defined and unblocks the rest of the pipeline. I force Keras to use the TensorFlow backend and use `tf_keras` (which is installed) to keep your exact architecture/training loop semantics while avoiding the protobuf/Keras3 crash. I also change the checkpoint filename to the Keras v3-compatible `.keras` format and load that same file, ensuring the submission is always written as `submission.csv` with the correct columns and row alignment. No score-tuning changes are introduced beyond making the training/inference actually run end-to-end.'
- What this solution (achieved 0.47796) has done: 'I fix the protobuf/Keras crash by removing the Keras 3 dependency entirely and using TensorFlow’s built-in `tf.keras` consistently (same layers/architecture/training loop), which avoids the `MessageFactory.GetPrototype` error in this environment. I also make the train/valid split stratified by the multi-label “argmax class” to stabilize ROC AUC without changing the core modeling approach, and ensure labels are fed as float32. Finally, I keep the checkpoint/save/load path consistent and always write `submission.csv` with the exact column order from `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"

train = pd.read_csv(PATH + "train.csv")
test = pd.read_csv(PATH + "test.csv")

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
target = train[target_cols]
test_ids = test["image_id"]

train_len = train.shape[0]
test_len = test.shape[0]

train.describe()



## === cell 1
from PIL import Image
from tqdm.notebook import tqdm

SIZE = 224

train_images = np.empty((train_len, SIZE, SIZE, 3), dtype=np.float32)
for i, img_id in enumerate(tqdm(train["image_id"].values, total=train_len)):
    img = Image.open(PATH + f"images/{img_id}.jpg").convert("RGB").resize((SIZE, SIZE))
    train_images[i] = np.asarray(img, dtype=np.float32) / 255.0

test_images = np.empty((test_len, SIZE, SIZE, 3), dtype=np.float32)
for i, img_id in enumerate(tqdm(test["image_id"].values, total=test_len)):
    img = Image.open(PATH + f"images/{img_id}.jpg").convert("RGB").resize((SIZE, SIZE))
    test_images[i] = np.asarray(img, dtype=np.float32) / 255.0

train_images.shape, test_images.shape



## === cell 2
from sklearn.model_selection import train_test_split

y_all = target.to_numpy(dtype=np.float32)
stratify_labels = np.argmax(y_all, axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    train_images,
    y_all,
    test_size=0.2,
    random_state=289,
    stratify=stratify_labels,
)

x_train.shape, x_test.shape, y_train.shape, y_test.shape



## === cell 3
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

tf.random.set_seed(289)
np.random.seed(289)

rlr = ReduceLROnPlateau(patience=3, verbose=1)
es = EarlyStopping(patience=7, restore_best_weights=True, verbose=1)

mc_path = "model.keras"
mc = ModelCheckpoint(mc_path, save_best_only=True, verbose=1)

filters = 16

base = Sequential()
base.add(Conv2D(filters, 3, activation="relu", input_shape=(SIZE, SIZE, 3)))
base.add(Conv2D(filters, 3, activation="relu"))
base.add(Conv2D(filters, 5, activation="relu"))
base.add(MaxPooling2D())
base.add(Dropout(0.5))
base.add(BatchNormalization())

filters *= 2
base.add(Conv2D(filters, 3, activation="relu"))
base.add(Conv2D(filters, 3, activation="relu"))
base.add(Conv2D(filters, 5, activation="relu"))
base.add(MaxPooling2D())
base.add(Dropout(0.5))
base.add(BatchNormalization())

filters *= 2
base.add(Conv2D(filters, 3, activation="relu"))
base.add(Conv2D(filters, 3, activation="relu"))
base.add(Conv2D(filters, 5, activation="relu"))
base.add(MaxPooling2D())
base.add(Dropout(0.5))
base.add(BatchNormalization())

filters *= 2
base.add(Conv2D(filters, 3, activation="relu"))
base.add(Conv2D(filters, 3, activation="relu"))
base.add(Conv2D(filters, 5, activation="relu"))
base.add(MaxPooling2D())
base.add(Dropout(0.5))
base.add(BatchNormalization())

base.add(Flatten())
base.add(Dense(16, activation="sigmoid"))

inp = Input(shape=(SIZE, SIZE, 3))
im_model = base(inp)

out_1 = Dense(1, activation="sigmoid", name="out_1")(im_model)
out_2 = Dense(1, activation="sigmoid", name="out_2")(im_model)
out_3 = Dense(1, activation="sigmoid", name="out_3")(im_model)
out_4 = Dense(1, activation="sigmoid", name="out_4")(im_model)

model = Model(inputs=inp, outputs=[out_1, out_2, out_3, out_4])
model.summary()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

y_train = y_train.astype(np.float32, copy=False)
y_test = y_test.astype(np.float32, copy=False)

train_dict = {"out_" + str(i + 1): y_train[:, i] for i in range(4)}
test_dict = {"out_" + str(i + 1): y_test[:, i] for i in range(4)}

history = model.fit(
    x_train,
    train_dict,
    epochs=60,
    batch_size=32,
    verbose=1,
    callbacks=[rlr, es, mc],
    validation_data=(x_test, test_dict),
)

model = load_model(mc_path)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/696639904.py in <cell line: 0>()
      8 test_dict = {"out_" + str(i + 1): y_test[:, i] for i in range(4)}
      9 
---> 10 history = model.fit(
     11     x_train,
     12     train_dict,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in _build_metrics_set(self, metrics, num_outputs, output_names, y_true, y_pred, argument_name)
    252             if isinstance(metrics, (list, tuple)):
    253                 if len(metrics) != len(y_pred):
--> 254                     raise ValueError(
    255                         "For a model with multiple outputs, "
    256                         f"when providing the `{argument_name}` argument as a "

ValueError: For a model with multiple outputs, when providing the `metrics` argument as a list, it should have as many entries as the model has outputs. Received:
metrics=['acc']
of length 1 whereas the model has 4 outputs.

## === cell 5
from sklearn.metrics import roc_auc_score

pred_test = model.predict(x_test, verbose=0)
roc_sum = 0.0
for i in range(4):
    score = roc_auc_score(y_test[:, i], np.asarray(pred_test[i]).ravel())
    roc_sum += score
    print(f"{score:.4f}")

roc_sum /= 4.0
print(f"totally:{roc_sum:.4f}")



## === cell 6
pred = model.predict(test_images, verbose=0)

res = pd.DataFrame(
    {
        "image_id": test_ids.values,
        "healthy": np.asarray(pred[0]).ravel(),
        "multiple_diseases": np.asarray(pred[1]).ravel(),
        "rust": np.asarray(pred[2]).ravel(),
        "scab": np.asarray(pred[3]).ravel(),
    }
)

sample = pd.read_csv(PATH + "sample_submission.csv")
res = res[sample.columns]

res.to_csv("submission.csv", index=False)
res.head(40)
