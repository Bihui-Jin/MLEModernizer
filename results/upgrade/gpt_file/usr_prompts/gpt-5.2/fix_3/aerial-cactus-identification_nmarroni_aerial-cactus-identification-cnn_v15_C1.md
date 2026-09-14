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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9368

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the TensorFlow import crash by pinning protobuf to a compatible version at runtime before importing TensorFlow (this is the root cause of the `MessageFactory` error in this environment). Then I fix the Keras 3+ `ModelCheckpoint` requirement by giving the checkpoint path a proper `.keras` suffix, and ensure the extracted zip folder paths match what the generator and inference code expect. Finally, I make submission generation robust by using `sample_submission.csv` to guarantee correct `id` order and only predict for those ids, writing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the root cause of the training crash (“PyDataset has length 0”) by ensuring the extracted image directories match what `flow_from_dataframe` expects, and by making `steps_per_epoch` never evaluate to 0 (which is what creates a zero-length dataset in Keras 3). I keep your model, augmentations, optimizer, and training loop intact, only adjusting directory resolution and the `steps_per_epoch` calculation. After training succeeds, the inference cell use the same extracted test directory and write a valid `submission.csv` in the required `id,has_cactus` format. These changes should also move the score up from 0.5 (which indicates effectively random/constant predictions due to no training) toward the target.'

# 9. Code solution

## === cell 0
try:
    get_ipython().run_line_magic("config", "Completer.use_jedi = False")
except Exception:
    pass

import os
import sys
import subprocess
import zipfile
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))




## === cell 1
def _ensure_protobuf_compatible():
    """
    Bug fix: TF import issues sometimes happen with protobuf>=5 in some Kaggle images.
    In this provided environment TF 2.18 is installed, but we keep this guard minimal.
    """
    try:
        import google.protobuf  # noqa: F401

        ver = getattr(google.protobuf, "__version__", "")
        major = int(ver.split(".")[0]) if ver else 0
    except Exception:
        ver, major = "", 0

    if major >= 5:
        print(
            f"Downgrading protobuf from {ver} to 4.25.3 for TensorFlow compatibility..."
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib
        import google.protobuf as gp

        importlib.reload(gp)


_ensure_protobuf_compatible()

import tensorflow as tf
from tensorflow import keras
import datetime

print("TensorFlow:", tf.__version__)



## === cell 2
TMP_DIR = "/kaggle/tmp"
TRAIN_ZIP = "/kaggle/input/aerial-cactus-identification/train.zip"
TEST_ZIP = "/kaggle/input/aerial-cactus-identification/test.zip"

os.makedirs(TMP_DIR, exist_ok=True)

with zipfile.ZipFile(TRAIN_ZIP, "r") as z:
    z.extractall(TMP_DIR)

with zipfile.ZipFile(TEST_ZIP, "r") as z:
    z.extractall(TMP_DIR)

print(
    "Extracted dirs:",
    [d for d in os.listdir(TMP_DIR) if os.path.isdir(os.path.join(TMP_DIR, d))][:20],
)


def _resolve_extracted_dir(root, leaf):
    direct = os.path.join(root, leaf)
    if os.path.isdir(direct) and len(os.listdir(direct)) > 0:
        return direct
    nested = os.path.join(root, "aerial-cactus-identification", leaf)
    if os.path.isdir(nested) and len(os.listdir(nested)) > 0:
        return nested
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == leaf and len(filenames) > 0:
            return dirpath
    return direct  # last resort


train_dir = _resolve_extracted_dir(TMP_DIR, "train")
test_dir = _resolve_extracted_dir(TMP_DIR, "test")

print(
    "Resolved train_dir:",
    train_dir,
    "exists:",
    os.path.isdir(train_dir),
    "n_files:",
    len(os.listdir(train_dir)) if os.path.isdir(train_dir) else 0,
)
print(
    "Resolved test_dir :",
    test_dir,
    "exists:",
    os.path.isdir(test_dir),
    "n_files:",
    len(os.listdir(test_dir)) if os.path.isdir(test_dir) else 0,
)




## === cell 3
class Cnn_Model:
    def __init__(self, train_dir, test_dir):
        self.BATCH_SZ = 32
        self.LR_RATE = 0.0001
        self.EPOCHS = 5

        self.MODEL_CKP_PATH = "/kaggle/tmp/model_ckpoint.keras"

        self.MODEL_TRAIN_DATA = train_dir
        self.MODEL_TEST_DATA = test_dir

        self.model = None
        self.df = None
        self.train_gen = None
        self.train_direcIter = None

    def get_model(self):
        return self.model

    def load_dataframe(self):
        self.df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
        self.df.has_cactus = self.df.has_cactus.astype(str)

    def make_train_gen(self):
        self.train_gen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            zoom_range=0.2,
            width_shift_range=0.4,
            height_shift_range=0.4,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=60,
            brightness_range=[0.8, 1.1],
        )

        self.train_direcIter = self.train_gen.flow_from_dataframe(
            dataframe=self.df,
            directory=self.MODEL_TRAIN_DATA,
            x_col="id",
            y_col="has_cactus",
            target_size=(32, 32),
            batch_size=self.BATCH_SZ,
            shuffle=True,
            class_mode="binary",
        )

    def build_model(self):
        self.model = keras.models.Sequential(
            [
                keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.2),
                keras.layers.Conv2D(64, (3, 3)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.1),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.summary()

    def compile_model(self):
        optz = keras.optimizers.Adam(learning_rate=self.LR_RATE)
        self.model.compile(
            optimizer=optz,
            loss=keras.losses.binary_crossentropy,
            metrics=["acc"],
        )

    def train_model(self):
        def scheduler(epoch, lr):
            return lr * tf.math.exp(-0.01)

        callback_list = [
            keras.callbacks.LearningRateScheduler(scheduler),
        ]

        steps = max(1, int(np.ceil(self.train_direcIter.samples / self.BATCH_SZ)))

        self.model.fit(
            x=self.train_direcIter,
            epochs=self.EPOCHS,
            callbacks=callback_list,
            steps_per_epoch=steps,
        )

    def save_model(self):
        self.model.save(self.MODEL_CKP_PATH)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.MODEL_CKP_PATH)
        print("> Model load done...")




## === cell 4
cnn_obj = Cnn_Model(train_dir=train_dir, test_dir=test_dir)
cnn_obj.load_dataframe()
cnn_obj.make_train_gen()

cnn_obj.build_model()
cnn_obj.compile_model()
cnn_obj.train_model()

cnn_obj.save_model()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/141309305.py in <cell line: 0>()
      5 cnn_obj.build_model()
      6 cnn_obj.compile_model()
----> 7 cnn_obj.train_model()
      8 
      9 cnn_obj.save_model()

/tmp/ipykernel_11/609157174.py in train_model(self)
     87         steps = max(1, int(np.ceil(self.train_direcIter.samples / self.BATCH_SZ)))
     88 
---> 89         self.model.fit(
     90             x=self.train_direcIter,
     91             epochs=self.EPOCHS,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 5
model = cnn_obj.get_model()

sample_path = "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
sub = pd.read_csv(sample_path)

results = []
missing = 0

for file_name in sub["id"].tolist():
    fullpath_img = os.path.join(test_dir, file_name)
    if not os.path.exists(fullpath_img):
        missing += 1
        results.append(0.5)  # neutral fallback; should not happen if paths resolved
        continue

    image = tf.keras.preprocessing.image.load_img(fullpath_img, target_size=(32, 32))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    pred = model.predict(input_arr, verbose=0)[0][0]
    results.append(float(pred))

if missing:
    print("Warning: missing test images:", missing)

sub["has_cactus"] = results
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
