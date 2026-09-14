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

3.11

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

0.4994

# 6. Current score

0.44244

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.44244) has done: 'I fix the environment/runtime break at import time by pinning protobuf’s pure-Python implementation before TensorFlow loads, which avoids the `MessageFactory.GetPrototype` crash. Then I make Keras 3 checkpointing/loading compatible by saving to a `.keras` file and loading that exact file. Finally, I make unzip destinations deterministic (to `/kaggle/working/train` and `/kaggle/working/test`) and ensure the test generator preserves filename order so predictions align with `id`, producing a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import shutil
import zipfile
from pathlib import Path

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
data.sample(5)



## === cell 2
data = data.astype({"id": str, "has_cactus": str})



## === cell 3
data.sample(5)




## === cell 4
def unzip_to(zip_path: str, dst_dir: str):
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


unzip_to(
    "/kaggle/input/aerial-cactus-identification/train.zip", "/kaggle/working/train"
)
unzip_to("/kaggle/input/aerial-cactus-identification/test.zip", "/kaggle/working/test")

print("Train dir exists:", Path("/kaggle/working/train").exists())
print("Test dir exists:", Path("/kaggle/working/test").exists())



## === cell 5
idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.0, validation_split=0.1
)



## === cell 6
train_idg = idg.flow_from_dataframe(
    data,
    "/kaggle/working/train",
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    batch_size=64,
    subset="training",
    seed=42,
    class_mode="categorical",
    shuffle=True,
)



## === cell 7
val_idg = idg.flow_from_dataframe(
    data,
    "/kaggle/working/train",
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    batch_size=64,
    subset="validation",
    seed=42,
    class_mode="categorical",
    shuffle=False,
)



## === cell 8
pass



## === cell 9
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input((32, 32, 3), name="InputLayer"))
model.add(tf.keras.layers.Flatten(name="Flat"))
model.add(tf.keras.layers.Dropout(0.25, name="Drop1"))
model.add(tf.keras.layers.Dense(512, "relu", name="D1"))
model.add(tf.keras.layers.Dense(128, "relu", name="D2"))
model.add(tf.keras.layers.Dense(2, "softmax", name="Output"))

model.summary()



## === cell 10
model.compile(
    tf.keras.optimizers.SGD(),
    tf.keras.losses.categorical_crossentropy,
    [tf.keras.metrics.AUC(200, "ROC", name="AUC"), "acc"],
)



## === cell 11
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(data["has_cactus"]),
    y=data["has_cactus"],
)
class_weights = dict(enumerate(class_weights))



## === cell 12
class_weights



## === cell 13
ckpt_path = "/kaggle/working/BestModelAsPerValAUC.keras"
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor="val_AUC",
    save_best_only=True,
    mode="max",
    verbose=1,
)



## === cell 14
history = model.fit(
    train_idg,
    epochs=15,
    validation_data=val_idg,
    class_weight=class_weights,
    callbacks=[model_ckpt],
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/953128869.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_idg,
      3     epochs=15,
      4     validation_data=val_idg,
      5     class_weight=class_weights,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in build(self, y_true, y_pred)
    606                 flat_loss_weights = tree.flatten(loss_weights)
    607                 if len(tree.flatten(loss)) != len(flat_loss_weights):
--> 608                     raise ValueError(
    609                         f"`loss_weights` must match the number of losses, "
    610                         f"got {len(tree.flatten(loss))} losses "

ValueError: `loss_weights` must match the number of losses, got 1 losses and 2 weights.

## === cell 15
model = tf.keras.models.load_model(ckpt_path)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/482181871.py in <cell line: 0>()
      1 # Fix: load the `.keras` checkpoint saved by ModelCheckpoint (Keras 3 compatible).
----> 2 model = tf.keras.models.load_model(ckpt_path)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/working/BestModelAsPerValAUC.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 16
test_dir = "/kaggle/working/test"
test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_result = pd.DataFrame(test_files, columns=["id"])
test_result.head()



## === cell 17
test_idg = idg.flow_from_dataframe(
    test_result,
    test_dir,
    batch_size=64,
    x_col="id",
    target_size=(32, 32),
    class_mode=None,
    shuffle=False,
)



## === cell 18
test_pred = model.predict(test_idg, verbose=1)



## === cell 19
test_pred.shape



## === cell 20
type(test_pred)



## === cell 21
test_pred[:5]



## === cell 22
train_idg.class_indices



## === cell 23
test_pred[:, 1].shape



## === cell 24
test_result["has_cactus"] = test_pred[:, 1].astype(np.float32)



## === cell 25
test_result.sample(5)



## === cell 26
pass



## === cell 27
sub_path = "/kaggle/working/submission.csv"
test_result.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(test_result))



## === cell 28
df = pd.read_csv(sub_path)
df.head()
