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

3.9

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0

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

0.48912

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from PIL import Image

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA JIT can reduce step overhead on CPU/GPU.
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_file = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
path = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
sub_path = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
test_file = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"




## === cell 2
df = pd.read_csv(train_file)
df.head()




## === cell 3
df_test = pd.read_csv(test_file)
df_test.head()




## === cell 4
colnames = df.columns.to_list()
colnames.remove("image_id")
colnames




## === cell 5
def get_label(row):
    if row["healthy"]:
        return "healthy"
    elif row["multiple_diseases"]:
        return "multiple_diseases"
    elif row["rust"]:
        return "rust"
    elif row["scab"]:
        return "scab"
    return "healthy"




## === cell 6
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
df["label"] = df[label_cols].astype(np.int8).idxmax(axis=1)
df[["image_id", "label"]].head()




## === cell 7
df["file_name"] = df["image_id"].astype(str) + ".jpg"
df_test["file_name"] = df_test["image_id"].astype(str) + ".jpg"




## === cell 8
df_train, df_validate = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

print(f"Training Size : {len(df_train)}")
print(f"Validation Size : {len(df_validate)}")




## === cell 9
im = Image.open(os.path.join(path, "Train_1.jpg"))
width, height = im.size
print("Original size:", width, height)




## === cell 10
BATCH = 32

new_width = int(width / 1.5)
new_height = int(height / 1.5)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=path,
    x_col="file_name",
    y_col="label",
    target_size=(new_height, new_width),
    batch_size=BATCH,
    class_mode="categorical",
    classes=["healthy", "multiple_diseases", "rust", "scab"],
    shuffle=True,
    seed=SEED,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

val_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validate,
    directory=path,
    x_col="file_name",
    y_col="label",
    target_size=(new_height, new_width),
    batch_size=BATCH,
    class_mode="categorical",
    classes=["healthy", "multiple_diseases", "rust", "scab"],
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, new_height, new_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    ),
).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, new_height, new_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    ),
).prefetch(AUTOTUNE)




## === cell 11
base_model = ResNet50(weights="imagenet", include_top=False)
print("Base model layers:", len(base_model.layers))




## === cell 12
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
preds = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)




## === cell 13
base_model.trainable = False




## === cell 14
n_epochs = 20
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["categorical_accuracy"],
)




## === cell 15
history = model.fit(
    train_ds,
    epochs=n_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_ds,
    validation_steps=len(val_generator),
    verbose=1,
    workers=min(8, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=32,
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/792283956.py in <cell line: 0>()
      1 # --- Speed: use multiprocessing workers for generator-backed training to reduce Python bottleneck.
      2 # Core training loop/epochs/steps are unchanged; only input pipeline parallelism is improved.
----> 3 history = model.fit(
      4     train_ds,
      5     epochs=n_epochs,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 16
submit_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

submit_generator = submit_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=path,
    x_col="file_name",
    y_col=None,
    target_size=(new_height, new_width),
    batch_size=BATCH,
    class_mode=None,
    shuffle=False,
)

submit_ds = tf.data.Dataset.from_generator(
    lambda: submit_generator,
    output_signature=tf.TensorSpec(
        shape=(None, new_height, new_width, 3), dtype=tf.float32
    ),
).prefetch(AUTOTUNE)

y_pred = model.predict(
    submit_ds,
    steps=len(submit_generator),
    verbose=1,
    workers=min(8, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=32,
)

y_pred = y_pred[: len(df_test)]
print("Pred shape:", y_pred.shape)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1004262233.py in <cell line: 0>()
     20 ).prefetch(AUTOTUNE)
     21 
---> 22 y_pred = model.predict(
     23     submit_ds,
     24     steps=len(submit_generator),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 17
submit = pd.concat(
    [
        df_test[["image_id"]].reset_index(drop=True),
        pd.DataFrame(y_pred, columns=["healthy", "multiple_diseases", "rust", "scab"]),
    ],
    axis=1,
)

submit = submit[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
submit.head()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805880673.py in <cell line: 0>()
      2     [
      3         df_test[["image_id"]].reset_index(drop=True),
----> 4         pd.DataFrame(y_pred, columns=["healthy", "multiple_diseases", "rust", "scab"]),
      5     ],
      6     axis=1,

NameError: name 'y_pred' is not defined

## === cell 18
out_path = "/kaggle/working/submit.csv"
submit.to_csv(out_path, index=False)
print("Wrote submission to", out_path, "with shape:", submit.shape)
print("Columns:", submit.columns.tolist())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2301698882.py in <cell line: 0>()
      1 out_path = "/kaggle/working/submit.csv"
----> 2 submit.to_csv(out_path, index=False)
      3 print("Wrote submission to", out_path, "with shape:", submit.shape)
      4 print("Columns:", submit.columns.tolist())

NameError: name 'submit' is not defined
