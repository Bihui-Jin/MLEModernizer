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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.82373

# 6. Current score

0.17487

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12567) has done: 'I remove the IPython magic that caused the import error, protect the optional visualisation cell that fails on missing files, correct the validation image size, fix the loss‑function mismatch (softmax + `from_logits=True`), and keep the rest of the pipeline unchanged. These changes eliminate runtime failures and let the model train correctly, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.12644) has done: 'I fix the protobuf import error by setting the required environment variable before importing TensorFlow, add a small data‑augmentation block to the model to boost validation accuracy, and give the early‑stopping callback a larger patience so the model can train longer. These changes resolve the runtime crash and modestly improve the score while preserving the original architecture and workflow.'
- What this solution (achieved 0.1299) has done: 'I replace the simple CNN with a transfer‑learning model (MobileNetV2) that leverages ImageNet weights, increase training epochs and adjust early‑stopping patience so the network can learn effectively. The rest of the pipeline (data loading, preprocessing, and submission creation) stays unchanged, ensuring a valid CSV is produced while moving the validation accuracy toward the target score.'
- What this solution (achieved 0.13336) has done: 'Implemented minimal yet essential fixes to unblock the pipeline and improve model performance:

- Set the protobuf environment variable **before** any imports and switched to TensorFlow‑native `EarlyStopping` to avoid the import error.
- Rewrote the test‑set loading logic using `tf.data.Dataset.from_tensor_slices` to correctly handle images that reside directly in the test folder (no sub‑directories).
- Ensured predictions stay aligned with the original `sample_submission.csv` order when creating the final CSV.

These changes resolve the runtime crash, correctly generate a valid `submission.csv`, and retain the MobileNetV2‑based model that should raise validation accuracy toward the target score.'
- What this solution (achieved 0.12029) has done: 'The changes add mixed‑precision training, speed up data preprocessing with parallel `map` calls, and increase the batch size to cut the number of steps per epoch. These tweaks keep the exact model architecture, loss, and training schedule, so the learned results remain the same while reducing the total runtime enough to finish under the 600 s limit.'
- What this solution (achieved 0.13605) has done: 'Implemented a small environment‑variable fix to prevent the protobuf version clash that caused TensorFlow import to fail. The added `TF_DISABLE_PROTOBUF_VERSION_CHECK` flag is set before any TensorFlow import, allowing the rest of the original pipeline to run unchanged and produce a proper `submission.csv`. No other logic was altered, preserving the model architecture and training behavior.'
- What this solution (achieved 0.13413) has done: 'Implemented fixes to unblock the pipeline and boost validation accuracy while preserving the original model architecture:

- Removed mixed‑precision policy (kept float32) to avoid numerical instability that was hurting training.
- Adjusted training schedules: increased base‑model epochs to 30 with early‑stopping patience 5, then fine‑tuned for 15 epochs with patience 10 using a lower learning rate.  
- Created a fresh EarlyStopping callback for the fine‑tuning phase so it works correctly.  

These changes resolve the import error, ensure stable training, and are expected to raise the validation accuracy toward the target score.'
- What this solution (achieved 0.15565) has done: 'Implemented fixes to unblock the notebook and boost validation accuracy:

- Set required TensorFlow environment variables **before any imports** and limit the initial imports to TensorFlow‑related objects to avoid the protobuf `AttributeError`.
- Added a proper MobileNetV2 preprocessing step (`preprocess_input`) after rescaling, which matches the model’s expected input range and improves learning.
- Adjusted the data pipeline for the test set to use the same preprocessing.
- Increased early‑stopping patience for the frozen‑base phase to let the model train longer.
- Kept the overall architecture unchanged while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.17487) has done: 'The changes switch the dataset caches from disk‑based files to in‑memory caches, eliminating costly disk I/O on every epoch while preserving the exact training data and order. The rest of the pipeline, model architecture, epochs, and augmentation remain unchanged, so model results are identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_DISABLE_PROTOBUF_VERSION_CHECK"] = "1"

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(f"GPU memory growth not set: {e}")

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib




## === cell 2
data = pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")
data.head()




## === cell 3
data.shape




## === cell 4
data["label"].unique().tolist()




## === cell 5
data["variety"].unique().tolist()




## === cell 6
data.age.describe()




## === cell 7
fig, ax = plt.subplots(1, 1, figsize=(21, 7))
sns.histplot(x="variety", data=data, ax=ax)
plt.title("Variety distribution in the dataset")
plt.show()




## === cell 8
fig, ax = plt.subplots(1, 1, figsize=(21, 7))
sns.histplot(x="label", data=data, ax=ax)
plt.title("Disease distribution in the dataset")
plt.show()




## === cell 9
normal = data[data["label"] == "normal"]
normal = normal[normal["variety"] == "ADT45"]
five_normals = normal.image_id[:5].values
five_normals.tolist()




## === cell 10
dead = data[data["label"] == "dead_heart"]
dead = dead[dead["variety"] == "ADT45"]
five_deads = dead.image_id[:5].values
five_deads.tolist()




## === cell 11
try:
    plt.figure(figsize=(20, 10))
    columns = 5
    path = "/kaggle/input/paddy-disease-classification/train_images/"
    for i, image_loc in enumerate(np.concatenate((five_normals, five_deads))):
        plt.subplot(10 // columns + 1, columns, i + 1)
        if i < 5:
            image = plt.imread(path + "normal/" + image_loc)
            plt.title("normal")
        else:
            image = plt.imread(path + "dead_heart/" + image_loc)
            plt.title("dead_heart")
        plt.imshow(image)
except Exception as e:
    print(f"Visualization skipped due to: {e}")




## === cell 12
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["label"])
data["variety"] = encoder.fit_transform(data["variety"])
data.head()




## === cell 13
batch_size = 64
img_height = 224
img_width = 224




## === cell 14
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/kaggle/input/paddy-disease-classification/train_images/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)




## === cell 15
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/kaggle/input/paddy-disease-classification/train_images/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)




## === cell 16
class_names = train_ds.class_names
print(class_names)




## === cell 17
for image_batch, label_batch in train_ds.take(1):
    print(image_batch.shape)
    print(label_batch.shape)




## === cell 18
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

preprocess_layer = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    ]
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze the pretrained weights

model = tf.keras.Sequential(
    [
        data_augmentation,
        preprocess_layer,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            len(class_names), activation="softmax", dtype="float32"
        ),  # Cast back to float32 for stability
    ]
)




## === cell 19
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)




## === cell 20
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

early_stop_frozen = EarlyStopping(patience=15, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,  # unchanged
    callbacks=[early_stop_frozen],
    verbose=2,
)




## === cell 21
base_model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

early_stop_finetune = EarlyStopping(patience=20, restore_best_weights=True)

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  # unchanged
    callbacks=[early_stop_finetune],
    verbose=2,
)




## === cell 22
loss, accu = model.evaluate(val_ds, verbose=0)
print(f"The validation loss is {loss:.4f}")
print(f"The validation accuracy is {accu*100:.2f}%")




## === cell 23
test_data_dir = "/kaggle/input/paddy-disease-classification/test_images/"




## === cell 24
test_path = pathlib.Path(test_data_dir)
test_files = sorted([str(p) for p in test_path.glob("*.jpg")])

test_ds = tf.data.Dataset.from_tensor_slices(test_files)


def load_and_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.keras.layers.Rescaling(1.0 / 255)(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image


test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)




## === cell 25
y_pred = model.predict(test_ds, verbose=1)
y_pred_classes = y_pred.argmax(axis=1)




## === cell 26
y_class_names = [class_names[idx] for idx in y_pred_classes]




## === cell 27
submission = pd.read_csv(
    "/kaggle/input/paddy-disease-classification/sample_submission.csv"
)
submission["label"] = y_class_names
submission.to_csv("submission.csv", index=False)
