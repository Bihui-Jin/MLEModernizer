# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.16935

# 6. Current score

0.13028

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12221) has done: 'Diagnosis: Cell 0 fails immediately with a `SyntaxError` because it contains un-commented narrative text (including smart quotes), which Python tries to interpret as code. Additionally, the cell includes the IPython magic `%matplotlib inline`, which also raise a syntax error in a plain Python execution context.  
Patch summary: In cell 0, convert the narrative lines into Python comments and replace `%matplotlib inline` with a safe Matplotlib backend call so the cell runs under standard Python while preserving the same imports and variables (`pd`, etc.) used by cell 1.  
Updated cells: Only cell 0 is changed.  
Compatibility notes for cell k+1: Cell 1 requires `pd` from `import pandas as pd`; this remains unchanged and now execute because cell 0 no longer crashes.  
Assumptions: This environment executes cells as standard Python (not IPython), so IPython magics must be avoided; `plt` is available and setting the backend is acceptable.'
- What this solution (achieved 0.12913) has done: 'Diagnosis: Cell 11 crashes because at least one of the hard-coded example image paths does not exist in this environment (e.g., `.../train_images/tungro/109629.jpg`), causing `plt.imread()` to raise `FileNotFoundError`. The dataset layout is present, but individual filenames can differ, so relying on specific IDs is brittle.

Patch summary: In cell 11 only, keep the same intent (show one sample image per disease) but make it deterministic and robust by selecting the first available `.jpg` file from each disease subfolder under the known `train_images/` directory. This avoids missing-file crashes without changing any downstream variables/interfaces (cell 12 does not depend on cell 11 outputs).

Updated cells: See patched cell 11 below.

Compatibility notes for cell k+1: Cell 12 only uses `data` and `LabelEncoder`; it does not depend on `images`/`diseases` from cell 11, so this change is fully compatible.

Assumptions: Each listed disease has at least one `.jpg` file in its corresponding `/kaggle/input/paddy-disease-classification/train_images/<disease>/` directory.'
- What this solution (achieved 0.12875) has done: 'I make two minimal fixes that directly improve the score: (1) correct the loss configuration (`from_logits=False`) so it matches your model’s `softmax` output, and (2) fix a bug in the validation dataset image size (it currently uses `(img_height, img_height)` instead of `(img_height, img_width)`), which otherwise degrades validation/training consistency and generalization. I also remove the IPython-only `%%time` magic so the script runs reliably in a plain Python Kaggle execution context, without changing the training loop/approach. These changes preserve your exact architecture and training semantics while making the optimization/evaluation behave as intended, which should move accuracy upward toward your target.'
- What this solution (achieved 0.12798) has done: 'We make two small, score-relevant fixes without changing your model or training loop: (1) add lightweight data augmentation (random flip/rotation) to improve generalization on this small CNN, and (2) ensure the cached dataset is deterministic and correctly ordered by adding `shuffle()` explicitly to the training pipeline while keeping validation unchanged. These are minimal changes that typically lift accuracy from the current ~0.129 toward your ~0.169 target, without altering loss/architecture or using approximations. The submission generation remains the same, still writing a valid `submission.csv`.'
- What this solution (achieved 0.12106) has done: 'To move your accuracy up toward the 0.16935 target with minimal risk, I make two small, score-relevant pipeline changes without altering your CNN architecture or loss. First, I apply the same normalization + augmentation to the `train_ds` via `map()` (instead of relying on the model’s first layers), which makes caching/prefetching more effective and ensures augmentation is only applied to training, not validation/test. Second, I add standard dataset performance options (parallel mapping + deterministic ordering for val/test) to reduce input bottlenecks and stabilize evaluation, which can slightly improve convergence at the same epoch budget. The submission logic and class-name mapping remain unchanged, still writing a valid `submission.csv`.'
- What this solution (achieved 0.13144) has done: 'We make two minimal, score-relevant fixes that keep your exact CNN and training loop intact while improving generalization toward the 0.16935 target. First, we add an explicit `InputLayer((224,224,3))` so the model shape is fixed and consistent (right now the first Conv2D sees an implicit input, which can lead to subtle shape/tracing differences across runs). Second, we set `EarlyStopping(monitor="val_accuracy", mode="max", restore_best_weights=True)` so you submit the best-validation checkpoint (same training process/epochs, but you avoid ending with worse weights than the best point). Everything else (augmentation, preprocessing, optimizer, loss, dataset loading, submission format/path) stays the same.'
- What this solution (achieved 0.12875) has done: 'We’re currently below the 0.16935 target (0.13144), so we should make a small, legitimate change that typically improves generalization without altering your CNN architecture, loss, or training loop. The most score-relevant minimal fix here is to add `RandomZoom` to your existing augmentation, since leaf disease cues vary with scale and your model is small/overfits easily. I also make sure the test preprocessing matches train/val preprocessing exactly (float cast + divide by 255) and keep everything else unchanged so the evaluation semantics remain identical. The submission generation remains the same and still write a valid `submission.csv`.'
- What this solution (achieved 0.12606) has done: 'We’re below the 0.16935 target (current 0.12875), so we make the smallest changes that typically improve generalization without changing your CNN architecture, loss, or training loop. The main score-relevant fix is to add a tiny amount of L2 weight decay to the Conv/Dense layers (regularization only; same layers/structure) to reduce overfitting on this small model. We also set TensorFlow/Keras random seeds for more stable convergence (often nudges accuracy upward at fixed settings) without changing semantics. Submission generation stays identical and still write `submission.csv` with the required columns.'
- What this solution (achieved 0.11222) has done: 'We’re currently below the target (0.12606 vs 0.16935), so we make the smallest legitimate change that typically improves accuracy without changing your CNN architecture, loss, or training loop. The most impactful minimal fix here is to add class weighting during `model.fit()` to counter the strong class imbalance visible in the label histogram; this often yields a measurable accuracy lift on underrepresented classes with the same model. We compute `class_weight` from the training directory labels (aligned to `train_ds.class_names`) and pass it into `fit`, leaving everything else untouched. Submission writing remains identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.13144) has done: 'To move your accuracy up toward the 0.16935 target with minimal, score-relevant changes, I fix two issues that quietly hurt convergence and test-time prediction quality. First, I stop installing an incompatible `protobuf<5` (your environment uses TF 2.18 + protobuf 6.x), because forcing a downgrade can break/disable TensorFlow optimizations and cause unstable behavior. Second, I compute `class_weight` from the original `train.csv` label distribution (same label set/order as `train_ds.class_names`) instead of iterating `train_ds.unbatch()` after augmentation/shuffle/cache; the current approach is slow and can be subtly wrong/unstable, while CSV-derived weights are deterministic and aligned to the training labels. Everything else (CNN architecture, loss, augmentation, training loop, submission writing) stays the same and it still produces `submission.csv`.'
- What this solution (achieved 0.13374) has done: 'Diagnosis: The crash happens immediately when importing TensorFlow/Keras in cell 0, before any of your notebook logic runs. With protobuf==6.33.0 in this environment, TensorFlow 2.18 can hit an internal incompatibility that raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The minimal, deterministic workaround is to force TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which avoids the missing `GetPrototype` path.  

Patch summary: In cell 0 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version `2`) via `os.environ` before importing TensorFlow, then keep the rest of the imports and random seed logic unchanged.  

Updated cells: Only cell 0 is modified below.  

Compatibility notes for cell k+1: Cell 1 still receives the same `tf`, `pd`, and other imports from cell 0; no variable names or semantics change besides preventing the import-time crash.  

Assumptions: The environment allows setting `os.environ` before TensorFlow import (standard in notebooks), and you are not relying on protobuf’s C++ implementation performance for correctness (this change is purely to avoid the crash).'
- What this solution (achieved 0.13028) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 0, before any model/data code runs. With `protobuf==6.33.0`, TensorFlow 2.18 can hit a known incompatibility that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The two `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION*` environment variables set in cell 0 are not sufficient to force a compatible protobuf runtime in this environment. The minimal fix is to pin protobuf to a TensorFlow-compatible 4.x version at runtime (before importing TensorFlow) and then proceed as originally.

Patch summary: In cell 0 only, add a small pre-import check that detects protobuf>=5 and installs `protobuf==4.25.3` via pip, then restart the protobuf import state before importing TensorFlow. Keep the rest of the cell’s logic unchanged (same imports, seeds, plotting backend).

Updated cells: cell 0 only.

Compatibility notes for cell k+1: All variables and imports used by cell 1 (`pd`) remain defined exactly as before, and TensorFlow import succeed so later cells can execute.

Assumptions: The environment allows `pip install` during execution and has access to the prebuilt wheel for `protobuf==4.25.3` compatible with Python 3.12.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf as _pb  # noqa: F401
    from google.protobuf import __version__ as _pb_version

    _major = int(_pb_version.split(".", 1)[0])
    if _major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for _m in list(sys.modules):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.switch_backend("Agg")

tf.keras.utils.set_random_seed(123)


## === cell 1
data = pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")
data.head()



## === cell 2
data.shape



## === cell 3
data["label"].unique().tolist()



## === cell 4
data["variety"].unique().tolist()



## === cell 5
data.age.describe()



## === cell 6
fig, ax = plt.subplots(1, 1, figsize=(21, 7))
sns.histplot(x="variety", data=data, ax=ax)
plt.title("Variety distribution in the dataset")
plt.show()



## === cell 7
fig, ax = plt.subplots(1, 1, figsize=(21, 7))
sns.histplot(x="label", data=data, ax=ax)
plt.title("Disease distribution in the dataset")
plt.show()



## === cell 8
normal = data[data["label"] == "normal"]
normal = normal[normal["variety"] == "ADT45"]
five_normals = normal.image_id[:5].values
five_normals.tolist()



## === cell 9
dead = data[data["label"] == "dead_heart"]
dead = dead[dead["variety"] == "ADT45"]
five_deads = dead.image_id[:5].values
five_deads.tolist()



## === cell 10
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



## === cell 11
import os
import glob

base_dir = "/kaggle/input/paddy-disease-classification/train_images"

class_names = [
    "hispa",
    "tungro",
    "bacterial_leaf_blight",
    "downy_mildew",
    "blast",
    "bacterial_leaf_streak",
    "normal",
    "brown_spot",
    "dead_heart",
    "bacterial_panicle_blight",
]

images = []
for cls in class_names:
    cls_dir = os.path.join(base_dir, cls)
    candidates = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
    if not candidates:
        raise FileNotFoundError(f"No .jpg images found in: {cls_dir}")
    images.append(candidates[0])

diseases = [cls + " image" for cls in class_names]
plt.figure(figsize=(20, 10))
columns = 5
for i, image_loc in enumerate(images):
    plt.subplot(len(images) // columns + 1, columns, i + 1)
    image = plt.imread(image_loc)
    plt.title(diseases[i])
    plt.imshow(image)



## === cell 12
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["label"])
data["variety"] = encoder.fit_transform(data["variety"])
data.head()



## === cell 13
batch_size = 32
img_height = 224
img_width = 224



## === cell 14
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)



## === cell 15
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=path,
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
for image_batch, label_batch in train_ds:
    print(image_batch.shape)
    print(label_batch.shape)
    break



## === cell 18
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)



## === cell 19
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, label_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))



## === cell 20
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
    ],
    name="data_augmentation",
)


def preprocess_train(x, y):
    x = tf.cast(x, tf.float32)
    x = data_augmentation(x, training=True)
    x = x / 255.0
    return x, y


def preprocess_eval(x, y):
    x = tf.cast(x, tf.float32)
    x = x / 255.0
    return x, y


train_ds = train_ds.shuffle(2048, seed=123, reshuffle_each_iteration=True)
train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)

val_ds = val_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE)

options_train = tf.data.Options()
options_train.experimental_deterministic = False
train_ds = train_ds.with_options(options_train)

options_eval = tf.data.Options()
options_eval.experimental_deterministic = True
val_ds = val_ds.with_options(options_eval)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



## === cell 21
num_classes = len(class_names)

l2 = tf.keras.regularizers.l2(1e-4)

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)



## === cell 22
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)



## === cell 23
from collections import Counter

label_counts = Counter(
    pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")[
        "label"
    ].tolist()
)
total = sum(label_counts.values())
class_weight = {
    i: (total / (num_classes * label_counts.get(cls_name, 1)))
    for i, cls_name in enumerate(class_names)
}
print("Computed class_weight (by class index):", class_weight)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping],
    class_weight=class_weight,
)

loss = model.evaluate(val_ds)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper right")
plt.show()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="lower left")
plt.show()



## === cell 24
model.summary()



## === cell 25
loss, accu = model.evaluate(val_ds)
print(f"the Testing loss is {loss:.2f}")
print(f"The testing accuracy is {accu*100:.2f}%")



## === cell 26
test_data_dir = "/kaggle/input/paddy-disease-classification/test_images/"



## === cell 27
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
)


def preprocess_test(x):
    x = tf.cast(x, tf.float32)
    x = x / 255.0
    return x


test_ds = test_ds.map(preprocess_test, num_parallel_calls=AUTOTUNE).with_options(
    options_eval
)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)



## === cell 28
y_pred = model.predict(test_ds, batch_size=batch_size, verbose=1)
y_pred.shape



## === cell 29
y_pred_classes = y_pred.argmax(axis=1)
y_pred_classes.shape



## === cell 30
y_classes_names = [class_names[x] for x in y_pred_classes]



## === cell 31
predictions = pd.read_csv(
    "/kaggle/input/paddy-disease-classification/sample_submission.csv"
)
predictions["label"] = y_classes_names
predictions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predictions.shape)
print(predictions.head())
