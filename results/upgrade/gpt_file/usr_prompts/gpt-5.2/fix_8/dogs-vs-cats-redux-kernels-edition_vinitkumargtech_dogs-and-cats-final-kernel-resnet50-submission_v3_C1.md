# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("PYTHONHASHSEED", "42")

import math
import random

import numpy as np
import pandas as pd

print("Listing ../input:")
print(os.listdir("../input"))



## === cell 1
from matplotlib import pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None

try:
    from tqdm import tqdm  # noqa: F401
except Exception:
    tqdm = None

random.seed(42)
np.random.seed(42)



## === cell 2
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

tf.random.set_seed(42)
print("TensorFlow:", tf.__version__)



## === cell 3
batch_size = 32
epochs = 20
num_classes = 2

path = "../input/dogs-vs-cats-redux-kernels-edition/"
train_data_dir = os.path.join(path, "train")

test_dir_candidates = [
    os.path.join(path, "test"),
    os.path.join(path, "test", "test"),
]
test_dir = None
for cand in test_dir_candidates:
    if os.path.isdir(os.path.join(cand, "unknown")):
        test_dir = cand
        break
if test_dir is None:
    test_dir = os.path.join(path, "test")

img_size = 224

print("train_data_dir:", train_data_dir, "exists:", os.path.isdir(train_data_dir))
print("test_dir:", test_dir, "exists:", os.path.isdir(test_dir))
print(
    "train subdirs:",
    os.listdir(train_data_dir)[:10] if os.path.isdir(train_data_dir) else None,
)
print("test subdirs:", os.listdir(test_dir)[:10] if os.path.isdir(test_dir) else None)



## === cell 4
classes = ["cat", "dog"]

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset="training",
    classes=classes,
)

val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=42,
    subset="validation",
    classes=classes,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
)

filename = test_generator.filenames
print("Example test filenames:", filename[:5])

print("train class_indices:", train_generator.class_indices)
print("val class_indices:", val_generator.class_indices)
print("test class_indices:", test_generator.class_indices)



## === cell 5
inputs = tf.keras.Input(shape=(img_size, img_size, 3))
base = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_tensor=inputs
)
base.trainable = False  # keep core logic: train only the classifier head

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("Built model:", model.name)
print("Output shape:", model.output_shape)



## === cell 6
steps_per_epoch = max(1, math.ceil(train_generator.samples / batch_size))
validation_steps = max(1, math.ceil(val_generator.samples / batch_size))

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 7
test_generator.reset()
pred = model.predict(
    test_generator,
    steps=math.ceil(test_generator.samples / test_generator.batch_size),
    verbose=1,
)

pred = np.asarray(pred)
if pred.ndim != 2 or pred.shape[1] != 2:
    raise ValueError(
        f"Expected model to output shape (N, 2) softmax probabilities, got {pred.shape}"
    )

train_class_indices = train_generator.class_indices
if not isinstance(train_class_indices, dict) or ("dog" not in train_class_indices):
    raise ValueError(
        f"Unexpected train class_indices (need 'dog'): {train_class_indices}"
    )
dog_idx = int(train_class_indices["dog"])

dog_probs = pred[:, dog_idx].astype(np.float64)
dog_probs = np.clip(dog_probs, 1e-7, 1 - 1e-7)

predicted_class_indices = np.argmax(pred, axis=1)
inv_map = {v: k for k, v in train_class_indices.items()}
new_preds = [inv_map.get(int(k), str(int(k))) for k in predicted_class_indices]

print("dog_idx:", dog_idx)
print("dog_probs sample:", dog_probs[:5])




## === cell 8
def display_testdata(testdata, filenames):
    f, ax = plt.subplots(5, 5, figsize=(15, 15))
    i = 0
    for a, b in zip(testdata, filenames):
        pred_label = a
        fname = b
        title = "Prediction :{}".format(pred_label)
        original = load_img("{}/{}".format(test_dir, fname))
        ax[i // 5, i % 5].axis("off")
        ax[i // 5, i % 5].set_title(title)
        ax[i // 5, i % 5].imshow(original)
        i = i + 1
        if i >= 25:
            break
    plt.show()




## === cell 9
start = 0
end = min(25, len(new_preds))
if end - start >= 1:
    display_testdata(new_preds[start:end], filename[start:end])




## === cell 10
def create_submission(filenames, probs):
    file_index = []
    for b in filenames:
        base = os.path.basename(b)  # e.g. '900.jpg' or 'unknown/900.jpg'
        img_id = os.path.splitext(base)[0]
        try:
            file_index.append(int(img_id))
        except ValueError:
            digits = "".join(ch for ch in img_id if ch.isdigit())
            if digits == "":
                raise ValueError(f"Could not parse numeric id from filename: {b}")
            file_index.append(int(digits))

    if len(file_index) != len(probs):
        raise ValueError(
            f"Mismatch: {len(file_index)} ids vs {len(probs)} predictions."
        )

    results = pd.DataFrame({"id": file_index, "label": probs})
    results = results.sort_values("id").reset_index(drop=True)

    results["id"] = results["id"].astype(int)
    results["label"] = results["label"].astype(float)

    results.to_csv("submission.csv", index=False)
    return results


sub = create_submission(filename, dog_probs)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.isfile("submission.csv"))
