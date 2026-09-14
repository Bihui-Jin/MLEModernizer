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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7826038781163449

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'I remove the unnecessary KaggleDatasets import that crashes, guard the model loading so the notebook continues even when the 4SMResNet50.h5 file is missing, and generate a fallback prediction (assigning “healthy” to every test image) to guarantee that the lengths match and a valid submission.csv is written. This makes the script runnable end‑to‑end while preserving the original structure; if the model file is present the original logic is used, otherwise the safe fallback ensures a correctly‑formatted submission.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

print(tf.__version__)
print(tf.keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib




## === cell 2
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
source = os.path.abspath("../input/plant-pathology-2021-fgvc8/test_images")
IMAGE_PATHS = [
    os.path.join(source, f)
    for f in os.listdir(source)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.IGNORECASE)
]



## === cell 5
IMAGE_PATHS[:5]



## === cell 6
AUTO = tf.data.experimental.AUTOTUNE



## === cell 7
test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)



## === cell 8
import tensorflow as tf
from tensorflow import keras




## === cell 9
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 10
model_path = "../input/4smresnet50/4SMResNet50.h5"
try:
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    use_model = True
    print("Pretrained model loaded.")
except Exception as e:
    model = None
    use_model = False
    print(f"Could not load pretrained model ({e}); using fallback predictions.")



## === cell 11
if use_model:
    probs = model.predict(test_dataset, verbose=0)
else:
    probs = np.zeros((len(IMAGE_PATHS), 5), dtype=np.float32)

temp_probs = probs



## === cell 12
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
threshold2 = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}

pred_string = []
for line in temp_probs:
    s = ""
    count = 0
    for i in range(5):
        if line[i] > threshold[i]:
            s += name[i] + " "
    for i in range(5):
        if line[i] > threshold2[i]:
            count += 1
    if count >= 2:
        notComplex = True
        for i in range(5):
            if line[i] > threshold[i] and name[i] == "complex":
                notComplex = False
                break
        if notComplex:
            s += "complex" + " "
    if s.strip() == "":
        s = name[6]  # default to 'healthy'
    pred_string.append(s.strip())




## === cell 13
print("Number of images:", len(IMAGE_PATHS))
print("Number of predictions:", len(pred_string))



## === cell 14
df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in IMAGE_PATHS], "labels": pred_string}
)
submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
display(df.head())
