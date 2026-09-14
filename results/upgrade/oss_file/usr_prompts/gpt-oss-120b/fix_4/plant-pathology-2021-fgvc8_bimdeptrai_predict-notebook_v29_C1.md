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

0.8086426592797806

# 6. Current score

0.12086

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10024) has done: 'I remove the problematic `tensorflow_addons` import, add a safe fallback when the pretrained model file is missing (building a simple DenseNet‑201 model with ImageNet weights), and fix the prediction‑to‑submission logic (use proper assignment, correct indexing and generate labels via argmax). This ensures the notebook runs end‑to‑end, creates a valid `submission.csv`, and keeps the original workflow intact.'
- What this solution (achieved 0.19782) has done: 'I fix the TensorFlow import error by setting the protobuf implementation flag before importing TensorFlow, adjust the fallback model to use a sigmoid + binary‑cross‑entropy output for true multi‑label prediction, and change the prediction‑to‑submission logic to apply a 0.5 threshold, collect all labels above the threshold (or fallback to “healthy”), and join them as a space‑delimited string. These minimal changes resolve the runtime crash and produce more appropriate multi‑label predictions, moving the F1 score toward the target.'
- What this solution (achieved 0.12086) has done: 'I adjust the prediction‑to‑submission step to use an argmax‑based label selection with a low confidence fallback to “healthy”. This typically improves multi‑label F1 when most images have a single dominant class, moving the score toward the target while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")




## === cell 2
label_split = train["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)
labels_df = pd.DataFrame(mlb.transform(label_split), columns=mlb.classes_)




## === cell 3
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")




## === cell 4
h_target, w_target = 256, 256
batch_size = 32




## === cell 5
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    submissions,
    directory="../input/plant-pathology-2021-fgvc8/test_images",
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    shuffle=False,
    batch_size=batch_size,
)




## === cell 6
model_path = "../input/densenet201-512-to-256/Densenet201.h5"
try:
    model = keras.models.load_model(model_path)
except Exception as e:
    print(
        f"Could not load model from '{model_path}'. Using fallback model. Reason: {e}"
    )
    base = keras.applications.DenseNet201(
        weights="imagenet", include_top=False, input_shape=(h_target, w_target, 3)
    )
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    output = keras.layers.Dense(labels_df.shape[1], activation="sigmoid")(x)
    model = keras.Model(inputs=base.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy")




## === cell 7
preds = model.predict(test_generator, verbose=1)




## === cell 8
fallback_label = "healthy"
confidence_threshold = 0.3  # low confidence cutoff

pred_labels_list = []
for probs in preds:
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]
    if max_prob < confidence_threshold:
        pred_labels_list.append(fallback_label)
    else:
        pred_labels_list.append(labels_df.columns[max_idx])
submissions["labels"] = pred_labels_list




## === cell 9
submissions.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
