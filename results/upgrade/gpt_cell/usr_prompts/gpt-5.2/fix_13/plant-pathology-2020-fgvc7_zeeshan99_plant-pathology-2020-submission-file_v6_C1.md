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

0.83351

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'You currently don’t yield a valid Kaggle submission because `processAndWriteDf()` drops the `image_id` column and saves only the 4 target columns, which breaks the required submission schema. I change the pipeline to always start from `sample_submission.csv` and write out a `submission.csv` with the exact same columns/order as the sample (including `image_id`). Since you don’t have a scored model running (TPU flag is off and the referenced model path may not exist), I produce a valid baseline submission by keeping the sample’s probabilities; this won’t maximize score, but it generate a valid submission so you can obtain a current score and iterate toward the target. All changes are minimal and only to ensure correct submission formatting and end-to-end execution.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from submitting constant probabilities (the sample submission), not from a working model. To move toward the 0.83351 target with minimal changes and without altering the model/loss logic, I (1) enable the prediction branch by default, (2) make the model/data paths robust to the provided `/kaggle/...` layout, and (3) fix the test generator settings to match common Keras expectations (avoid an invalid `y_col=[]` and use a standard `target_size` so inference is feasible within the time limit). If the pretrained model file isn’t present, the code still fall back to producing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'We need to move your score up from 0.5 toward 0.83351, and the smallest legitimate improvement is to ensure the TPU inference path actually finds and uses a real model when present, rather than silently falling back to the constant-probability sample submission. I keep your model loading/inference logic the same, but broaden the model search paths to include the common “working” and “input” locations you actually have, and add a lightweight debug print of the resolved paths so you can confirm you’re not accidentally in the fallback. I also make the prediction-to-column mapping robust by deriving the label column order from `sample_submission.csv` (still the same 4 targets), avoiding accidental class-order mismatches that can cap AUC. The submission writer remain schema-safe and still fall back to sample if model loading fails, but it should now much more reliably use the real predictions to increase score.'

# 9. Code solution

## === cell 0
import os
import zipfile

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        return self.pool.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd




## === cell 1
def processAndWriteDf(df, out_path="./submission.csv"):
    candidate_sample_paths = [
        "../input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
    ]
    sample_path = None
    for p in candidate_sample_paths:
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        raise FileNotFoundError(
            "Could not locate sample_submission.csv in expected input paths."
        )

    sample = pd.read_csv(sample_path)
    required_cols = list(sample.columns)

    if "image_id" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'image_id' column for a valid submission."
        )

    for c in required_cols:
        if c not in df.columns:
            if c == "image_id":
                continue
            df[c] = sample[c].iloc[0]

    test_paths = [
        "../input/plant-pathology-2020-fgvc7/test.csv",
        "../input/test.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/test.csv",
    ]
    test_path = None
    for p in test_paths:
        if os.path.exists(p):
            test_path = p
            break
    if test_path is None:
        raise FileNotFoundError("Could not locate test.csv in expected input paths.")

    test_df = pd.read_csv(test_path)[["image_id"]]
    merged = test_df.merge(df, on="image_id", how="left")

    for c in required_cols:
        if c != "image_id":
            merged[c] = merged[c].fillna(sample[c].iloc[0])

    submission = merged[required_cols].copy()
    submission.to_csv(out_path, index=False)
    print(submission.head(5))
    print(
        f"write done -> {out_path} (rows={len(submission)}, cols={len(submission.columns)})"
    )
    return submission




## === cell 2
def getPredictionFromTPUModel():
    candidate_model_paths = [
        "../input/plant-pathology-2020-tpu/my_modelv5.h5",
        "/kaggle/input/plant-pathology-2020-tpu/my_modelv5.h5",
        "/kaggle/data/plant-pathology-2020-tpu/my_modelv5.h5",
        "./my_modelv5.h5",
        "/kaggle/working/my_modelv5.h5",
        "/kaggle/working/plant-pathology-2020-fgvc7/my_modelv5.h5",
        "/kaggle/working/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7/my_modelv5.h5",
    ]
    model_path = None
    for p in candidate_model_paths:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(
            "Could not locate my_modelv5.h5 in expected paths. Tried: "
            + ", ".join(candidate_model_paths)
        )

    print("Using model:", model_path)
    model = tf.keras.models.load_model(model_path)

    candidate_base_dirs = [
        "../input/plant-pathology-2020-fgvc7/images/",
        "/kaggle/input/plant-pathology-2020-fgvc7/images/",
        "/kaggle/data/images/",
        "/kaggle/data/plant-pathology-2020-fgvc7/images/",
    ]
    base_dir = None
    for d in candidate_base_dirs:
        if os.path.isdir(d):
            base_dir = d
            break
    if base_dir is None:
        raise FileNotFoundError(
            "Could not locate images/ directory. Tried: "
            + ", ".join(candidate_base_dirs)
        )

    candidate_test_csv = [
        "../input/plant-pathology-2020-fgvc7/test.csv",
        "../input/test.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/test.csv",
    ]
    test_csv_path = None
    for p in candidate_test_csv:
        if os.path.exists(p):
            test_csv_path = p
            break
    if test_csv_path is None:
        raise FileNotFoundError("Could not locate test.csv in expected paths.")

    print("Using images dir:", base_dir)
    print("Using test.csv:", test_csv_path)

    candidate_sample_paths = [
        "../input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
    ]
    sample_path = None
    for p in candidate_sample_paths:
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        raise FileNotFoundError(
            "Could not locate sample_submission.csv in expected input paths."
        )
    sample = pd.read_csv(sample_path)
    label_cols = [c for c in sample.columns if c != "image_id"]

    test_csv = pd.read_csv(test_csv_path)
    test_csv["image_id"] = test_csv["image_id"] + ".jpg"

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    batchSize = 16

    try:
        ishape = model.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        h, w = ishape[1], ishape[2]
        if h is None or w is None:
            raise ValueError("Dynamic input shape")
        target_size = (int(h), int(w))
    except Exception:
        target_size = (224, 224)

    test_generator = test_datagen.flow_from_dataframe(
        test_csv,
        directory=base_dir,
        x_col="image_id",
        y_col=None,
        target_size=target_size,
        batch_size=batchSize,
        class_mode=None,
        shuffle=False,
    )

    p = model.predict(test_generator, verbose=0)

    if p.shape[-1] != len(label_cols):
        raise ValueError(
            f"Model output has {p.shape[-1]} columns but submission expects {len(label_cols)}: {label_cols}"
        )

    p_df = pd.DataFrame(p, columns=label_cols)
    for c in label_cols:
        p_df[c] = p_df[c].clip(0.0, 1.0)

    test_csv_clean = pd.read_csv(test_csv_path)
    result = pd.concat([test_csv_clean[["image_id"]], p_df], axis=1)
    return result




## === cell 3
isTPU = True

if isTPU:
    try:
        df_pred = getPredictionFromTPUModel()
        processAndWriteDf(df_pred, out_path="./submission.csv")
    except Exception as e:
        print("Model inference failed; falling back to sample_submission baseline.")
        print("Reason:", repr(e))

        candidate_sample_paths = [
            "../input/plant-pathology-2020-fgvc7/sample_submission.csv",
            "../input/sample_submission.csv",
            "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
            "/kaggle/input/sample_submission.csv",
            "/kaggle/data/sample_submission.csv",
            "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
        ]
        sample_path = None
        for p in candidate_sample_paths:
            if os.path.exists(p):
                sample_path = p
                break
        if sample_path is None:
            raise FileNotFoundError(
                "Could not find sample_submission.csv. Tried: "
                + ", ".join(candidate_sample_paths)
            )

        df = pd.read_csv(sample_path)
        processAndWriteDf(df, out_path="./submission.csv")
else:
    candidate_sample_paths = [
        "../input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
    ]
    sample_path = None
    for p in candidate_sample_paths:
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        raise FileNotFoundError(
            "Could not find sample_submission.csv. Tried: "
            + ", ".join(candidate_sample_paths)
        )

    df = pd.read_csv(sample_path)
    processAndWriteDf(df, out_path="./submission.csv")
