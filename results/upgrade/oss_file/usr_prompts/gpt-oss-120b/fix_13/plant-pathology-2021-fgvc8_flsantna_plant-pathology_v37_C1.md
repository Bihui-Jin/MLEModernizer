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

0.8117451523545736

# 6. Current score

0.57503

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I set the protobuf implementation before importing TensorFlow to avoid the MessageFactory error, replace the faulty model loading with a simple placeholder that assigns the “healthy” label to every test image (ensuring a valid CSV submission), and clean up the inference loop accordingly. This fixes the runtime crashes while still producing a correctly‑formatted submission file.'
- What this solution (achieved 0.58833) has done: 'The fix adds a lightweight transfer‑learning pipeline using TensorFlow’s built‑in MobileNetV2 to generate image embeddings, creates class centroids from a small sampled subset of the training set, and assigns each test image the label of the nearest centroid. This replaces the “always‑healthy” placeholder with a data‑driven prediction, keeps the original file‑handling logic, and still writes a correctly formatted `submission.csv`. The changes also keep the protobuf environment setting and guard against GPU use.'
- What this solution (achieved 0.56755) has done: 'I keep the original pipeline but fix the protobuf import issue by setting the environment variable before importing TensorFlow (as already done) and make two small, score‑oriented tweaks: increase the number of sampled training images per class to improve centroid quality, and change the inference step to output the two most similar disease labels per image (space‑delimited) instead of a single label. These adjustments stay within the existing logic, avoid any major architectural changes, and should raise the F1 score toward the target while still writing a correct `submission.csv`.'
- What this solution (achieved 0.6244) has done: 'I keep the overall centroid‑based approach but tighten it up to improve F1: increase the number of training samples per class, predict only the single most‑similar label (top‑1) instead of two, and use np.argmax for a reliable highest‑similarity choice. These small tweaks stay within the original pipeline while boosting precision, and the script still write a correctly‑named submission.csv file.'
- What this solution (achieved 0.58061) has done: 'The changes parallelize image loading for both training and test data using a thread pool, which removes the Python‑level I/O bottleneck while keeping the exact same preprocessing, feature extraction, and prediction logic. By loading images in batches before feeding them to the MobileNetV2 (or histogram) extractor, we keep all subsequent computations identical, preserving model results and accuracy but fitting comfortably within the 600‑second limit.'
- What this solution (achieved 0.63188) has done: 'I guard the TensorFlow import and any TensorFlow‑specific utilities so the script can run when TF cannot be loaded, switch image loading to Pillow when TF is unavailable, increase histogram resolution for better discriminative power, and limit predictions to the single most similar label (TOP_K_PRED = 1) which generally improves the multilabel F1 score.'
- What this solution (achieved 0.63188) has done: 'I make the TensorFlow import safer by setting the protobuf environment variables *before* any imports and by wrapping the import in a try‑except that falls back to a TF‑less mode. This prevents the protobuf‑related crash while still allowing TensorFlow to be used when it can be loaded, preserving the existing feature‑extraction logic.'
- What this solution (achieved 0.35257) has done: 'Implemented safer TensorFlow import with protected GPU visibility call, added similarity‑threshold based multi‑label prediction (up to five labels per image), and introduced configurable constants for threshold and max predictions. These changes fix the protobuf crash, allow richer label sets to improve the mean F1 score, and keep the core centroid‑based logic unchanged while still writing a correct `submission.csv`.'
- What this solution (achieved 0.48597) has done: 'I lower the label‑selection logic to always return the top K most similar classes (removing the similarity‑threshold filter) and set TOP_K_PRED to 3. This keeps the centroid‑based approach unchanged while improving recall, which should raise the mean F1‑score toward the target. The rest of the pipeline and file handling stay the same.'
- What this solution (achieved 0.35257) has done: 'The fix adds a monkey‑patch for the missing `GetPrototype` method in protobuf so TensorFlow can be imported safely, and improves prediction by returning all labels whose cosine similarity exceeds the defined threshold (up to five labels) instead of a fixed top‑K list. This keeps the original centroid‑based pipeline while enabling the MobileNetV2 feature extractor for better embeddings, and aligns the output with the multilabel F1 metric.'
- What this solution (achieved 0.57503) has done: 'I fixed the protobuf monkey‑patch that caused an AttributeError, simplified the training data loading to keep each image with its full multilabel string, replaced the centroid‑based classifier with a fast cosine‑similarity nearest‑neighbour lookup (using the full training set), and lowered the similarity threshold logic to always return the most common labels from the top‑K similar images (up to five). These changes keep the original feature extraction (MobileNetV2 when TensorFlow is available, otherwise a colour histogram) while improving label prediction, moving the score toward the target and guaranteeing a correctly formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

try:
    from google.protobuf import message_factory as mf

    if not hasattr(mf, "MessageFactory"):
        pass
    else:
        pass
except Exception:
    pass

try:
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    tf_available = True
except Exception:  # pragma: no cover
    tf_available = False

import numpy as np
import pandas as pd

if tf_available:
    from tensorflow.keras.preprocessing import image
else:
    from PIL import Image

from concurrent.futures import ThreadPoolExecutor  # parallel image loading
from collections import Counter



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_images_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

default_label = "healthy"
if default_label not in dataset_labels:
    default_label = dataset_labels[0]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TOP_K_NEIGHBORS = 5  # number of nearest neighbours to consider
MAX_LABELS_PER_IMAGE = 5  # maximum labels to output per image
SIMILARITY_THRESHOLD = 0.3  # lower to increase recall when using thresholds

if tf_available:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    feature_extractor = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=IMG_SIZE + (3,),
    )


def load_and_preprocess(img_path):
    """Load an image file, resize to IMG_SIZE and return a uint8 ndarray."""
    if tf_available:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        arr = image.img_to_array(img).astype(np.uint8)
    else:
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img).astype(np.uint8)
    return arr


def histogram_feature(arr):
    """Create a finer color histogram (64 bins per channel)."""
    hists = []
    for ch in range(3):
        hist, _ = np.histogram(arr[..., ch], bins=64, range=(0, 256))
        hists.append(hist)
    return np.concatenate(hists).astype(np.float32)


path_label_pairs = []
for _, row in data_set.iterrows():
    img_name = row["image"]
    label_str = row["labels"]
    img_path = os.path.join(train_images_dir, img_name)
    if os.path.isfile(img_path):
        path_label_pairs.append((img_path, label_str))

features_list = []
train_labels = []  # keep the full space‑delimited label string per image


def load_pair(pair):
    img_path, label_str = pair
    try:
        arr = load_and_preprocess(img_path)
    except Exception:
        return None, None
    if tf_available:
        arr_pp = preprocess_input(arr.astype(np.float32))
        return arr_pp, label_str
    else:
        return histogram_feature(arr), label_str


with ThreadPoolExecutor() as executor:
    for feat, lbl in executor.map(load_pair, path_label_pairs, chunksize=64):
        if feat is not None:
            features_list.append(feat)
            train_labels.append(lbl)

if len(features_list) == 0:
    raise RuntimeError("No training features could be loaded.")

if tf_available:
    embeddings = np.stack(features_list, axis=0)
    train_features = feature_extractor.predict(
        embeddings, batch_size=BATCH_SIZE, verbose=0
    )
else:
    train_features = np.stack(features_list, axis=0)

train_feats_norm = train_features / (
    np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-10
)



## === cell 2
if __name__ == "__main__":
    test_images = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    test_paths = [os.path.join(test_dir, name) for name in test_images]

    def load_test(img_path):
        try:
            arr = load_and_preprocess(img_path)
        except Exception:
            return None
        if tf_available:
            return preprocess_input(arr.astype(np.float32))
        else:
            return histogram_feature(arr)

    with ThreadPoolExecutor() as executor:
        loaded = list(executor.map(load_test, test_paths, chunksize=64))

    predictions = []
    batch_imgs = []
    batch_names = []

    def predict_batch(feats_batch, names_batch):
        feats_norm = feats_batch / (
            np.linalg.norm(feats_batch, axis=1, keepdims=True) + 1e-10
        )
        sim = np.dot(feats_norm, train_feats_norm.T)  # shape (batch, N_train)

        for name, sim_row in zip(names_batch, sim):
            if TOP_K_NEIGHBORS >= sim_row.shape[0]:
                top_idxs = np.argsort(-sim_row)
            else:
                top_idxs = np.argpartition(-sim_row, TOP_K_NEIGHBORS)[:TOP_K_NEIGHBORS]
                top_idxs = top_idxs[np.argsort(-sim_row[top_idxs])]

            label_counter = Counter()
            for idx in top_idxs:
                neigh_labels = train_labels[idx].split()
                label_counter.update(neigh_labels)

            selected = [
                lbl for lbl, _ in label_counter.most_common(MAX_LABELS_PER_IMAGE)
            ]
            if not selected:
                selected = [default_label]

            predictions.append([name, " ".join(selected)])

    for name, feat_arr in zip(test_images, loaded):
        if feat_arr is None:
            predictions.append([name, default_label])
            continue

        batch_imgs.append(feat_arr)
        batch_names.append(name)

        if len(batch_imgs) == BATCH_SIZE:
            batch_arr = np.stack(batch_imgs)
            if tf_available:
                feats = feature_extractor.predict(
                    batch_arr, batch_size=BATCH_SIZE, verbose=0
                )
            else:
                feats = batch_arr
            predict_batch(feats, batch_names)
            batch_imgs, batch_names = [], []

    if batch_imgs:
        batch_arr = np.stack(batch_imgs)
        if tf_available:
            feats = feature_extractor.predict(
                batch_arr, batch_size=len(batch_imgs), verbose=0
            )
        else:
            feats = batch_arr
        predict_batch(feats, batch_names)

    submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
