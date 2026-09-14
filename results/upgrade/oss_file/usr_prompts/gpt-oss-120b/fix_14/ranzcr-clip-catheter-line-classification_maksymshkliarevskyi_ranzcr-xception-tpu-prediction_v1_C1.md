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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9393387801547622

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I guard TensorFlow imports so they don’t raise the protobuf error, skip loading the missing pre‑trained model, and replace the model‑based prediction with a simple baseline that uses the mean label frequencies from the training set. This removes the runtime crashes, ensures a valid `submission.csv` is written, and keeps the core logic intact while providing reasonable predictions.'
- What this solution (achieved 0.5176) has done: 'The script’s main bottleneck is the sequential reading of ~30 k images to compute mean intensities for the logistic‑regression baseline. By parallelizing this I/O‑bound work with a thread pool we dramatically cut the runtime while keeping the exact same feature (image mean) and thus preserving prediction accuracy. The only change is a faster `_compute_mean_intensity` implementation; all other logic, model loading, and output generation remain untouched.'
- What this solution (achieved 0.5) has done: 'The changes remove the expensive Xception model loading (which isn’t used for the final predictions) and replace the thread‑based image‑feature extraction with a process‑based pool that fully utilizes all CPU cores, dramatically cutting the I/O‑bound image‑reading time while keeping the exact same statistics. The rest of the workflow—including dataset creation, logistic‑regression training, and submission generation—remains unchanged.'
- What this solution (achieved 0.5) has done: 'Implemented a faster feature extraction by converting the path series to a plain list, using a `ProcessPoolExecutor` with a higher default worker count, and simplifying the worker function to avoid repeated attribute lookups. Added a brief comment explaining the change. All other logic, model handling, and prediction steps remain unchanged, preserving exact semantics and results.'
- What this solution (achieved 0.5) has done: 'Implemented two critical fixes and a modest feature enhancement to boost the baseline score:

1. **Robust image handling** – `Image` from Pillow is now always imported, preventing a `NameError` when OpenCV is available.
2. **Richer image features** – added per‑channel 16‑bin histograms to the feature set (total 53 features) while keeping the original statistics. This improves the Logistic Regression baseline without altering the core model logic.

These changes ensure the script runs end‑to‑end and produces a valid `submission.csv`, while nudging the AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I fix the import guard for TensorFlow, add per‑channel standard deviations to the image feature vector, and train each Logistic Regression with `class_weight="balanced"` and a standard‑scaler to improve AUC while keeping the overall baseline logic unchanged. The script now run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'The fix disables TensorFlow entirely to avoid the protobuf import error and forces the script to use the logistic‑regression baseline, while a modest increase of the regularisation strength (`C=10.0`) nudges the model toward a higher AUC. All other logic remains unchanged, and a valid `submission.csv` is written.'
- What this solution (achieved 0.5) has done: 'The fix removes the problematic TensorFlow import (which caused a protobuf error) and forces `tf` to `None` while keeping all other logic unchanged. This allows the script to run end‑to‑end, compute richer image features, train the logistic‑regression baseline, and write a valid `submission.csv` without runtime crashes.'
- What this solution (achieved 0.5) has done: 'I enhance the image feature extractor by adding the overall median intensity (a cheap, informative statistic) and raise the logistic‑regression regularisation strength (C) from 10.0 to 100.0. These minimal tweaks keep the same baseline logic while providing the model with a slightly richer signal, which should raise the AUC toward the target without altering the overall workflow.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import cv2
except Exception:
    cv2 = None
from PIL import Image

tf = None

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler




## === cell 1
def build_decoder(with_labels=True, target_size=(512, 512), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
):
    if tf is None:
        raise RuntimeError(
            "TensorFlow is unavailable; dataset utilities cannot be used."
        )
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)
    return dset




## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("Working directory contents:", os.listdir(WORK_DIR))




## === cell 3
print("Train images: %d" % len(os.listdir(os.path.join(WORK_DIR, "train"))))




## === cell 4
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_images = (
    os.path.join(WORK_DIR, "train") + "/" + train["StudyInstanceUID"] + ".jpg"
)
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
test_images = os.path.join(WORK_DIR, "test") + "/" + ss["StudyInstanceUID"] + ".jpg"

label_cols = ss.columns[1:]
labels = train[label_cols].values

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

print("Labels:\n", "*" * 20, "\n", label_cols.values)
print("*" * 50)
print(train.head())




## === cell 5
sns.set_style("whitegrid")
fig = plt.figure(figsize=(15, 12), dpi=300)
plt.suptitle("Labels count", fontfamily="serif", size=15)

for ind, i in enumerate(label_cols):
    fig.add_subplot(4, 3, ind + 1)
    sns.countplot(
        train[i],
        edgecolor="black",
        palette=list(reversed(sns.color_palette("viridis", 2))),
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontfamily="serif", size=10)
    plt.yticks(fontfamily="serif", size=10)
    plt.title(i, fontfamily="serif", size=10)
plt.show()




## === cell 6
sample = train.sample(9)
plt.figure(figsize=(10, 7), dpi=300)
for ind, image_id in enumerate(sample.StudyInstanceUID):
    plt.subplot(3, 3, ind + 1)
    img_path = os.path.join(WORK_DIR, "train", image_id + ".jpg")
    if cv2 is not None:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(img_path).convert("RGB"))
    plt.imshow(img)
    plt.axis("off")
plt.show()




## === cell 7
BATCH_SIZE = 16
STEPS_PER_EPOCH = len(train) * 0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train) * 0.2 / BATCH_SIZE
EPOCHS = 15
TARGET_SIZE = 512




## === cell 8
if tf is not None:
    test_df = build_dataset(
        test_images,
        bsize=BATCH_SIZE,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
    )
else:
    test_df = None
    print("Skipping dataset creation because TensorFlow is unavailable.")




## === cell 9
model = None
print("Model loading skipped to reduce runtime.")




## === cell 10
if model is not None:
    print("Our Xception CNN has %d layers" % len(model.layers))
else:
    print("No model available; skipping layer count.")




## === cell 11
def activation_layer_vis(img, activation_layer=0, layers=10):
    if model is None:
        raise RuntimeError("Model not loaded; cannot visualize activations.")
    layer_outputs = [layer.output for layer in model.layers[:layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)
    rows = int(activations[activation_layer].shape[3] / 3)
    cols = int(activations[activation_layer].shape[3] / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15 * cols))
    axes = axes.flatten()
    for i, ax in zip(range(activations[activation_layer].shape[3]), axes):
        ax.matshow(activations[activation_layer][0, :, :, i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def all_activations_vis(img, layers=10):
    if model is None:
        raise RuntimeError("Model not loaded; cannot visualize activations.")
    layer_outputs = [layer.output for layer in model.layers[:layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)
    layer_names = [layer.name for layer in model.layers[:layers]]
    images_per_row = 3
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std() + 1e-5
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[
                    col * size : (col + 1) * size, row * size : (row + 1) * size
                ] = channel_image
        scale = 1.0 / size
        plt.figure(
            figsize=(
                scale * 5 * display_grid.shape[1],
                scale * 5 * display_grid.shape[0],
            )
        )
        plt.title(layer_name)
        plt.grid(False)
        plt.axis("off")
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
    plt.show()




## === cell 12
if tf is not None:
    img_tensor = build_dataset(
        pd.Series(test_images[0]),
        bsize=1,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
    )
else:
    img_tensor = None




## === cell 13
if tf is not None:
    img_tensor = build_dataset(
        pd.Series(test_images[0]),
        bsize=1,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
    )
else:
    img_tensor = None




## === cell 14
if tf is not None:
    img_tensor = build_dataset(
        pd.Series(test_images[0]),
        bsize=1,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
    )
else:
    img_tensor = None




## === cell 15
if model is not None and img_tensor is not None:
    all_activations_vis(img_tensor, 5)
else:
    print("Skipping activation visualization because model or data is unavailable.")




## === cell 16
def _compute_image_features(path_series, max_workers=None):
    """
    Compute richer image statistics for each path:
    - overall mean intensity
    - overall median intensity   <-- new feature
    - overall standard deviation
    - mean intensity per channel (R, G, B)
    - std intensity per channel (R, G, B)
    - per‑channel 16‑bin histograms (48 values)
    Returns an (N, 60) array.
    """
    paths = list(path_series)
    n = len(paths)
    feats = np.empty((n, 60), dtype=np.float32)

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) * 2)

    cv2_local = cv2
    Image_local = Image

    bins = np.arange(0, 257, 16)  # 16 bins of width 16

    def _worker(idx_path):
        idx, p = idx_path
        if cv2_local is not None:
            img = cv2_local.imread(p)
            if img is None:
                return idx, None
            img = cv2_local.cvtColor(img, cv2_local.COLOR_BGR2RGB)
        else:
            try:
                img = np.array(Image_local.open(p).convert("RGB"))
            except Exception:
                return idx, None
        overall_mean = float(img.mean())
        overall_median = float(np.median(img))
        overall_std = float(img.std())
        mean_r = float(img[:, :, 0].mean())
        mean_g = float(img[:, :, 1].mean())
        mean_b = float(img[:, :, 2].mean())
        std_r = float(img[:, :, 0].std())
        std_g = float(img[:, :, 1].std())
        std_b = float(img[:, :, 2].std())
        hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 256))
        hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 256))
        hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 256))
        hist_r = hist_r.astype(np.float32) / (hist_r.sum() + 1e-6)
        hist_g = hist_g.astype(np.float32) / (hist_g.sum() + 1e-6)
        hist_b = hist_b.astype(np.float32) / (hist_b.sum() + 1e-6)

        feature_vec = np.concatenate(
            [
                [
                    overall_mean,
                    overall_median,  # new
                    overall_std,
                    mean_r,
                    mean_g,
                    mean_b,
                    std_r,
                    std_g,
                    std_b,
                ],
                hist_r,
                hist_g,
                hist_b,
            ]
        )
        return idx, feature_vec

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, feature_vec in executor.map(_worker, enumerate(paths), chunksize=32):
            if feature_vec is None:
                feats[idx] = np.zeros(60, dtype=np.float32)
            else:
                feats[idx] = feature_vec

    return feats


try:
    if LogisticRegression is not None:
        print("Computing richer image features for training data...")
        train_feat = _compute_image_features(train_images)
        print("Computing richer image features for test data...")
        test_feat = _compute_image_features(test_images)

        scaler = StandardScaler()
        train_feat_scaled = scaler.fit_transform(train_feat)
        test_feat_scaled = scaler.transform(test_feat)

        preds = np.empty((len(test_feat_scaled), len(label_cols)), dtype=np.float32)
        for i, col in enumerate(label_cols):
            lr = LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
                C=100.0,  # stronger model
            )
            lr.fit(train_feat_scaled, train[col].values)
            preds[:, i] = lr.predict_proba(test_feat_scaled)[:, 1]

        ss.loc[:, label_cols] = preds
        print("Logistic‑regression baseline with richer features applied.")
    else:
        raise RuntimeError("LogisticRegression not available")
except Exception as e:
    print("Fallback to mean‑baseline due to error:", e)
    mean_probs = train[label_cols].mean()
    ss.loc[:, label_cols] = mean_probs.values

submission_path = "submission.csv"
ss.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")




## === cell 17
print(ss.head())
