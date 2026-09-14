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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
PyYAML==6.0.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.19042

# 6. Current score

0.16124

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.48372) has done: 'I fix the environment-breaking import issue by switching the Keras import path to `tf_keras` (compatible with the Kaggle TensorFlow/Keras stack) and remove notebook-only magics. Then I fix the core data loading bug (images have varying shapes) by avoiding `np.array(...)` on variable-sized images and instead grouping indices by shape while keeping the generator-based training logic intact. Finally, I make inference and submission writing robust: read test images as a list, apply the same padding/normalization used in training, clip outputs to [0,1], and write `submission.csv` with exactly `id,value` as required.'
- What this solution (achieved 0.1111) has done: 'I fix the runtime failure that happens before any training by removing the problematic protobuf-dependent import chain (matplotlib/animation/IPython) from the early cells, since it is not required for training or submission generation. Then I fix the `model.fit(...)` crash by omitting the invalid `workers=0` argument (tf_keras expects `workers>=1` if provided), which allow training to run and produce `log` for the later plotting cell. Finally, I keep the existing autoencoder/generator core logic intact and ensure the pipeline always writes a valid `submission.csv` with `id,value` and clipped predictions in `[0,1]`.'
- What this solution (achieved 0.12785) has done: 'I fix the environment-breaking `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by preventing optional visualization imports (matplotlib/IPython) from being imported at module import time, since that import chain is what triggers the protobuf incompatibility in this Kaggle image. I also ensure we always use the correct competition input paths (`/kaggle/input/denoising-dirty-documents/...`) deterministically, and keep the training/inference/submission core logic unchanged. Finally, I keep submission generation stable (correct `id,value`, clipping to `[0,1]`) and write to `submission.csv` in the working directory.'
- What this solution (achieved 0.38708) has done: 'I fix the protobuf-related crash that happens at import time by avoiding the `tf_keras` package (which is pulling in an incompatible protobuf chain here) and instead importing Keras via TensorFlow’s built-in `tensorflow.keras`, which is stable in Kaggle TF environments. This is a runtime-only change; the model architecture, training loop, data generator, and loss stay identical, so it should be score-neutral and just restore end-to-end execution. I also keep visualization imports fully optional so they can’t break training/submission generation. Finally, I ensure the script always writes `submission.csv` with exactly `id,value` and clipped predictions in `[0,1]`.'
- What this solution (achieved 0.28616) has done: 'I fix the import-time protobuf crash by switching to the Kaggle-stable `tf_keras` package (and defensively falling back to `tensorflow.keras` if needed). Then I fix the training crash caused by variable image shapes: Keras’ generator adapter requires consistent tensor shapes across batches, so I keep your generator-based training logic but pad every batch to a single global target size (computed once from the dataset and rounded to the model’s reduction factor). This also preserves the model’s core architecture/training loop while enabling full end-to-end execution. Finally, I align inference padding with the same global target size and keep the submission writer identical (`id,value`, clipped to `[0,1]`) to improve RMSE toward the target by ensuring consistent training/inference geometry and removing generator shape mismatches.'
- What this solution (achieved 0.28616) has done: 'I fix the import-time protobuf crash by making sure we consistently import layers/models from the same Keras package instance (`tf_keras` or the fallback) instead of mixing `keras` and `tf_keras`, which is what triggers the `MessageFactory` error in this environment. Then I make the submission writer match the competition’s required pixel id set by iterating through `sampleSubmission.csv` and filling values in that exact order (this avoids shape/size mismatches and typically improves RMSE versus writing only the raw predicted pixels). Finally, I keep the autoencoder, training loop, padding strategy, and loss exactly the same, only adding robust, score-relevant postprocessing for submission creation.'
- What this solution (achieved 0.11241) has done: 'I fix the import-time protobuf crash by ensuring we never mix standalone `keras` with `tf_keras`/`tensorflow.keras`, and by avoiding importing visualization stacks that can trigger the protobuf error. Then I make the encoder/decoder builders safe by not (ab)using `__init__` as a factory (which can silently break weight transfer), while keeping the same architecture and weights. Finally, I keep training/inference/submission logic intact, but make submission generation slightly more score-aligned by outputting grayscale predictions without changing the model (still trained on RGB), which is consistent with the competition’s grayscale target and should nudge RMSE down toward the target.'
- What this solution (achieved 0.15302) has done: 'I fix the import-time protobuf crash by ensuring we never import from standalone `keras` (Keras 3) while using `tf_keras`/`tensorflow.keras`, and instead consistently import layers/models from the same backend (`tf_keras` preferred, fallback to `tensorflow.keras`). This change is runtime-stability focused and keeps your model architecture/training loop identical, so it should be score-neutral (your current score is already better than the target band for a lower-is-better metric). I also make the TF/Keras seed setting and backend selection deterministic and keep all paths and submission-writing logic unchanged so `submission.csv` is always produced with the required `id,value` format.'
- What this solution (achieved 0.11325) has done: 'I fix the import-time protobuf crash by ensuring we never import `tensorflow.keras` symbols when `tf_keras` is in use (mixing them is what triggers the `MessageFactory` error in this Kaggle image). This is a runtime-stability change that preserves your exact model architecture/training loop and should be score-neutral (your current score is already better than the target band for a lower-is-better metric, so we avoid any intentional score improvements). I also keep visualization/animation imports optional and delayed so they can’t break training/submission generation. The pipeline still train, run inference, and always write a valid `submission.csv` with `id,value` using the sampleSubmission order.'
- What this solution (achieved 0.13565) has done: 'I fix the import-time crash by ensuring we never import from standalone `keras` (Keras 3) while also using `tf_keras`/`tensorflow.keras`, which is what triggers the protobuf `MessageFactory` error in this environment. Concretely, I route `Conv2D`, `Input`, `Model`, etc. through the same `keras` object that was successfully imported, and keep the rest of the training/inference/submission logic unchanged (so the score should remain close to your current 0.11325, which is already better than the target). I also keep all visualization-related imports optional/delayed as they already are, to avoid reintroducing the protobuf chain. The script still write a valid `submission.csv` with exactly `id,value` using `sampleSubmission.csv` row order.'
- What this solution (achieved 0.12006) has done: 'I fix the import-time protobuf crash by ensuring we never import from standalone `keras` (Keras 3) after selecting `tf_keras`/`tensorflow.keras`, and instead route `layers`/`Model` imports through the same `keras` object. This is a runtime-stability fix and should be score-neutral (your current RMSE 0.13565 is already better than the target 0.19042 for a lower-is-better metric, so we avoid any intentional score-changing edits). I also keep the visualization/animation cells optional and non-fatal as they already are, so training and submission generation always complete. The script still write a valid `submission.csv` with exactly `id,value` in the sampleSubmission row order.'
- What this solution (achieved 0.11897) has done: 'The crash happens immediately at import time due to an incompatibility in this Kaggle image between the `tf_keras` package import chain and the installed protobuf runtime (the `MessageFactory.GetPrototype` error). To make the notebook run end-to-end reliably, I switch the Keras import to TensorFlow’s bundled `tensorflow.keras` only (no mixing), which is the most stable option here and keeps your model/training/inference logic unchanged. Because your current score (0.12006, lower-is-better) is already substantially better than the target (0.19042), I not make any score-improving changes; this patch is intended to be score-neutral and only restore execution + valid `submission.csv` writing. All paths, padding strategy, generator training loop, and submission formatting remain the same.'
- What this solution (achieved 0.13301) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by preventing TensorFlow from using the C++ protobuf implementation in this Kaggle image (forcing the pure-Python protobuf runtime), which resolves the incompatibility without changing your model/training logic. I keep the exact same autoencoder, generator, padding, loss, and submission-writing approach to avoid intentionally shifting the score (your current 0.11897 is already better than the target 0.19042 for a lower-is-better metric). I also make the Keras import path deterministic and consistent (TensorFlow bundled Keras only) and ensure `submission.csv` is always written with `id,value` in the sampleSubmission order. No training/inference hyperparameters or postprocessing semantics are changed beyond stability.'
- What this solution (achieved 0.12622) has done: 'I fix the import-time protobuf crash by ensuring the protobuf implementation environment variable is set before any TensorFlow/Keras import (your current code sets it too late because `os.environ.setdefault` doesn’t override Kaggle’s preset). I also make the TensorFlow/Keras imports consistent and add a small safety fallback so the notebook always runs end-to-end in this environment without changing the model, training loop, padding, loss, or submission-writing semantics. Because your current score (0.13301, lower-is-better) is already better than the target (0.19042) and within the ±10% band, I avoid any score-changing modifications and focus strictly on stability and producing a valid `submission.csv`. The rest of the pipeline remains identical: same autoencoder, same generator logic, same inference cropping, and same sampleSubmission-ordered CSV writer.'
- What this solution (achieved 0.16124) has done: 'I fix the import-time protobuf crash by forcing the pure-Python protobuf implementation *before* TensorFlow is imported (and by setting it unconditionally, not with `setdefault`). I also prevent any accidental mixing of standalone `keras` (Keras 3) with `tensorflow.keras` by consistently using `tensorflow.keras` only, which is score-neutral and stabilizes execution. Since your current score (0.12622, lower-is-better) is already better than the target band around 0.19042, I not change the model/training/inference semantics; the goal is only to make it run end-to-end reliably and always write a valid `submission.csv` with `id,value` in the exact sampleSubmission order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import (
    keras,
)  # Use TF-bundled Keras only; do not import standalone `keras` (Keras 3).

layers = keras.layers
Model = keras.models.Model
Conv2D = layers.Conv2D
UpSampling2D = layers.UpSampling2D
MaxPooling2D = layers.MaxPooling2D
Input = layers.Input
Conv2DTranspose = layers.Conv2DTranspose

np.random.seed(42)
tf.random.set_seed(42)

BASE_INPUT = "/kaggle/input"
COMP_DIR = os.path.join(BASE_INPUT, "denoising-dirty-documents")

if os.path.isdir(COMP_DIR):
    TRAIN_DIR = os.path.join(COMP_DIR, "train")
    TRAIN_CLEAN_DIR = os.path.join(COMP_DIR, "train_cleaned")
    TEST_DIR = os.path.join(COMP_DIR, "test")
    SAMPLE_SUB_PATH = os.path.join(COMP_DIR, "sampleSubmission.csv")
else:
    TRAIN_DIR = os.path.join(BASE_INPUT, "train")
    TRAIN_CLEAN_DIR = os.path.join(BASE_INPUT, "train_cleaned")
    TEST_DIR = os.path.join(BASE_INPUT, "test")
    SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sampleSubmission.csv")

print("Using paths:")
print("TRAIN_DIR      :", TRAIN_DIR)
print("TRAIN_CLEAN_DIR:", TRAIN_CLEAN_DIR)
print("TEST_DIR       :", TEST_DIR)
print("SAMPLE_SUB     :", SAMPLE_SUB_PATH)


def imread_rgb(path):
    """Read image as RGB float32 in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0


def pad_to_shape(img, target_h, target_w, pad_value=1.0):
    """Pad an (H,W,C) image on bottom/right to (target_h,target_w,C) with pad_value."""
    h, w, c = img.shape
    if h > target_h or w > target_w:
        img = img[: min(h, target_h), : min(w, target_w), :]
        h, w, c = img.shape
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def round_up_to_multiple(x, m):
    return int(((int(x) + m - 1) // m) * m)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
clean_files = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))

train_map = {os.path.basename(p): p for p in train_files}
clean_map = {os.path.basename(p): p for p in clean_files}
common_names = sorted(set(train_map.keys()).intersection(clean_map.keys()))
if len(common_names) == 0:
    raise RuntimeError("No matching train/train_cleaned filenames found. Check paths.")

X = [imread_rgb(train_map[n]) for n in common_names[:4]]
y = [imread_rgb(clean_map[n]) for n in common_names[:4]]

print("Loaded sample shapes (X, y):")
for i in range(len(X)):
    print(i, X[i].shape, y[i].shape)




## === cell 2
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 8))
    for i in range(min(4, len(X))):
        plt.subplot(241 + i)
        fig = plt.imshow(X[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(245 + i)
        fig = plt.imshow(y[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.show()
except Exception as e:
    print("Skipping visualization due to:", repr(e))




## === cell 3
shapes_list = []
for p in train_files:
    img = imread_rgb(p)
    shapes_list.append(img.shape[:2])

heights = [s[0] for s in shapes_list]
widths = [s[1] for s in shapes_list]

reduction_factor = 4
TARGET_H = round_up_to_multiple(max(heights), reduction_factor)
TARGET_W = round_up_to_multiple(max(widths), reduction_factor)

print("Train image H range:", (min(heights), max(heights)))
print("Train image W range:", (min(widths), max(widths)))
print("Using global padded target (H,W):", (TARGET_H, TARGET_W))




## === cell 4
class Autoencoder:
    def __init__(
        self,
        dimensions_factor=2,
        layers=2,
        k=3,
        filter_size=None,
        pooling_factor=None,
        only_decoder=False,
        only_encoder=False,
        loss="mean_squared_error",
        channels=3,
    ):

        if not (only_decoder or only_encoder):
            self.channels = channels
            self.layers = layers
            self.k = k
            self.filter_size = (
                [(k, k)] * (layers * 2 + 1) if (filter_size is None) else filter_size
            )
            self.pooling_factor = (
                [2] * (layers * 2) if (pooling_factor is None) else pooling_factor
            )
            self.dimensions_factor = dimensions_factor

        if not only_decoder:
            input_img = Input(shape=(None, None, self.channels))
            x = input_img
            for i in range(self.layers):
                filters = int(
                    self.pooling_factor[i] * int(x.shape[-1]) * self.dimensions_factor
                )
                x = Conv2D(
                    filters,
                    self.filter_size[i],
                    activation="relu",
                    padding="same",
                    name="encoding_conv_" + str(i),
                )(x)
                x = MaxPooling2D(
                    (self.pooling_factor[i], self.pooling_factor[i]),
                    padding="valid",
                    name="encoding_pool_" + str(i),
                )(x)
            self.code_filters = filters
        else:
            input_img = Input(shape=(None, None, self.code_filters))
            x = input_img

        if not only_encoder:
            for i in range(self.layers, 2 * self.layers):
                filters = int(
                    int(x.shape[-1]) / (self.pooling_factor[i]) * self.dimensions_factor
                )
                x = Conv2DTranspose(
                    filters,
                    self.filter_size[i],
                    activation="relu",
                    padding="same",
                    name="decoding_conv_" + str(i),
                )(x)
                x = UpSampling2D(
                    (self.pooling_factor[i], self.pooling_factor[i]),
                    name="decoding_pool_" + str(i),
                )(x)

            x = Conv2D(
                self.channels,
                (self.filter_size[-1]),
                activation="sigmoid",
                padding="same",
                name="decoded",
            )(x)

        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer="adam", loss=loss, metrics=["mse"])

        if not (only_decoder or only_encoder):
            self.model = autoencoder
        else:
            return autoencoder

    def _build_encoder_model(self):
        input_img = Input(shape=(None, None, self.channels))
        x = input_img
        for i in range(self.layers):
            filters = int(
                self.pooling_factor[i] * int(x.shape[-1]) * self.dimensions_factor
            )
            x = Conv2D(
                filters,
                self.filter_size[i],
                activation="relu",
                padding="same",
                name="encoding_conv_" + str(i),
            )(x)
            x = MaxPooling2D(
                (self.pooling_factor[i], self.pooling_factor[i]),
                padding="valid",
                name="encoding_pool_" + str(i),
            )(x)
        encoder = Model(input_img, x)
        return encoder

    def _build_decoder_model(self):
        input_img = Input(shape=(None, None, self.code_filters))
        x = input_img
        for i in range(self.layers, 2 * self.layers):
            filters = int(
                int(x.shape[-1]) / (self.pooling_factor[i]) * self.dimensions_factor
            )
            x = Conv2DTranspose(
                filters,
                self.filter_size[i],
                activation="relu",
                padding="same",
                name="decoding_conv_" + str(i),
            )(x)
            x = UpSampling2D(
                (self.pooling_factor[i], self.pooling_factor[i]),
                name="decoding_pool_" + str(i),
            )(x)
        x = Conv2D(
            self.channels,
            (self.filter_size[-1]),
            activation="sigmoid",
            padding="same",
            name="decoded",
        )(x)
        decoder = Model(input_img, x)
        return decoder

    def get_encoder(self):
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not created")
        encoder = self._build_encoder_model()
        for layer in encoder.layers:
            if layer.name in [l.name for l in self.model.layers]:
                try:
                    layer.set_weights(self.model.get_layer(layer.name).get_weights())
                except Exception:
                    pass
        return encoder

    def get_decoder(self):
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not created")
        decoder = self._build_decoder_model()
        for layer in decoder.layers:
            if layer.name in [l.name for l in self.model.layers]:
                try:
                    layer.set_weights(self.model.get_layer(layer.name).get_weights())
                except Exception:
                    pass
        return decoder




## === cell 5
class Image_generator:
    """
    Ensure generator yields consistent tensor shapes across ALL batches by padding
    every image to a single global (TARGET_H, TARGET_W) rounded to reduction_factor.
    """

    def __init__(self, path, val_percentage=0.115, batch_size=4, reduction_factor=4):
        self.base_path = path
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor

        train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
        clean_files = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))

        train_map = {os.path.basename(p): p for p in train_files}
        clean_map = {os.path.basename(p): p for p in clean_files}
        self.names = sorted(set(train_map.keys()).intersection(clean_map.keys()))
        if len(self.names) == 0:
            raise RuntimeError("No matching train/train_cleaned filenames found.")

        self.train_paths = [train_map[n] for n in self.names]
        self.clean_paths = [clean_map[n] for n in self.names]

        self.X = [imread_rgb(p) for p in self.train_paths]
        self.y = [imread_rgb(p) for p in self.clean_paths]

        self.idx_train, self.idx_val = self.split_batches(val_percentage)

        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)

    def split_batches(self, val_percentage):
        idx = np.arange(len(self.X), dtype=int)
        np.random.shuffle(idx)
        num_full = (len(idx) // self.batch_size) * self.batch_size
        idx = idx[:num_full]
        if num_full == 0:
            raise RuntimeError("Not enough images to form a single full batch.")
        idx_batches = idx.reshape(-1, self.batch_size)
        np.random.shuffle(idx_batches)

        samples = len(idx_batches)
        val_samples = int(samples * val_percentage)
        if val_samples < 1 and samples > 1:
            val_samples = 1
        if val_samples == 0:
            val_samples = 1

        return idx_batches[:-val_samples], idx_batches[-val_samples:]

    def check_size(self, batch_x, batch_y, axis):
        for each_axis in axis:
            while (batch_x.shape[each_axis] % self.reduction_factor) != 0:
                batch_x = np.insert(
                    batch_x, batch_x.shape[each_axis], 1.0, axis=each_axis
                )
                batch_y = np.insert(
                    batch_y, batch_y.shape[each_axis], 1.0, axis=each_axis
                )
        return batch_x, batch_y

    def _make_batch(self, batch_ids):
        bx = []
        by = []
        for i in batch_ids:
            xi = pad_to_shape(self.X[i], TARGET_H, TARGET_W, pad_value=1.0)
            yi = pad_to_shape(self.y[i], TARGET_H, TARGET_W, pad_value=1.0)
            bx.append(xi)
            by.append(yi)
        batch_x = np.stack(bx, axis=0)
        batch_y = np.stack(by, axis=0)
        batch_x, batch_y = self.check_size(batch_x, batch_y, [-2, -3])
        return batch_x, batch_y

    def get_train_batch(self):
        while True:
            batch_ids = self.idx_train[0]
            batch_x, batch_y = self._make_batch(batch_ids)
            self.idx_train = np.roll(self.idx_train, 1, axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            batch_ids = self.idx_val[0]
            batch_x, batch_y = self._make_batch(batch_ids)
            self.idx_val = np.roll(self.idx_val, 1, axis=0)
            yield batch_x, batch_y




## === cell 6
data_generator = Image_generator(os.getcwd(), reduction_factor=reduction_factor)
autoencoder = Autoencoder(loss="mse")

print(
    "Train steps:", data_generator.train_steps, "Val steps:", data_generator.val_steps
)

log = autoencoder.model.fit(
    x=data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=20,
    shuffle=False,
    validation_data=data_generator.get_val_batch(),
    validation_steps=data_generator.val_steps,
    verbose=1,
)




## === cell 7
try:
    import matplotlib.pyplot as plt

    plt.figure()
    for k, v in log.history.items():
        plt.plot(v, label=str(k))
    plt.legend()
    plt.show()
except Exception as e:
    print("Skipping training plot due to:", repr(e))




## === cell 8
def build_pred_map(pred_list, ids):
    """
    Build dict: image_id -> predicted grayscale (H,W) in [0,1].
    """
    pred_map = {}
    for i, each in enumerate(pred_list):
        each = np.clip(each, 0.0, 1.0)
        gray = np.mean(each, axis=-1).astype("float32")
        pred_map[str(ids[i])] = gray
    return pred_map


def to_csv_from_sample(
    pred_map, sample_sub_path=SAMPLE_SUB_PATH, out_path="submission.csv"
):
    """
    Create submission by iterating over sampleSubmission ids, ensuring exact row count/order.
    Missing pixels (if any) are filled with 1.0 (white), consistent with padding value.
    """
    import pandas as pd

    sample = pd.read_csv(sample_sub_path)
    parts = sample["id"].str.split("_", expand=True)
    img_ids = parts[0].astype(str).values
    rows = parts[1].astype(int).values - 1
    cols = parts[2].astype(int).values - 1

    values = np.empty(len(sample), dtype="float32")
    for i in range(len(sample)):
        img = pred_map.get(img_ids[i], None)
        if img is None:
            values[i] = 1.0
        else:
            r = rows[i]
            c = cols[i]
            if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
                values[i] = float(img[r, c])
            else:
                values[i] = 1.0

    sample["value"] = np.clip(values, 0.0, 1.0)
    sample.to_csv(out_path, index=False)
    return sample.shape


test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
if len(test_files) == 0:
    raise RuntimeError(f"No test images found in {TEST_DIR}")

ids = [os.path.basename(p)[:-4] for p in test_files]

predictions = []
for i, p in enumerate(test_files, start=1):
    x = imread_rgb(p)
    original_shape = x.shape

    x_pad = pad_to_shape(x, TARGET_H, TARGET_W, pad_value=1.0)
    x_in = x_pad.reshape((1,) + x_pad.shape)

    prediction = autoencoder.model.predict(x_in, verbose=0)[0]
    prediction = prediction[: original_shape[0], : original_shape[1], :]
    predictions.append(prediction)

    print(f"{i} of {len(test_files)} predictions calculated", end="\r")

print("\nSaving...")
pred_map = build_pred_map(predictions, ids)
shape_written = to_csv_from_sample(
    pred_map, sample_sub_path=SAMPLE_SUB_PATH, out_path="submission.csv"
)
print("Saved submission.csv with shape:", shape_written)




## === cell 9
try:
    encoder = autoencoder.get_encoder()
    example = imread_rgb(test_files[min(10, len(test_files) - 1)])
    example_pad = pad_to_shape(example, TARGET_H, TARGET_W, pad_value=1.0)
    ex_in = example_pad.reshape((1,) + example_pad.shape)
    filters = encoder.predict(ex_in, verbose=0)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(example)
    plt.axis("off")
    plt.show()

    print("Encoder output shape:", filters.shape)
except Exception as e:
    filters = None
    print("Skipping encoder visualization due to:", repr(e))




## === cell 10
try:
    if filters is None:
        raise RuntimeError("No filters to animate.")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure(figsize=(4, 4))
    ims = []
    max_frames = min(filters.shape[-1], 32)
    for i in range(max_frames):
        im = plt.imshow(filters[0, :, :, i], animated=True, cmap="viridis")
        plt.axis("off")
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=0)
except Exception as e:
    ani = None
    print("Skipping animation due to:", repr(e))




## === cell 11
try:
    if ani is not None:
        from IPython.display import HTML

        HTML(ani.to_jshtml())
except Exception as e:
    print("Animation display not available:", repr(e))
