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
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
        input/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
            test/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
            train/
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

# 5. Target score

0.28311

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob, csv
import numpy as np
from cv2 import imread, resize

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D

import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # avoid interactive backend issues


def get_data_root():
    candidates = [
        "/kaggle/input/denoising-dirty-documents",  # typical Kaggle kernel location
        "./kaggle/input/denoising-dirty-documents",
        "./input/denoising-dirty-documents",
        "./working/denoising-dirty-documents",
        "./denoising-dirty-documents",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Data root for denoising-dirty-documents not found.")


DATA_ROOT = get_data_root()
print("Data root resolved to:", DATA_ROOT)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_HEIGHT, IMG_WIDTH = 128, 128  # can be changed later if needed


def load_and_resize(path_pattern, limit=None, resize_flag=True):
    """Load PNG images, optionally resizing them to IMG_HEIGHT×IMG_WIDTH.
    Returns a NumPy array of shape (N, H, W, C) with float32 values in [0,1].
    If no images are found, returns an empty array with shape (0,)."""
    files = sorted(glob.glob(path_pattern))
    if limit:
        files = files[:limit]
    imgs = []
    for f in files:
        img = imread(f)
        if img is None:
            continue
        if resize_flag:
            img = resize(img, (IMG_WIDTH, IMG_HEIGHT))  # cv2 uses (w, h)
        imgs.append(img.astype("float32") / 255.0)
    if not imgs:
        return np.empty((0,))
    return np.stack(imgs, axis=0)


X_sample = load_and_resize(os.path.join(DATA_ROOT, "train/*.png"), limit=4)
y_sample = load_and_resize(os.path.join(DATA_ROOT, "train_cleaned/*.png"), limit=4)




## === cell 2
if X_sample.size and y_sample.size:
    plt.figure(figsize=(16, 8))
    for i in range(len(X_sample)):
        plt.subplot(2, len(X_sample), i + 1)
        plt.imshow(X_sample[i])
        plt.axis("off")
        plt.subplot(2, len(y_sample), i + 1 + len(X_sample))
        plt.imshow(y_sample[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 3
class Autoencoder:
    def __init__(
        self,
        dimensions_factor=2,
        layers=2,
        k=3,
        filter_size=None,
        pooling_factor=None,
        loss="mean_squared_error",
        channels=3,
    ):
        self.channels = channels
        self.layers = layers
        self.k = k
        self.filter_size = (
            [(k, k)] * (layers * 2 + 1) if filter_size is None else filter_size
        )
        self.pooling_factor = (
            [2] * (layers * 2) if pooling_factor is None else pooling_factor
        )
        self.dimensions_factor = dimensions_factor

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
            self.filter_size[-1],
            activation="sigmoid",
            padding="same",
            name="decoded",
        )(x)

        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer="adam", loss=loss, metrics=["mse"])
        self.model = autoencoder

    def get_encoder(self):
        raise NotImplementedError("Encoder extraction not required for this script.")

    def get_decoder(self):
        raise NotImplementedError("Decoder extraction not required for this script.")




## === cell 4
class Image_generator:
    def __init__(
        self, base_path, val_percentage=0.115, batch_size=4, reduction_factor=4
    ):
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor

        self.y = load_and_resize(os.path.join(DATA_ROOT, "train_cleaned/*.png"))
        self.X = load_and_resize(os.path.join(DATA_ROOT, "train/*.png"))

        if self.X.size == 0 or self.y.size == 0:
            raise RuntimeError("Training images could not be loaded. Check paths.")

        self.idx_train, self.idx_val = self.split_batches(val_percentage)
        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)

    def split_batches(self, val_percentage):
        shape = self.X[0].shape
        all_idx = np.arange(len(self.X))
        np.random.shuffle(all_idx)
        batches = np.array_split(all_idx, max(1, len(all_idx) // self.batch_size))
        batches = [b for b in batches if len(b) == self.batch_size]
        np.random.shuffle(batches)
        total = len(batches)
        val_cnt = int(total * val_percentage)
        return np.array(batches[:-val_cnt]), np.array(batches[-val_cnt:])

    def check_size(self, batch_x, batch_y):
        for axis in (-3, -2):  # height, width
            while batch_x.shape[axis] % self.reduction_factor != 0:
                pad_shape = list(batch_x.shape)
                pad_shape[axis] = 1
                batch_x = np.concatenate(
                    [batch_x, np.ones(pad_shape, dtype=batch_x.dtype)], axis=axis
                )
                batch_y = np.concatenate(
                    [batch_y, np.ones(pad_shape, dtype=batch_y.dtype)], axis=axis
                )
        return batch_x, batch_y

    def get_train_batch(self):
        while True:
            batch_indices = self.idx_train[0]
            batch_x = np.stack(self.X[batch_indices])
            batch_y = np.stack(self.y[batch_indices])
            batch_x, batch_y = self.check_size(batch_x, batch_y)
            self.idx_train = np.roll(self.idx_train, -1, axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            batch_indices = self.idx_val[0]
            batch_x = np.stack(self.X[batch_indices])
            batch_y = np.stack(self.y[batch_indices])
            batch_x, batch_y = self.check_size(batch_x, batch_y)
            self.idx_val = np.roll(self.idx_val, -1, axis=0)
            yield batch_x, batch_y




## === cell 5
np.random.seed(42)
data_generator = Image_generator(os.getcwd())
autoencoder = Autoencoder(loss="mse")




## === cell 6
log = autoencoder.model.fit(
    data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=5,  # modest number of epochs for quick turnaround
    validation_data=data_generator.get_val_batch(),
    validation_steps=data_generator.val_steps,
    shuffle=False,
    verbose=2,
)




## === cell 7
plt.figure()
for k, v in log.history.items():
    plt.plot(v, label=str(k))
plt.legend()
plt.show()




## === cell 8
def to_csv(npdata, ids, outfile="submission.csv"):
    """Write predictions to the Kaggle submission format."""
    with open(outfile, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["id", "value"])
        for i, img in enumerate(npdata):
            rows, cols, _ = img.shape
            for r in range(rows):
                for c in range(cols):
                    pid = f"{ids[i]}_{r+1}_{c+1}"
                    val = img[r, c, :].mean()
                    csvwriter.writerow([pid, f"{val:.6f}"])


def load_test_images(path_pattern):
    """Load test PNGs while preserving original dimensions (no resizing)."""
    files = sorted(glob.glob(path_pattern))
    imgs = []
    for f in files:
        img = imread(f)
        if img is None:
            continue
        imgs.append(img.astype("float32") / 255.0)
    return imgs


X_test_files = sorted(glob.glob(os.path.join(DATA_ROOT, "test/*.png")))
ids = [os.path.splitext(os.path.basename(p))[0] for p in X_test_files]

X_test = load_test_images(os.path.join(DATA_ROOT, "test/*.png"))

predictions = []
for idx, x in enumerate(X_test):
    pred = autoencoder.model.predict(x[np.newaxis, ...], verbose=0)[0]
    predictions.append(pred)
    print(f"Processed {idx + 1}/{len(X_test)}", end="\r")
print("\nSaving submission...")
to_csv(np.array(predictions), ids, outfile="submission.csv")
print("Submission saved as submission.csv")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2302946655.py in <cell line: 0>()
     36     print(f"Processed {idx + 1}/{len(X_test)}", end="\r")
     37 print("\nSaving submission...")
---> 38 to_csv(np.array(predictions), ids, outfile="submission.csv")
     39 print("Submission saved as submission.csv")
     40 

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (29,) + inhomogeneous part.

## === cell 9
if len(X_test) > 0 and len(predictions) > 0:
    example_idx = 0
    example_input = X_test[example_idx]
    example_output = predictions[example_idx]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Noisy Input")
    plt.imshow(example_input)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Denoised Output")
    plt.imshow(example_output)
    plt.axis("off")
    plt.show()
