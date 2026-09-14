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
Identify contrails in satellite imagery.

## Metric
Global Dice coefficient. The Dice coefficient formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where X is the entire set of predicted contrail pixels for all observations in the test data and Y is the ground truth set of all contrail pixels in the test data.

## Submission Format
Use run-length encoding. For example, '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

Use a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

Empty predictions must be marked with '-' in the submission file.

The file should contain a header and have the following format:

```
record_id,encoded_pixels  
1000834164244036115,1 1 5 1  
1002653297254493116,-  
etc.
```

## Dataset 
Some key labeling guidance:
- Contrails must contain at least 10 pixels
- At some time in their life, Contrails must be at least 3x longer than they are wide
- Contrails must either appear suddenly or enter from the sides of the image
- Contrails should be visible in at least two image

A sequence of images at 10-minute intervals are provided. Each example (`record_id`) contains exactly one labeled frame.

- **train/** - the training set; each folder represents a `record_id` and contains the following data:
    - **band_{08-16}.npy**: array with size of `H x W x T`, where `T = n_times_before + n_times_after + 1`, representing the number of images in the sequence. There are `n_times_before` and `n_times_after` images before and after the labeled frame respectively. In our dataset all examples have `n_times_before=4` and `n_times_after=3`. Each band represents an infrared channel at different wavelengths and is converted to brightness temperatures based on the calibration parameters. The number in the filename corresponds to the GOES-16 ABI band number. Details of the ABI bands can be found [here](https://www.goes-r.gov/mission/ABI-bands-quick-info.html).
    - **human_individual_masks.npy**: array with size of `H x W x 1 x R`. Each example is labeled by `R` individual human labelers. `R` is not the same for all samples. The labeled masks have value either 0 or 1 and correspond to the `(n_times_before+1)`-th image in `band_{08-16}.npy`. They are available only in the training set.
    - **human_pixel_masks.npy**: array with size of `H x W x 1` containing the binary ground truth. A pixel is regarded as contrail pixel in evaluation if it is labeled as contrail by more than half of the labelers.
- **validation/** - the same as the training set, without the individual label annotations; it is permitted to use this as training data if desired
- **test/** - the test set; your objective is to identify contrails found in these records.
- **{train|validation}_metadata.json** - metadata information for each record; contains the timestamps and the projection parameters to reproduce the satellite images.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (129 lines)
            sample_submission.csv (1857 lines)
            sample_submission.csv.zip (21.4 kB)
            test.zip (26.9 GB)
            train.zip (270.6 GB)
            train_metadata.json (1 lines)
            validation.zip (27.0 GB)
            validation_metadata.json (1 lines)
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
            test/
                1006714073984511039/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                1011991214639847439/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                ... and 1855 other folders
            train/
                1000216489776414077/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                1000603527582775543/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                ... and 18672 other folders
            validation/
                1000834164244036115/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                1002653297254493116/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                ... and 1854 other folders
        input/
            description.md (129 lines)
            sample_submission.csv (1857 lines)
            sample_submission.csv.zip (21.4 kB)
            test.zip (26.9 GB)
            train.zip (270.6 GB)
            train_metadata.json (1 lines)
            validation.zip (27.0 GB)
            validation_metadata.json (1 lines)
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
            test/
                1006714073984511039/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                1011991214639847439/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                ... and 1855 other folders
            train/
                1000216489776414077/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                1000603527582775543/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                ... and 18672 other folders
            validation/
                1000834164244036115/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                1002653297254493116/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                ... and 1854 other folders
        working/
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
```

-> data/google-research-identify-contrails-reduce-global-warming/sample_submission.csv has 1856 rows and 4 columns.
The columns are: record_id, encoded_pixels, height, width

-> data/google-research-identify-contrails-reduce-global-warming/train_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/google-research-identify-contrails-reduce-global-warming/validation_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/sample_submission.csv has 1856 rows and 4 columns.
The columns are: record_id, encoded_pixels, height, width

-> data/train_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/validation_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> (stopped after 10 files for performance)

# 5. Target score

0.00344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap, math, pathlib, random, shutil
import numpy as np
import pandas as pd



## === cell 1
print("CWD:", os.getcwd())



## === cell 2
try:
    import google.protobuf as _pb

    _pb_ver = getattr(_pb, "__version__", "unknown")
    print("Detected protobuf version:", _pb_ver)
except Exception as e:
    print("Protobuf import check skipped due to:", repr(e))



## === cell 3
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt




## === cell 4
class ABI:
    bands = {
        name: idx
        for idx, name in enumerate(
            ["08", "09", "10", "11", "12", "13", "14", "15", "16"]
        )
    }
    colors = {
        name: idx
        for idx, name in enumerate(
            [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "cyan",
                "magenta",
                "yellow",
                "black",
            ]
        )
    }




## === cell 5
N_TIMES_BEFORE = 4
N_TIMES_AFTER = 3



## === cell 6
WORK_DIR = "/kaggle/working"  # preserved if notebook is saved
TEMP_DIR = "/kaggle/temp"  # just during current session

DATA_DIR = "/kaggle/input/google-research-identify-contrails-reduce-global-warming"


class Paths:
    train = os.path.join(DATA_DIR, "train")
    valid = os.path.join(DATA_DIR, "validation")
    test = os.path.join(DATA_DIR, "test")




## === cell 7
DRAW = False



## === cell 8
train_ids = sorted([str(x) for x in os.listdir(Paths.train)])
valid_ids = sorted([str(x) for x in os.listdir(Paths.valid)])
test_ids = sorted([str(x) for x in os.listdir(Paths.test)])
print(len(train_ids), len(valid_ids), len(test_ids))



## === cell 9
pixel_mask = np.load(
    os.path.join(DATA_DIR, "train", train_ids[3], "human_pixel_masks.npy")
)
print(pixel_mask.shape)
print(pixel_mask.min(), pixel_mask.max())



## === cell 10
sample_ids = train_ids




## === cell 11
def plot_bands_over_time(sample_id, split_dir):
    """
    Adapted from:
    https://www.kaggle.com/code/pranavnadimpali/comprehensive-eda-submission
    """
    fig, axs = plt.subplots(8, len(ABI.bands), figsize=(16, 16))
    for band, j in ABI.bands.items():
        img = np.load(os.path.join(DATA_DIR, split_dir, sample_id, f"band_{band}.npy"))
        for i in range(8):
            axs[i, j].imshow(img[..., i])
            axs[i, j].set_title(f"Band {band}\nTime Step {i+1}")
    plt.tight_layout()
    plt.show()


if DRAW:
    plot_bands_over_time(sample_ids[3], "train")




## === cell 12
def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)




## === cell 13
def _as_record_id(sample_id) -> str:
    return os.path.basename(str(sample_id).rstrip("/"))


def _split_dir_name(split_dir: str) -> str:
    return os.path.basename(str(split_dir).rstrip("/"))


def _npy_path(split_dir: str, sample_id: str, filename: str) -> str:
    split_name = _split_dir_name(split_dir)
    rid = _as_record_id(sample_id)
    return os.path.join(DATA_DIR, split_name, rid, filename)


def get_ash_colors(sample_id, split_dir):
    """
    Based on bands: 11, 14, 15
    """
    band15 = np.load(_npy_path(split_dir, sample_id, "band_15.npy"))
    band14 = np.load(_npy_path(split_dir, sample_id, "band_14.npy"))
    band11 = np.load(_npy_path(split_dir, sample_id, "band_11.npy"))

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    ash_colors = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return ash_colors




## === cell 14
def get_individual_mask(sample_id, split_dir):
    masks_path = _npy_path(split_dir, sample_id, "human_individual_masks.npy")
    pixel_mask = np.load(masks_path)
    return pixel_mask




## === cell 15
def get_pixel_mask(sample_id, split_dir):
    masks_path = _npy_path(split_dir, sample_id, "human_pixel_masks.npy")
    pixel_mask = np.load(masks_path)
    return pixel_mask




## === cell 16
def plot_ash_colors(sample_id, split_dir, plot, time_step=4):
    ash_colors = get_ash_colors(sample_id, split_dir)
    img = ash_colors[..., time_step]  # 5th image corresponds to ground truth
    ground_truth = get_pixel_mask(sample_id, split_dir)

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(16, 8))
        axs[0].imshow(img)
        axs[0].set_title("Ash Color Image")
        axs[1].imshow(ground_truth)
        axs[1].set_title("Ground Truth")
        axs[2].imshow(img)
        axs[2].imshow(ground_truth, cmap="Reds", alpha=0.3, interpolation="none")
        axs[2].set_title("Contrail mask on ash color image")
        plt.tight_layout()
        plt.show()
    return img


if DRAW:
    plot_ash_colors(sample_ids[3], "train", True)




## === cell 17
def plot_three_bands(sample_id, split_dir, bands, timestep=4):
    """
    Adapted from:
    https://www.kaggle.com/code/pranavnadimpali/comprehensive-eda-submission
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 16))
    for j, band in enumerate(bands):
        img = np.load(_npy_path(split_dir, sample_id, f"band_{band}.npy"))
        axs[j].imshow(img[..., timestep])
        axs[j].set_title(f"Band {band}\nTime Step {timestep+1}")
    plt.tight_layout()
    plt.show()


if DRAW:
    plot_three_bands(sample_ids[3], "train", ["11", "14", "15"])
    plot_three_bands(sample_ids[3], "train", ["08", "09", "10"])
    plot_three_bands(sample_ids[3], "train", ["12", "13", "16"])



## === cell 18
ash_path = pathlib.Path(os.path.join(TEMP_DIR, "ash_colors_images"))
ash_path.mkdir(exist_ok=True, parents=True)
print("ash_path:", ash_path.resolve())



## === cell 19
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as backend

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 20
SEED = 42


class Config:
    img_size = (256, 256)
    train = True
    num_epochs = 1  # keep as provided
    num_classes = 1
    batch_size = 32
    warmup = 0
    lr = 3e-4
    seed = SEED


keras.utils.set_random_seed(Config.seed)




## === cell 21
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    previous_block_activation = x  # residual

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


keras.backend.clear_session()
model = get_model(Config.img_size, Config.num_classes)
model.summary()




## === cell 22
class AshColorSingleFrames(keras.utils.Sequence):
    """Iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, sample_ids, split_dir, n_samples=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.split_dir = split_dir
        self.sample_ids = (
            sample_ids[:n_samples] if n_samples is not None else sample_ids
        )

    def __len__(self):
        return math.ceil(len(self.sample_ids) / self.batch_size)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_sample_ids = self.sample_ids[i : i + self.batch_size]
        bs = len(batch_sample_ids)

        x = np.zeros((bs,) + self.img_size + (3,), dtype="float32")
        for j, sample_id in enumerate(batch_sample_ids):
            img = get_ash_colors(sample_id, self.split_dir)
            x[j] = img[..., N_TIMES_BEFORE]

        y = np.zeros((bs,) + self.img_size + (1,), dtype="uint8")
        if _split_dir_name(self.split_dir) != "test":
            for j, sample_id in enumerate(batch_sample_ids):
                m = get_pixel_mask(sample_id, self.split_dir)
                y[j] = m

        return x, y




## === cell 23
train_set = AshColorSingleFrames(
    Config.batch_size, Config.img_size, train_ids, "train", n_samples=500
)
print("number of train batches:", len(train_set))

valid_set = AshColorSingleFrames(
    Config.batch_size, Config.img_size, valid_ids, "validation", n_samples=100
)
print("number of valid batches:", len(valid_set))

test_set = AshColorSingleFrames(Config.batch_size, Config.img_size, test_ids, "test")
print("number of test batches:", len(test_set))



## === cell 24
print(train_set[0][0].shape, train_set[0][1].shape)




## === cell 25
def dice_coef(y_true, y_pred, threshold=0.5, smooth=0.001):
    y_true_f = backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = backend.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth
    )
    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)




## === cell 26
sample_id = train_ids[3]
merged_mask = get_pixel_mask(sample_id, "train")
indiv_masks = get_individual_mask(sample_id, "train")

print(
    "dice(merged, merged):",
    float(
        dice_coef(tf.convert_to_tensor(merged_mask), tf.convert_to_tensor(merged_mask))
    ),
)
n_labelers = indiv_masks.shape[-1]
for idv in range(n_labelers):
    print(
        f"dice(merged, indiv {idv}):",
        float(
            dice_coef(
                tf.convert_to_tensor(merged_mask),
                tf.convert_to_tensor(indiv_masks[..., idv]),
            )
        ),
    )



## === cell 27
decay_steps = 1000
initial_learning_rate = 0
warmup_steps = 1000
target_learning_rate = 0.1
scheduler = keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps)



## === cell 28
model.compile(optimizer="rmsprop", loss=dice_loss, metrics=[dice_coef])

callbacks = [keras.callbacks.ModelCheckpoint("contrails-unet.h5", save_best_only=True)]

model.fit(
    train_set, epochs=Config.num_epochs, validation_data=valid_set, callbacks=callbacks
)



## === cell 29
predictions = model.predict(test_set, verbose=1)
print("predictions:", predictions.shape, predictions.dtype)




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_12/2681824410.py in <cell line: 0>()
      1 # Bugfix: inference now uses correct split directory and record_id path handling via _npy_path().
----> 2 predictions = model.predict(test_set, verbose=1)
      3 print("predictions:", predictions.shape, predictions.dtype)
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/google-research-identify-contrails-reduce-global-warming/test/test/band_15.npy'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py", line 248, in _finite_generator
    yield self._standardize_batch(self.py_dataset[i])
                                  ~~~~~~~~~~~~~~~^^^

  File "/tmp/ipykernel_12/2530477111.py", line 22, in __getitem__
    img = get_ash_colors(sample_id, self.split_dir)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_12/180161896.py", line 21, in get_ash_colors
    band15 = np.load(_npy_path(split_dir, sample_id, "band_15.npy"))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/google-research-identify-contrails-reduce-global-warming/test/test/band_15.npy'


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 30
def rle_encode(x, fg_val=1):
    """
    Args:
        x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encoding as list

    Pixels numbered top-to-bottom then left-to-right => flatten in Fortran order (column-major),
    implemented via x.T.flatten() on a C-contiguous (H,W) array.
    """
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """Empty list returns '-'."""
    if x:
        return str(x).replace("[", "").replace("]", "").replace(",", "")
    return "-"




## === cell 31
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

sub = sample_sub[["record_id", "encoded_pixels"]].copy()
sub["record_id"] = sub["record_id"].astype(str)

bin_masks = (predictions[..., 0] > 0.5).astype(np.uint8)

id_to_idx = {str(rid): i for i, rid in enumerate(test_ids)}

encoded = []
for rid in sub["record_id"].tolist():
    i = id_to_idx.get(rid, None)
    if i is None:
        encoded.append("-")
    else:
        rle = rle_encode(bin_masks[i])
        encoded.append(list_to_string(rle))

sub["encoded_pixels"] = encoded

out_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print("Submission shape:", sub.shape)
print("Encoded '-' count:", int((sub["encoded_pixels"] == "-").sum()))

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/4253095468.py in <cell line: 0>()
      6 
      7 # Post-process predictions to binary masks for RLE (threshold 0.5).
----> 8 bin_masks = (predictions[..., 0] > 0.5).astype(np.uint8)
      9 
     10 id_to_idx = {str(rid): i for i, rid in enumerate(test_ids)}

NameError: name 'predictions' is not defined
