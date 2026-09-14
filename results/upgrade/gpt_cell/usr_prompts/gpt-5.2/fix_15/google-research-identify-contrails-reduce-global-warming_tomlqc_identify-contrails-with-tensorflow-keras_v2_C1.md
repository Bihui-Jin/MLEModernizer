# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 1
%pwd


## === cell 2

import math
import pathlib
import random
import shutil

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 5
class ABI:
    bands = {name: idx for idx, name in enumerate([
        '08', '09', '10', '11', '12', '13', '14', '15', '16'])}
    colors = {name: idx for idx, name in enumerate([
        'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black'])}


## === cell 6
N_TIMES_BEFORE = 4
N_TIMES_AFTER = 3


## === cell 7
WORK_DIR = '/kaggle/working'  # preserved if notebook is saved
TEMP_DIR = '/kaggle/temp'  # just during current session

DATA_DIR = '/kaggle/input/google-research-identify-contrails-reduce-global-warming'

class Paths:
    train = os.path.join(DATA_DIR, 'train')
    valid = os.path.join(DATA_DIR, 'validation')
    test = os.path.join(DATA_DIR, 'test')


## === cell 8
DRAW = False


## === cell 9
train_ids = os.listdir(Paths.train)
valid_ids = os.listdir(Paths.valid)
test_ids = os.listdir(Paths.test)
print(len(train_ids), len(valid_ids), len(test_ids))


## === cell 10
pixel_mask = np.load(os.path.join(DATA_DIR, 'train', train_ids[3], 'human_pixel_masks.npy'))
print(pixel_mask.shape)
print(pixel_mask.min(), pixel_mask.max())


## === cell 11
sample_ids = train_ids


## === cell 12
def plot_bands_over_time(sample_id, split_dir):
    """
    
    Adapted from:
    https://www.kaggle.com/code/pranavnadimpali/comprehensive-eda-submission
    
    Args: 
        sample_id(str): The id of the example i.e. '1000216489776414077'
        split_dir(str): The split directoryu i.e. 'test', 'train', 'val'
    """
    fig, axs = plt.subplots(8, len(ABI.bands), figsize=(16, 16)) 

    for band, j in ABI.bands.items():
        img = np.load(DATA_DIR + f"/{split_dir}/{sample_id}/band_{band}.npy")
        for i in range(8):
            axs[i, j].imshow(img[..., i]) 
            axs[i, j].set_title(f"Band {band}\nTime Step {i+1}") 

    plt.tight_layout()  
    plt.show()
    
if DRAW:
    plot_bands_over_time(sample_ids[3], 'train')


## === cell 13
def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

def get_ash_colors(sample_id, split_dir):
    """
    Based on bands: 11, 14, 15
    
    Args:
        sample_id(str): The id of the example i.e. '1000216489776414077'
        split_dir(str): The split directoryu i.e. 'test', 'train', 'val'
    """
    band15 = np.load(DATA_DIR + f"/{split_dir}/{sample_id}/band_15.npy")
    band14 = np.load(DATA_DIR + f"/{split_dir}/{sample_id}/band_14.npy")
    band11 = np.load(DATA_DIR + f"/{split_dir}/{sample_id}/band_11.npy")

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    ash_colors = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    
    return ash_colors


## === cell 14
def get_individual_mask(sample_id, split_dir):
    masks_path = DATA_DIR + f"/{split_dir}/{sample_id}/human_individual_masks.npy"
    pixel_mask = np.load(masks_path)
    return pixel_mask


## === cell 15
def get_pixel_mask(sample_id, split_dir):
    masks_path = DATA_DIR + f"/{split_dir}/{sample_id}/human_pixel_masks.npy"
    pixel_mask = np.load(masks_path)
    return pixel_mask


## === cell 16
def plot_ash_colors(sample_id, split_dir, plot, time_step=4):

    ash_colors = get_ash_colors(sample_id, split_dir)
    img = ash_colors[..., time_step] # 5th image corresponds to ground truth
    
    ground_truth = get_pixel_mask(sample_id, split_dir)
    
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(16, 8))

        axs[0].imshow(img)
        axs[0].set_title("Ash Color Image")

        axs[1].imshow(ground_truth)
        axs[1].set_title("Ground Truth")

        axs[2].imshow(img)
        axs[2].imshow(ground_truth, cmap='Reds', alpha=.3, interpolation='none')
        axs[2].set_title('Contrail mask on ash color image')


        plt.tight_layout() 
        plt.show()

    return img
    
if DRAW:
    plot_ash_colors(sample_ids[3], 'train', True)


## === cell 17
def plot_three_bands(sample_id, split_dir, bands, timestep=4):
    """
    
    Adapted from:
    https://www.kaggle.com/code/pranavnadimpali/comprehensive-eda-submission
    
    Args: 
        sample_id(str): The id of the example i.e. '1000216489776414077'
        split_dir(str): The split directoryu i.e. 'test', 'train', 'val'
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 16)) 

    for j, band in enumerate(bands):
        img = np.load(DATA_DIR + f"/{split_dir}/{sample_id}/band_{band}.npy")
        axs[j].imshow(img[..., timestep]) 
        axs[j].set_title(f"Band {band}\nTime Step {timestep+1}") 

    plt.tight_layout()  
    plt.show()
    
if DRAW:
    plot_three_bands(sample_ids[3], 'train', ['11', '14', '15'])
    plot_three_bands(sample_ids[3], 'train', ['08', '09', '10'])
    plot_three_bands(sample_ids[3], 'train', ['12', '13', '16'])


## === cell 18
ash_path = pathlib.Path(os.path.join(TEMP_DIR, 'ash_colors_images'))
ash_path.mkdir(exist_ok=True, parents=True)

ash_path.resolve()  # get absolute path


## === cell 20
split_dir = 'train'


if False:

    print('convert to ash colors')
    for sample_id in tqdm(train_ids[:100]):

        ash_colors = get_ash_colors(sample_id, split_dir)
    

if False:
    print('convert to ash colors and write')
    for sample_id in tqdm(train_ids[:100]):

        ash_colors = get_ash_colors(sample_id, split_dir)

        ash_colors = ash_colors.astype(np.float16)

        image_path = ash_path/f"{sample_id}.npy"
        np.save(str(image_path), ash_colors)
        
        
if False:
    print('read')
    for sample_id in tqdm(train_ids[:100]):

        image_path = ash_path/f"{sample_id}.npy"
        np.load(str(image_path))
    


## === cell 21
import os

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as backend


## === cell 22
tf.__version__


## === cell 23
SEED = 42


## === cell 24
class Config:
    
    img_size = (256, 256)
    
    train = True
    
    num_epochs = 1  # 10
    num_classes = 1
    batch_size = 32
    
    warmup = 0
    lr = 3e-4

    seed = SEED


## === cell 25

keras.utils.set_random_seed(Config.seed)



## === cell 26
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))


    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    previous_block_activation = x  # Set aside residual

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
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


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
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


keras.backend.clear_session()

model = get_model(Config.img_size, Config.num_classes)
model.summary()


## === cell 27
class AshColorSingleFrames(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, sample_ids, split_dir, n_samples=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.split_dir = split_dir
        self.sample_ids = sample_ids[:n_samples]

    def __len__(self):
        return math.ceil(len(self.sample_ids) / self.batch_size)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_sample_ids = self.sample_ids[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, sample_id in enumerate(batch_sample_ids):
            img = get_ash_colors(sample_id, self.split_dir)
            x[j] = img[..., N_TIMES_BEFORE]

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        if self.split_dir != 'test':
            for j, sample_id in enumerate(batch_sample_ids):
                img = get_pixel_mask(sample_id, self.split_dir)
                y[j] = img
        
        return x, y


## === cell 28
train_set = AshColorSingleFrames(Config.batch_size, Config.img_size, train_ids, 'train', n_samples=500)
print('number of batches:', len(train_set))


## === cell 29
valid_set = AshColorSingleFrames(Config.batch_size, Config.img_size, valid_ids, 'validation', n_samples=100)
print('number of batches:', len(valid_set))


## === cell 30
test_set = AshColorSingleFrames(Config.batch_size, Config.img_size, test_ids, 'test')
print('number of batches:', len(test_set))


## === cell 31
train_set[0][0].shape, train_set[0][1].shape


## === cell 32
def dice_coef(y_true, y_pred, threshold=0.5, smooth=0.001):
    y_true_f = backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


## === cell 33
sample_id = train_ids[3]

merged_mask = get_pixel_mask(sample_id, "train")
indiv_masks = get_individual_mask(sample_id, "train")

print(dice_coef(tf.convert_to_tensor(merged_mask), tf.convert_to_tensor(merged_mask)))

n_individual = indiv_masks.shape[-1]
for idv in range(n_individual):
    print(
        dice_coef(
            tf.convert_to_tensor(merged_mask),
            tf.convert_to_tensor(indiv_masks[..., idv]),
        )
    )


## === cell 34
decay_steps = 1000
initial_learning_rate = 0
warmup_steps = 1000
target_learning_rate = 0.1
scheduler = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps
)


## === cell 35
model.compile(optimizer="rmsprop", loss=dice_loss, metrics=[dice_coef])

callbacks = [
    keras.callbacks.ModelCheckpoint("contrails-unet.h5", save_best_only=True)
]

model.fit(train_set, epochs=Config.num_epochs, validation_data=valid_set, callbacks=callbacks)


## === cell 37
class AshColorSingleFrames(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, sample_ids, split_dir, n_samples=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.split_dir = split_dir

        if isinstance(sample_ids, (list, tuple, np.ndarray)):
            ids = list(sample_ids)
        else:
            ids = [sample_ids]

        self.sample_ids = ids if n_samples is None else ids[:n_samples]

    def __len__(self):
        return math.ceil(len(self.sample_ids) / self.batch_size)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_sample_ids = self.sample_ids[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, sample_id in enumerate(batch_sample_ids):
            img = get_ash_colors(sample_id, self.split_dir)
            x[j] = img[..., N_TIMES_BEFORE]

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        if self.split_dir != "test":
            for j, sample_id in enumerate(batch_sample_ids):
                img = get_pixel_mask(sample_id, self.split_dir)
                y[j] = img

        return x, y


## === cell 38
if "predictions" not in globals():
    predictions = model.predict(test_set, verbose=0)

len(predictions)


## --- ERROR in cell 38, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mUnknownError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/414172570.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# Generate predictions using the already-defined trained model and test_set generator.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mif[0m [0;34m"predictions"[0m [0;32mnot[0m [0;32min[0m [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mpredictions[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mtest_set[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mlen[0m[0;34m([0m[0mpredictions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36mraise_from_not_ok_status[0;34m(e, name)[0m
[1;32m   6000[0m [0;32mdef[0m [0mraise_from_not_ok_status[0m[0;34m([0m[0me[0m[0;34m,[0m [0mname[0m[0;34m)[0m [0;34m->[0m [0mNoReturn[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6001[0m   [0me[0m[0;34m.[0m[0mmessage[0m [0;34m+=[0m [0;34m([0m[0;34m" name: "[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mname[0m [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0;34m""[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6002[0;31m   [0;32mraise[0m [0mcore[0m[0;34m.[0m[0m_status_to_exception[0m[0;34m([0m[0me[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m  [0;31m# pylint: disable=protected-access[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6003[0m [0;34m[0m[0m
[1;32m   6004[0m [0;34m[0m[0m

[0;31mUnknownError[0m: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/google-research-identify-contrails-reduce-global-warming/test/test/band_15.npy'
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

  File "/tmp/ipykernel_11/2019322289.py", line 20, in __getitem__
    img = get_ash_colors(sample_id, self.split_dir)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/821513533.py", line 17, in get_ash_colors
    band15 = np.load(DATA_DIR + f"/{split_dir}/{sample_id}/band_15.npy")
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/google-research-identify-contrails-reduce-global-warming/test/test/band_15.npy'


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 39
def rle_encode(x, fg_val=1):
    """
    Args:
        x:  numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encoding as list
    """

    dots = np.where(
        x.T.flatten() == fg_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """
    Converts list to a string representation
    Empty list returns '-'
    """
    if x: # non-empty list
        s = str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        s = '-'
    return s
