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


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as backend


## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 22
tf.__version__
