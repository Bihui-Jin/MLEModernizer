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
Detect functional tissue units (FTUs) across different tissue preparation pipelines. An FTU is defined as a "three-dimensional block of cells centered around a capillary, such that each cell in this block is within diffusion distance from any other cell in the same block".

## Metric
Dice coefficient.

## Submission Format
Use run-length encoding on the pixel values. Submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
img,pixels\
1,1 1 5 1\
2,1 1\
3,1 1\
etc.
```

## Dataset
The training set includes annotations in both RLE-encoded and unencoded (JSON) forms. The annotations denote segmentations of glomeruli.

Both the training and public test sets also include anatomical structure segmentations. They are intended to help you identify the various parts of the tissue.

### File structure
The JSON files are structured as follows, with each feature having:

-   A `type` (`Feature`) and object type `id` (`PathAnnotationObject`). Note that these fields are the same between all files and do not offer signal.
-   A `geometry` containing a `Polygon` with `coordinates` for the feature's enclosing volume
-   Additional `properties`, including the name and color of the feature in the image.
-   The `IsLocked` field is the same across file types (locked for glomerulus, unlocked for anatomical structure) and is not signal-bearing.

Note that the objects themselves do NOT have unique IDs. The expected prediction for a given image is an RLE-encoded mask containing ALL objects in the image. The mask, as mentioned in the Evaluation page, should be binary when encoded - with `0` indicating the lack of a masked pixel, and `1` indicating a masked pixel.

`train.csv` contains the unique IDs for each image, as well as an RLE-encoded representation of the mask for the objects in the image.

`HuBMAP-20-dataset_information.csv` contains additional information (including anonymized patient data) about each image.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            HuBMAP-20-dataset_information.csv (16 lines)
            HuBMAP-20-dataset_information.csv.zip (987 Bytes)
            description.md (211 lines)
            sample_submission.csv (4 lines)
            sample_submission.csv.zip (238 Bytes)
            test.zip (3.3 GB)
            train.csv (13 lines)
            train.csv.zip (5.0 MB)
            train.zip (16.5 GB)
            hubmap-kidney-segmentation/
                HuBMAP-20-dataset_information.csv (16 lines)
                HuBMAP-20-dataset_information.csv.zip (987 Bytes)
                ... and 7 other files
                hubmap-kidney-segmentation/
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
            test/
                0486052bb-anatomical-structure.json (161 lines)
                0486052bb.json (11619 lines)
                ... and 8 other files
                test/
            train/
                1e2425f28-anatomical-structure.json (127 lines)
                1e2425f28.json (30956 lines)
                ... and 34 other files
                train/
        input/
            HuBMAP-20-dataset_information.csv (16 lines)
            HuBMAP-20-dataset_information.csv.zip (987 Bytes)
            description.md (211 lines)
            sample_submission.csv (4 lines)
            sample_submission.csv.zip (238 Bytes)
            test.zip (3.3 GB)
            train.csv (13 lines)
            train.csv.zip (5.0 MB)
            train.zip (16.5 GB)
            hubmap-kidney-segmentation/
                HuBMAP-20-dataset_information.csv (16 lines)
                HuBMAP-20-dataset_information.csv.zip (987 Bytes)
                ... and 7 other files
                hubmap-kidney-segmentation/
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
            test/
                0486052bb-anatomical-structure.json (161 lines)
                0486052bb.json (11619 lines)
                ... and 8 other files
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
            train/
                1e2425f28-anatomical-structure.json (127 lines)
                1e2425f28.json (30956 lines)
                ... and 34 other files
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
        working/
            hubmap-kidney-segmentation/
                HuBMAP-20-dataset_information.csv (16 lines)
                HuBMAP-20-dataset_information.csv.zip (987 Bytes)
                ... and 7 other files
                hubmap-kidney-segmentation/
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
```

-> data/HuBMAP-20-dataset_information.csv has 15 rows and 16 columns.
The columns are: image_file, width_pixels, height_pixels, anatomical_structures_segmention_file, glomerulus_segmentation_file, patient_number, race, ethnicity, sex, age, weight_kilograms, height_centimeters, bmi_kg/m^2, laterality, percent_cortex... and 1 more columns

-> data/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv has 15 rows and 16 columns.
The columns are: image_file, width_pixels, height_pixels, anatomical_structures_segmention_file, glomerulus_segmentation_file, patient_number, race, ethnicity, sex, age, weight_kilograms, height_centimeters, bmi_kg/m^2, laterality, percent_cortex... and 1 more columns

-> data/hubmap-kidney-segmentation/sample_submission.csv has 3 rows and 2 columns.
The columns are: id, predicted

-> data/hubmap-kidney-segmentation/test/0486052bb-anatomical-structure.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "integer"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/0486052bb.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "number"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/095bf7a1f-anatomical-structure.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "integer"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/095bf7a1f.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "number"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/8242609fa-anatomical-structure.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "integer"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> (stopped after 10 files for performance)

# 5. Target score

0.876686722278975

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I add the missing imports, replace the faulty test‑dataset handling with a simple iterator over the IDs from the sample submission, and adjust `make_submission` to generate a valid CSV (using empty RLE strings) without relying on the undefined `HuBMAPDatasetPreprocessing`. This fixes the NameError and ensures a submission file is produced.'

# 9. Code solution

## === cell 0
import torch
from torch import nn, optim
from torch.utils import data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.segmentation import fcn_resnet101

import json
import torch.nn.functional as F
import glob
import os
import pandas as pd
import numpy as np
from skimage import io
import csv
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import binary_dilation

""" Parts of the U‑Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x.clone())


class Model(nn.Module):
    """U‑Net encoder‑decoder."""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, X):
        x1 = self.inc(X)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def collate_fn(x):
    x = list(filter(lambda x: x[-1] != -1, x))
    if x:
        return data.dataloader.default_collate(x)
    else:
        return []


def get_test_ids():
    """
    Yields image identifiers present in the sample submission file.
    """
    submission_path = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    with open(submission_path, newline="") as sub_file:
        reader = csv.reader(sub_file)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            yield image_id


def find_image_path(image_id, base_dir):
    """
    Return the full path to a test image given its ID.
    Supports both .tiff and .tif extensions.
    """
    for ext in (".tiff", ".tif"):
        candidate = os.path.join(base_dir, f"{image_id}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def rle_encode(mask):
    """
    Convert a binary mask to run‑length encoding as required by the competition.
    The mask is flattened column‑wise (Fortran order) because pixels are numbered
    top‑to‑bottom then left‑to‑right.
    """
    pixels = mask.T.flatten()  # column‑major order
    padded = np.concatenate([[0], pixels, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decode a run‑length encoded string into a binary mask.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def dice_score(pred, true):
    """
    Compute Dice coefficient for two binary masks.
    """
    pred = pred.astype(np.bool_)
    true = true.astype(np.bool_)
    intersection = np.logical_and(pred, true).sum()
    if pred.sum() + true.sum() == 0:
        return 1.0
    return 2.0 * intersection / (pred.sum() + true.sum())


def estimate_threshold(model, device, num_samples=30):
    """
    Run a small validation loop on a few training images to find a threshold
    that yields the best Dice score. Returns the best threshold (float).
    """
    meta_path = (
        "/kaggle/input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
    )
    train_csv = "/kaggle/input/hubmap-kidney-segmentation/train.csv"
    meta_df = pd.read_csv(meta_path)
    train_df = pd.read_csv(train_csv)

    id_to_shape = {
        os.path.splitext(row["image_file"])[0]: (
            row["height_pixels"],
            row["width_pixels"],
        )
        for _, row in meta_df.iterrows()
    }
    id_to_rle = dict(zip(train_df["id"], train_df["encoding"]))

    train_img_dir = "/kaggle/input/hubmap-kidney-segmentation/train"
    img_paths = glob.glob(os.path.join(train_img_dir, "*.*"))  # any extension
    if not img_paths:
        return 0.5  # fallback

    thresholds = np.arange(0.3, 0.71, 0.05)
    best_thr = 0.5
    best_score = -1.0

    for img_path in img_paths[:num_samples]:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        if img_id not in id_to_shape or img_id not in id_to_rle:
            continue
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

        true_mask = rle_decode(id_to_rle[img_id], id_to_shape[img_id])

        for thr in thresholds:
            pred_mask = (probs > thr).astype(np.uint8)
            score = dice_score(pred_mask, true_mask)
            if score > best_score:
                best_score = score
                best_thr = thr

    return best_thr


def _process_single(image_path, model, device, threshold):
    """Helper for parallel inference; returns (image_id, rle)."""
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    img = io.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        pred_mask = (probs > threshold).astype(np.uint8)

    pred_mask = binary_dilation(pred_mask, structure=np.ones((3, 3))).astype(np.uint8)

    rle = rle_encode(pred_mask)
    return image_id, rle


def make_submission(model, threshold):
    """
    Generates a submission.csv using the (potentially pretrained) model.
    Parallelises inference over available CPU cores to stay within time limits.
    """
    dev = torch.device("cpu")
    model.to(dev)
    model.eval()

    test_image_dir = "/kaggle/input/hubmap-kidney-segmentation/test"
    names, preds = [], []

    for image_id in get_test_ids():
        img_path = find_image_path(image_id, test_image_dir)
        if img_path is None:
            names.append(image_id)
            preds.append("")
            continue

        _, rle = _process_single(img_path, model, dev, threshold)
        names.append(image_id)
        preds.append(rle)

    df = pd.DataFrame({"id": names, "predicted": preds})
    submission_path = "/kaggle/working/submission.csv"
    df.to_csv(submission_path, index=False)
    print("submission.csv created with", len(df), "rows (threshold =", threshold, ")")


def find_checkpoint(root_dirs=None):
    """
    Search for the first *.pkl checkpoint under the given directories.
    Returns the full path if found, otherwise None.
    """
    if root_dirs is None:
        root_dirs = [
            "/kaggle/input/hubmapmodel",
            "/kaggle/input/hubmap-kidney-segmentation",
        ]
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(".pkl"):
                    return os.path.join(dirpath, f)
    return None


torch.set_num_threads(os.cpu_count() or 1)

model = Model()

checkpoint_path = find_checkpoint()
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading model weights from {checkpoint_path}")
    try:
        data_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if "model_state_dict" in data_dict:
            state_dict = data_dict["model_state_dict"]
        elif "state_dict" in data_dict:
            state_dict = data_dict["state_dict"]
        else:
            state_dict = data_dict
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(
            f"Failed to load checkpoint due to {e}. Proceeding with an untrained model."
        )
else:
    print("No pretrained checkpoint found – proceeding with an untrained model.")

device = torch.device("cpu")
opt_thr = estimate_threshold(model, device)
print(f"Using estimated threshold: {opt_thr}")

make_submission(model, opt_thr)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/130587520.py in <cell line: 0>()
    360 
    361 device = torch.device("cpu")
--> 362 opt_thr = estimate_threshold(model, device)
    363 print(f"Using estimated threshold: {opt_thr}")
    364 

/tmp/ipykernel_55/130587520.py in estimate_threshold(model, device, num_samples)
    243         if img_id not in id_to_shape or img_id not in id_to_rle:
    244             continue
--> 245         img = io.imread(img_path)
    246         if img.ndim == 2:
    247             img = np.stack([img] * 3, axis=-1)

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in fixed_func(*args, **kwargs)
    326                     kwargs[self.new_name] = deprecated_value
    327 
--> 328             return func(*args, **kwargs)
    329 
    330         if self.modify_docstring and func.__doc__ is not None:

/usr/local/lib/python3.11/dist-packages/skimage/io/_io.py in imread(fname, as_gray, plugin, **plugin_args)
     80 
     81     with file_or_url_context(fname) as fname, _hide_plugin_deprecation_warnings():
---> 82         img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
     83 
     84     if not hasattr(img, 'ndim'):

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in wrapped(*args, **kwargs)
    536             stacklevel = 1 + self.get_stack_length(func) - stack_rank
    537             warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
--> 538             return func(*args, **kwargs)
    539 
    540         # modify docstring to display deprecation warning

/usr/local/lib/python3.11/dist-packages/skimage/io/manage_plugins.py in call_plugin(kind, *args, **kwargs)
    252             raise RuntimeError(f'Could not find the plugin "{plugin}" for {kind}.')
    253 
--> 254     return func(*args, **kwargs)
    255 
    256 

/usr/local/lib/python3.11/dist-packages/skimage/io/_plugins/imageio_plugin.py in imread(*args, **kwargs)
      9 @wraps(imageio_imread)
     10 def imread(*args, **kwargs):
---> 11     out = np.asarray(imageio_imread(*args, **kwargs))
     12     if not out.flags['WRITEABLE']:
     13         out = out.copy()

/usr/local/lib/python3.11/dist-packages/imageio/v3.py in imread(uri, index, plugin, extension, format_hint, **kwargs)
     51         call_kwargs["index"] = index
     52 
---> 53     with imopen(uri, "r", **plugin_kwargs) as img_file:
     54         return np.asarray(img_file.read(**call_kwargs))
     55 

/usr/local/lib/python3.11/dist-packages/imageio/core/imopen.py in imopen(uri, io_mode, plugin, extension, format_hint, legacy_mode, **kwargs)
    279 
    280     request.finish()
--> 281     raise err_type(err_msg)

OSError: Could not find a backend to open `/kaggle/input/hubmap-kidney-segmentation/train/4ef6695ce.json`` with iomode `r`.

## === cell 1
"""
from skimage import io
import matplotlib.pyplot as plt
image = io.imread('/kaggle/input/compressedhubmap/train/compressed-0486052bb.tiff')
"""



## === cell 2
"""
from skimage import io
import torch
import matplotlib.pyplot as plt
subimage = image[500:1500,2500:3500]
f,subplots = plt.subplots(1, 3)
subplots[0].imshow(subimage)
activation = {} # dictionary to store the activation of a layer
torch.cuda.empty_cache()

def create_hook(name):
    def hook(m, i, o):
        # copy the output of the given layer
        activation[name] = o

    return hook

model = Model()
data_dict = torch.load('../input/hubmapmodel/model (14).pkl', map_location=torch.device('cpu'))
model.load_state_dict(data_dict['model_state_dict'])
input = torch.Tensor(subimage).permute(2,0,1).unsqueeze(0)
prediction = model(input)

pred = prediction.detach().cpu().squeeze()
subplots[1].imshow(pred)
model_weights = []
conv_layers = []

def rec(model):
    global model_weights, conv_layers
    if isinstance(model, nn.Conv2d):
        model_weights.append(model.weight)
        conv_layers.append(model)
        return
    if isinstance(model, nn.Upsample):
        conv_layers.append(model)
    for child in model.children():
        rec(child)

rec(model)
print(f"Total convolutional layers: {len(conv_layers)}")
print(f"Total layers: {len(model_weights)}")
for weight, conv in zip(model_weights, conv_layers):
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

results = [conv_layers[0](input)]
for i in range(1, len(conv_layers)):
    res = conv_layers[i](results[-1])
    results.append(res)

for num_layer, out in enumerate(results):
    plt.figure(figsize=(30, 30))
    layer_viz = out[0].cpu().numpy()
    for i in range(min(64, layer_viz.shape[0])):
        plt.subplot(8, 8, i + 1)
        plt.imshow(layer_viz[i], cmap='gray')
        plt.axis("off")
    plt.savefig(f"layer_{num_layer}.png")
    plt.close()
"""



## === cell 3
"""
import csv
import os
import pandas as pd
import numpy as np

test_csv = '../input/hubmapmodel/submission (3).csv'
original_csv = '../input/hubmap-kidney-segmentation/train.csv'
ds_metadata = '../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv'

ds_meta = pd.read_csv(ds_metadata)
test_value = pd.read_csv(test_csv)
original_value = pd.read_csv(original_csv)


def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


image_to_shape = {}
image_to_result = {}
for _, row in ds_meta.iterrows():
    image_to_shape[os.path.splitext(row['image_file'])[0]] = (row['width_pixels'], row['height_pixels'])

for _, row in original_value.iterrows():
    image_to_result[row['id']] = row['encoding']

for _, row in test_value.iterrows():
    img_id = row['id']
    if img_id in image_to_shape and img_id in image_to_result:
        original = rle_decode(image_to_result[img_id], image_to_shape[img_id])
        result = rle_decode(row['encoding'], image_to_shape[img_id])
"""
