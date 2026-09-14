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

0.9005890780432344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import cv2
import csv
import json
import glob
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision

from scipy.ndimage import rotate as sp_rotate
from skimage import io

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


def crop_center(img, cropx, cropy):
    y, x, z = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


class HuBMAPDatasetPreprocessing(Dataset):
    """
    Bugfixes:
    1) OpenCV cv2.imread can crash on very large TIFFs due to CV_IO_MAX_IMAGE_PIXELS.
       Use skimage.io.imread as primary loader in this environment, fallback to cv2 if needed.
    2) Fix polygon rasterization: directly fill into the full-size mask with cv2.fillPoly.
    Core logic preserved: tiling (sz=256*reduce), padding, downscale (reduce=4 if not compressed), normalization, and jitter.
    """

    def __init__(
        self, image, category="train", use_compressed=False, apply_jitter=False
    ):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category
        self.reduce = 4 if not use_compressed else 1
        self.apply_jitter = apply_jitter

        self._img = None  # loaded full image (H,W,3) uint8
        self._mask_full = None  # loaded full mask (H,W) bool for train only

        self._get_image()

    def _imread_tiff(self, file_path):
        img = None
        try:
            img = io.imread(file_path)
        except Exception:
            img = None

        if img is None:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {file_path}")

        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] >= 3:
            pass
        return img

    def _get_image(self):
        if self._img is not None:
            return self._img

        file_path = os.path.join(self.folder, self.category, f"{self.image}.tiff")
        img = self._imread_tiff(file_path)

        self._img = img

        if self._img.ndim == 2:
            self._img = np.repeat(self._img[..., None], 3, axis=2)
        elif self._img.ndim == 3 and self._img.shape[2] == 4:
            self._img = self._img[..., :3]
        elif self._img.ndim == 3 and self._img.shape[2] > 4:
            self._img = self._img[..., :3]

        if self._img.dtype != np.uint8:
            if self._img.max() <= 255:
                self._img = self._img.astype(np.uint8)
            else:
                self._img = (self._img / self._img.max() * 255.0).astype(np.uint8)

        self.shape = (self._img.shape[0], self._img.shape[1])  # (H,W)

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        if self.category == "train":
            self._mask_full = np.zeros(self.shape, dtype=np.uint8)
            ann_path = os.path.join(
                self.mask_folder, self.category, f"{self.image}.json"
            )
            features = json.load(open(ann_path, "r"))
            for f in features:
                geom = f.get("geometry", {})
                if geom.get("type") != "Polygon":
                    continue
                coords = geom.get("coordinates", [])
                if not coords:
                    continue
                ring = coords[0]
                if len(ring) < 3:
                    continue
                poly = np.array(ring, dtype=np.float32)
                poly = np.round(poly).astype(np.int32)
                cv2.fillPoly(self._mask_full, [poly.reshape(-1, 1, 2)], 1)
            self._mask_full = self._mask_full.astype(bool)

        return self._img

    def close(self):
        self._img = None
        self._mask_full = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return int(self.n0max * self.n1max)

    def __getitem__(self, idx):
        img_full = self._get_image()
        H, W = self.shape

        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, H)
        p10, p11 = max(0, y0), min(y0 + self.sz, W)

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_full[
            p00:p01, p10:p11, :
        ]

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask = np.zeros((self.sz, self.sz, 1), np.uint8)
            mfull = self._mask_full.astype(np.uint8)
            mask[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0), 0] = mfull[
                p00:p01, p10:p11
            ]
            if self.reduce != 1:
                mask = cv2.resize(
                    mask,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )
                if mask.ndim == 2:
                    mask = mask[..., None]
        else:
            mask = torch.zeros(1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        result_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        result_tensor = torchvision.transforms.functional.normalize(
            result_tensor / 255.0, mean, std
        ).float()

        if self.category == "train" and self.apply_jitter is True:
            p0 = random.random()
            p1 = random.random()
            p2 = random.random()
            if p0 <= 0.7:
                if p1 <= 0.3:
                    result_tensor = torchvision.transforms.functional.hflip(
                        result_tensor
                    )
                    mask = np.fliplr(mask)
                if p2 <= 0.3:
                    result_tensor = torchvision.transforms.functional.vflip(
                        result_tensor
                    )
                    mask = np.flipud(mask)
            else:
                angle = random.randint(-20, 20)
                result_tensor = torchvision.transforms.functional.rotate(
                    result_tensor, angle, expand=True
                )
                result_tensor = torchvision.transforms.functional.center_crop(
                    result_tensor, (256, 256)
                )
                mask = sp_rotate(mask, angle, order=0, mode="constant", cval=0.0)
                x, y, z = mask.shape
                crop = (x - 256) // 2
                leftover = (x - 2 * crop) - 256
                if crop != 0:
                    mask = mask[crop + leftover : -crop, crop + leftover : -crop]

        if self.category == "train":
            mask = mask > 0
            if (s > 40).sum() <= 1000 or img.sum() <= 1000:
                return result_tensor, mask, -1
            else:
                return result_tensor, mask, idx
        else:
            if (s > 40).sum() <= 1000 or img.sum() <= 1000:
                return result_tensor, mask, -1
            else:
                return result_tensor, mask, idx




## === cell 1
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', 'train', apply_jitter=True, use_compressed=True)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 10)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    sample = sample.permute(1,2,0)
    if idx == -1 or mask.sum() == 0:
        continue
    if c == 10:
        break
    plots[0][c].imshow(255. * sample.numpy())
    plots[0][c].set_title(f'Sample #{c}')
    plots[1][c].imshow(mask.squeeze().astype(np.uint8)*255, cmap='gray')
    plots[1][c].set_title(f'Mask #{c}')
    c += 1
"""


## === cell 2
"""
dataset_test = HuBMAPDatasetPreprocessing('0486052bb', 'test', use_compressed=True)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset_test[i]
    sample = sample.permute(1,2,0)
    if idx == -1:
        continue
    if c == 5:
        break
    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    c += 1
"""


## === cell 3
import torch
from torch import nn, optim
from torch.utils import data
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

        self.skip_conf = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x2 = self.skip_conf(x2)
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
    """Encoder-decoder architecture (unchanged)."""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512)
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




## === cell 4
import os
import gc

OUTPUT_PATH = "/kaggle/working/models"

VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(
                name, category="train", apply_jitter=False, use_compressed=True
            )
            yield dataset


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(
            image, category="train", use_compressed=True
        )
        yield dataset


def dice_loss(pred, target, device):
    numerator = 2 * torch.sum(pred.float() * target.float())
    denominator = torch.sum(pred.float() + target.float())
    return numerator / denominator


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)

        self.model = Model().to(self.device)
        self.epochs = 90
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (17).pkl"

        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []

        self.loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([5]).to(self.device)
        )

        if os.path.exists(self._original_model_path):
            data_dict = torch.load(self._original_model_path, map_location=self.device)
            self.start_positions = int(data_dict.get("epoch", 0))
            self.model.load_state_dict(data_dict["model_state_dict"])
        else:
            self.start_positions = 0

    def evaluate(self, epoch, display=False):
        val_set = get_val_set()
        numerator, denominator = 0, 0
        TP, FP, FN = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for ds in val_set:
                print(f"started processing image {ds.image}")
                loader = data.DataLoader(ds, batch_size=9, pin_memory=True)

                for batch_ndx, sample in enumerate(loader):
                    if sample is None or (
                        isinstance(sample, (list, tuple)) and len(sample) == 0
                    ):
                        continue
                    target, mask, idx = sample
                    target = target.to(self.device)
                    mask = torch.as_tensor(mask).to(self.device)
                    mask = (mask[..., :1] > 0).squeeze(3)  # (B,H,W)

                    res = self.model(target)
                    result = (res > 0).squeeze(1)  # (B,H,W)

                    for i in range(mask.shape[0]):
                        inputs = mask[i].int()
                        targets = result[i].int()
                        tp = inputs * targets
                        fp = (1 - targets) * inputs
                        fn = targets * (1 - inputs)

                        TP += tp.sum()
                        FP += fp.sum()
                        FN += fn.sum()
                        numerator += 2 * torch.sum(mask[i].float() * result[i].float())
                        denominator += torch.sum(mask[i].float() + result[i].float())

                    del mask, target, result, res
                    if self.is_cuda_available:
                        torch.cuda.empty_cache()
                    gc.collect()

        score = (numerator / denominator).item() if denominator != 0 else 0.0
        print(f"epoch {epoch} evaluation", score)
        print(f"epoch {epoch} TP={TP}, FP={FP}, FN={FN}")
        self.eval_metrics.append(
            (
                score,
                TP.item() if torch.is_tensor(TP) else TP,
                FP.item() if torch.is_tensor(FP) else FP,
                FN.item() if torch.is_tensor(FP) else FP,
            )
        )
        gc.collect()
        self.model.train()

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        print(f"start position {self.start_positions}")
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            self.model.train()

            for dataset in get_tiff_images():
                loader = None
                try:
                    loader = data.DataLoader(dataset, batch_size=12, pin_memory=True)

                    for batch_ndx, sample in enumerate(loader):
                        self.optim.zero_grad()
                        if sample is None or (
                            isinstance(sample, (list, tuple)) and len(sample) == 0
                        ):
                            continue
                        target, mask, idx = sample
                        mask = torch.as_tensor(mask)
                        mask = mask[..., :1] > 0

                        result = self.model(target.to(self.device))

                        mask = mask.to(self.device).float().permute(0, 3, 1, 2)
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()

                        del mask, target
                        if self.is_cuda_available:
                            torch.cuda.empty_cache()
                        gc.collect()

                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset, loader
                    if self.is_cuda_available:
                        torch.cuda.empty_cache()
                    gc.collect()

            self.evaluate(i)
            with open(
                os.path.join(OUTPUT_PATH, "eval_metrics.csv"), "w+", encoding="utf-8"
            ) as fw:
                fw.write(
                    "\n".join(
                        "\t".join(map(str, metric)) for metric in self.eval_metrics
                    )
                )

            print(f"saving epoch {i}")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                os.path.join(OUTPUT_PATH, f"model_{i}.pkl"),
            )
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                self._model_path,
            )


def display_predictions(tp, fp, fn, subplot):
    result = np.zeros(fp.shape)
    result += tp.cpu().numpy() * 255
    result += fn.cpu().numpy() * 64
    subplot.imshow(result)




## === cell 5
torch.cuda.empty_cache()
trainer = Trainer()



## === cell 6
from torch.utils import data


def collate_fn(x):
    x = list(filter(lambda t: t[-1] != -1, x))
    if len(x) == 0:
        return None
    return data.dataloader.default_collate(x)


def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    sub = pd.read_csv(submission_file)
    for image_id in sub["id"].tolist():
        print(f"openning {image_id}")
        dataset = HuBMAPDatasetPreprocessing(
            image_id, category="test", use_compressed=True
        )
        yield dataset


def rle_encode_less_memory(img):
    """
    img: torch/numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    Pixels are read top-to-bottom then left-to-right => transpose then flatten.
    """
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    pixels = img.T.flatten().astype(np.uint8)
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _find_best_checkpoint():
    candidates = []
    p_main = os.path.join(OUTPUT_PATH, "model.pkl")
    if os.path.exists(p_main):
        return p_main

    for p in glob.glob(os.path.join(OUTPUT_PATH, "model_*.pkl")):
        base = os.path.basename(p)
        try:
            ep = int(os.path.splitext(base)[0].split("_")[-1])
        except Exception:
            ep = -1
        candidates.append((ep, p))
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]
    return None


def make_submission(model, allow_empty_fallback=True):
    names, preds = [], []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    if model is not None:
        model = model.to(dev)
        model.eval()
    elif not allow_empty_fallback:
        raise ValueError("model is None but allow_empty_fallback is False")

    with torch.no_grad():
        for ds in get_test_dataset():
            print(ds.image, len(ds))

            if model is None:
                names.append(ds.image)
                preds.append("")
                del ds
                gc.collect()
                continue

            loader = data.DataLoader(ds, batch_size=32, collate_fn=collate_fn)

            tile_sz = ds.sz // ds.reduce
            tile_mask = torch.zeros((len(ds), tile_sz, tile_sz), dtype=torch.uint8)

            for batch_num, batch in enumerate(loader):
                if batch is None:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = batch
                image = image.to(dev)

                logits = model(image)  # (B,1,tile_sz,tile_sz)
                logits = logits.squeeze(1)  # (B,tile_sz,tile_sz)

                for i, ndx in enumerate(idx.tolist()):
                    tile_mask[ndx] = (logits[i] > 0).to(torch.uint8)

                del logits, image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            mask = (
                tile_mask.view(ds.n0max, ds.n1max, tile_sz, tile_sz)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * tile_sz, ds.n1max * tile_sz)
            )

            r0s = (ds.pad0 // 2) // ds.reduce
            r0e = (
                -((ds.pad0 - ds.pad0 // 2) // ds.reduce)
                if ds.pad0 > 0
                else mask.shape[0]
            )
            r1s = (ds.pad1 // 2) // ds.reduce
            r1e = (
                -((ds.pad1 - ds.pad1 // 2) // ds.reduce)
                if ds.pad1 > 0
                else mask.shape[1]
            )
            mask = mask[r0s:r0e, r1s:r1e]

            rle = rle_encode_less_memory(mask)
            names.append(ds.image)
            preds.append(rle)

            del tile_mask, mask, ds
            gc.collect()

    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(df), "rows")


model = Model()
ckpt_path = _find_best_checkpoint()
if ckpt_path is not None:
    print("Loading checkpoint:", ckpt_path)
    data_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"])
    else:
        model.load_state_dict(data_dict)
    make_submission(model, allow_empty_fallback=False)
else:
    print(
        "WARNING: No checkpoint found under",
        OUTPUT_PATH,
        "- writing empty-mask baseline submission.",
    )
    make_submission(None, allow_empty_fallback=True)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/2291195775.py in <cell line: 0>()
    153         "- writing empty-mask baseline submission.",
    154     )
--> 155     make_submission(None, allow_empty_fallback=True)
    156 

/tmp/ipykernel_55/2291195775.py in make_submission(model, allow_empty_fallback)
     70 
     71     with torch.no_grad():
---> 72         for ds in get_test_dataset():
     73             print(ds.image, len(ds))
     74 

/tmp/ipykernel_55/2291195775.py in get_test_dataset()
     15     for image_id in sub["id"].tolist():
     16         print(f"openning {image_id}")
---> 17         dataset = HuBMAPDatasetPreprocessing(
     18             image_id, category="test", use_compressed=True
     19         )

/tmp/ipykernel_55/3547757833.py in __init__(self, image, category, use_compressed, apply_jitter)
     50         self._mask_full = None  # loaded full mask (H,W) bool for train only
     51 
---> 52         self._get_image()
     53 
     54     def _imread_tiff(self, file_path):

/tmp/ipykernel_55/3547757833.py in _get_image(self)
     73 
     74         file_path = os.path.join(self.folder, self.category, f"{self.image}.tiff")
---> 75         img = self._imread_tiff(file_path)
     76 
     77         self._img = img

/tmp/ipykernel_55/3547757833.py in _imread_tiff(self, file_path)
     60 
     61         if img is None:
---> 62             img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
     63             if img is None:
     64                 raise FileNotFoundError(f"Failed to read image: {file_path}")

error: OpenCV(4.12.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:79: error: (-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS in function 'validateInputImageSize'


## === cell 7
"""
from skimage import io
import matplotlib.pyplot as plt
image = io.imread('/kaggle/input/hubmap-kidney-segmentation/train/0486052bb.tiff')
"""


## === cell 8
"""
(unchanged exploratory cell)
"""


## === cell 9
"""
(unchanged exploratory cell)
"""
