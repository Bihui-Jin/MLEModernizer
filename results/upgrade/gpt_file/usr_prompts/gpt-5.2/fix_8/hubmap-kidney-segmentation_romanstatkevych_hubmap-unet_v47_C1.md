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

0.8895511033929748

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
import glob
import json
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)


def _load_tiff_rgb(path: str, prefer_lowres: bool = False) -> np.ndarray:
    """
    Robust TIFF reader for Kaggle HuBMAP whole-slide-like TIFFs.

    Returns:
    - HxWx3 uint8 in BGR order (to preserve downstream cv2.cvtColor(img, BGR2HSV) behavior).

    Fix:
    - Avoid cv2.imread on huge images (OpenCV CV_IO_MAX_IMAGE_PIXELS assertion).
    - Use tifffile as the primary loader; optionally pick a lower-resolution level/page if present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not read image (missing): {path}")

    img = None

    try:
        import tifffile as tiff

        with tiff.TiffFile(path) as tf:
            if prefer_lowres and len(tf.pages) > 1:
                page = tf.pages[-1]
                img = page.asarray()
            else:
                img = tf.asarray()

        if img is not None and img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]
            img = img[:, :, ::-1].copy()
        elif img is not None and img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        elif img is not None and img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
    except Exception:
        img = None

    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape {img.shape} for {path}")

        if img.shape[2] > 3:
            img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def _rasterize_json_to_mask(
    json_path: str, height: int, width: int, reduce: int = 1
) -> np.ndarray:
    """
    Create a single binary mask (HxW uint8 {0,1}) from the annotation list by unioning all polygons.
    """
    with open(json_path, "r") as f:
        feats = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)

    for feat in feats:
        geom = feat.get("geometry", {})
        if geom.get("type", "") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue

        ring = coords[0]
        if not ring:
            continue

        pts = np.array(ring, dtype=np.float32)  # [[x,y],...]
        if reduce != 1:
            pts[:, 0] = np.floor(pts[:, 0] / reduce)
            pts[:, 1] = np.floor(pts[:, 1] / reduce)

        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        pts_i = pts.astype(np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(mask, [pts_i], 1)

    return mask


class HuBMAPDatasetPreprocessing(Dataset):
    def __init__(self, image, category="train", use_compressed=False):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category

        self.reduce = 4 if use_compressed else 1

        self._img = None
        self._mask = None
        self._get_image_and_mask()

    def _get_image_and_mask(self):
        if self._img is not None:
            return

        if self.use_compressed:
            file_path = os.path.join(
                self.folder, self.category, f"compressed-{self.image}.tiff"
            )
            if not os.path.exists(file_path):
                file_path = os.path.join(
                    self.folder, self.category, f"{self.image}.tiff"
                )
        else:
            file_path = os.path.join(self.folder, self.category, f"{self.image}.tiff")

        img_full = _load_tiff_rgb(file_path, prefer_lowres=False)  # HxWx3 uint8 (BGR)

        if self.reduce != 1:
            img_full = cv2.resize(
                img_full,
                (img_full.shape[1] // self.reduce, img_full.shape[0] // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        self.shape = (
            img_full.shape[0],
            img_full.shape[1],
        )  # (H,W) at working resolution

        self.sz = (
            256 * 1
        )  # tiles are 256 at working resolution (consistent with ds.reduce handling downstream)
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        self._img = img_full

        if self.category == "train":
            json_path = os.path.join(
                self.mask_folder, self.category, f"{self.image}.json"
            )
            full_mask = _rasterize_json_to_mask(
                json_path,
                height=self.shape[0] * self.reduce,
                width=self.shape[1] * self.reduce,
                reduce=1,
            ).astype(np.uint8)
            if self.reduce != 1:
                full_mask = cv2.resize(
                    full_mask,
                    (self.shape[1], self.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            self._mask = full_mask

    def close(self):
        pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        self._img = None
        self._mask = None

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        self._get_image_and_mask()
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz

        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self._img[
            p00:p01, p10:p11, :
        ]

        if self.category == "train":
            mask = np.zeros((self.sz, self.sz), np.uint8)
            mask[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self._mask[
                p00:p01, p10:p11
            ]
            mask = np.repeat(mask[:, :, None], 3, axis=2)
        else:
            mask = torch.zeros(1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        result_tensor = (
            torch.from_numpy((img.astype(np.float32) / 255.0 - mean) / std)
            .float()
            .permute(2, 0, 1)
        )

        if (s > 40).sum() <= 1000 or img.sum() <= 1000:
            return result_tensor, mask, -1
        else:
            return result_tensor, mask, idx




## === cell 1
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', use_compressed=True)

fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    sample = sample.permute(1,2,0)
    
    if idx == -1 or (hasattr(mask, "sum") and mask.sum() == 0):
        continue
    if c == 5:
        break

    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    mask = [np.array(m)*255 for m in mask]
    plots[1][c].imshow(mask)
    plots[1][c].set_title(f'Sample #{c}')
    c += 1
"""


## === cell 2
"""
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset_test[i]
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
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.segmentation import fcn_resnet101

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
    """The base class for the encoder-decoder architecture."""

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
import numpy

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    x = filter(lambda x: x[-1] != -1, x)
    ls = list(x)
    if ls:
        return data.dataloader.default_collate(ls)
    else:
        return []


VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(
                os.path.splitext(bname)[0], use_compressed=True
            )
            yield dataset


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(image, use_compressed=True)
        yield dataset


def dice_loss(pred, target, device):
    numerator = 2 * torch.sum(pred.float() * target.float())
    denominator = torch.sum(pred.float() + target.float())
    return numerator / denominator


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        device=None,
        alpha=0.7,
        beta=0.3,
        gamma=0.75,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + (self.alpha) * FP + (self.beta) * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


import collections


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)
        self.model = Model()
        self.model.to(self.device)
        self.epochs = 60
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (11).pkl"
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []
        if os.path.exists(self._original_model_path):
            if not self.is_cuda_available:
                self._data_dict = torch.load(
                    self._original_model_path, map_location=torch.device("cpu")
                )
            else:
                self._data_dict = torch.load(self._original_model_path)
            self.start_positions = self._data_dict["epoch"]
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )
            self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )

    def evaluate(self, epoch, display=False):
        val_set = get_val_set()
        numerator, denominator = 0, 0
        TP, FP, FN = 0, 0, 0
        for ds in val_set:
            print(f"started processing image {ds.image}")
            loader = data.DataLoader(ds, 9, pin_memory=True, collate_fn=collate_fn)

            for batch_ndx, sample in enumerate(loader):
                if not sample:
                    continue
                target, mask, idx = sample
                target = target.to(self.device)
                mask = mask.to(self.device)
                mask = (mask[:, :, :, :1] > 0).squeeze()
                res = self.model(target)
                result = (res > 0).squeeze()
                for i in range(len(mask)):
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
                torch.cuda.empty_cache()
                gc.collect()

        print(f"epoch {epoch} evaluation", numerator / denominator)
        print(f"epoch {epoch} TP={TP}, FP={FP}, FN={FN}")
        self.eval_metrics.append((numerator / denominator, TP, FP, FN))
        del ds
        gc.collect()

    def predict(self, image):
        transforms_list = []
        image = image.to(self.device)
        first_res = self.model(image)
        for t in transforms_list:
            res = self.model(t(image))
            first_res += res
            del res
            gc.collect()

        denom = len(transforms_list) + 1
        preds = first_res / denom
        del image, first_res
        gc.collect()
        return preds

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")

            for dataset in get_tiff_images():
                try:
                    loader = data.DataLoader(
                        dataset, 12, pin_memory=True, collate_fn=collate_fn
                    )

                    for batch_ndx, sample in enumerate(loader):
                        self.optim.zero_grad()
                        if not sample:
                            continue
                        target, mask, idx = sample
                        mask = mask[:, :, :, :1] > 0

                        result = self.model.forward(target.to(self.device)).to(
                            self.device
                        )

                        mask = mask.to(self.device).float().permute(0, 3, 1, 2)
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()

                        del mask, target
                        torch.cuda.empty_cache()
                        gc.collect()

                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset, loader
                    torch.cuda.empty_cache()
                    gc.collect()
            self.evaluate(i)
            with open(os.path.join(OUTPUT_PATH, "eval_metrics.csv"), "w+") as fw:
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
import pandas as pd


def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    sub = pd.read_csv(submission_file)
    for image_id in sub["id"].astype(str).tolist():
        print(f"openning {image_id}")
        dataset = HuBMAPDatasetPreprocessing(
            image_id, category="test", use_compressed=True
        )
        yield dataset


def rle_encode_less_memory(img):
    """
    img: torch/numpy 2D mask (1 - mask, 0 - background)
    Returns run length as string formatted.

    Competition expects pixels numbered top-to-bottom then left-to-right.
    Common Kaggle encoding for this competition uses transpose + flatten (Fortran-like order).
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)

    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _resolve_checkpoint_path():
    candidates = [
        "/kaggle/input/hubmapmodel/model (12).pkl",
        "/kaggle/input/hubmapmodel/model (11).pkl",
        "/kaggle/input/hubmapmodel/model.pkl",
        "/kaggle/working/models/model.pkl",
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    for p in glob.glob("/kaggle/input/**/model*.pkl", recursive=True):
        return p

    return None


def make_submission(model):
    names, preds = [], []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    model.eval()

    with torch.no_grad():
        for ds in get_test_dataset():
            print(ds.image, len(ds))
            loader = data.DataLoader(
                ds,
                batch_size=32,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )

            tile_sz = ds.sz  # working tile size (256)
            mask_tiles = torch.zeros((len(ds), tile_sz, tile_sz), dtype=torch.uint8)

            for batch_num, data_ in enumerate(loader):
                if not data_:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = data_
                image = image.to(dev)

                out = model(image)  # Bx1x(tile_sz)x(tile_sz)
                out = out.squeeze(1)

                for i, ndx in enumerate(idx):
                    mask_tiles[int(ndx)] = (out[i] > 0).to(torch.uint8).cpu()

                del out, image, data_
                torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            full_mask_small = (
                mask_tiles.view(ds.n0max, ds.n1max, tile_sz, tile_sz)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * tile_sz, ds.n1max * tile_sz)
            )

            pad0s = ds.pad0
            pad1s = ds.pad1

            full_mask_small = full_mask_small[
                pad0s // 2 : -(pad0s - pad0s // 2) if pad0s > 0 else ds.n0max * tile_sz,
                pad1s // 2 : -(pad1s - pad1s // 2) if pad1s > 0 else ds.n1max * tile_sz,
            ]

            if ds.reduce != 1:
                full_mask = cv2.resize(
                    full_mask_small.numpy().astype(np.uint8),
                    (ds.shape[1] * ds.reduce, ds.shape[0] * ds.reduce),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                full_mask = full_mask_small.numpy().astype(np.uint8)

            rle = rle_encode_less_memory(full_mask)
            names.append(ds.image)
            preds.append(rle)

            del mask_tiles, full_mask_small, full_mask, ds
            gc.collect()

    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(df), "rows")


model = Model()
ckpt_path = _resolve_checkpoint_path()
if ckpt_path is None:
    print("WARNING: No checkpoint found. Proceeding with randomly initialized weights.")
else:
    print("Loading checkpoint:", ckpt_path)
    data_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"])
    else:
        model.load_state_dict(data_dict)

make_submission(model)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3133061953.py in <cell line: 0>()
    137         model.load_state_dict(data_dict)
    138 
--> 139 make_submission(model)
    140 

/tmp/ipykernel_55/3133061953.py in make_submission(model)
     59 
     60     with torch.no_grad():
---> 61         for ds in get_test_dataset():
     62             print(ds.image, len(ds))
     63             loader = data.DataLoader(

/tmp/ipykernel_55/3133061953.py in get_test_dataset()
      8     for image_id in sub["id"].astype(str).tolist():
      9         print(f"openning {image_id}")
---> 10         dataset = HuBMAPDatasetPreprocessing(
     11             image_id, category="test", use_compressed=True
     12         )

/tmp/ipykernel_55/3233476279.py in __init__(self, image, category, use_compressed)
    127         self._img = None
    128         self._mask = None
--> 129         self._get_image_and_mask()
    130 
    131     def _get_image_and_mask(self):

/tmp/ipykernel_55/3233476279.py in _get_image_and_mask(self)
    146         # Fix: avoid OpenCV pixel-limit crash by using tifffile-based loader; for test, prefer lowres if pyramidal.
    147         # Also: for use_compressed=True, we will actually work on the reduced-resolution image for consistent tiling.
--> 148         img_full = _load_tiff_rgb(file_path, prefer_lowres=False)  # HxWx3 uint8 (BGR)
    149 
    150         # Fix: if use_compressed=True, reduce the working image resolution and set shape accordingly.

/tmp/ipykernel_55/3233476279.py in _load_tiff_rgb(path, prefer_lowres)
     57     if img is None:
     58         # This may still fail for huge TIFFs; tifffile path above should handle competition data.
---> 59         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
     60 
     61         if img is None:

error: OpenCV(4.12.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:79: error: (-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS in function 'validateInputImageSize'


## === cell 7
import pandas as pd

sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.columns.tolist(), "rows:", len(sub))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2952113428.py in <cell line: 0>()
      1 import pandas as pd
      2 
----> 3 sub = pd.read_csv("submission.csv")
      4 print(sub.head())
      5 print(sub.columns.tolist(), "rows:", len(sub))

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

## === cell 8
"""
(from skimage import io
 import matplotlib.pyplot as plt
 ...debug cell unchanged...)
"""



## === cell 9
"""
(debug cell unchanged...)
"""
