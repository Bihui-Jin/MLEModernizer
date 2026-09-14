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

0.8641630081338946

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import gc
import json
import csv

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import tifffile as tiff

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)


class HuBMAPDatasetPreprocessing(Dataset):
    """
    TIFF reading is done via tifffile to avoid OpenCV max-pixels assertion errors.
    Core tiling/padding/reduce logic and normalization are preserved.

    Bugfix (critical): compute the copy region using an explicit intersection between
    the requested tile window [r0:r1, c0:c1] and the image bounds [0:H, 0:W].
    Then map that intersection into both destination and source slices with identical
    sizes, preventing broadcasting errors.
    """

    def __init__(self, image, category="train", use_compressed=False):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category

        self.reduce = 1 if use_compressed else 4

        self._img = None
        self._load_image()

        self.sz = 256 * self.reduce

        self.pad_h = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad_w = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad_h) // self.sz  # rows tiles
        self.n1max = (self.shape[1] + self.pad_w) // self.sz  # cols tiles

        if self.category == "train":
            self.mask = self._build_mask()  # H,W boolean

    def _image_path(self):
        if self.use_compressed:
            p = os.path.join(
                self.folder, self.category, f"compressed-{self.image}.tiff"
            )
            if os.path.exists(p):
                return p
        return os.path.join(self.folder, self.category, f"{self.image}.tiff")

    def _load_image(self):
        if self._img is not None:
            return
        fp = self._image_path()
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Image path does not exist: {fp}")

        img = tiff.imread(fp)

        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.shape[-1] >= 3:
            img = img[..., :3]
        img = np.ascontiguousarray(img)

        if img.dtype != np.uint8:
            img_min = float(img.min())
            img_max = float(img.max())
            if img_max > img_min:
                img = (
                    ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        img = img[..., ::-1]  # RGB->BGR for OpenCV consistency

        self._img = img
        self.shape = self._img.shape[:2]  # (H,W)

    def close(self):
        self._img = None

    def _get_masks(self):
        file_path = os.path.join(self.mask_folder, self.category, f"{self.image}.json")
        with open(file_path, "r") as f:
            return json.load(f)

    def _build_mask(self):
        h, w = self.shape
        m = np.zeros((h, w), dtype=np.uint8)
        features = self._get_masks()

        for feat in features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", None)
            if not coords:
                continue
            poly = coords[0]
            if len(poly) < 3:
                continue
            pts = np.asarray(poly, dtype=np.float32)
            xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
            cv2.fillPoly(m, [np.stack([xs, ys], axis=1)], 1)

        return m > 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return int(self.n0max * self.n1max)

    def __getitem__(self, idx):
        self._load_image()
        n0, n1 = idx // self.n1max, idx % self.n1max

        r0 = int(-self.pad_h // 2 + n0 * self.sz)
        c0 = int(-self.pad_w // 2 + n1 * self.sz)
        r1 = int(r0 + self.sz)
        c1 = int(c0 + self.sz)

        H, W = self.shape

        sr0 = max(0, r0)
        sc0 = max(0, c0)
        sr1 = min(H, r1)
        sc1 = min(W, c1)

        if sr0 >= sr1 or sc0 >= sc1:
            dummy = torch.zeros(
                (3, self.sz // self.reduce, self.sz // self.reduce), dtype=torch.float32
            )
            if self.category == "train":
                mask = np.zeros(
                    (self.sz // self.reduce, self.sz // self.reduce, 1), np.uint8
                )
            else:
                mask = torch.zeros(1)
            return dummy, mask, -1

        dr0 = sr0 - r0
        dc0 = sc0 - c0
        dr1 = dr0 + (sr1 - sr0)
        dc1 = dc0 + (sc1 - sc0)

        dr0 = int(np.clip(dr0, 0, self.sz))
        dc0 = int(np.clip(dc0, 0, self.sz))
        dr1 = int(np.clip(dr1, 0, self.sz))
        dc1 = int(np.clip(dc1, 0, self.sz))

        copy_h = dr1 - dr0
        copy_w = dc1 - dc0
        if copy_h <= 0 or copy_w <= 0 or (sr1 - sr0) != copy_h or (sc1 - sc0) != copy_w:
            dummy = torch.zeros(
                (3, self.sz // self.reduce, self.sz // self.reduce), dtype=torch.float32
            )
            if self.category == "train":
                mask = np.zeros(
                    (self.sz // self.reduce, self.sz // self.reduce, 1), np.uint8
                )
            else:
                mask = torch.zeros(1)
            return dummy, mask, -1

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[dr0:dr1, dc0:dc1] = self._img[sr0:sr1, sc0:sc1]

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask = np.zeros((self.sz, self.sz), np.uint8)
            mask[dr0:dr1, dc0:dc1] = self.mask[sr0:sr1, sc0:sc1].astype(np.uint8)
            if self.reduce != 1:
                mask = cv2.resize(
                    mask,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask[..., None]
        else:
            mask = torch.zeros(1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        result_tensor = torch.from_numpy(
            ((img.astype(np.float32) / 255.0) - mean) / std
        ).float()
        result_tensor = result_tensor.permute(2, 0, 1).contiguous()

        if (s > 40).sum() <= 1000 or img.sum() <= 1000:
            return result_tensor, mask, -1
        else:
            return result_tensor, mask, idx




## === cell 1
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', category='test', use_compressed=False)

fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(1, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    if idx == -1:
        continue
    if c == 5:
        break
    plots[c].imshow(sample.permute(1,2,0).numpy())
    plots[c].set_title(f'Sample #{c}')
    c += 1
"""




## === cell 2
"""
# Placeholder visualization cell kept from original notebook (no-op by default).
"""




## === cell 3
import torch
from torch import nn, optim
from torch.utils import data
import torchvision
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
        self.image_size = (128, 128)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, X):
        x1 = self.inc(X)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




## === cell 4
OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    ls = [t for t in x if t[-1] != -1]
    if len(ls) == 0:
        return None
    return data.dataloader.default_collate(ls)


VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name in VALSET:
            continue
        try:
            dataset = HuBMAPDatasetPreprocessing(
                name, category="train", use_compressed=False
            )
            yield dataset
        except Exception as e:
            print(
                f"Warning: skipping image {name} due to error during dataset init: {repr(e)}"
            )
            continue


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(
            image, category="train", use_compressed=False
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
        self.model = Model()
        self.model.to(self.device)
        self.epochs = 40
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")

        self.optim = optim.Adam(self.model.parameters())

        if os.path.exists(self._model_path):
            self._data_dict = torch.load(self._model_path, map_location="cpu")
            self.start_positions = int(self._data_dict.get("epoch", 0))
            self.loss = self._data_dict.get(
                "loss",
                torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.Tensor([5]).to(self.device)
                ),
            )
            if "optimizer_state_dict" in self._data_dict:
                self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            if "model_state_dict" in self._data_dict:
                self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )

    def evaluate(self, epoch):
        val_set = get_val_set()
        numerator, denominator = 0, 0
        self.model.eval()
        with torch.no_grad():
            for ds in val_set:
                print(f"started processing image {ds.image}")
                loader = data.DataLoader(
                    ds, 16, pin_memory=True, collate_fn=collate_fn, num_workers=0
                )
                for batch_ndx, sample in enumerate(loader):
                    if sample is None:
                        continue
                    target, mask, idx = sample
                    target = target.to(self.device)
                    mask = torch.as_tensor(mask).to(self.device)
                    mask = (mask[:, :, :, :1] > 0).squeeze(3)  # B,H,W
                    res = self.model(target)
                    result = (res > 0).squeeze(1)  # B,H,W
                    for i in range(len(mask)):
                        numerator += 2 * torch.sum(mask[i].float() * result[i].float())
                        denominator += torch.sum(mask[i].float() + result[i].float())
                    del mask, target, result, res
                    torch.cuda.empty_cache()
                    gc.collect()
        self.model.train()
        print(f"epoch {epoch} evaluation", (numerator / (denominator + 1e-8)).item())
        gc.collect()

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            for dataset in get_tiff_images():
                loader = None
                try:
                    loader = data.DataLoader(
                        dataset,
                        16,
                        pin_memory=True,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )
                    for batch_ndx, sample in enumerate(loader):
                        self.optim.zero_grad()
                        if sample is None:
                            continue
                        target, mask, idx = sample
                        mask = torch.as_tensor(mask)
                        mask = mask[:, :, :, :1] > 0  # B,H,W,1
                        result = self.model.forward(target.to(self.device))
                        mask = (
                            mask.to(self.device).float().permute(0, 3, 1, 2)
                        )  # B,1,H,W
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()

                        del mask, target, result, loss_result
                        torch.cuda.empty_cache()
                        gc.collect()

                    print(f"data processed for {dataset.image}")

                except Exception as e:
                    print(
                        f"Warning: skipping dataset {getattr(dataset, 'image', '?')} due to error: {repr(e)}"
                    )
                finally:
                    try:
                        del dataset, loader
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    gc.collect()

            self.evaluate(i)

            print(f"saving epoch {i}")
            state = {
                "epoch": i,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "loss": self.loss,
            }
            torch.save(state, os.path.join(OUTPUT_PATH, f"model_{i}.pkl"))
            torch.save(state, self._model_path)


def display_predictions(masks):
    masks = masks.squeeze()
    masks = masks > 0.5
    mask = 255 * masks
    plt.imshow(mask)




## === cell 5
torch.cuda.empty_cache()
trainer = Trainer()




## === cell 6
def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    with open(submission_file, newline="") as sub_file:
        reader = csv.reader(sub_file)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            print(f"openning {image_id}")
            dataset = HuBMAPDatasetPreprocessing(
                image_id, category="test", use_compressed=False
            )
            yield dataset


def rle_encode_less_memory(img):
    """
    img: 2D numpy array, 1 - mask, 0 - background
    Returns run length as string formatted

    Note: img is expected in (H,W). We keep the original transpose+flatten
    because this competition numbers pixels top-to-bottom then left-to-right.

    Bugfix: handle empty images safely.
    """
    if img is None:
        return ""
    if img.size == 0:
        return ""
    pixels = img.T.flatten().astype(np.uint8)
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _find_best_checkpoint(output_path):
    cand = os.path.join(output_path, "model.pkl")
    if os.path.exists(cand):
        return cand
    model_glob = sorted(glob.glob(os.path.join(output_path, "model_*.pkl")))
    if model_glob:
        return model_glob[-1]
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
                ds, 32, collate_fn=collate_fn, pin_memory=True, num_workers=0
            )

            tile_sz_full = ds.sz
            tile_sz_model = ds.sz // ds.reduce

            tiles_mask_small = torch.zeros(
                (len(ds), tile_sz_model, tile_sz_model), dtype=torch.uint8
            )

            for batch_num, batch in enumerate(loader):
                if batch is None:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = batch
                image = image.to(dev)

                logits = model(image)  # B,1,tile_sz_model,tile_sz_model
                pred_small = (logits > 0).squeeze(1).to("cpu").to(torch.uint8)

                if (
                    pred_small.shape[-1] != tile_sz_model
                    or pred_small.shape[-2] != tile_sz_model
                ):
                    pred_np = pred_small.numpy()
                    pred_resized = []
                    for k in range(pred_np.shape[0]):
                        pred_resized.append(
                            cv2.resize(
                                pred_np[k],
                                (tile_sz_model, tile_sz_model),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
                    pred_small = torch.from_numpy(np.stack(pred_resized, axis=0)).to(
                        torch.uint8
                    )

                for i, ndx in enumerate(idx.tolist()):
                    tiles_mask_small[ndx] = pred_small[i]

                del image, idx, logits, pred_small
                torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            full_small = (
                tiles_mask_small.view(ds.n0max, ds.n1max, tile_sz_model, tile_sz_model)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * tile_sz_model, ds.n1max * tile_sz_model)
            )

            if ds.reduce != 1:
                full = torch.from_numpy(
                    cv2.resize(
                        full_small.numpy(),
                        (ds.n1max * tile_sz_full, ds.n0max * tile_sz_full),
                        interpolation=cv2.INTER_NEAREST,
                    )
                ).to(torch.uint8)
            else:
                full = full_small

            top = ds.pad_h // 2
            bottom_cut = ds.pad_h - ds.pad_h // 2
            left = ds.pad_w // 2
            right_cut = ds.pad_w - ds.pad_w // 2
            bottom = None if bottom_cut == 0 else -bottom_cut
            right = None if right_cut == 0 else -right_cut
            full = full[top:bottom, left:right]

            rle = rle_encode_less_memory(full.numpy())
            names.append(ds.image)
            preds.append(rle)

            del tiles_mask_small, full_small, full, ds
            gc.collect()

    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(df), "rows")


ckpt_path = _find_best_checkpoint(OUTPUT_PATH)
if ckpt_path is None:
    print("No local checkpoint found; starting training to produce one.")
    trainer.train()
    ckpt_path = _find_best_checkpoint(OUTPUT_PATH)

model = Model()
if ckpt_path is not None and os.path.exists(ckpt_path):
    print("Loading checkpoint:", ckpt_path)
    data_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data_dict["model_state_dict"])
else:
    print(
        "Warning: checkpoint not found; using randomly initialized weights (submission will score poorly)."
    )

make_submission(model)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1330291350.py in <cell line: 0>()
    144 if ckpt_path is None:
    145     print("No local checkpoint found; starting training to produce one.")
--> 146     trainer.train()
    147     ckpt_path = _find_best_checkpoint(OUTPUT_PATH)
    148 

/tmp/ipykernel_55/1746535422.py in train(self)
    159                     gc.collect()
    160 
--> 161             self.evaluate(i)
    162 
    163             print(f"saving epoch {i}")

/tmp/ipykernel_55/1746535422.py in evaluate(self, epoch)
     88                     ds, 16, pin_memory=True, collate_fn=collate_fn, num_workers=0
     89                 )
---> 90                 for batch_ndx, sample in enumerate(loader):
     91                     if sample is None:
     92                         continue

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/4088936641.py in __getitem__(self, idx)
    185 
    186         img = np.zeros((self.sz, self.sz, 3), np.uint8)
--> 187         img[dr0:dr1, dc0:dc1] = self._img[sr0:sr1, sc0:sc1]
    188 
    189         if self.reduce != 1:

ValueError: could not broadcast input array from shape (3,16180,3) into shape (1,1,3)
