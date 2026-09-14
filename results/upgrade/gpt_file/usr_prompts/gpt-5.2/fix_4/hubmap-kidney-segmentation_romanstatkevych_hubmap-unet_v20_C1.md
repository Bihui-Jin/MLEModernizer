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

0.7962667518195039

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
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io as skio

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)


class HuBMAPDatasetPreprocessing(Dataset):
    """
    Bugfix: OpenCV cv2.imread can crash on very large TIFFs due to CV_IO_MAX_IMAGE_PIXELS.
    Fix: use skimage.io.imread for TIFF reading (no such limit), then convert to uint8 RGB.
    Core logic preserved: same tiling/padding/reduction, HSV filter, normalization, and JSON polygon rasterization.
    """

    def __init__(self, image, category="train", use_compressed=False):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category

        self.reduce = 4 if not use_compressed else 1
        self._img_full = None
        self._mask_full = None
        self._get_image()

    def _get_image_path(self):
        if self.use_compressed:
            return os.path.join(
                self.folder, self.category, f"compressed-{self.image}.tiff"
            )
        return os.path.join(self.folder, self.category, f"{self.image}.tiff")

    def _get_masks(self):
        file_path = os.path.join(self.mask_folder, self.category, f"{self.image}.json")
        with open(file_path, "r") as f:
            return json.load(f)

    def _read_geometry_coords(self):
        feats = self._get_masks()
        polys = []
        for f in feats:
            geom = f.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates", [])
            if not coords:
                continue
            ring = coords[0]
            if len(ring) < 3:
                continue
            arr = np.asarray(ring, dtype=np.float32)  # [[x,y], ...]
            if self.use_compressed:
                arr = arr / 4.0
            polys.append(arr)
        return polys

    def _rasterize_polygons(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self._read_geometry_coords():
            pts = np.round(poly).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)

    def _read_tiff_rgb_uint8(self, path):
        img = skio.imread(path)  # can be (H,W), (H,W,C), dtype uint8/uint16/...
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]
        else:
            raise ValueError(f"Unexpected TIFF shape {img.shape} for {path}")

        if img.dtype != np.uint8:
            mx = float(np.max(img)) if np.max(img) > 0 else 1.0
            img = (img.astype(np.float32) / mx * 255.0).clip(0, 255).astype(np.uint8)

        return img

    def _get_image(self):
        if self._img_full is not None:
            return self._img_full

        path = self._get_image_path()
        img = self._read_tiff_rgb_uint8(path)  # RGB uint8

        self._img_full = img
        self.shape = (img.shape[0], img.shape[1])  # (H,W)

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        if self.category == "train":
            m = self._rasterize_polygons(self.shape[0], self.shape[1])
            self._mask_full = np.stack([m, m, m], axis=0)  # (3,H,W) bool

        return self._img_full

    def close(self):
        return

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        self._img_full = None
        self._mask_full = None

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        img_full = self._get_image()
        n0, n1 = idx // self.n1max, idx % self.n1max

        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_full[
            p00:p01, p10:p11
        ]

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask = np.zeros((3, self.sz, self.sz), np.uint8)
            mask[:, (p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self._mask_full[
                :, p00:p01, p10:p11
            ].astype(np.uint8)
            mask = np.moveaxis(mask, 0, -1)  # (sz,sz,3)
            if self.reduce != 1:
                mask = cv2.resize(
                    mask,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )
        else:
            mask = torch.zeros(1)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        result_tensor = torch.from_numpy(
            ((img.astype(np.float32) / 255.0) - mean) / std
        ).float()
        result_tensor = result_tensor.permute(2, 0, 1).contiguous().float()

        if (s > 40).sum() <= 1000 or img.sum() <= 1000:
            return result_tensor, mask, -1
        else:
            return result_tensor, mask, idx




## === cell 1
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', use_compressed=True)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    sample = sample.permute(1,2,0)
    
    if idx == -1 or (isinstance(mask, np.ndarray) and mask.sum() == 0):
        continue
    if c == 5:
        break

    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    if isinstance(mask, np.ndarray):
        plots[1][c].imshow(mask[...,0])
    plots[1][c].set_title(f'Sample #{c}')
    c += 1
"""


## === cell 2
"""
# Visualization cell (kept commented)
"""


## === cell 3
import torch
from torch import nn, optim
from torch.utils import data
import torchvision
import torch.nn.functional as F


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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
import os
import gc
import glob
import numpy
import torchvision
from torch.utils import data

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    x = list(filter(lambda t: t[-1] != -1, x))
    if len(x) == 0:
        return []
    return data.dataloader.default_collate(x)


VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(name, use_compressed=True)
            yield dataset


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(image, use_compressed=True)
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
        self.epochs = 20
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self.optim = optim.Adam(self.model.parameters())

        if os.path.exists(self._model_path):
            self._data_dict = torch.load(self._model_path, map_location=self.device)
            self.start_positions = self._data_dict["epoch"]
            self.loss = self._data_dict["loss"]
            self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
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
                loader = data.DataLoader(ds, 16, pin_memory=True, collate_fn=collate_fn)
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
                        numerator += 2 * torch.sum(mask[i].float() * result[i].float())
                        denominator += torch.sum(mask[i].float() + result[i].float())

                    del mask, target, result
                    torch.cuda.empty_cache()
                    gc.collect()

        print(f"epoch {epoch} evaluation", numerator / denominator)
        gc.collect()
        self.model.train()

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            self.model.train()
            for dataset in get_tiff_images():
                try:
                    loader = data.DataLoader(
                        dataset, 16, pin_memory=True, collate_fn=collate_fn
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

            print(f"saving epoch {i}")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    "loss": self.loss,
                },
                os.path.join(OUTPUT_PATH, f"model_{i}.pkl"),
            )
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    "loss": self.loss,
                },
                self._model_path,
            )


def display_predictions(masks):
    import matplotlib.pyplot as plt

    masks = masks.squeeze()
    masks = masks > 0.5
    mask = 255 * masks
    plt.imshow(mask)




## === cell 5
torch.cuda.empty_cache()
trainer = Trainer()




## === cell 6
import pandas as pd
import csv
from torch.utils import data


def collate_fn(x):
    x = list(filter(lambda t: t[-1] != -1, x))
    if len(x) == 0:
        return []
    return data.dataloader.default_collate(x)


def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    with open(submission_file, newline="") as sub_file:
        reader = csv.reader(sub_file)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            print(f"openning {image_id}")
            dataset = HuBMAPDatasetPreprocessing(
                image_id, category="test", use_compressed=True
            )
            yield dataset


def rle_encode_less_memory(img):
    if img.ndim != 2:
        img = np.squeeze(img)
    img = img.astype(np.uint8, copy=False)
    if img.max() == 0:
        return ""
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def make_submission(model):
    names, preds = [], []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    model.eval()

    with torch.no_grad():
        for ds in get_test_dataset():
            print(ds.image, len(ds))
            loader = data.DataLoader(ds, 32, collate_fn=collate_fn, pin_memory=True)

            tile_sz_full = ds.sz
            mask_tiles = torch.zeros(
                len(ds), tile_sz_full, tile_sz_full, dtype=torch.uint8
            )

            for batch_num, batch in enumerate(loader):
                if not batch:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = batch
                image = image.to(dev)

                logits = model(image)
                up = torch.nn.functional.interpolate(
                    logits, scale_factor=ds.reduce, mode="bilinear", align_corners=False
                )
                up = up.squeeze(1)  # (B,H_full,W_full) where H_full==ds.sz

                up_cpu = up.detach().cpu()
                for i, ndx in enumerate(idx):
                    mask_tiles[int(ndx)] = (up_cpu[i] > 0).to(torch.uint8)

                del logits, up, up_cpu, image
                torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            mask = (
                mask_tiles.view(ds.n0max, ds.n1max, tile_sz_full, tile_sz_full)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * tile_sz_full, ds.n1max * tile_sz_full)
            )

            mask = mask[
                ds.pad0
                // 2 : (
                    -(ds.pad0 - ds.pad0 // 2)
                    if ds.pad0 > 0
                    else ds.n0max * tile_sz_full
                ),
                ds.pad1
                // 2 : (
                    -(ds.pad1 - ds.pad1 // 2)
                    if ds.pad1 > 0
                    else ds.n1max * tile_sz_full
                ),
            ]

            rle = rle_encode_less_memory(mask.numpy())
            names.append(ds.image)
            preds.append(rle)

            del mask_tiles, mask, ds
            gc.collect()

    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", df.shape)


model = Model()

candidate_paths = [
    "/kaggle/input/hubmapmodel/model (3).pkl",  # original path in the notebook
    "/kaggle/working/models/model.pkl",  # locally trained checkpoint (if user runs training)
]

ckpt_path = None
for p in candidate_paths:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is not None:
    data_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"])
        print("Loaded checkpoint:", ckpt_path)
    else:
        model.load_state_dict(data_dict)
        print("Loaded raw state_dict:", ckpt_path)
else:
    print(
        "No checkpoint found; using untrained model (submission will be low-scoring but valid)."
    )

make_submission(model)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2230281059.py in <cell line: 0>()
    140     )
    141 
--> 142 make_submission(model)
    143 
    144 

/tmp/ipykernel_55/2230281059.py in make_submission(model)
     49 
     50     with torch.no_grad():
---> 51         for ds in get_test_dataset():
     52             print(ds.image, len(ds))
     53             loader = data.DataLoader(ds, 32, collate_fn=collate_fn, pin_memory=True)

/tmp/ipykernel_55/2230281059.py in get_test_dataset()
     21                 continue
     22             print(f"openning {image_id}")
---> 23             dataset = HuBMAPDatasetPreprocessing(
     24                 image_id, category="test", use_compressed=True
     25             )

/tmp/ipykernel_55/1816549586.py in __init__(self, image, category, use_compressed)
     30         self._img_full = None
     31         self._mask_full = None
---> 32         self._get_image()
     33 
     34     def _get_image_path(self):

/tmp/ipykernel_55/1816549586.py in _get_image(self)
     95 
     96         path = self._get_image_path()
---> 97         img = self._read_tiff_rgb_uint8(path)  # RGB uint8
     98 
     99         self._img_full = img

/tmp/ipykernel_55/1816549586.py in _read_tiff_rgb_uint8(self, path)
     73 
     74     def _read_tiff_rgb_uint8(self, path):
---> 75         img = skio.imread(path)  # can be (H,W), (H,W,C), dtype uint8/uint16/...
     76         if img is None:
     77             raise FileNotFoundError(f"Could not read image: {path}")

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

/usr/local/lib/python3.11/dist-packages/skimage/io/_plugins/tifffile_plugin.py in imread(fname, **kwargs)
     72         kwargs['key'] = kwargs.pop('img_num')
     73 
---> 74     return tifffile_imread(fname, **kwargs)

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in imread(files, selection, aszarr, key, series, level, squeeze, maxworkers, buffersize, mode, name, offset, size, pattern, axesorder, categories, imread, imreadargs, sort, container, chunkshape, chunkdtype, axestiled, ioworkers, chunkmode, fillvalue, zattrs, multiscales, omexml, out, out_inplace, _multifile, _useframes, **kwargs)
   1205 
   1206         if isinstance(files, str) or not isinstance(files, Sequence):
-> 1207             with TiffFile(
   1208                 files,
   1209                 mode=mode,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in __init__(self, file, mode, name, offset, size, omexml, _multifile, _useframes, _parent, **is_flags)
   4280             self.is_ome = True
   4281 
-> 4282         fh = FileHandle(file, mode=mode, name=name, offset=offset, size=size)
   4283         self._fh = fh
   4284         self._multifile = True if _multifile is None else bool(_multifile)

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in __init__(self, file, mode, name, offset, size)
  13325         self._close = True
  13326         self._lock = NullContext()
> 13327         self.open()
  13328         assert self._fh is not None
  13329 

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in open(self)
  13344             self._file = os.path.realpath(self._file)
  13345             self._dir, self._name = os.path.split(self._file)
> 13346             self._fh = open(self._file, self._mode, encoding=None)
  13347             self._close = True
  13348             self._offset = max(0, self._offset)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hubmap-kidney-segmentation/test/compressed-8242609fa.tiff'

## === cell 7
"""
# Debug/visualization cell kept commented
"""
