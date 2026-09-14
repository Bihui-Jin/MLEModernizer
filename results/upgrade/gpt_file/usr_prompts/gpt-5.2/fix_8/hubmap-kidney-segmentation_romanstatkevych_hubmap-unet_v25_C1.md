# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8792900293798971

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your pipeline currently can fail to yield a valid Kaggle score because the submission column names don’t match the competition’s expected schema (the provided sample has `id,predicted`, but your code writes `img,pixels`). I keep your model, tiling, thresholding, and RLE logic intact, and only adjust submission formatting to exactly follow the sample submission header and ID alignment. I also make the test loader robust by always using the sample’s ID column and emitting predictions in the same order, which avoids accidental misalignment (a common silent score killer). These minimal changes should turn “Not yielded” into a valid submission and thus move the score toward the target by enabling evaluation.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid/empty prediction problem rather than model quality: the test loader drops many “non-tissue” tiles (idx = -1) and never writes them back into the reconstruction, leaving large parts of the final mask as all-zeros. I keep your model, thresholds, tiling, and RLE exactly the same, but make reconstruction robust by tracking each tile’s true grid position and filling skipped tiles as zeros in the correct place (instead of shrinking the canvas and misassembling). I also fix a subtle but critical bug: the canvas is currently sized to `len(ds)` (number of tiles after padding), but you then reshape using `ds.n0max*ds.n1max`; those are not equal, causing incorrect assembly (or runtime issues) and leading to bad/empty RLEs. Finally, I keep the submission schema aligned to `sample_submission.csv` (`id,predicted`) and preserve row order.'

# 9. Code solution

## === cell 0
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)  # effectively "no limit"

import glob
import json
import gc

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    cv2.setNumThreads(0)
except Exception:
    pass

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)


class HuBMAPDatasetPreprocessing(Dataset):
    def __init__(self, image, category="train", use_compressed=False):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = (
            "/kaggle/input/compressedhubmap"
            if use_compressed
            else "/kaggle/input/hubmap-kidney-segmentation"
        )
        self.use_compressed = use_compressed
        self.image = image
        self.category = category
        self.reduce = 4 if not use_compressed else 1

        self._img = None
        self._get_image()

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        if self.category == "train":
            self.mask = self._read_mask_fullres()  # HxW bool
        else:
            self.mask = None

    def _tiff_path(self):
        if self.use_compressed:
            p = os.path.join(
                self.folder, self.category, f"compressed-{self.image}.tiff"
            )
            if os.path.exists(p):
                return p
        return os.path.join(
            "/kaggle/input/hubmap-kidney-segmentation",
            self.category,
            f"{self.image}.tiff",
        )

    def _get_image(self):
        if self._img is not None:
            return self._img
        path = self._tiff_path()
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise FileNotFoundError(f"Could not read TIFF via OpenCV: {path}")
        self._img = img
        self.shape = img.shape[:2]  # (H, W)
        return self._img

    def close(self):
        self._img = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_masks(self):
        file_path = os.path.join(self.mask_folder, self.category, f"{self.image}.json")
        with open(file_path, "r") as f:
            return json.load(f)

    def _read_mask_fullres(self):
        H, W = self.shape
        m = np.zeros((H, W), dtype=np.uint8)

        features = self._get_masks()
        for feat in features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", None)
            if not coords:
                continue
            poly = coords[0]
            if len(poly) < 3:
                continue

            xs = []
            ys = []
            for x, y in poly:
                if self.use_compressed:
                    x = x // 4
                    y = y // 4
                xs.append(float(x))
                ys.append(float(y))

            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            cv2.fillPoly(m, [pts], 1)

        return m.astype(bool)

    def __len__(self):
        return int(self.n0max * self.n1max)

    def __getitem__(self, idx):
        img_full = self._get_image()
        H, W = self.shape

        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, H)
        p10, p11 = max(0, y0), min(y0 + self.sz, W)

        tile = np.zeros((self.sz, self.sz, 3), np.uint8)
        tile[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_full[
            p00:p01, p10:p11
        ]

        if self.reduce != 1:
            tile = cv2.resize(
                tile,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask_tile = np.zeros((self.sz, self.sz), np.uint8)
            mask_tile[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self.mask[
                p00:p01, p10:p11
            ].astype(np.uint8)
            if self.reduce != 1:
                mask_tile = cv2.resize(
                    mask_tile,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )
            mask_out = np.repeat(mask_tile[..., None], 3, axis=2)
        else:
            mask_out = torch.zeros(1)

        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        result_tensor = torch.from_numpy(
            (tile.astype(np.float32) / 255.0 - mean) / std
        ).float()
        result_tensor = result_tensor.permute(2, 0, 1).contiguous()

        if (s > 40).sum() <= 1000 or tile.sum() <= 1000:
            return result_tensor, mask_out, -1
        return result_tensor, mask_out, idx




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
    plots[1][c].imshow(mask[...,0]*255, cmap='gray')
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
import torchvision.transforms.functional as F_tv

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
import os
import gc
import re
import torchvision
from torch.utils import data

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    x = list(filter(lambda z: z[-1] != -1, x))
    if not x:
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
        self.model = Model().to(self.device)
        self.epochs = 60
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "/kaggle/input/hubmapmodel/model (5).pkl"
        self.optim = optim.Adam(self.model.parameters())

        if os.path.exists(self._original_model_path):
            self._data_dict = torch.load(
                self._original_model_path, map_location=self.device
            )
            self.start_positions = int(self._data_dict.get("epoch", 0))
            self.loss = self._data_dict.get(
                "loss",
                nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5]).to(self.device)),
            )
            if "optimizer_state_dict" in self._data_dict:
                self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            self.model.load_state_dict(self._data_dict["model_state_dict"])
        elif os.path.exists(self._model_path):
            self._data_dict = torch.load(self._model_path, map_location=self.device)
            self.start_positions = int(self._data_dict.get("epoch", 0))
            self.loss = self._data_dict.get(
                "loss",
                nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5]).to(self.device)),
            )
            if "optimizer_state_dict" in self._data_dict:
                self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = nn.BCEWithLogitsLoss(
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
                    mask = torch.as_tensor(mask).to(self.device)
                    mask = (mask[:, :, :, :1] > 0).squeeze(3)  # B,H,W
                    res = self.model(target)
                    result = (res > 0).squeeze(1)  # B,H,W

                    for i in range(mask.shape[0]):
                        numerator += 2 * torch.sum(mask[i].float() * result[i].float())
                        denominator += torch.sum(mask[i].float() + result[i].float())

                    del mask, target, result, res
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        print(
            f"epoch {epoch} evaluation",
            (numerator / denominator).item() if denominator != 0 else 0.0,
        )
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
                        if not sample:
                            continue
                        self.optim.zero_grad()
                        target, mask, idx = sample
                        mask = torch.as_tensor(mask)
                        mask = (
                            (mask[:, :, :, :1] > 0).float().permute(0, 3, 1, 2)
                        )  # B,1,H,W

                        result = self.model(target.to(self.device))
                        loss_result = self.loss(result, mask.to(self.device))
                        loss_result.backward()
                        self.optim.step()

                        del mask, target, result, loss_result
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset, loader
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            self.evaluate(i)

            print(f"saving epoch {i}")
            payload = {
                "epoch": i,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "loss": self.loss,
            }
            torch.save(payload, os.path.join(OUTPUT_PATH, f"model_{i}.pkl"))
            torch.save(payload, self._model_path)


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
from skimage.color import rgb2gray  # noqa: F401


def collate_fn(x):
    x = list(filter(lambda z: z[-1] != -1, x))
    if x:
        return data.dataloader.default_collate(x)
    return []


def get_test_ids_and_schema(
    submission_file="/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv",
):
    sub = pd.read_csv(submission_file)
    id_col = "id" if "id" in sub.columns else sub.columns[0]
    pred_col = "predicted" if "predicted" in sub.columns else sub.columns[1]
    ids = sub[id_col].astype(str).tolist()
    return ids, id_col, pred_col


def get_test_dataset():
    ids, _, _ = get_test_ids_and_schema()
    for image_id in ids:
        print(f"openning {image_id}")
        dataset = HuBMAPDatasetPreprocessing(
            image_id, category="test", use_compressed=False
        )
        yield dataset


def rle_encode_less_memory(img):
    if img is None:
        return ""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.size == 0:
        return ""
    pixels = img.T.flatten().astype(np.uint8, copy=False)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _find_best_local_checkpoint(models_dir="/kaggle/working/models"):
    if not os.path.isdir(models_dir):
        return None
    candidates = glob.glob(os.path.join(models_dir, "model_*.pkl"))
    if not candidates:
        rolling = os.path.join(models_dir, "model.pkl")
        return rolling if os.path.exists(rolling) else None

    def _epoch_num(p):
        m = re.search(r"model_(\d+)\.pkl$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    candidates.sort(key=_epoch_num)
    return candidates[-1]


def make_submission(model):
    ids, id_col, pred_col = get_test_ids_and_schema()
    preds_map = {}

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()

    with torch.no_grad():
        for ds in get_test_dataset():
            print(ds.image, len(ds))
            loader = data.DataLoader(ds, 32, collate_fn=collate_fn)

            total_tiles = int(ds.n0max * ds.n1max)
            canvas = torch.zeros((total_tiles, ds.sz, ds.sz), dtype=torch.uint8)

            for batch_num, data_ in enumerate(loader):
                if not data_:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = data_
                image = image.to(dev)

                logits = model(image)  # B,1,h,w
                if ds.reduce != 1:
                    logits = F.interpolate(
                        logits,
                        scale_factor=ds.reduce,
                        mode="bilinear",
                        align_corners=False,
                    )

                prob = (logits > 0).squeeze(1).to("cpu")  # B,H,W bool
                for i, ndx in enumerate(idx):
                    canvas[int(ndx)] = prob[i].to(torch.uint8)

                del logits, prob, image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            mask = (
                canvas.view(ds.n0max, ds.n1max, ds.sz, ds.sz)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
            )
            h0 = ds.pad0 // 2
            h1 = ds.pad0 - ds.pad0 // 2
            w0 = ds.pad1 // 2
            w1 = ds.pad1 - ds.pad1 // 2
            mask = mask[
                h0 : mask.shape[0] - h1 if h1 > 0 else mask.shape[0],
                w0 : mask.shape[1] - w1 if w1 > 0 else mask.shape[1],
            ]

            rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
            preds_map[str(ds.image)] = rle

            del canvas, mask, ds
            gc.collect()

    preds = [preds_map.get(str(i), "") for i in ids]
    df = pd.DataFrame({id_col: ids, pred_col: preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", df.shape)
    print(df.head())


model = Model()

ckpt_path = None
external_ckpt = "/kaggle/input/hubmapmodel/model (5).pkl"
if os.path.exists(external_ckpt):
    ckpt_path = external_ckpt
else:
    ckpt_path = _find_best_local_checkpoint("/kaggle/working/models")

if ckpt_path is not None:
    data_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data_dict["model_state_dict"])
else:
    print(
        "WARNING: No checkpoint found. Proceeding with randomly initialized model to produce a valid submission.csv."
    )

make_submission(model)
