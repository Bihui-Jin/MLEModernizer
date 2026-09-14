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
import cv2
import numpy as np
import torch

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # rasterio not installed in the environment
    rasterio = None

    class Window:
        @staticmethod
        def from_slices(row_slice, col_slice):
            return (row_slice, col_slice)


from skimage import io, draw
import matplotlib.pyplot as plt
import dataclasses
import os
import glob
import json
import gc
from torch.utils.data import Dataset

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


class HuBMAPDatasetPreprocessing(Dataset):
    def __init__(self, image, category="train", use_compressed=False):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        if use_compressed:
            self.folder = "/kaggle/input/compressedhubmap"
        else:
            self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category
        self._file_ids = []
        self.reduce = 4 if not use_compressed else 1
        self._image = None
        self._get_image()
        if self._image and getattr(self._image, "count", None) != 3:
            subdatasets = self._image.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))

    def _read_geometry(self):
        coords = [f["geometry"] for f in self._get_masks()]
        if self.use_compressed:
            for geom in coords:
                for coord in geom["coordinates"][0]:
                    coord[0] = coord[0] // 4
                    coord[1] = coord[1] // 4
        return coords

    def _get_image(self):
        if self._image:
            return self._image
        if self.use_compressed:
            file_path = os.path.join(
                self.folder, self.category, f"compressed-{self.image}.tiff"
            )
        else:
            file_path = os.path.join(self.folder, self.category, f"{self.image}.tiff")
        if rasterio is None:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image not found: {file_path}")
            self._image = img
            self.shape = img.shape[:2]
        else:
            self._image = rasterio.open(file_path)
            self.shape = self._image.shape
        self.sz = 256 * self.reduce
        self.pad0 = self.sz - self.shape[0] % self.sz
        self.pad1 = self.sz - self.shape[1] % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz
        if self.category == "train":
            self.mask = rasterio.mask.mask(self._image, self._read_geometry())[0] > 0
        return self._image

    def close(self):
        if rasterio is not None and hasattr(self._image, "close"):
            self._image.close()

    def _get_masks(self):
        file_path = os.path.join(self.mask_folder, self.category, f"{self.image}.json")
        return json.load(open(file_path))

    def __del__(self):
        self.close()
        del self._image

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        image = self._get_image()
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        if rasterio is None:
            img_slice = image[p00:p01, p10:p11, :]
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_slice
        else:
            if image.count == 3:
                img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = np.moveaxis(
                    image.read(
                        [1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))
                    ),
                    0,
                    -1,
                )
            else:
                for i, layer in enumerate(self.layers):
                    img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0), i] = (
                        layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))
                    )

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )
        if self.category == "train":
            mask = np.zeros((3, self.sz, self.sz), np.uint8)
            mask[:, (p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self.mask[
                :, p00:p01, p10:p11
            ]
            mask = np.moveaxis(mask, 0, -1)
            if self.reduce != 1:
                mask = cv2.resize(
                    mask,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )
        else:
            mask = torch.zeros(1)

        result_tensor = torch.from_numpy((img / 255.0 - mean) / std).float()
        result_tensor = result_tensor.permute(2, 0, 1).float()
        if (img[..., 1] > 40).sum() <= 1000 or img.sum() <= 1000:
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
    
    if idx == -1 or mask.sum() == 0:
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
import torchvision.transforms.functional as TF
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.segmentation import fcn_resnet101

import json
import torch.nn.functional as F

""" Parts of the U-Net model """


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




## === cell 4
import pandas as pd
import csv
import cv2
from skimage import io as skio


def rle_encode_less_memory(img):
    """
    img: numpy array with shape (H,W), values {0,1}
    Returns run‑length encoding string.
    """
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_test_ids():
    """Yield image IDs from the sample submission file."""
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    with open(submission_file) as sf:
        reader = csv.reader(sf)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            yield image_id


def load_image(image_id):
    """Load a .tiff image, down‑scale if it exceeds a safe size, and return a torch tensor."""
    path = f"/kaggle/input/hubmap-kidney-segmentation/test/{image_id}.tiff"
    try:
        img = skio.imread(path)
    except Exception as e:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(
                f"Image not found or unsupported format: {path}"
            ) from e

    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.ndim == 2:  # grayscale fallback
        img = np.stack([img] * 3, axis=-1)

    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024 / max_dim
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img
        new_h, new_w = h, w

    img_resized = img_resized.astype(np.float32) / 255.0
    norm = (img_resized - mean) / std
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor, (h, w), (new_h, new_w)


def make_submission(model):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ids, preds = [], []
    for image_id in get_test_ids():
        print(f"Processing {image_id}")
        img_tensor, (orig_h, orig_w), (proc_h, proc_w) = load_image(image_id)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            prob = torch.sigmoid(logits)
            mask = (prob > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        if (proc_h, proc_w) != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        rle = rle_encode_less_memory(mask)
        ids.append(image_id)
        preds.append(rle)

    submission = pd.DataFrame({"id": ids, "predicted": preds})
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


model = Model()
pretrained_path = "/kaggle/input/hubmapmodel/model (12).pkl"
if os.path.exists(pretrained_path):
    try:
        data_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
        model.load_state_dict(data_dict["model_state_dict"])
        print("Pretrained weights loaded.")
    except Exception as e:
        print(f"Could not load pretrained weights ({e}); using untrained model.")
else:
    print("Pretrained weight file not found; using untrained model.")

make_submission(model)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/947104322.py in load_image(image_id)
     35     try:
---> 36         img = skio.imread(path)
     37     except Exception as e:

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in fixed_func(*args, **kwargs)
    327 
--> 328             return func(*args, **kwargs)
    329 

/usr/local/lib/python3.11/dist-packages/skimage/io/_io.py in imread(fname, as_gray, plugin, **plugin_args)
     81     with file_or_url_context(fname) as fname, _hide_plugin_deprecation_warnings():
---> 82         img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
     83 

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in wrapped(*args, **kwargs)
    537             warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
--> 538             return func(*args, **kwargs)
    539 

/usr/local/lib/python3.11/dist-packages/skimage/io/manage_plugins.py in call_plugin(kind, *args, **kwargs)
    253 
--> 254     return func(*args, **kwargs)
    255 

/usr/local/lib/python3.11/dist-packages/skimage/io/_plugins/tifffile_plugin.py in imread(fname, **kwargs)
     73 
---> 74     return tifffile_imread(fname, **kwargs)

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in imread(files, selection, aszarr, key, series, level, squeeze, maxworkers, buffersize, mode, name, offset, size, pattern, axesorder, categories, imread, imreadargs, sort, container, chunkshape, chunkdtype, axestiled, ioworkers, chunkmode, fillvalue, zattrs, multiscales, omexml, out, out_inplace, _multifile, _useframes, **kwargs)
   1237                     return zarr_selection(store, selection, out=out)
-> 1238                 return tif.asarray(
   1239                     key=key,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, key, series, level, squeeze, out, maxworkers, buffersize)
   4541                 raise ValueError('page is None')
-> 4542             result = page0.asarray(
   4543                 out=out, maxworkers=maxworkers, buffersize=buffersize

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, out, squeeze, lock, maxworkers, buffersize)
   8884 
-> 8885             for _ in self.segments(
   8886                 func=func,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in segments(self, lock, maxworkers, func, sort, buffersize, _fullsize)
   8698                 ):
-> 8699                     yield from executor.map(decode, segments)
   8700 

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    316         try:
--> 317             return fut.result(timeout)
    318         finally:

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    400             try:
--> 401                 raise self._exception
    402             finally:

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode(args, decodeargs, decode)
   8671             def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
-> 8672                 return func(decode(*args, **decodeargs))
   8673 

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode_raise_compression(exc, *args, **kwargs)
   8093             def decode_raise_compression(*args, exc=str(exc)[1:-1], **kwargs):
-> 8094                 raise ValueError(f'{exc}')
   8095 

ValueError: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package

During handling of the above exception, another exception occurred:

error                                     Traceback (most recent call last)
/tmp/ipykernel_55/947104322.py in <cell line: 0>()
    105     print("Pretrained weight file not found; using untrained model.")
    106 
--> 107 make_submission(model)
    108 
    109 

/tmp/ipykernel_55/947104322.py in make_submission(model)
     72     for image_id in get_test_ids():
     73         print(f"Processing {image_id}")
---> 74         img_tensor, (orig_h, orig_w), (proc_h, proc_w) = load_image(image_id)
     75         img_tensor = img_tensor.to(device)
     76 

/tmp/ipykernel_55/947104322.py in load_image(image_id)
     37     except Exception as e:
     38         # Fallback to OpenCV when skimage cannot handle JPEG‑compressed TIFFs
---> 39         img = cv2.imread(path, cv2.IMREAD_COLOR)
     40         if img is None:
     41             raise FileNotFoundError(

error: OpenCV(4.12.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:79: error: (-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS in function 'validateInputImageSize'


## === cell 5
"""
from skimage import io
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

trainer = Trainer()
#model = Model()
#data_dict = torch.load('models/model5661.pkl')
#model.load_state_dict(data_dict['model_state_dict'])
#model#.to('cuda:0')
input = torch.Tensor(subimage).permute(2,0,1).unsqueeze(0)#.to('cuda:0')
prediction = trainer.predict(input)

pred = prediction.detach().cpu().squeeze().permute(1,2,0)
subplots[1].imshow(pred)
pred_2 = trainer.model(input).detach().cpu().squeeze()
counter = 0
print(pred_2)
subplots[2].imshow(pred_2 > 0)
model_weights = []
conv_layers = []
# append all the conv layers and their respective weights to the list
def rec(model):
    global counter, model_weights, conv_layers
    if type(model) in [nn.Conv2d]:
        counter += 1
        model_weights.append(model.weight)
        conv_layers.append(model)
        return
    if type(model) == nn.Upsample:
        conv_layers.append(model)
    model_children = list(model.children())
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                rec(model_children[i][j])
        else:
            rec(model_children[i])

rec(model)
print(f"Total convolutional layers: {len(conv_layers)}")
print(f"Total layers: {len(model_weights)}")
# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    # visualize the first conv layer filters

#plt.figure(figsize=(20, 17))
#for i in enumerate(model_weights[0]):
    #plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)%%
    #print(filter[0, :,:])
    #plt.imshow(filter[0, :, :].detach(), cmap='gray')
    #plt.axis('off')
    #plt.savefig('filter.png')
#plt.show()    
# pass the image through all the layers
results = [conv_layers[0](input)]
try:
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        res = conv_layers[i](results[-1])
        print(f'calculated {i}, conv_layert: {conv_layers[i]}, result_shape: {res.shape}')
        results.append(res)
except:
    pass
# make a copy of the `results`
outputs = results
# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print('size', layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"layer_{num_layer}.png")
    # plt.show()
    plt.close()
"""
