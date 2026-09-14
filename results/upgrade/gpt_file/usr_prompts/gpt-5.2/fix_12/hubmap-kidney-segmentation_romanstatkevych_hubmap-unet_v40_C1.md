# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import cv2
import csv
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

import tifffile

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

try:
    import rasterio
    from rasterio.windows import Window
except Exception as _e:
    rasterio = None
    Window = None


def _ensure_hwc_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """
    Bug fix:
    - Some TIFFs decode to 4D/5D arrays (pages/levels/channels) depending on codec.
    - Convert to a single HxWx3 RGB uint8 image by selecting a reasonable slice/page.
    This preserves core logic (still using raw image pixels) but prevents crashes.
    """
    if img is None:
        raise ValueError("img is None")

    img = np.asarray(img)

    while img.ndim > 3:
        img = img[0]

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        if img.shape[-1] >= 4:
            img = img[..., :3]

        if img.shape[-1] != 3:
            ch0 = img[..., 0]
            img = np.stack([ch0, ch0, ch0], axis=-1)
    else:
        raise ValueError(f"Unexpected image ndim={img.ndim}")

    if img.dtype != np.uint8:
        imgf = img.astype(np.float32)
        mx = float(np.max(imgf)) if imgf.size else 0.0
        if mx <= 1.0:
            imgf = imgf * 255.0
        elif mx > 255.0:
            imgf = imgf / mx * 255.0
        img = np.clip(imgf, 0, 255).astype(np.uint8)

    return img


def _safe_imread_rgb(path: str) -> np.ndarray:
    """
    Robust TIFF reader returning HxWx3 uint8 RGB.

    Bug fix:
    - tifffile requires 'imagecodecs' for JPEG-compressed TIFFs; Kaggle env here lacks it.
    - Use OpenCV's imreadmulti which can decode many compressed TIFFs without imagecodecs.
    - Additionally, handle multi-page / multi-dimensional TIFFs (ndim>3) by selecting a slice.
    """
    img = None

    if rasterio is not None:
        try:
            with rasterio.open(path) as src:
                count = src.count
                if count >= 3:
                    arr = src.read([1, 2, 3])  # (3,H,W)
                    img = np.transpose(arr, (1, 2, 0))
                else:
                    arr1 = src.read(1)
                    img = np.stack([arr1, arr1, arr1], axis=-1)
        except Exception:
            img = None

    if img is None:
        try:
            ok, pages = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
            if ok and pages:
                img0 = pages[0]
                img0 = _ensure_hwc_rgb_uint8(img0)
                img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        except Exception:
            img = None

    if img is None:
        try:
            img = tifffile.imread(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read TIFF. Path={path}. "
                f"OpenCV+rasterio+tifffile all failed. Last error={e}"
            )

    img = _ensure_hwc_rgb_uint8(img)

    return img


def rle_decode(mask_rle: str, shape: tuple) -> np.ndarray:
    """
    Decode run-length encoding into a binary mask.
    Kaggle convention for this competition: pixels numbered top-to-bottom then left-to-right,
    which corresponds to flattening the transposed array (img.T.flatten()) used in encoding.
    """
    if mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)
    s = str(mask_rle).strip()
    if s == "" or s.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    arr = np.asarray(list(map(int, s.split())), dtype=np.int64)
    starts = arr[0::2] - 1
    lengths = arr[1::2]
    ends = starts + lengths

    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        if lo < 0:
            lo = 0
        if hi > flat.size:
            hi = flat.size
        if lo < hi:
            flat[lo:hi] = 1

    return flat.reshape((shape[1], shape[0])).T


class HuBMAPDatasetPreprocessing(Dataset):
    """
    Core logic preserved: same tiling/padding, same normalization,
    same "skip low saturation" filter, and same downscale behavior via self.reduce.

    Score fix:
    - For training, actually populate tile masks from train.csv RLE so training can learn.
    """

    def __init__(self, image, category="train", use_compressed=False, rle=None):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed  # kept for API compatibility
        self.image = str(image)
        self.category = category
        self.reduce = 4 if not use_compressed else 1

        candidates = []
        if self.category == "train":
            candidates = [
                os.path.join(self.folder, "train_images", f"{self.image}.tiff"),
                os.path.join(self.folder, "train", f"{self.image}.tiff"),
            ]
        else:
            candidates = [
                os.path.join(self.folder, "test_images", f"{self.image}.tiff"),
                os.path.join(self.folder, "test", f"{self.image}.tiff"),
            ]

        self.file_path = None
        for p in candidates:
            if os.path.exists(p):
                self.file_path = p
                break
        if self.file_path is None:
            raise FileNotFoundError(
                f"Missing TIFF for id={self.image}. Tried: {candidates}"
            )

        self._img_cache = _safe_imread_rgb(self.file_path)  # HxWx3 uint8
        self.shape = (self._img_cache.shape[0], self._img_cache.shape[1])

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        self._full_mask = None
        if self.category == "train":
            if rle is not None:
                self._full_mask = rle_decode(rle, self.shape)  # HxW uint8
            else:
                self._full_mask = np.zeros(self.shape, dtype=np.uint8)

    def close(self):
        self._img_cache = None
        self._full_mask = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz

        img = np.zeros((self.sz, self.sz, 3), np.uint8)

        H, W = self.shape
        x1 = x0 + self.sz
        y1 = y0 + self.sz

        px0, px1 = max(0, x0), min(x1, H)
        py0, py1 = max(0, y0), min(y1, W)

        if px0 < px1 and py0 < py1:
            img[(px0 - x0) : (px1 - x0), (py0 - y0) : (py1 - y0), :] = self._img_cache[
                px0:px1, py0:py1, :
            ]

        if self.category == "train":
            m = np.zeros((self.sz, self.sz), np.uint8)
            if self._full_mask is not None and px0 < px1 and py0 < py1:
                m[(px0 - x0) : (px1 - x0), (py0 - y0) : (py1 - y0)] = self._full_mask[
                    px0:px1, py0:py1
                ]
        else:
            m = None

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )
            if self.category == "train":
                m = cv2.resize(
                    m,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_NEAREST,
                )

        if self.category == "train":
            mask = torch.from_numpy(m[None, ...].astype(np.float32))  # 1xhxw float
        else:
            mask = torch.zeros(1)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    sample = sample.permute(1,2,0)

    if idx == -1:
        continue
    if c == 5:
        break

    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    c += 1
"""



## === cell 2
"""
import matplotlib.pyplot as plt
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
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data


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
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
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
        super().__init__()
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
import collections
import pandas as pd

OUTPUT_PATH = "/kaggle/working/models"
VALSET = ["e79de561c", "cb2d976f4"]


def collate_fn(x):
    x = list(filter(lambda z: z[-1] != -1, x))
    if not x:
        return []
    return data.dataloader.default_collate(x)


TRAIN_CSV_PATH = "/kaggle/input/hubmap-kidney-segmentation/train.csv"
if os.path.exists(TRAIN_CSV_PATH):
    _train_df = pd.read_csv(TRAIN_CSV_PATH)
    _id_col = "id" if "id" in _train_df.columns else _train_df.columns[0]
    _rle_col = (
        "encoding"
        if "encoding" in _train_df.columns
        else ("rle" if "rle" in _train_df.columns else _train_df.columns[1])
    )
    TRAIN_RLE_MAP = dict(
        zip(_train_df[_id_col].astype(str).tolist(), _train_df[_rle_col].tolist())
    )
else:
    TRAIN_RLE_MAP = {}


def get_tiff_images():
    """
    Bug fix:
    - Skip any single unreadable train TIFF instead of crashing training.
    """
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train_images/*.tiff"
    fallback_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    paths = glob.glob(input_folder)
    if not paths:
        paths = glob.glob(fallback_folder)
    for images in paths:
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name in VALSET:
            continue
        try:
            dataset = HuBMAPDatasetPreprocessing(
                name,
                category="train",
                use_compressed=True,
                rle=TRAIN_RLE_MAP.get(name, ""),
            )
            yield dataset
        except Exception as e:
            print(f"WARNING: skipping train image {name} due to read error: {e}")
            continue


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(
            image,
            category="train",
            use_compressed=True,
            rle=TRAIN_RLE_MAP.get(str(image), ""),
        )
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


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)
        self.model = Model().to(self.device)

        self.epochs = 2  # minimal to move score off 0.0 within time constraints
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (9).pkl"
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []

        self.loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([5]).to(self.device)
        )
        if os.path.exists(self._original_model_path):
            self._data_dict = torch.load(
                self._original_model_path, map_location=self.device
            )
            self.start_positions = self._data_dict.get("epoch", 0)
            if "optimizer_state_dict" in self._data_dict:
                self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            if "model_state_dict" in self._data_dict:
                self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0

    def evaluate(self, epoch):
        val_set = get_val_set()
        self.model.eval()
        with torch.no_grad():
            for ds in val_set:
                loader = data.DataLoader(ds, 8, pin_memory=True, collate_fn=collate_fn)
                for sample in loader:
                    if not sample:
                        continue
                    x, y, idx = sample
                    x = x.to(self.device)
                    _ = self.model(x)
                    del x, y, idx, _
        print(f"epoch {epoch} evaluation done (metric not computed here).")
        self.model.train()

    def train(self):
        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        self.model.train()
        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            losses_stats = collections.defaultdict(float)
            n_batches = 0

            for dataset in get_tiff_images():
                try:
                    loader = data.DataLoader(
                        dataset, 8, pin_memory=True, collate_fn=collate_fn
                    )
                    for batch_ndx, sample in enumerate(loader):
                        if not sample:
                            continue
                        x, y, idx = sample
                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)

                        self.optim.zero_grad(set_to_none=True)
                        logits = self.model(x)
                        loss_val = self.loss(logits, y)
                        loss_val.backward()
                        self.optim.step()

                        losses_stats["bce"] += float(loss_val.detach().cpu().item())
                        n_batches += 1

                        del x, y, idx, logits, loss_val
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            if n_batches > 0:
                losses_stats["bce"] /= n_batches
            print("losses_stats", dict(losses_stats))

            self.evaluate(i)

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




## === cell 5
torch.cuda.empty_cache()
trainer = Trainer()
trainer.train()



## === cell 6
import pandas as pd


def collate_fn(x):
    x = list(filter(lambda z: z[-1] != -1, x))
    if not x:
        return []
    return data.dataloader.default_collate(x)


def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    df = pd.read_csv(submission_file)

    id_col = (
        "id"
        if "id" in df.columns
        else ("img" if "img" in df.columns else df.columns[0])
    )

    for image_id in df[id_col].astype(str).tolist():
        print(f"openning {image_id}")
        try:
            dataset = HuBMAPDatasetPreprocessing(
                image_id, category="test", use_compressed=False
            )
            yield dataset
        except Exception as e:
            print(f"WARNING: failed to open {image_id}: {e}")
            yield (image_id, None)


def rle_encode_less_memory(img):
    """
    img: 2D numpy array or torch tensor, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if img is None:
        return ""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = (img > 0).astype(np.uint8)
    pixels = img.T.flatten()
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def _find_best_checkpoint():
    candidates = []
    if os.path.exists(OUTPUT_PATH):
        candidates.extend(sorted(glob.glob(os.path.join(OUTPUT_PATH, "model_*.pkl"))))
        if os.path.exists(os.path.join(OUTPUT_PATH, "model.pkl")):
            candidates.append(os.path.join(OUTPUT_PATH, "model.pkl"))
    best = None
    best_epoch = -1
    for p in candidates:
        bn = os.path.basename(p)
        if bn.startswith("model_") and bn.endswith(".pkl"):
            try:
                ep = int(bn[len("model_") : -len(".pkl")])
                if ep > best_epoch:
                    best_epoch = ep
                    best = p
            except Exception:
                continue
    if best is None:
        mp = os.path.join(OUTPUT_PATH, "model.pkl")
        if os.path.exists(mp):
            best = mp
    return best


def make_submission(model):
    sample_path = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    sample = pd.read_csv(sample_path)
    id_col = (
        "id"
        if "id" in sample.columns
        else ("img" if "img" in sample.columns else sample.columns[0])
    )

    names, preds = [], []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    model.eval()

    thr = 0.5

    with torch.no_grad():
        for ds in get_test_dataset():
            if isinstance(ds, tuple) and ds[1] is None:
                image_id = ds[0]
                names.append(image_id)
                preds.append("")  # empty mask if unreadable
                continue

            print(ds.image, len(ds))
            loader = data.DataLoader(ds, 16, collate_fn=collate_fn)

            tile_sz = ds.sz // ds.reduce
            tile_mask = torch.zeros((len(ds), tile_sz, tile_sz), dtype=torch.uint8)

            for batch_num, data_ in enumerate(loader):
                if not data_:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = data_
                image = image.to(dev)

                logits = model(image)  # (B,1,tile_sz,tile_sz)
                prob = torch.sigmoid(logits.squeeze(1))
                pred = (prob > thr).to(torch.uint8).cpu()

                for i, ndx in enumerate(idx):
                    tile_mask[int(ndx)] = pred[i]

                del image, logits, prob, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            full_mask_small = (
                tile_mask.view(ds.n0max, ds.n1max, tile_sz, tile_sz)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * tile_sz, ds.n1max * tile_sz)
            )

            if ds.reduce != 1:
                full_mask = torch.from_numpy(
                    cv2.resize(
                        full_mask_small.numpy(),
                        (ds.n1max * ds.sz, ds.n0max * ds.sz),
                        interpolation=cv2.INTER_NEAREST,
                    )
                )
            else:
                full_mask = full_mask_small

            full_mask = full_mask[
                ds.pad0
                // 2 : (-(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz),
                ds.pad1
                // 2 : (-(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz),
            ]

            rle = rle_encode_less_memory(full_mask)
            names.append(ds.image)
            preds.append(rle)

            del tile_mask, full_mask_small, full_mask, ds
            gc.collect()

    sub = pd.DataFrame(
        {
            id_col: names,
            ("predicted" if "predicted" in sample.columns else "pixels"): preds,
        }
    )

    if "id" in sample.columns and "predicted" in sample.columns:
        sub = sub.rename(columns={id_col: "id"})
        if "predicted" not in sub.columns:
            sub = sub.rename(columns={"pixels": "predicted"})
        sub = sub[["id", "predicted"]]
    else:
        if id_col != "img":
            sub = sub.rename(columns={id_col: "img"})
        if "pixels" not in sub.columns:
            sub = sub.rename(columns={"predicted": "pixels"})
        sub = sub[["img", "pixels"]]

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(sub), "rows and columns:", list(sub.columns))


model = trainer.model  # use the just-trained model
ckpt = _find_best_checkpoint()
if ckpt is not None and os.path.exists(ckpt):
    print("Loading checkpoint:", ckpt)
    data_dict = torch.load(ckpt, map_location=torch.device("cpu"))
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"])
    else:
        model.load_state_dict(data_dict)
else:
    print("No checkpoint found; using current model weights.")

make_submission(model)
