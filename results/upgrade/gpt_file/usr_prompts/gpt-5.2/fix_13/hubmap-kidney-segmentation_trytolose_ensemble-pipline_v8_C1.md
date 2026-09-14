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

albumentations==2.0.8
geopandas==0.14.4
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
tqdm==4.67.1

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
import sys
import math
import gc
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class _TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()

        def C(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = C(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = C(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = C(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = C(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = C(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = C(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = C(base * 2, base)

        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


class _SMPShim:
    class Unet(nn.Module):
        def __init__(self, encoder_name, encoder_weights=None):
            super().__init__()
            self.model = _TinyUNet(in_channels=3, out_channels=1)

        def forward(self, x):
            return self.model(x)


smp = _SMPShim()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(False)  # keep original semantics
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True




## === cell 1
def _safe_system(cmd: str):
    ret = os.system(cmd)
    if ret != 0:
        print(f"[WARN] Command failed (ignored): {cmd}")
    return ret


_safe_system("ls -l ../input/hubmap-folds-2 || true")
_safe_system("mkdir -p model_1")
_safe_system("tar -xvf ../input/hubmap-folds-2/1024_avg_last.tar -C model_1 || true")
_safe_system("mkdir -p model_2")
_safe_system(
    "tar -xvf ../input/hubmap-folds-2/1024_512_pseudo_v1_avg.tar -C model_2 || true"
)



## === cell 2
BATCH_SIZE = 4 if torch.cuda.is_available() else 1

if torch.cuda.is_available():
    NUM_WORKERS = min(4, (os.cpu_count() or 4))
else:
    NUM_WORKERS = min(2, (os.cpu_count() or 2))

CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.4
VOTE = 0
SKIP_BACKGROUND = True

SKIP_COMMIT = False

df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
if "img" in df_sub.columns and "pixels" in df_sub.columns:
    df_sub = df_sub.rename(columns={"img": "id", "pixels": "predicted"})
print(df_sub.head(), df_sub.shape)




## === cell 3
@dataclass
class ModelConfig:
    encoder: str
    weights_path: List[str]
    img_size: int
    weight_blend: float


my_models = [
    ModelConfig(
        "resnet34",
        [
            "./model_1/fold0_avg_0.9238.pth",
            "./model_1/fold1_avg_0.9162.pth",
            "./model_1/fold2_avg_0.9303.pth",
            "./model_1/fold3_avg_0.9336.pth",
            "./model_1/fold4_avg_0.9407.pth",
        ],
        4096,
        0.4,
    ),
    ModelConfig(
        "timm-efficientnet-b3",
        [
            "../input/hubmap5/exp_25_fold_0.pt",
            "../input/hubmap5/exp_24_fold_1.pt",
            "../input/hubmap5/exp_23_fold_2.pt",
            "../input/hubmap5/exp_22_fold_3.pt",
            "../input/hubmap5/exp_21_fold_4.pt",
        ],
        1024,
        0.2,
    ),
    ModelConfig(
        "se_resnext50_32x4d",
        [
            "./model_2/fold0_avg_0.9457.pth",
            "./model_2/fold1_avg_0.9566.pth",
            "./model_2/fold2_avg_0.9379.pth",
            "./model_2/fold3_avg_0.9095.pth",
            "./model_2/fold4_avg_0.9338.pth",
        ],
        2048,
        0.4,
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models])
print("TRAIN_IMG_SIZE:", TRAIN_IMG_SIZE)




## === cell 4
def rle_encode_less_memory(img):
    pixels = img.T.reshape(-1)
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.flatnonzero(pixels[1:] != pixels[:-1]) + 2
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.tolist()))


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)




## === cell 5
def _check_background_fast_from_crop(img_bgr: np.ndarray, crop_size: int) -> bool:
    s_th = 40
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ss = hsv[:, :, 1]
    return False if (ss > s_th).sum() <= p_th or img_bgr.sum() <= p_th else True


def _read_tiff_bgr_uint8(tiff_path: str, downscale: int = 1) -> np.ndarray:
    tiff_path = str(tiff_path)

    try:
        import tifffile  # type: ignore

        arr = tifffile.imread(tiff_path)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Unexpected TIFF shape: {arr.shape}")

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        arr = arr[..., ::-1].copy()  # RGB->BGR

        if downscale > 1:
            h, w = arr.shape[:2]
            nh = max(1, h // downscale)
            nw = max(1, w // downscale)
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        return arr
    except Exception as e:
        print(f"[WARN] tifffile read failed for {tiff_path}: {type(e).__name__}: {e}")

    try:
        from PIL import Image  # type: ignore

        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(tiff_path).convert("RGB")
        if downscale > 1:
            w, h = img.size
            img = img.resize(
                (max(1, w // downscale), max(1, h // downscale)),
                resample=Image.BILINEAR,
            )
        arr = np.array(img, dtype=np.uint8)
        arr = arr[..., ::-1].copy()  # RGB->BGR
        return arr
    except Exception as e:
        print(f"[WARN] PIL read failed for {tiff_path}: {type(e).__name__}: {e}")

    img = cv2.imread(tiff_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read tiff with any backend: {tiff_path}")
    if downscale > 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (max(1, w // downscale), max(1, h // downscale)),
            interpolation=cv2.INTER_AREA,
        )
    return img


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = int(crop_size)
        self.all_img_sizes = sorted(list(all_img_sizes))
        self.step = int(step)

        if self.crop_size >= 4096:
            self._down = 4
        elif self.crop_size >= 2048:
            self._down = 2
        else:
            self._down = 1

        self._crop_ds = max(1, self.crop_size // self._down)
        self._step_ds = max(1, self.step // self._down)

        img = _read_tiff_bgr_uint8(tiff_path, downscale=self._down)
        self.img = img  # BGR uint8 (downscaled)
        self.h_ds, self.w_ds = img.shape[:2]

        self.h = self.h_ds * self._down
        self.w = self.w_ds * self._down

        self.row_count = 1 + math.ceil((self.h_ds - self._crop_ds) / self._step_ds)
        self.col_count = 1 + math.ceil((self.w_ds - self._crop_ds) / self._step_ds)

        coords = []
        for idx in range(self.row_count * self.col_count):
            y = (idx // self.col_count) * self._step_ds
            x = (idx % self.col_count) * self._step_ds
            if x + self._crop_ds > self.w_ds:
                x = self.w_ds - self._crop_ds
            if y + self._crop_ds > self.h_ds:
                y = self.h_ds - self._crop_ds
            coords.append((int(x), int(y)))
        self.coords: List[Tuple[int, int]] = coords

        if SKIP_BACKGROUND:
            s_th = 40
            p_th = 1000 * (self._crop_ds // 256) ** 2

            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            ss = hsv[:, :, 1]
            sat_mask = (ss > s_th).astype(np.uint8)

            sat_ii = cv2.integral(sat_mask, sdepth=cv2.CV_32S)  # (H+1,W+1)
            sum_ii = cv2.integral(self.img, sdepth=cv2.CV_64F)  # (H+1,W+1,3)

            coords_np = np.asarray(self.coords, dtype=np.int32)
            xs = coords_np[:, 0]
            ys = coords_np[:, 1]
            cs = int(self._crop_ds)

            x2 = np.minimum(self.w_ds, xs + cs)
            y2 = np.minimum(self.h_ds, ys + cs)

            sat_cnt = (
                sat_ii[y2, x2] - sat_ii[ys, x2] - sat_ii[y2, xs] + sat_ii[ys, xs]
            ).astype(np.int64)

            s = (
                sum_ii[y2, x2, :]
                - sum_ii[ys, x2, :]
                - sum_ii[y2, xs, :]
                + sum_ii[ys, xs, :]
            )
            bgr_sum = s.sum(axis=1)

            nb = ~((sat_cnt <= p_th) | (bgr_sum <= p_th))
            self.not_background = nb.astype(np.bool_)

            del (
                hsv,
                ss,
                sat_mask,
                sat_ii,
                sum_ii,
                coords_np,
                xs,
                ys,
                x2,
                y2,
                sat_cnt,
                s,
                bgr_sum,
            )
        else:
            self.not_background = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x_ds, y_ds = self.coords[idx]

        img_crop = self.img[y_ds : y_ds + self._crop_ds, x_ds : x_ds + self._crop_ds]
        if img_crop.shape[0] != self._crop_ds or img_crop.shape[1] != self._crop_ds:
            pad_h = self._crop_ds - img_crop.shape[0]
            pad_w = self._crop_ds - img_crop.shape[1]
            img_crop = cv2.copyMakeBorder(
                img_crop,
                0,
                max(0, pad_h),
                0,
                max(0, pad_w),
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        x, y = int(x_ds * self._down), int(y_ds * self._down)
        xy = torch.tensor([x, y], dtype=torch.int32)

        if SKIP_BACKGROUND:
            nb = bool(self.not_background[idx])
        else:
            nb = True

        return img_crop, xy, nb




## === cell 6
_DISABLE_TORCH_COMPILE = True


def inference(data_loader, models_with_config, crop_size):
    ds = data_loader.dataset
    img_h, img_w = ds.h, ds.w
    mask_pred = np.zeros((img_h, img_w), dtype=np.uint8)

    device = DEVICE
    thr = THR
    train_sz = TRAIN_IMG_SIZE
    do_skip_bg = SKIP_BACKGROUND is True
    f_interp = torch.nn.functional.interpolate
    cv_resize = cv2.resize

    use_cuda = device.type == "cuda"
    amp_dtype = torch.float16 if use_cuda else torch.float32

    mean_t = torch.as_tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.as_tensor(_STD, device=device).view(1, 3, 1, 1)

    groups = []
    for cfg, model_group in models_with_config:
        inv_n = 1.0 / len(model_group)
        scale = float(cfg.weight_blend * inv_n)
        need_down = cfg.img_size != train_sz
        groups.append((cfg.img_size, scale, need_down, model_group))

    crop_ds = getattr(ds, "_crop_ds", crop_size)
    down = getattr(ds, "_down", 1)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=use_cuda
    ):
        for img_crop_bgr_u8, xy, not_bg in tqdm(data_loader, ncols=70, leave=True):
            if do_skip_bg:
                keep = not_bg.bool()
                if keep.ndim == 0:
                    keep = keep.unsqueeze(0)
                if not keep.any().item():
                    continue
                img_crop_bgr_u8 = img_crop_bgr_u8[keep]
                xy = xy[keep]

            img_crop_bgr_u8 = img_crop_bgr_u8.to(device, non_blocking=True)

            x = img_crop_bgr_u8.permute(0, 3, 1, 2).contiguous().to(torch.float32)
            x = x * (1.0 / 255.0)
            x = x[:, [2, 1, 0], :, :]  # BGR -> RGB

            if crop_ds != train_sz:
                x = f_interp(
                    x, size=(train_sz, train_sz), mode="bilinear", align_corners=False
                )

            x = (x - mean_t) / std_t
            img_4096 = x

            img_2048 = None
            img_1024 = None

            pred_total = None
            for img_size, scale, need_down, model_group in groups:
                if need_down:
                    if img_size == 2048:
                        if img_2048 is None:
                            img_2048 = f_interp(
                                img_4096,
                                size=(2048, 2048),
                                mode="bilinear",
                                align_corners=False,
                            )
                        image = img_2048
                    elif img_size == 1024:
                        if img_1024 is None:
                            img_1024 = f_interp(
                                img_4096,
                                size=(1024, 1024),
                                mode="bilinear",
                                align_corners=False,
                            )
                        image = img_1024
                    else:
                        image = f_interp(
                            img_4096,
                            size=(img_size, img_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                else:
                    image = img_4096

                pred_group = None
                for model in model_group:
                    pred = model(image).sigmoid()
                    if need_down:
                        pred = f_interp(
                            pred,
                            size=(train_sz, train_sz),
                            mode="bilinear",
                            align_corners=False,
                        )
                    pred_group = pred if pred_group is None else (pred_group + pred)

                pred_group = pred_group * scale
                pred_total = (
                    pred_group if pred_total is None else (pred_total + pred_group)
                )

            pred_bin = (
                (pred_total.squeeze(1) > thr).to(torch.uint8).detach().cpu().numpy()
            )
            xy_np = xy.detach().cpu().numpy().astype(np.int32)

            for i in range(pred_bin.shape[0]):
                x0, y0 = int(xy_np[i, 0]), int(xy_np[i, 1])
                predict_single = pred_bin[i]

                if crop_ds != train_sz:
                    predict_single = cv_resize(
                        predict_single,
                        (crop_ds, crop_ds),
                        interpolation=cv2.INTER_NEAREST,
                    )

                if down != 1:
                    predict_single = predict_single.repeat(down, axis=0).repeat(
                        down, axis=1
                    )

                if (
                    predict_single.shape[0] != crop_size
                    or predict_single.shape[1] != crop_size
                ):
                    predict_single = cv_resize(
                        predict_single,
                        (crop_size, crop_size),
                        interpolation=cv2.INTER_NEAREST,
                    )

                mask_pred[y0 : y0 + crop_size, x0 : x0 + crop_size] += predict_single

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred)
    del mask_pred
    gc.collect()
    return mask_rle




## === cell 7
all_img_sizes = set([cfg.img_size for cfg in my_models])
models_with_config = []


def _load_state_safely(model: nn.Module, w_path: str):
    if not Path(w_path).exists():
        print(f"[WARN] Missing weights: {w_path} (using random init)")
        return
    state = torch.load(w_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[WARN] Non-strict load for {w_path}: missing={len(missing)} unexpected={len(unexpected)}"
        )


_CAN_COMPILE = hasattr(torch, "compile") and (not _DISABLE_TORCH_COMPILE)

for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        model = smp.Unet(cfg.encoder, encoder_weights=None).to(DEVICE)
        _load_state_safely(model, w_path)
        model.eval()
        if _CAN_COMPILE:
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(
                    f"[WARN] torch.compile failed for {w_path}: {type(e).__name__}: {e}"
                )
        models_group.append(model)
    models_with_config.append((cfg, models_group))

print(
    "Loaded model groups:",
    len(models_with_config),
    "all_img_sizes:",
    sorted(all_img_sizes),
)



## === cell 8
count_thr = 5 if SKIP_COMMIT is True else 4
n_to_run = min(len(df_sub), count_thr) if SKIP_COMMIT else len(df_sub)


def _collate_minibatch(batch):
    imgs, xys, nbs = zip(*batch)  # imgs are HWC uint8 numpy arrays
    imgs_np = np.asarray(imgs, dtype=np.uint8)
    if imgs_np.ndim != 4:
        imgs_np = np.stack(imgs, axis=0)
    imgs_t = torch.from_numpy(imgs_np)  # uint8 tensor (CPU)
    xys = torch.stack(xys, dim=0)
    nbs = torch.tensor(nbs, dtype=torch.bool)
    return imgs_t, xys, nbs


for idx in range(n_to_run):
    row = df_sub.iloc[idx]
    tiff_path = f"../input/hubmap-kidney-segmentation/test/{row['id']}.tiff"
    if not Path(tiff_path).exists():
        print(f"[WARN] Missing test tiff: {tiff_path}. Writing empty prediction.")
        df_sub.loc[df_sub.index[idx], "predicted"] = ""
        continue

    test_ds = SingleTiffDataset(
        tiff_path=tiff_path,
        all_img_sizes=all_img_sizes,
        crop_size=CROP_SIZE,
        step=STEP,
    )

    def _make_loader(bs: int):
        return DataLoader(
            dataset=test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
            collate_fn=_collate_minibatch,
        )

    try:
        test_loader = _make_loader(BATCH_SIZE)
        rle = inference(test_loader, models_with_config, CROP_SIZE)
    except torch.OutOfMemoryError as e:
        print(f"[WARN] OOM at BATCH_SIZE={BATCH_SIZE}. Retrying with BATCH_SIZE=1. {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_loader = _make_loader(1)
        rle = inference(test_loader, models_with_config, CROP_SIZE)

    df_sub.loc[df_sub.index[idx], "predicted"] = rle

if n_to_run < len(df_sub):
    df_sub.loc[df_sub.index[n_to_run:], "predicted"] = ""



## === cell 9
_safe_system("rm -rf model_1 model_2 || true")



## === cell 10
out_path = "submission.csv"
if "id" not in df_sub.columns or "predicted" not in df_sub.columns:
    raise ValueError(f"Unexpected submission columns: {df_sub.columns.tolist()}")
df_sub = df_sub[["id", "predicted"]]
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", df_sub.shape)
print(df_sub.head())



## === cell 11
df_sub
