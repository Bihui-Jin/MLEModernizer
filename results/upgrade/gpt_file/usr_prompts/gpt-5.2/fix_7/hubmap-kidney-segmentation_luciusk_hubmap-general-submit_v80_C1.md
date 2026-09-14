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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
tifffile==2025.6.11
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

# 5. Target score

0.946787482005875

# 6. Current score

0.08161

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.08161) has done: 'Your current script doesn’t yield a score mainly because it’s likely not producing a valid (non-empty) submission: it points to `../input/hubmap-kidney-segmentation/test/*.tiff`, but this dataset version is JSON-based (no TIFFs), so inference falls into the exception path and writes empty RLEs. To keep core logic intact while making it run end-to-end, I add a minimal fallback that rasterizes the provided anatomical-structure JSON polygons into a binary mask (same output semantics: one binary mask per image, then RLE). I also fix the submission column names to match the provided `sample_submission.csv` (`id,predicted`) to ensure Kaggle accepts the file. This should move the score from “not yielded” to a valid baseline (not target-level, but toward it) without changing your model architecture/training logic.'

# 9. Code solution

## === cell 0
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
import gc
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import json

import warnings

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # type: ignore
except ModuleNotFoundError:

    class _SimpleUNet(nn.Module):
        def __init__(self, in_ch=3, out_ch=1, base=32):
            super().__init__()

            def conv_block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )

            self.enc1 = conv_block(in_ch, base)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = conv_block(base, base * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = conv_block(base * 2, base * 4)
            self.pool3 = nn.MaxPool2d(2)
            self.enc4 = conv_block(base * 4, base * 8)
            self.pool4 = nn.MaxPool2d(2)

            self.bottleneck = conv_block(base * 8, base * 16)

            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = conv_block(base * 16, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = conv_block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = conv_block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = conv_block(base * 2, base)

            self.outc = nn.Conv2d(base, out_ch, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            b = self.bottleneck(self.pool4(e4))

            d4 = self.up4(b)
            d4 = torch.cat([d4, e4], dim=1)
            d4 = self.dec4(d4)

            d3 = self.up3(d4)
            d3 = torch.cat([d3, e3], dim=1)
            d3 = self.dec3(d3)

            d2 = self.up2(d3)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)

            return self.outc(d1)

    class _SMPFallback:
        class Unet(_SimpleUNet):
            def __init__(
                self,
                encoder_name=None,
                encoder_weights=None,
                classes=1,
                in_channels=3,
                **kwargs,
            ):
                super().__init__(in_ch=in_channels, out_ch=classes, base=32)

    smp = _SMPFallback()



## === cell 1
sz = 256  # the size of tiles (model input)
reduce = 4  # reduce the original images by 4 times before model
TH = 0.5  # threshold for positive predictions
TRAIN = False

if TRAIN:
    DATA = "../input/hubmap-kidney-segmentation/train/"
    df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/train.csv")
else:
    DATA = "../input/hubmap-kidney-segmentation/test/"
    df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

MODELS = [
    f"../input/b2datanew/efficientnet-b2-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]
bs = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b2"
EXPAND = 4
minoverlap = 1 / 16
TTA = False
tile_size = int(sz * EXPAND)  # 1024 (what model sees after resize)
tile_resized = int(tile_size * reduce)  # 4096 (read from tiff before resize)




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None or (isinstance(enc, float) and np.isnan(enc)):
            continue
        s = str(enc).split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start : start + length] = 1 + m
    return img.reshape(shape).T


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(" ".join(str(x) for x in runs))
    return encs


def rle_encode_less_memory(img):
    if img is None:
        return ""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.max() == 0:
        return ""
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array of size (N,4), where N is number of tiles.
    Each row: x1,x2,y1,y2 on full-res image coordinates.
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)

    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


def _read_tiff_as_hwc(path):
    """
    Returns HxWxC uint8 array in RGB order.
    Tries PIL first (handles JPEG-compressed TIFF without imagecodecs),
    then falls back to tifffile and OpenCV.
    """
    img = None

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            img = np.array(im, dtype=np.uint8)
            return img
    except Exception:
        pass

    try:
        img = tiff.imread(path)
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.moveaxis(img, 0, -1)
    except Exception:
        img = None

    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]

    if img.dtype != np.uint8:
        if img.max() > 255:
            img = (img / (img.max() + 1e-6) * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


def _load_image_info():
    info_paths = [
        "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv",
        "../input/HuBMAP-20-dataset_information.csv",
    ]
    for p in info_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


_INFO_DF = _load_image_info()


def _get_hw_for_id(idx):
    if _INFO_DF is None:
        return None
    key = f"{idx}.tiff"
    m = _INFO_DF[_INFO_DF["image_file"] == key]
    if len(m) == 0:
        return None
    w = int(m.iloc[0]["width_pixels"])
    h = int(m.iloc[0]["height_pixels"])
    return h, w


def _rasterize_anatomical_json_to_mask(idx):
    hw = _get_hw_for_id(idx)
    if hw is None:
        return None
    h, w = hw
    jpath = os.path.join(DATA, f"{idx}-anatomical-structure.json")
    if not os.path.exists(jpath):
        return None

    with open(jpath, "r") as f:
        data = json.load(f)

    mask = np.zeros((h, w), dtype=np.uint8)
    for feat in data:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = coords[0]
        if ring is None or len(ring) < 3:
            continue
        pts = np.asarray(ring, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts_i], 1)
    return mask


class HuBMAPDataset(Dataset):
    """
    Core logic preserved for TIFF path; if TIFF not present, dataset init will fail and caller will fallback.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")
        self.img = _read_tiff_as_hwc(self.path)
        self.shape = self.img.shape[:2]  # (H, W)
        self.reduce = reduce
        self.sz = reduce * sz  # 1024
        self.mask_grid = make_grid(
            self.shape, window=tile_resized, min_overlap=int(tile_resized * minoverlap)
        )

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self.img[x1:x2, y1:y2]

        if img.shape[0] != tile_resized or img.shape[1] != tile_resized:
            pad_h = tile_resized - img.shape[0]
            pad_w = tile_resized - img.shape[1]
            if pad_h < 0 or pad_w < 0:
                img = img[:tile_resized, :tile_resized]
                pad_h = tile_resized - img.shape[0]
                pad_w = tile_resized - img.shape[1]
            if pad_h > 0 or pad_w > 0:
                img = np.pad(
                    img,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

        if self.reduce != 1:
            img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        img_c = np.ascontiguousarray(img)

        hsv = cv2.cvtColor(img_c, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        vertices = torch.tensor([x1, x2, y1, y2])

        img_f = img_c.astype(np.float32)
        if img_f.max() > 1.5:
            img_f = img_f / 255.0

        if (s > s_th).sum() <= p_th or img_f.sum() <= p_th:
            return img2tensor((img_f - mean) / std), vertices, -1
        else:
            return img2tensor((img_f - mean) / std), vertices, idx


class Model_pred:
    def __init__(self, models, dl, tta: bool = TTA, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, z, y in iter(self.dl):
                if (y >= 0).sum() > 0 and len(self.models) > 0:
                    x = x[y >= 0].to(device)
                    z = z[y >= 0]
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()

                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        py = p if py is None else (py + p)

                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= 1 + len(flips)

                    py /= len(self.models)

                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu().squeeze(-1).numpy()
                    z = z.numpy()

                    for i in range(len(py)):
                        yield py[i], z[i], y[i].item()

    def __len__(self):
        return len(self.dl.dataset)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _discover_model_paths(default_paths):
    existing = [p for p in default_paths if os.path.exists(p)]
    if len(existing) > 0:
        return existing

    found = []
    for root, _, files in os.walk("../input"):
        for fn in files:
            if fn.endswith(".pth") and (
                "unet" in fn.lower()
                or "hubmap" in fn.lower()
                or "efficientnet" in fn.lower()
            ):
                found.append(os.path.join(root, fn))
    found = sorted(found)
    return found




## === cell 4
model_paths = _discover_model_paths(MODELS)
print(f"Using {len(model_paths)} model checkpoint(s).")
if len(model_paths) == 0:
    print(
        "WARNING: No .pth checkpoints found under ../input/. Will use JSON anatomical-structure fallback masks."
    )

models = []
for path in model_paths:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    state_dict = _extract_state_dict(ckpt)

    model = smp.Unet(model_name, encoder_weights=None, classes=1)

    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_sd[nk] = v
        state_dict = new_sd

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)

    model.float()
    model.eval()
    model.to(device)
    models.append(model)

if "ckpt" in locals():
    del ckpt
if "state_dict" in locals():
    del state_dict
gc.collect()

names, preds = [], []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"] if "id" in df_sample.columns else row.get("img")

    rle = ""
    try:
        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds,
            batch_size=bs,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            num_workers=0,
        )
        mp = Model_pred(models, dl)

        mask = np.zeros(ds.shape, dtype=np.uint8)

        if len(models) > 0:
            for pred, vert, _i in iter(mp):
                x1, x2, y1, y2 = vert
                mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

            mask = (mask > 0.5).astype(np.uint8)
            rle = rle_encode_less_memory(mask)
        else:
            m = _rasterize_anatomical_json_to_mask(idx)
            rle = rle_encode_less_memory(m) if m is not None else ""
    except Exception:
        m = _rasterize_anatomical_json_to_mask(idx)
        rle = rle_encode_less_memory(m) if m is not None else ""

    names.append(idx)
    preds.append(rle)

    for v in ["ds", "dl", "mp", "mask", "m"]:
        if v in locals():
            try:
                del locals()[v]
            except Exception:
                pass
    gc.collect()



## === cell 5
sub_df = pd.DataFrame({"id": names, "predicted": preds})

sample_ids = (
    df_sample["id"].tolist() if "id" in df_sample.columns else df_sample["img"].tolist()
)
sub_df = sub_df.drop_duplicates(subset=["id"], keep="last")
sub_df = sub_df.set_index("id").reindex(sample_ids).fillna("").reset_index()

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print(f"Wrote submission.csv with {len(sub_df)} rows")
