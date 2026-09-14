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

0.9353995603741084

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TIFF reading failure by adding a safe reader that can handle JPEG-compressed TIFFs without `imagecodecs` by falling back to extracting/reading the embedded image bytes via OpenCV (and only if that fails, output an empty mask for that tile/image). I also correct the submission column name to exactly match the competition’s expected `predicted` column and ensure the output file is written as `submission.csv` in the working directory. These changes preserve your model/inference logic and only affect data I/O robustness so the pipeline runs end-to-end and always produces a valid CSV. Because your current score is “Not yielded”, the priority is producing a valid submission; the fallback behavior is deterministic and conservative.'
- What this solution (achieved 0.03738) has done: 'Your 0.0 score is consistent with effectively predicting an empty/near-empty mask everywhere, which is happening because the code is not actually loading the intended EfficientNet-B4 UNet weights (it instantiates a different SimpleUNet and uses `strict=False`, so most weights won’t match and the model behaves like random/untrained). The smallest score-improving change that preserves your inference pipeline is to (1) stop silently accepting mismatched checkpoints by requiring a near-complete key match, and (2) if those external weights aren’t available, fall back to a deterministic non-empty mask derived from the provided anatomical-structure JSON (a legitimate signal source in this competition) so you don’t submit blanks. I also fix the TTA averaging bug (you were not dividing by the number of models inside TTA), which is a minimal correctness fix that can materially improve Dice without changing architecture or training. These changes keep your tiling, preprocessing, thresholding, and RLE logic intact and still produce `submission.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by (1) heavy per-tile OpenCV work in `__getitem__` (especially the extra 64×64 resize + sat-count per tile) and (2) Python loops that paste every predicted tile into the full-resolution mask. The changes below keep the exact same model/inference logic and thresholds, but reduce overhead by caching the “keep/discard” decision per tile (computed once per image using a fast integral-image on a 64×64 proxy) and by pasting tiles via a single vectorized `np.maximum.at` scatter instead of a Python loop. I also avoid redundant `.pin_memory()` calls on already-pinned DataLoader batches and reduce DataLoader worker overhead while preserving determinism. All I/O paths, model code, TTA behavior, and RLE semantics are unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly coming from effectively empty predictions (either because no real model weights are being used, or because the strict-ish loader rejects them and the fallback produces near-empty masks). The smallest score-improving change that preserves your inference/tiling/RLE logic is to (1) correctly locate the TIFFs by pointing `DATA` at the dataset’s actual `test/` folder, and (2) load checkpoints in a way that accepts common key-prefix/format differences (while still requiring a high match ratio), so you actually use the trained weights instead of falling back. As a safety net to avoid blank masks when weights still can’t be loaded, the anatomical-structure fallback is kept, but we also OR it into the prediction when models are used (a minimal post-processing change aligned with the competition’s intent to use those structures). These changes should move Dice substantially upward toward your target without changing the core model/inference semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, classes=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16 + base * 8, base * 8)
        self.up2 = Up(base * 8 + base * 4, base * 4)
        self.up3 = Up(base * 4 + base * 2, base * 2)
        self.up4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class DummyModel(nn.Module):
    """Deterministic fallback (no randomness) to ensure a valid submission is always produced."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3]),
            -10.0,
            device=x.device,
            dtype=x.dtype,
        )


def seed_everything(seed: int = 42):
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
sz = 256  # tile size at network input
reduce = 4  # downsample factor from original TIFF tiles to network input
TH = 0.35  # threshold for positive predictions

DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [
    f"../input/all-data/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
shift = True
minoverlap = 300

USE_TTA = True


def _resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("../input/"):
        alt = p.replace("../input/", "/kaggle/input/")
        if os.path.exists(alt):
            return alt
    return p


def _resolve_data_dir(base: str) -> str:
    candidates = [
        base,
        base.rstrip("/") + "/",
        "../input/hubmap-kidney-segmentation/test/",
        "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
        "/kaggle/input/hubmap-kidney-segmentation/test/",
        "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
    ]
    for c in candidates:
        c = _resolve_path(c)
        if os.path.isdir(c):
            try:
                if any(fn.endswith(".tiff") for fn in os.listdir(c)):
                    return c if c.endswith("/") else c + "/"
            except Exception:
                pass
    c = _resolve_path(base)
    return c if c.endswith("/") else c + "/"


DATA = _resolve_data_dir(DATA)
MODELS = [_resolve_path(p) for p in MODELS]
df_sample_path = _resolve_path(
    "../input/hubmap-kidney-segmentation/sample_submission.csv"
)
df_sample = pd.read_csv(df_sample_path)

print("Resolved DATA:", DATA)
print("Sample rows:", len(df_sample))




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = enc.split()
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
    pixels = img.T.reshape(-1).astype(np.uint8, copy=False)
    if pixels.size == 0:
        return ""
    idx = np.flatnonzero(pixels)
    if idx.size == 0:
        return ""
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate(([idx[0]], idx[breaks]))
    ends = np.concatenate((idx[breaks - 1], [idx[-1]]))
    lengths = ends - starts + 1
    starts = starts + 1
    runs = np.empty(starts.size * 2, dtype=np.int64)
    runs[0::2] = starts
    runs[1::2] = lengths
    return " ".join(map(str, runs.tolist()))




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40
p_th = 1000 * (sz // 256) ** 2

NET_SZ = sz  # network input spatial size
TILE_SZ = reduce * sz  # window size read from full-res image
RESIZE_DSIZE = (TILE_SZ // reduce, TILE_SZ // reduce)  # equals (sz,sz)

P_TH_SMALL = int(p_th * (64 * 64) / float(NET_SZ * NET_SZ))
IMG_SUM_THRESH = int(p_th * int(reduce * reduce))


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    x, y = shape
    step = window - min_overlap

    nx = x // step + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)

    ny = y // step + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)

    X1, Y1 = np.meshgrid(x1, y1, indexing="ij")
    X2, Y2 = np.meshgrid(x2, y2, indexing="ij")
    slices = np.stack([X1, X2, Y1, Y2], axis=-1).reshape(-1, 4)
    return slices.astype(np.int64, copy=False)


class TiffMemmapReader:
    def __init__(self, path: str):
        self.path = path
        self._memmapped = False
        self.arr = None

        try:
            arr = tiff.memmap(path)
            self._memmapped = True
            self.arr = arr
        except Exception:
            try:
                self.arr = tiff.imread(path)
                self._memmapped = False
            except Exception as e_imread:
                try:
                    with tiff.TiffFile(path) as tif:
                        page = tif.pages[0]
                        segs = [seg for seg in page.segments()]
                        encoded = b"".join(segs)
                    buf = np.frombuffer(encoded, dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError("cv2.imdecode returned None")
                    if img.ndim == 2:
                        img = img[..., None]
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    if img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    self.arr = img
                    self._memmapped = False
                except Exception as e_cv:
                    raise ValueError(
                        f"Failed to read TIFF without imagecodecs. imread error: {e_imread}; opencv-bytes error: {e_cv}"
                    )

        if self.arr.ndim == 3:
            if self.arr.shape[0] in (3, 4) and self.arr.shape[2] not in (3, 4):
                self.arr = np.moveaxis(self.arr, 0, -1)
        elif self.arr.ndim == 2:
            self.arr = self.arr[..., None]
        else:
            raise ValueError(f"Unexpected TIFF shape for {path}: {self.arr.shape}")

        self.shape = (self.arr.shape[0], self.arr.shape[1])  # (H,W)
        self.count = self.arr.shape[2]

    def read_window(self, x1, x2, y1, y2):
        return self.arr[x1:x2, y1:y2, :]


_ANAT_MASK_CACHE = {}
_KERNEL_CACHE = {}
_TIFF_READER_CACHE = {}


def anatomical_json_to_mask(idx: str, shape_hw):
    cache_key = (idx, int(shape_hw[0]), int(shape_hw[1]))
    if cache_key in _ANAT_MASK_CACHE:
        return _ANAT_MASK_CACHE[cache_key]

    h, w = int(shape_hw[0]), int(shape_hw[1])
    jpath = os.path.join(DATA, f"{idx}-anatomical-structure.json")
    jpath = _resolve_path(jpath)
    if not os.path.exists(jpath):
        _ANAT_MASK_CACHE[cache_key] = None
        return None

    try:
        with open(jpath, "r") as f:
            feats = json.load(f)
    except Exception:
        _ANAT_MASK_CACHE[cache_key] = None
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    for feat in feats:
        geom = feat.get("geometry", {})
        if geom.get("type", "") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = coords[0]
        if ring is None or len(ring) < 3:
            continue
        pts = np.asarray(ring, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue
        pts = pts[:, :2]
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

    k = max(3, (min(h, w) // 512) * 2 + 3)
    kkey = (k,)
    kernel = _KERNEL_CACHE.get(kkey)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        _KERNEL_CACHE[kkey] = kernel

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        out = mask.astype(np.uint8)
        _ANAT_MASK_CACHE[cache_key] = out
        return out

    min_area = int(0.0005 * h * w)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1

    _ANAT_MASK_CACHE[cache_key] = out
    return out




## === cell 4
def _compute_keep_flags_from_proxy(reader: TiffMemmapReader, grid: np.ndarray):
    h, w = reader.shape
    arr = reader.arr
    if arr.shape[-1] > 3:
        arr3 = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr3 = np.repeat(arr, 3, axis=-1)
    else:
        arr3 = arr

    if arr3.dtype != np.uint8:
        if arr3.dtype == np.uint16:
            arr3 = (arr3 / 257).astype(np.uint8)
        else:
            arr3 = np.clip(arr3, 0, 255).astype(np.uint8)

    proxy = cv2.resize(arr3, (64, 64), interpolation=cv2.INTER_AREA)

    mx = proxy.max(axis=2).astype(np.uint16)
    mn = proxy.min(axis=2).astype(np.uint16)
    diff = mx - mn
    sat_mask = ((mx > 0) & (255 * diff > (s_th * mx))).astype(np.uint8)

    sat_ii = cv2.integral(sat_mask, sdepth=cv2.CV_32S)  # shape (65,65)

    sum_img = arr3.astype(np.uint32).sum(axis=2)  # (H,W), exact integer sum per pixel
    sum_ii = cv2.integral(sum_img, sdepth=cv2.CV_64F)  # robust for large sums

    x1 = grid[:, 0].astype(np.int64, copy=False)
    x2 = grid[:, 1].astype(np.int64, copy=False)
    y1 = grid[:, 2].astype(np.int64, copy=False)
    y2 = grid[:, 3].astype(np.int64, copy=False)

    x1p = (x1 * 64) // h
    x2p = (x2 * 64 + (h - 1)) // h
    y1p = (y1 * 64) // w
    y2p = (y2 * 64 + (w - 1)) // w
    x1p = np.clip(x1p, 0, 64)
    x2p = np.clip(x2p, 0, 64)
    y1p = np.clip(y1p, 0, 64)
    y2p = np.clip(y2p, 0, 64)

    sat_cnt = (
        sat_ii[x2p, y2p] - sat_ii[x1p, y2p] - sat_ii[x2p, y1p] + sat_ii[x1p, y1p]
    ).astype(np.int64)

    tile_sum = (
        sum_ii[x2, y2] - sum_ii[x1, y2] - sum_ii[x2, y1] + sum_ii[x1, y1]
    ).astype(np.float64)

    keep = (sat_cnt > P_TH_SMALL) & (tile_sum > IMG_SUM_THRESH)
    return keep


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        tiff_path = os.path.join(DATA, idx + ".tiff")
        tiff_path = _resolve_path(tiff_path)
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"Missing TIFF: {tiff_path}")

        reader = _TIFF_READER_CACHE.get(idx)
        if reader is None:
            reader = TiffMemmapReader(tiff_path)
            _TIFF_READER_CACHE[idx] = reader

        self.data = reader
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce * sz  # read window size at original scale
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

        self._keep_flags = _compute_keep_flags_from_proxy(self.data, self.mask_grid)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self.data.read_window(int(x1), int(x2), int(y1), int(y2))

        if img.shape[-1] > 3:
            img = img[..., :3]
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        if img.dtype != np.uint8:
            if img.dtype == np.uint16:
                img = (img / 257).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        img_net = cv2.resize(img, RESIZE_DSIZE, interpolation=cv2.INTER_AREA)

        if not bool(self._keep_flags[idx]):
            return (
                img2tensor((img_net.astype(np.float32) / 255.0 - mean) / std),
                np.asarray([x1, x2, y1, y2], dtype=np.int64),
                -1,
            )

        return (
            img2tensor((img_net.astype(np.float32) / 255.0 - mean) / std),
            np.asarray([x1, x2, y1, y2], dtype=np.int64),
            idx,
        )


class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.inference_mode():
            for x, z, y in iter(self.dl):
                keep = y >= 0
                if int(keep.sum().item()) == 0:
                    continue

                x = x[keep]
                z = z[keep].numpy()

                if device.type == "cuda":
                    x = x.to(device, non_blocking=True)
                    x = x.contiguous(memory_format=torch.channels_last)
                else:
                    x = x.to(device)

                if self.half:
                    x = x.half()

                if not self.tta:
                    py = None
                    for model in self.models:
                        p = torch.sigmoid(model(x))
                        py = p if py is None else (py + p)
                    py /= float(len(self.models))
                else:
                    x0 = x
                    x1 = torch.flip(x, [-1])
                    x2 = torch.flip(x, [-2])
                    x3 = torch.flip(x, [-2, -1])
                    x_cat = torch.cat([x0, x1, x2, x3], dim=0)

                    py = None
                    for model in self.models:
                        p_cat = model(x_cat)
                        p0, p1, p2, p3 = torch.chunk(p_cat, 4, dim=0)
                        p1 = torch.flip(p1, [-1])
                        p2 = torch.flip(p2, [-2])
                        p3 = torch.flip(p3, [-2, -1])
                        p = (
                            torch.sigmoid(p0)
                            + torch.sigmoid(p1)
                            + torch.sigmoid(p2)
                            + torch.sigmoid(p3)
                        )
                        py = p if py is None else (py + p)

                    py /= float(len(self.models) * 4)

                py = F.interpolate(
                    py, scale_factor=reduce, mode="bilinear", align_corners=False
                )
                py = py.squeeze(1).float().cpu().numpy()  # (B, H, W)
                yield py, z

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
def _maybe_extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    prefixes = ["model.", "net.", "module.model.", "seg_model."]
    for pref in prefixes:
        if any(k.startswith(pref) for k in state_dict.keys()):
            return {k[len(pref) :]: v for k, v in state_dict.items()}
    return state_dict


def _load_model_state_strictish(
    model: nn.Module, state_dict: dict, min_match_ratio: float = 0.60
):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    matched = model_keys.intersection(ckpt_keys)
    match_ratio = len(matched) / max(1, len(model_keys))
    if match_ratio < min_match_ratio:
        raise ValueError(f"Incompatible checkpoint: match_ratio={match_ratio:.3f}")
    model.load_state_dict(state_dict, strict=False)
    return match_ratio


models = []
missing = 0
loaded = 0

for path in MODELS:
    if not os.path.exists(path):
        missing += 1
        continue
    try:
        raw = torch.load(path, map_location=torch.device("cpu"))
        state_dict = _maybe_extract_state_dict(raw)
        state_dict = _strip_module_prefix(state_dict)
        state_dict = _strip_known_prefixes(state_dict)

        model = SimpleUNet(in_channels=3, classes=1, base=32)
        _ = _load_model_state_strictish(model, state_dict, min_match_ratio=0.60)

        model.float()
        model.eval()

        if device.type == "cuda":
            model = model.to(device, memory_format=torch.channels_last)
        else:
            model = model.to(device)

        models.append(model)
        loaded += 1
        del raw, state_dict
    except Exception:
        missing += 1

use_anatomical_fallback = False
if len(models) == 0:
    use_anatomical_fallback = True
    models = [DummyModel().to(device).eval()]

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

if device.type == "cuda":
    try:
        models = [
            torch.compile(m, mode="reduce-overhead", fullgraph=False) for m in models
        ]
    except Exception:
        pass

gc.collect()
print(f"Loaded {loaded} model(s); missing/unusable weights: {missing}")
print("Using anatomical-structure fallback:", use_anatomical_fallback)




## === cell 6
def _seed_worker(worker_id):
    base_seed = 42
    seed = base_seed + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_tiles_scatter_max(mask: np.ndarray, tiles: np.ndarray, boxes: np.ndarray):
    if tiles.size == 0:
        return
    x1 = boxes[:, 0].astype(np.int64, copy=False)
    y1 = boxes[:, 2].astype(np.int64, copy=False)
    Ht, Wt = int(tiles.shape[1]), int(tiles.shape[2])

    rr = np.arange(Ht, dtype=np.int64)[:, None]
    cc = np.arange(Wt, dtype=np.int64)[None, :]
    base_r = x1[:, None, None] + rr[None, :, :]
    base_c = y1[:, None, None] + cc[None, :, :]
    flat = (base_r * mask.shape[1] + base_c).reshape(-1)
    vals = tiles.reshape(-1)
    np.maximum.at(mask.reshape(-1), flat, vals)


names, preds = [], []

if device.type == "cuda":
    cpu = os.cpu_count() or 2
    num_workers = max(1, min(2, cpu // 4))
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2
else:
    num_workers = 0
    persistent_workers = False
    prefetch_factor = None

g = torch.Generator()
g.manual_seed(42)

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]

    ds = None
    dl = None
    mask = None

    try:
        ds = HuBMAPDataset(idx)

        if use_anatomical_fallback:
            mask = anatomical_json_to_mask(idx, ds.shape)
            if mask is None:
                mask = np.zeros(ds.shape, dtype=np.uint8)
        else:
            dl_kwargs = dict(
                batch_size=bs,
                pin_memory=(device.type == "cuda"),
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                worker_init_fn=_seed_worker if num_workers > 0 else None,
                generator=g if num_workers > 0 else None,
            )
            if prefetch_factor is not None and num_workers > 0:
                dl_kwargs["prefetch_factor"] = prefetch_factor

            dl = DataLoader(ds, **dl_kwargs)

            mp = Model_pred(models, dl, tta=USE_TTA)

            mask = np.zeros(ds.shape, dtype=np.uint8)

            for py, zv in mp:
                bin_batch = (py > TH).astype(np.uint8, copy=False)
                apply_tiles_scatter_max(mask, bin_batch, zv)

            anat = anatomical_json_to_mask(idx, ds.shape)
            if anat is not None:
                mask = np.maximum(mask, anat.astype(np.uint8, copy=False))

        rle = rle_encode_less_memory(mask)
    except Exception as e:
        print(f"[WARN] Failed on id={idx}: {e}")
        rle = ""

    names.append(idx)
    preds.append(rle)

    if dl is not None:
        del dl
    if ds is not None:
        del ds
    if mask is not None:
        del mask

gc.collect()

df = pd.DataFrame({"id": names, "predicted": preds})
df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv columns:", list(df.columns))
