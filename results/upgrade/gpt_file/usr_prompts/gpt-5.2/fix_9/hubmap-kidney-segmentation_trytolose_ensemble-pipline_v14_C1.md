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

# 5. Target score

0.9489723373737288

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the unavailable `segmentation_models_pytorch` dependency and replacing it with a tiny compatible UNet implementation that matches the `smp.Unet(...).cuda(); load_state_dict(...); model(image)` calling pattern. I also restore missing imports (they failed because the first cell crashed), make weight loading robust across CPU/GPU and common checkpoint key formats, and ensure the submission uses the required `id,predicted` columns and writes `submission.csv`. Finally, I keep the inference logic/thresholding/RLE exactly as-is to preserve evaluation semantics and only adjust what’s necessary for end-to-end execution.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by switching TIFF reading to OpenCV, and I reintroduce the missing torch/Dataset imports that failed because the first cell crashed. I keep the core inference logic (tiling, model ensembling, thresholding, voting, RLE) identical, only changing the dataset’s image I/O so the pipeline runs end-to-end. I also make the dataset robust to the actual test file extension by probing for `.tiff/.tif/.png/.jpg` and ensure `submission.csv` is always written with the required `id,predicted` columns. With weights present, this should move the score up from 0.0 to a meaningful Dice score instead of failing/empty masks.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with producing almost-all-empty masks, which here can happen if the test images are never actually loaded (OpenCV can’t read Kaggle’s WSIs directly) and/or if the weight files are missing so you fall back to empty predictions. I make two minimal, score-relevant fixes: (1) robustly locate and extract the actual test `.tiff` images from `../input/hubmap-kidney-segmentation/test.zip` into a local folder and resolve paths from there, and (2) ensure image reading works for TIFF by using `cv2.IMREAD_UNCHANGED` and converting to 3-channel BGR when needed (without changing any model/inference/threshold/RLE logic). These changes should move you away from empty/invalid inputs and toward a meaningful Dice score using your existing ensemble logic. The rest of the pipeline (tiling, blending, thresholding, voting, RLE) remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “empty/near-empty masks everywhere”, which in this pipeline can happen even when the model runs, due to (a) your `_check_background` logic being inverted (it currently flags tissue as background and then skips it), and (b) empty/invalid RLE strings caused by mutating the first/last pixel in-place inside `rle_encode_less_memory`. I make two minimal, score-directed fixes: correct the background check so tissue tiles are kept (without changing the overall skip mechanism), and make RLE encoding safe by adding sentinel zeros via concatenation rather than overwriting pixels. Everything else (tiling, blending, thresholding, voting, model loading, and submission schema) is left unchanged to preserve core evaluation semantics while moving the score up from 0.0 toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “valid submission but essentially empty masks everywhere”, which in this pipeline can happen if (a) background filtering is too strict and skips most tissue tiles, and (b) a too-large `RAMKA` border wipe removes most positive pixels from each crop (especially with `CROP_SIZE=4096`). I make two minimal, score-relevant adjustments that preserve your core inference/tiling/threshold/RLE logic: relax the background check thresholds so tissue tiles aren’t skipped, and scale `RAMKA` to crop size so we don’t erase most of the prediction. Everything else (models, blending, thresholding, voting, and RLE encoding) is kept identical to avoid unintended metric changes while moving the Dice up toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from an empty/near-empty prediction pipeline (either because no weights are actually being loaded, or because the “skip background” logic is currently skipping tissue tiles and leaving the final mask empty). I make two minimal, score-relevant fixes that keep your model/inference/tiling/threshold/RLE semantics the same: (1) robustly discover and load any available weight files by globbing inside the extracted `model_1/` and `model_2/` folders when the hardcoded filenames don’t exist, and (2) fix the background check to correctly return `True` for background (so your existing `if SKIP_BACKGROUND ...: continue` behaves as intended). These changes should move you away from producing empty masks everywhere and increase Dice toward your target without changing your core inference recipe.'
- What this solution (achieved 0.0) has done: 'I fix two score-critical logic bugs that currently make the pipeline skip almost all tissue tiles and therefore predict near-empty masks (leading to a 0.0 Dice). First, I correct the dataset flag so `not_background` is actually “not background” (tissue) rather than “is background”, aligning it with the existing `if SKIP_BACKGROUND ...: continue` logic without changing the skip mechanism itself. Second, I make the background threshold computation consistent across crop sizes so the heuristic doesn’t become unrealistically strict at 4096 crops. These are minimal changes that preserve your model/inference/thresholding/RLE logic while moving the score up toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 Dice is most consistent with the model weights not being loaded correctly (silently loading with many missing keys due to architecture mismatch) or producing near-empty masks from mis-thresholded logits. I keep your inference/tiling/threshold/RLE logic intact, but make weight loading score-relevant and robust by (1) inferring `in_channels` and `classes` from each checkpoint’s tensors and instantiating the same tiny UNet accordingly, and (2) enforcing strict loading when possible (and skipping checkpoints that clearly don’t match) instead of `strict=False` silently degrading to random weights. This should move you away from empty-mask predictions and toward a meaningful Dice, without changing the fundamental pipeline behavior. The submission format/path remains identical and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBNReLU(in_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _TinyUNet(nn.Module):
    def __init__(self, in_channels=3, classes=1, base=32):
        super().__init__()
        self.inc = _DoubleConv(in_channels, base)
        self.down1 = _Down(base, base * 2)
        self.down2 = _Down(base * 2, base * 4)
        self.down3 = _Down(base * 4, base * 8)
        self.down4 = _Down(base * 8, base * 8)

        self.up1 = _Up(base * 16, base * 4)
        self.up2 = _Up(base * 8, base * 2)
        self.up3 = _Up(base * 4, base)
        self.up4 = _Up(base * 2, base)

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


class smp:
    class Unet(nn.Module):
        def __init__(
            self,
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            **kwargs,
        ):
            super().__init__()
            base = 32
            if "efficientnet" in str(encoder_name).lower():
                base = 40
            elif (
                "resnext" in str(encoder_name).lower()
                or "se_resnext" in str(encoder_name).lower()
            ):
                base = 48
            self.net = _TinyUNet(in_channels=in_channels, classes=classes, base=base)

        def forward(self, x):
            return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



## === cell 1
import zipfile
from subprocess import check_call

Path("model_1").mkdir(exist_ok=True)
Path("model_2").mkdir(exist_ok=True)

tar1 = Path("../input/hubmap-folds-2/1024_avg_last.tar")
tar2 = Path("../input/hubmap-folds-2/1024_512_pseudo_v1_avg.tar")
if tar1.exists():
    check_call(["tar", "-xvf", str(tar1), "-C", "model_1"])
if tar2.exists():
    check_call(["tar", "-xvf", str(tar2), "-C", "model_2"])

TEST_ZIP = Path("../input/hubmap-kidney-segmentation/test.zip")
EXTRACT_TEST_DIR = Path("./_extracted_test_images")
EXTRACT_TEST_DIR.mkdir(parents=True, exist_ok=True)

already = any(EXTRACT_TEST_DIR.glob("*.tif")) or any(EXTRACT_TEST_DIR.glob("*.tiff"))
if TEST_ZIP.exists() and not already:
    with zipfile.ZipFile(TEST_ZIP, "r") as z:
        members = [m for m in z.namelist() if m.lower().endswith((".tif", ".tiff"))]
        z.extractall(EXTRACT_TEST_DIR)
    if (EXTRACT_TEST_DIR / "test").exists():
        EXTRACT_TEST_DIR = EXTRACT_TEST_DIR / "test"

print(
    "Using extracted test dir:", EXTRACT_TEST_DIR, "exists:", EXTRACT_TEST_DIR.exists()
)



## === cell 2
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.4
VOTE = 0
SKIP_BACKGROUND = True
SKIP_COMMIT = True

RAMKA_FRAC = (
    0.0625  # ~1/16 of crop, equals 256 when crop=4096, smaller for smaller crops
)

df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
assert {"id", "predicted"}.issubset(
    df_sub.columns
), f"Unexpected submission columns: {df_sub.columns.tolist()}"




## === cell 3
def resolve_test_image_path(img_id: str) -> str:
    exts = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]

    for ext in exts:
        p = EXTRACT_TEST_DIR / f"{img_id}{ext}"
        if p.exists():
            return str(p)

    base = Path("../input/hubmap-kidney-segmentation/test")
    for ext in exts:
        p = base / f"{img_id}{ext}"
        if p.exists():
            return str(p)

    return str(base / f"{img_id}.tiff")




## === cell 4
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
            f"../input/hubmap5/exp_25_fold_0.pt",
            f"../input/hubmap5/exp_24_fold_1.pt",
            f"../input/hubmap5/exp_23_fold_2.pt",
            f"../input/hubmap5/exp_22_fold_3.pt",
            f"../input/hubmap5/exp_21_fold_4.pt",
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
TRAIN_IMG_SIZE




## === cell 5
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if isinstance(ckpt, dict):
        new_sd = {}
        for k, v in ckpt.items():
            nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            new_sd[nk] = v
        ckpt = new_sd
    return ckpt


def _infer_in_out_channels_from_sd(sd: dict) -> Tuple[Optional[int], Optional[int]]:
    in_ch = None
    out_ch = None

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("inc.block.0.conv.weight") and v.ndim == 4:
            in_ch = int(v.shape[1])
            break
    if in_ch is None:
        candidates = []
        for k, v in sd.items():
            if torch.is_tensor(v) and v.ndim == 4 and int(v.shape[1]) <= 4:
                candidates.append((k, v))
        if candidates:
            k, v = sorted(candidates, key=lambda kv: int(kv[1].shape[0]))[0]
            in_ch = int(v.shape[1])

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("outc.weight") and v.ndim == 4:
            out_ch = int(v.shape[0])
            break

    return in_ch, out_ch


def _load_state_dict_flexible(
    model: nn.Module, ckpt_path: str, strict_prefer: bool = True
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise ValueError(
            f"Checkpoint does not contain a state_dict-like dict: {ckpt_path}"
        )

    if strict_prefer:
        model.load_state_dict(sd, strict=True)
        return [], []
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return missing, unexpected




## === cell 6
def rle_encode_less_memory(img):
    pixels = img.T.flatten().astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def valid_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )




## === cell 7
def _check_background(img, crop_size) -> bool:
    s_th = 20
    p_th = int(250 * (crop_size / 256.0) ** 2)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)

    tissue_present = True if ((ss > s_th).sum() > p_th and img.sum() > p_th) else False
    is_background = not tissue_present
    return is_background


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step

        if not Path(tiff_path).exists():
            raise FileNotFoundError(f"Image not found: {tiff_path}")

        img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed for: {tiff_path}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.img = img
        self.h, self.w = self.img.shape[:2]

        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size

        img = self.img[y : y + self.crop_size, x : x + self.crop_size]

        transormed_imgs = {
            img_size: valid_transform(img_size)(image=img)["image"]
            for img_size in self.all_img_sizes
        }
        transormed_imgs["crop_names"] = f"{x}_{y}"
        transormed_imgs["not_background"] = not _check_background(img, self.crop_size)
        return transormed_imgs




## === cell 8
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)
    mask_pred = np.zeros(img_size, dtype=np.uint8)

    ramka = int(round(crop_size * RAMKA_FRAC))
    ramka = max(0, min(ramka, crop_size // 2 - 1)) if crop_size >= 4 else 0

    for batch in tqdm(data_loader, ncols=70, leave=True):
        if SKIP_BACKGROUND is True and batch["not_background"].item() is False:
            continue
        with torch.no_grad():
            pred_total = None
            for cfg, model_group in models_with_config:
                image = batch[cfg.img_size].to(device, non_blocking=True)
                pred_group = None

                for model in model_group:
                    pred = model(image)
                    pred = pred.sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = torch.nn.functional.interpolate(
                            pred, size=TRAIN_IMG_SIZE, mode="bilinear"
                        )
                    if pred_group is None:
                        pred_group = pred
                    else:
                        pred_group += pred

                pred_group = pred_group.squeeze()
                if len(pred_group.shape) == 2:
                    pred_group = pred_group.unsqueeze(0)
                image = image.cpu()
                del image
                pred_group = cfg.weight_blend * (pred_group / len(model_group))
                if pred_total is None:
                    pred_total = pred_group
                else:
                    pred_total += pred_group

            pred_total = (pred_total.cpu().data.numpy() > THR).astype(np.uint8)
            for predict_single, crop_name in zip(pred_total, batch["crop_names"]):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])

                if crop_size != TRAIN_IMG_SIZE:
                    predict_single = cv2.resize(predict_single, (crop_size, crop_size))

                if ramka > 0:
                    predict_single[0:ramka] = 0
                    predict_single[-ramka:] = 0
                    predict_single[:, 0:ramka] = 0
                    predict_single[:, -ramka:] = 0

                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred)
    del mask_pred
    gc.collect()
    gc.collect()
    return mask_rle




## === cell 9
all_img_sizes = set([cfg.img_size for cfg in my_models])
models_with_config = []

missing_weights = []
skipped_incompatible = []
for cfg in my_models:
    models_group = []
    candidate_paths = [Path(p) for p in cfg.weights_path]
    if not any(p.exists() for p in candidate_paths):
        if "model_1" in str(cfg.weights_path[0]):
            candidate_paths = sorted(Path("./model_1").glob("**/*.pth")) + sorted(
                Path("./model_1").glob("**/*.pt")
            )
        elif "model_2" in str(cfg.weights_path[0]):
            candidate_paths = sorted(Path("./model_2").glob("**/*.pth")) + sorted(
                Path("./model_2").glob("**/*.pt")
            )

    for w_path in candidate_paths:
        w_path = str(w_path)
        if not Path(w_path).exists():
            missing_weights.append(w_path)
            continue

        try:
            ckpt = torch.load(w_path, map_location="cpu")
            sd = _extract_state_dict(ckpt)
            if not isinstance(sd, dict):
                skipped_incompatible.append((w_path, "no_state_dict"))
                continue
            in_ch, out_ch = _infer_in_out_channels_from_sd(sd)
            if in_ch is None:
                in_ch = 3
            if out_ch is None:
                out_ch = 1

            model = smp.Unet(
                cfg.encoder, encoder_weights=None, in_channels=in_ch, classes=out_ch
            ).to(device)
            _load_state_dict_flexible(model, w_path, strict_prefer=True)
            model.eval()
            models_group.append(model)
        except Exception as e:
            skipped_incompatible.append((w_path, str(e)))
            continue

    if len(models_group) > 0:
        models_with_config.append((cfg, models_group))

if len(models_with_config) == 0:
    print(
        "WARNING: No compatible model weights found. Will create an empty-mask submission."
    )
else:
    if missing_weights:
        print(
            f"WARNING: Some weights were missing ({len(missing_weights)}). Proceeding with available weights."
        )
    if skipped_incompatible:
        print(
            f"WARNING: Skipped incompatible checkpoints ({len(skipped_incompatible)})."
        )
        for p, msg in skipped_incompatible[:5]:
            print(" -", p, "->", msg)
    print("Loaded model groups:", [(c.encoder, len(g)) for c, g in models_with_config])



## === cell 10
count_thr = 5 if SKIP_COMMIT is True else 4

for idx, row in df_sub.iterrows():
    if len(models_with_config) == 0:
        df_sub.loc[idx, "predicted"] = ""
        continue

    tiff_path = resolve_test_image_path(row["id"])
    test_ds = SingleTiffDataset(
        tiff_path=tiff_path,
        all_img_sizes=all_img_sizes,
        crop_size=CROP_SIZE,
        step=STEP,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    rle = inference(test_loader, models_with_config, CROP_SIZE)
    df_sub.loc[idx, "predicted"] = rle



## === cell 11
import shutil

shutil.rmtree("model_1", ignore_errors=True)
shutil.rmtree("model_2", ignore_errors=True)



## === cell 12
df_sub[["id", "predicted"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)



## === cell 13
df_sub
