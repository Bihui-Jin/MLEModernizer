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

0.9470332366877252

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by reading the `.tiff` images with OpenCV and replicating the same sliding-window + resizing logic, which fixes the immediate runtime error and keeps the inference pipeline intact. I also fix the broken cell ordering/NameErrors by consolidating global constants and required imports into the first cell so later cells can run unmodified. To ensure a valid submission is always produced, the code fill missing predictions with empty strings and always write `submission.csv` with the required `id,predicted` columns. I not change thresholds/model blending logic (score semantics) beyond what’s necessary to make the code run end-to-end.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
import tarfile
from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DATA_ROOT = Path("../input/hubmap-kidney-segmentation")
WORKDIR = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", DEVICE)
print("DATA_ROOT exists:", DATA_ROOT.exists())
print("WORKDIR:", WORKDIR.resolve())

BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.35
VOTE = 0
SKIP_BACKGROUND = True
SKIP_COMMIT = True


def resolve_test_tiff(data_root: Path, image_id: str) -> Path:
    candidates = [
        data_root / "test" / f"{image_id}.tiff",
        data_root / "test_images" / f"{image_id}.tiff",
        data_root / "test" / f"{image_id}.tif",
        data_root / "test_images" / f"{image_id}.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]




## === cell 1
folds_dir = Path("../input/hubmap-folds-2")
print("folds_dir exists:", folds_dir.exists())
if folds_dir.exists():
    for p in sorted(folds_dir.iterdir()):
        print(p.name)



## === cell 2
model_1_dir = WORKDIR / "model_1"
model_2_dir = WORKDIR / "model_2"
model_1_dir.mkdir(exist_ok=True)
model_2_dir.mkdir(exist_ok=True)

tar1 = folds_dir / "1024_avg_last.tar"
tar2 = folds_dir / "1024_512_final.tar"

if tar1.exists():
    with tarfile.open(tar1, "r:*") as tf:
        tf.extractall(model_1_dir)
else:
    print("WARNING: missing", tar1)

if tar2.exists():
    with tarfile.open(tar2, "r:*") as tf:
        tf.extractall(model_2_dir)
else:
    print("WARNING: missing", tar2)

print("Extracted model_1 files:", len(list(model_1_dir.rglob("*"))))
print("Extracted model_2 files:", len(list(model_2_dir.rglob("*"))))



## === cell 3
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 4
df_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")
print(df_sub.head())



## === cell 5
from torchvision.models import resnet34
from torchvision.models import resnet50


class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet34(weights=None)
        self.enc0 = nn.Sequential(m.conv1, m.bn1, m.relu)  # /2, 64
        self.pool = m.maxpool  # /4
        self.enc1 = m.layer1  # /4, 64
        self.enc2 = m.layer2  # /8, 128
        self.enc3 = m.layer3  # /16, 256
        self.enc4 = m.layer4  # /32, 512

        self.center = nn.Sequential(ConvRelu(512, 512), ConvRelu(512, 256))

        self.up4 = UpBlock(256, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = nn.Sequential(ConvRelu(64, 32), ConvRelu(32, 32))

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)  # /2
        x1 = self.enc1(self.pool(x0))  # /4
        x2 = self.enc2(x1)  # /8
        x3 = self.enc3(x2)  # /16
        x4 = self.enc4(x3)  # /32

        c = self.center(x4)
        d4 = self.up4(c, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = F.interpolate(
            d1, scale_factor=2.0, mode="bilinear", align_corners=False
        )  # back to input size
        d0 = self.up0(d0)
        return self.final(d0)


class UNetResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=None)
        self.enc0 = nn.Sequential(m.conv1, m.bn1, m.relu)  # /2, 64
        self.pool = m.maxpool  # /4
        self.enc1 = m.layer1  # /4, 256
        self.enc2 = m.layer2  # /8, 512
        self.enc3 = m.layer3  # /16, 1024
        self.enc4 = m.layer4  # /32, 2048

        self.center = nn.Sequential(ConvRelu(2048, 512), ConvRelu(512, 256))

        self.up4 = UpBlock(256, 1024, 256)
        self.up3 = UpBlock(256, 512, 128)
        self.up2 = UpBlock(128, 256, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = nn.Sequential(ConvRelu(64, 32), ConvRelu(32, 32))
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)  # /2
        x1 = self.enc1(self.pool(x0))  # /4
        x2 = self.enc2(x1)  # /8
        x3 = self.enc3(x2)  # /16
        x4 = self.enc4(x3)  # /32
        c = self.center(x4)
        d4 = self.up4(c, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=False)
        d0 = self.up0(d0)
        return self.final(d0)


def build_model(encoder_name: str) -> nn.Module:
    if encoder_name == "resnet34":
        return UNetResNet34()
    if encoder_name in ("se_resnext50_32x4d", "timm-efficientnet-b3"):
        return UNetResNet50()
    raise ValueError(f"Unknown encoder: {encoder_name}")


def load_checkpoint_tolerant(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(missing) or len(unexpected):
        print(
            f"[load_checkpoint_tolerant] {Path(ckpt_path).name}: missing={len(missing)}, unexpected={len(unexpected)}"
        )




## === cell 6
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
        0.35,
    ),
    ModelConfig(
        "timm-efficientnet-b3",
        [
            "../input/hubmap6/exp_65_fold_0.pt",
            "../input/hubmap6/exp_63_fold_1.pt",
            "../input/hubmap6/exp_66_fold_2.pt",
            "../input/hubmap6/exp_64_fold_3.pt",
            "../input/hubmap6/exp_62_fold_4.pt",
        ],
        1024,
        0.3,
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
        0.35,
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models])
TRAIN_IMG_SIZE




## === cell 7
def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
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




## === cell 8
def _check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blancking threshold
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)

    background = False if (ss > s_th).sum() <= p_th or img.sum() <= p_th else True
    return background


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step

        img = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {tiff_path}")
        self.img = img
        self.h, self.w = img.shape[:2]

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
        transormed_imgs["not_background"] = _check_background(img, self.crop_size)
        return transormed_imgs




## === cell 9
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)
    mask_pred = np.zeros(img_size, dtype=np.uint8)

    for batch in tqdm(data_loader, ncols=70, leave=True):
        if SKIP_BACKGROUND is True and batch["not_background"].item() is False:
            continue

        with torch.no_grad():
            pred_total = None
            for cfg, model_group in models_with_config:
                if len(model_group) == 0:
                    continue
                image = batch[cfg.img_size].to(DEVICE, non_blocking=True)
                pred_group = None

                for model in model_group:
                    pred = model(image)
                    pred = pred.sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = F.interpolate(
                            pred,
                            size=TRAIN_IMG_SIZE,
                            mode="bilinear",
                            align_corners=False,
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

            if pred_total is None:
                continue

            pred_total = (pred_total.cpu().data.numpy() > THR).astype(np.uint8)

            for predict_single, crop_name in zip(pred_total, batch["crop_names"]):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])

                if crop_size != TRAIN_IMG_SIZE:
                    predict_single = cv2.resize(predict_single, (crop_size, crop_size))
                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred)
    del mask_pred
    gc.collect()
    gc.collect()
    return mask_rle




## === cell 10
all_img_sizes = set([cfg.img_size for cfg in my_models])

models_with_config = []
for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        if not Path(w_path).exists():
            print("WARNING: weight not found:", w_path)
            continue
        model = build_model(cfg.encoder).to(DEVICE)
        load_checkpoint_tolerant(model, w_path)
        model.eval()
        models_group.append(model)
    if len(models_group) == 0:
        print("WARNING: no weights loaded for encoder:", cfg.encoder)
    models_with_config.append((cfg, models_group))

print("Loaded model groups:", [(cfg.encoder, len(g)) for cfg, g in models_with_config])



## === cell 11
if "id" not in df_sub.columns:
    raise ValueError("Submission dataframe must contain 'id' column")
if "predicted" not in df_sub.columns:
    df_sub["predicted"] = ""

df_sub["predicted"] = df_sub["predicted"].fillna("")

for idx, row in df_sub.iterrows():
    tiff_path = resolve_test_tiff(DATA_ROOT, str(row["id"]))
    if not tiff_path.exists():
        print("WARNING: missing tiff:", tiff_path)
        df_sub.loc[idx, "predicted"] = ""
        continue

    test_ds = SingleTiffDataset(
        tiff_path=str(tiff_path),
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

out_path = WORKDIR / "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())
print(df_sub.head())




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/1618310032.py in <cell line: 0>()
     15         continue
     16 
---> 17     test_ds = SingleTiffDataset(
     18         tiff_path=str(tiff_path),
     19         all_img_sizes=all_img_sizes,

/tmp/ipykernel_55/4089340382.py in __init__(self, tiff_path, all_img_sizes, crop_size, step)
     15         self.step = step
     16 
---> 17         img = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)  # BGR uint8
     18         if img is None:
     19             raise FileNotFoundError(f"Failed to read image: {tiff_path}")

error: OpenCV(4.12.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:79: error: (-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS in function 'validateInputImageSize'


## === cell 12
def _rm_tree(p: Path):
    if not p.exists():
        return
    for child in sorted(p.rglob("*"), reverse=True):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        except Exception:
            pass
    try:
        p.rmdir()
    except Exception:
        pass


_rm_tree(model_1_dir)
_rm_tree(model_2_dir)
print("Cleanup done.")
