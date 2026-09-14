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

0.9465277303916344

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing dependency on `segmentation_models_pytorch` by using the installed `torchvision` to implement a minimal U-Net–like wrapper around a ResNet-34 backbone, and I route the `se_resnext50_32x4d` config to a supported torchvision backbone (`resnext50_32x4d`) so the provided weights can still be loaded when compatible. I also remove notebook shell commands (`!ls`, `!tar`, `!mkdir`, `!rm`) and replace them with pure-Python extraction via `tarfile`, using a robust search for the weight tar files in `/kaggle/input` so the notebook runs in this environment. Finally, I fix the cell ordering/NameErrors by consolidating all imports and definitions early, keep the inference logic and RLE submission format identical, and guarantee that `submission.csv` is written with `id` and `predicted` columns.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by switching tile reading to OpenCV and resizing the whole image to the expected training size before tiling, keeping the same sliding-window + model-ensemble + thresholding + RLE submission semantics. I also make the model-weight loading robust when the provided `1024.tar` / `512.tar` files (and referenced `.pth` names) are not present in this Kaggle dataset, by falling back to a valid “all-empty masks” submission rather than crashing (this still score 0.0 but run end-to-end and generate a valid CSV). Finally, I fix the cell ordering NameErrors by consolidating imports/definitions so `Dataset`, `nn`, etc. are always defined before use, and ensure `submission.csv` has exactly `id,predicted` columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from never actually running inference: `SKIP_COMMIT=True` causes the code to skip prediction whenever the test set has more than 5 rows (it does), leaving `predicted` empty. I make the smallest change to ensure full test inference runs (set `SKIP_COMMIT=False`) so you get non-empty RLE masks and a non-zero Dice score. I also fix the test image path to match the competition’s JSON-based test set (no `.tiff` files here), by rasterizing the provided `*-anatomical-structure.json` into an input image (preserving your existing tiling/inference core logic). Finally, I keep all thresholds/tiling/model/ensemble logic unchanged and still write `submission.csv` with `id,predicted`.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
import tarfile
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)




## === cell 1
def find_first(pattern: str, root: str = "/kaggle/input") -> Optional[Path]:
    matches = list(Path(root).rglob(pattern))
    if not matches:
        return None
    return matches[0]


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=out_dir)


MODEL1_DIR = Path("./model_1")
MODEL2_DIR = Path("./model_2")

tar_1024 = find_first("1024.tar")
tar_512 = find_first("512.tar")

print("Found 1024.tar:", tar_1024)
print("Found 512.tar:", tar_512)

if tar_1024 is not None:
    safe_extract_tar(tar_1024, MODEL1_DIR)
if tar_512 is not None:
    safe_extract_tar(tar_512, MODEL2_DIR)

print("Model_1 files:", len(list(MODEL1_DIR.rglob("*"))) if MODEL1_DIR.exists() else 0)
print("Model_2 files:", len(list(MODEL2_DIR.rglob("*"))) if MODEL2_DIR.exists() else 0)



## === cell 2
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.4
VOTE = 0
SKIP_BACKGROUND = True

SKIP_COMMIT = False  # was True

INPUT_DIR = Path("/kaggle/input/hubmap-kidney-segmentation")
df_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

if (
    "img" in df_sub.columns
    and "pixels" in df_sub.columns
    and ("id" not in df_sub.columns)
):
    df_sub = df_sub.rename(columns={"img": "id", "pixels": "predicted"})
if "predicted" not in df_sub.columns:
    df_sub["predicted"] = ""

print(df_sub.head())




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
            "./model_1/fold0_avg_0.9221.pth",
            "./model_1/fold1_avg_0.9176.pth",
            "./model_1/fold2_avg_0.9366.pth",
            "./model_1/fold3_avg_0.9235.pth",
            "./model_1/fold4_avg_0.9399.pth",
        ],
        4096,
        0.5,
    ),
    ModelConfig(
        "se_resnext50_32x4d",
        [
            "./model_2/fold0_avg_0.9399.pth",
            "./model_2/fold1_avg_0.9566.pth",
            "./model_2/fold2_avg_0.9381.pth",
            "./model_2/fold3_avg_0.9331.pth",
        ],
        2048,
        0.5,
    ),
]

TRAIN_IMG_SIZE = max(cfg.img_size for cfg in my_models)
print("TRAIN_IMG_SIZE:", TRAIN_IMG_SIZE)




## === cell 4
def rle_encode_less_memory(img: np.ndarray) -> str:
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def valid_transform(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def _check_background(img_bgr, crop_size) -> bool:
    s_th = 40  # saturation blanking threshold
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)
    background = False if (ss > s_th).sum() <= p_th or img_bgr.sum() <= p_th else True
    return background


def rasterize_anatomical_json_to_bgr(
    json_path: Path, out_hw: Optional[tuple] = None
) -> np.ndarray:
    """
    Minimal fix to get a real input image in this dataset:
    test images are provided as JSON polygons (no TIFFs here).
    We rasterize anatomical-structure polygons into a 3-channel BGR canvas.
    This preserves the rest of the pipeline (tiling -> transforms -> model inference -> RLE).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    all_xy = []
    for feat in data:
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue
        for ring in coords:
            for pt in ring:
                if len(pt) >= 2:
                    all_xy.append((float(pt[0]), float(pt[1])))

    if out_hw is None:
        if len(all_xy) == 0:
            h = w = TRAIN_IMG_SIZE
        else:
            max_x = int(max(x for x, _ in all_xy))
            max_y = int(max(y for _, y in all_xy))
            w = max_x + 2
            h = max_y + 2
    else:
        h, w = out_hw

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for feat in data:
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue

        color_int = (
            feat.get("properties", {}).get("classification", {}).get("colorRGB", None)
        )
        if isinstance(color_int, int):
            r = (color_int >> 16) & 255
            g = (color_int >> 8) & 255
            b = (color_int) & 255
            color = (b, g, r)
        else:
            color = (255, 255, 255)

        for ring in coords:
            pts = []
            for pt in ring:
                if len(pt) >= 2:
                    pts.append([int(round(pt[0])), int(round(pt[1]))])
            if len(pts) >= 3:
                pts = np.array(pts, dtype=np.int32)
                cv2.fillPoly(canvas, [pts], color=color)

    canvas = cv2.resize(
        canvas, (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), interpolation=cv2.INTER_AREA
    )
    return canvas




## === cell 5
class SingleTiffDataset(Dataset):
    """
    Reads an image canvas (BGR) and tiles it; we now support JSON-based test inputs by
    rasterizing them to a BGR canvas before tiling (no change to tiling/inference semantics).
    """

    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step

        tiff_path = str(tiff_path)
        p = Path(tiff_path)

        img_bgr = None
        if p.suffix.lower() in (".json",):
            img_bgr = rasterize_anatomical_json_to_bgr(p)
        else:
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {tiff_path}")

        img_bgr = cv2.resize(
            img_bgr, (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        self.img_bgr = img_bgr
        self.h, self.w = img_bgr.shape[:2]

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

        img = self.img_bgr[y : y + self.crop_size, x : x + self.crop_size]

        transformed_imgs = {
            img_size: valid_transform(img_size)(image=img)["image"]
            for img_size in self.all_img_sizes
        }
        transformed_imgs["crop_names"] = f"{x}_{y}"
        transformed_imgs["not_background"] = _check_background(img, self.crop_size)
        return transformed_imgs




## === cell 6
class SimpleUNet(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()

        if backbone_name == "resnet34":
            backbone = torchvision.models.resnet34(weights=None)
            enc_channels = [64, 64, 128, 256, 512]
        elif backbone_name in ("se_resnext50_32x4d", "resnext50_32x4d"):
            backbone = torchvision.models.resnext50_32x4d(weights=None)
            enc_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {backbone_name}")

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.maxpool = backbone.maxpool  # /4
        self.layer1 = backbone.layer1  # /4
        self.layer2 = backbone.layer2  # /8
        self.layer3 = backbone.layer3  # /16
        self.layer4 = backbone.layer4  # /32

        def conv_relu(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.up4 = conv_relu(enc_channels[4], 256)
        self.up3 = conv_relu(256 + enc_channels[3], 256)
        self.up2 = conv_relu(256 + enc_channels[2], 128)
        self.up1 = conv_relu(128 + enc_channels[1], 64)
        self.up0 = conv_relu(64 + enc_channels[0], 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)  # /2
        x1 = self.maxpool(x0)  # /4
        x1 = self.layer1(x1)  # /4
        x2 = self.layer2(x1)  # /8
        x3 = self.layer3(x2)  # /16
        x4 = self.layer4(x3)  # /32

        d4 = F.interpolate(
            self.up4(x4), scale_factor=2, mode="bilinear", align_corners=False
        )  # /16
        d3 = torch.cat([d4, x3], dim=1)
        d3 = F.interpolate(
            self.up3(d3), scale_factor=2, mode="bilinear", align_corners=False
        )  # /8
        d2 = torch.cat([d3, x2], dim=1)
        d2 = F.interpolate(
            self.up2(d2), scale_factor=2, mode="bilinear", align_corners=False
        )  # /4
        d1 = torch.cat([d2, x1], dim=1)
        d1 = F.interpolate(
            self.up1(d1), scale_factor=2, mode="bilinear", align_corners=False
        )  # /2
        d0 = torch.cat([d1, x0], dim=1)
        d0 = F.interpolate(
            self.up0(d0), scale_factor=2, mode="bilinear", align_corners=False
        )  # /1

        return self.head(d0)


def build_model(encoder_name: str) -> nn.Module:
    mapped = "resnext50_32x4d" if encoder_name == "se_resnext50_32x4d" else encoder_name
    return SimpleUNet(mapped)


def load_state_flexible(model: nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint type at {weights_path}: {type(state)}")

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}) for {weights_path}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}) for {weights_path}")




## === cell 7
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)
    mask_pred = np.zeros(img_size, dtype=np.uint8)

    for batch in tqdm(data_loader, ncols=70, leave=True):
        if SKIP_BACKGROUND is True and batch["not_background"].item() is False:
            continue

        with torch.no_grad():
            pred_total = None
            for cfg, model_group in models_with_config:
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
                    pred_group = pred if pred_group is None else (pred_group + pred)

                pred_group = pred_group.squeeze()
                if len(pred_group.shape) == 2:
                    pred_group = pred_group.unsqueeze(0)

                pred_group = cfg.weight_blend * (pred_group / len(model_group))
                pred_total = (
                    pred_group if pred_total is None else (pred_total + pred_group)
                )

            pred_total = (pred_total.detach().cpu().numpy() > THR).astype(np.uint8)

            for predict_single, crop_name in zip(pred_total, batch["crop_names"]):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])
                if crop_size != TRAIN_IMG_SIZE:
                    predict_single = cv2.resize(
                        predict_single,
                        (crop_size, crop_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred.astype(np.uint8))
    del mask_pred
    gc.collect()
    return mask_rle




## === cell 8
all_img_sizes = set(cfg.img_size for cfg in my_models)

models_with_config = []
missing_weights = []

for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        if not Path(w_path).exists():
            missing_weights.append(w_path)
            continue
        model = build_model(cfg.encoder).to(DEVICE)
        load_state_flexible(model, w_path)
        model.eval()
        models_group.append(model)
    if len(models_group) > 0:
        models_with_config.append((cfg, models_group))

if missing_weights:
    print(
        f"[WARN] Missing {len(missing_weights)} weight files. Example:",
        missing_weights[:3],
    )

print("Loaded model groups:", [(c.encoder, len(g)) for c, g in models_with_config])



## === cell 9
if len(models_with_config) == 0:
    print(
        "[WARN] No model weights available -> writing empty predictions for all test ids."
    )
    df_sub["predicted"] = ""
else:
    for idx, row in df_sub.iterrows():
        json_path = INPUT_DIR / "test" / f"{row['id']}-anatomical-structure.json"
        if not json_path.exists():
            json_path = INPUT_DIR / "test" / f"{row['id']}.json"

        test_ds = SingleTiffDataset(
            tiff_path=str(json_path),
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



## === cell 10
df_sub["predicted"] = df_sub["predicted"].fillna("").astype(str)

out_path = Path("submission.csv")
df_sub[["id", "predicted"]].to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())
print(df_sub.head())




## === cell 11
def rm_tree(p: Path):
    if not p.exists():
        return
    for child in p.rglob("*"):
        if child.is_file() or child.is_symlink():
            try:
                child.unlink()
            except Exception:
                pass
    for child in sorted(p.rglob("*"), reverse=True):
        if child.is_dir():
            try:
                child.rmdir()
            except Exception:
                pass
    try:
        p.rmdir()
    except Exception:
        pass


rm_tree(MODEL1_DIR)
rm_tree(MODEL2_DIR)



## === cell 12
df_sub
