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
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most plausibly coming from a semantic mismatch: the model was trained on real RGB histology TIFFs, but the current pipeline feeds rasterized anatomical-structure polygons as “images”, so predictions are effectively meaningless even though the submission is valid. The smallest change that should move Dice toward your target is to run inference on the actual `.tif` images (as in the original competition) by extracting `train.zip`/`test.zip` (or using already-extracted folders) and reading TIFFs with OpenCV, while keeping your tiling, thresholds, ensemble, and RLE encoding unchanged. I also keep the JSON rasterization as a fallback only if a TIFF truly isn’t found, so the script still runs end-to-end. Finally, I make the dataset use the real image dimensions from `HuBMAP-20-dataset_information.csv` when available so RLE indexing aligns with the true pixel grid.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script never actually using the intended pretrained models: the `1024.tar`/`512.tar` aren’t found in this dataset, so `models_with_config` becomes empty and you submit all-empty masks. I make the smallest change that moves you toward the target by (1) extracting the provided model tar files from common locations in `/kaggle/input` (including `.zip`-wrapped tars) and (2) falling back to a trivial-but-nonzero prediction only if weights still can’t be loaded (so you don’t get stuck at 0.0). I also fix one subtle submission-format pitfall: ensure we always use the competition’s `id,predicted` columns and don’t carry over `img,pixels` naming. Core tiling, thresholding, ensemble averaging, and RLE encoding stay the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely because you’re not actually using the intended pretrained weights: the expected `.pth` files don’t exist after extracting `1024.tar`/`512.tar`, so `models_with_config` ends up empty and you fall back to a tiny constant mask (near-zero Dice). I make the smallest change that restores real inference by (1) searching the extracted model directories for any `.pth` files and using them if the hardcoded paths are missing, and (2) ensuring we open TIFFs correctly (HubMAP TIFFs are often multi-page/16-bit, so `cv2.imread` can return `None`; using `cv2.imdecode` from bytes is a robust fix). These changes preserve your core tiling/inference/threshold/RLE logic and only improve the model-weight discovery and image loading so predictions become meaningful. The submission format and filename remain unchanged (`submission.csv` with `id,predicted`).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the pipeline not actually using valid pretrained weights (so it falls back to the tiny constant mask, which yields ~0 Dice), and/or failing to read the real TIFFs so it ends up predicting on rasterized anatomical-structure JSONs. I make minimal changes to (1) robustly locate and extract `1024.tar`/`512.tar` from all provided dataset locations (including inside the competition folder), (2) robustly read TIFFs using OpenCV’s `IMREAD_UNCHANGED` + channel conversion (more reliable than `imdecode` for TIFF here), and (3) ensure RLE encoding is correct and safe for empty/full masks without mutating the input array. Core model, tiling, thresholds, ensembling, and submission format remain unchanged; this should move you sharply upward toward the target by enabling real inference. The script still guarantees a valid `submission.csv` even if weights are truly absent.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with never actually loading the intended pretrained weights, causing the fallback tiny constant masks (near-zero Dice). I make the smallest changes that increase the chance of loading real weights by (1) extracting the correct model archives from more possible locations (including nested inside the dataset folder and any tar files found), and (2) making weight discovery stricter so we only use `.pth` files that match the requested fold naming (instead of mixing arbitrary `.pth` files across configs). I also add a minimal checkpoint-key remapping for common `segmentation_models_pytorch` naming (`encoder.*` / `decoder.*` / `segmentation_head.*`) to our wrapper’s keys, without changing the model forward/inference/RLE logic. Everything else (tiling, thresholds, blending, submission format) remains the same.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
import tarfile
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple
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
def find_first(pattern: str, roots: List[str]) -> Optional[Path]:
    for root in roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        matches = list(rootp.rglob(pattern))
        if matches:
            return matches[0]
    return None


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=out_dir)


def safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def try_prepare_model_dir(model_dir: Path, tar_name: str) -> Optional[Path]:
    """
    Change (score-relevant): broaden archive discovery so we actually extract the pretrained weights
    (prevents all-empty/tiny-mask fallback -> raises Dice from 0.0).
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [
        "/kaggle/input",
        "/kaggle/input/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
    ]

    tar_path = find_first(tar_name, search_roots)

    if tar_path is None:
        zip_path = find_first(f"{tar_name}.zip", search_roots)
        if zip_path is not None:
            tmp_dir = model_dir / "_zip_extract"
            safe_extract_zip(zip_path, tmp_dir)
            cand = list(tmp_dir.rglob(tar_name))
            if cand:
                tar_path = cand[0]

    if tar_path is None:
        all_tars = []
        for root in search_roots:
            rp = Path(root)
            if rp.exists():
                all_tars.extend(list(rp.rglob("*.tar")))
                all_tars.extend(list(rp.rglob("*.tar.gz")))
                all_tars.extend(list(rp.rglob("*.tgz")))
        all_tars = sorted(all_tars, key=lambda p: (tar_name not in p.name, len(p.name)))
        for cand_tar in all_tars[:50]:
            try:
                with tarfile.open(cand_tar, "r:*") as tar:
                    names = tar.getnames()
                if any(("fold" in n and n.endswith(".pth")) for n in names):
                    tar_path = cand_tar
                    print(f"[INFO] Using fallback tar for {tar_name}: {tar_path}")
                    break
            except Exception:
                continue

    if tar_path is None:
        return None

    safe_extract_tar(tar_path, model_dir)
    return tar_path


MODEL1_DIR = Path("./model_1")
MODEL2_DIR = Path("./model_2")

tar_1024 = try_prepare_model_dir(MODEL1_DIR, "1024.tar")
tar_512 = try_prepare_model_dir(MODEL2_DIR, "512.tar")

print("Prepared 1024.tar from:", tar_1024)
print("Prepared 512.tar from:", tar_512)
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

SKIP_COMMIT = False

INPUT_DIR = Path("/kaggle/input/hubmap-kidney-segmentation")
df_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

if (
    ("img" in df_sub.columns)
    and ("pixels" in df_sub.columns)
    and ("id" not in df_sub.columns)
):
    df_sub = df_sub.rename(columns={"img": "id", "pixels": "predicted"})
if "id" not in df_sub.columns and df_sub.shape[1] >= 1:
    df_sub = df_sub.rename(columns={df_sub.columns[0]: "id"})
if "predicted" not in df_sub.columns:
    df_sub["predicted"] = ""

df_sub = df_sub[["id", "predicted"]].copy()
print(df_sub.head())

WORK_DIR = Path("/kaggle/working/hubmap_data")
TEST_IMG_DIR = WORK_DIR / "test"
TRAIN_IMG_DIR = WORK_DIR / "train"
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)

test_zip = INPUT_DIR / "test.zip"
train_zip = INPUT_DIR / "train.zip"


def maybe_extract(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists():
        print("[WARN] zip not found:", zip_path)
        return
    if any(out_dir.rglob("*.tif")) or any(out_dir.rglob("*.tiff")):
        return
    print("Extracting:", zip_path, "->", out_dir)
    safe_extract_zip(zip_path, out_dir)


maybe_extract(test_zip, TEST_IMG_DIR)
maybe_extract(train_zip, TRAIN_IMG_DIR)

info_path = INPUT_DIR / "HuBMAP-20-dataset_information.csv"
df_info = None
if info_path.exists():
    df_info = pd.read_csv(info_path)
    df_info["id"] = (
        df_info["image_file"]
        .astype(str)
        .str.replace(".tif", "", regex=False)
        .str.replace(".tiff", "", regex=False)
    )
    id2wh: Dict[str, Tuple[int, int]] = {
        r["id"]: (int(r["width_pixels"]), int(r["height_pixels"]))
        for _, r in df_info.iterrows()
        if ("width_pixels" in r) and ("height_pixels" in r)
    }
else:
    id2wh = {}

print("Have size metadata for:", len(id2wh), "images")




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
    if img.size == 0:
        return ""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8, copy=False)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
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


def find_test_tiff(img_id: str) -> Optional[Path]:
    for ext in (".tif", ".tiff"):
        p = TEST_IMG_DIR / f"{img_id}{ext}"
        if p.exists():
            return p
    for ext in (".tif", ".tiff"):
        p = INPUT_DIR / "test" / f"{img_id}{ext}"
        if p.exists():
            return p
    hits = list(TEST_IMG_DIR.rglob(f"{img_id}.tif")) + list(
        TEST_IMG_DIR.rglob(f"{img_id}.tiff")
    )
    if hits:
        return hits[0]
    return None


def read_tiff_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]

        if img.dtype != np.uint8:
            imin = float(img.min())
            imax = float(img.max())
            if imax > imin:
                img = (
                    ((img.astype(np.float32) - imin) / (imax - imin) * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        return img
    except Exception:
        return None




## === cell 5
class SingleTiffDataset(Dataset):
    def __init__(self, img_id: str, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step
        self.img_id = img_id

        if img_id in id2wh:
            self.w, self.h = id2wh[img_id]
        else:
            self.h = self.w = TRAIN_IMG_SIZE

        img_bgr = None
        tiff_path = find_test_tiff(img_id)
        if tiff_path is not None:
            img_bgr = read_tiff_bgr(tiff_path)
            if img_bgr is None:
                img_bgr = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)

        if img_bgr is None:
            json_path = INPUT_DIR / "test" / f"{img_id}-anatomical-structure.json"
            if not json_path.exists():
                json_path = INPUT_DIR / "test" / f"{img_id}.json"
            img_bgr = rasterize_anatomical_json_to_bgr(
                json_path, out_hw=(self.h, self.w)
            )

        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image for id={img_id}")

        if img_bgr.shape[0] != self.h or img_bgr.shape[1] != self.w:
            img_bgr = cv2.resize(
                img_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA
            )

        self.img_bgr = img_bgr
        self.h, self.w = img_bgr.shape[:2]

        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)
        if self.h <= self.crop_size:
            self.row_count = 1
        if self.w <= self.crop_size:
            self.col_count = 1

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = max(0, self.w - self.crop_size)
        if y + self.crop_size > self.h:
            y = max(0, self.h - self.crop_size)

        img = self.img_bgr[y : y + self.crop_size, x : x + self.crop_size]

        if img.shape[0] != self.crop_size or img.shape[1] != self.crop_size:
            pad = np.zeros((self.crop_size, self.crop_size, 3), dtype=img.dtype)
            pad[: img.shape[0], : img.shape[1]] = img
            img = pad

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

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

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
        x0 = self.stem(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = F.interpolate(
            self.up4(x4), scale_factor=2, mode="bilinear", align_corners=False
        )
        d3 = torch.cat([d4, x3], dim=1)
        d3 = F.interpolate(
            self.up3(d3), scale_factor=2, mode="bilinear", align_corners=False
        )
        d2 = torch.cat([d3, x2], dim=1)
        d2 = F.interpolate(
            self.up2(d2), scale_factor=2, mode="bilinear", align_corners=False
        )
        d1 = torch.cat([d2, x1], dim=1)
        d1 = F.interpolate(
            self.up1(d1), scale_factor=2, mode="bilinear", align_corners=False
        )
        d0 = torch.cat([d1, x0], dim=1)
        d0 = F.interpolate(
            self.up0(d0), scale_factor=2, mode="bilinear", align_corners=False
        )

        return self.head(d0)


def build_model(encoder_name: str) -> nn.Module:
    mapped = "resnext50_32x4d" if encoder_name == "se_resnext50_32x4d" else encoder_name
    return SimpleUNet(mapped)


def _remap_smp_to_simpleunet_keys(
    state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Change (score-relevant): many public HubMAP weights are from segmentation_models_pytorch with keys:
    encoder.*, decoder.*, segmentation_head.*. We minimally remap the encoder to our torchvision backbone
    names and segmentation head to our head, so weights can actually load (reduces 0.0 risk).
    """
    out = {}
    for k, v in state.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]

        if nk.startswith("encoder."):
            nk2 = nk[len("encoder.") :]
            if nk2.startswith("conv1."):
                nk = "stem.0." + nk2[len("conv1.") :]
            elif nk2.startswith("bn1."):
                nk = "stem.1." + nk2[len("bn1.") :]
            elif nk2.startswith("layer1."):
                nk = "layer1." + nk2[len("layer1.") :]
            elif nk2.startswith("layer2."):
                nk = "layer2." + nk2[len("layer2.") :]
            elif nk2.startswith("layer3."):
                nk = "layer3." + nk2[len("layer3.") :]
            elif nk2.startswith("layer4."):
                nk = "layer4." + nk2[len("layer4.") :]
            elif nk2.startswith("relu."):
                nk = "stem.2." + nk2[len("relu.") :]
            elif nk2.startswith("maxpool."):
                nk = "maxpool." + nk2[len("maxpool.") :]
            else:
                nk = nk2  # best-effort

        if nk.startswith("segmentation_head."):
            nk2 = nk[len("segmentation_head.") :]
            if nk2 in ("0.weight", "0.bias"):
                nk = "head." + nk2.split(".")[1]
            elif nk2 in ("weight", "bias"):
                nk = "head." + nk2

        out[nk] = v
    return out


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

    state = _remap_smp_to_simpleunet_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
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
                y2 = min(y + crop_size, mask_pred.shape[0])
                x2 = min(x + crop_size, mask_pred.shape[1])
                mask_pred[y:y2, x:x2] += predict_single[: (y2 - y), : (x2 - x)]

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred.astype(np.uint8))
    del mask_pred
    gc.collect()
    return mask_rle




## === cell 8
def discover_pth_files(model_dir: Path) -> List[str]:
    if not model_dir.exists():
        return []
    pths = sorted([str(p) for p in model_dir.rglob("*.pth")])
    return pths


def patch_missing_weight_paths(cfg: ModelConfig) -> ModelConfig:
    """
    Change (score-relevant): avoid mixing arbitrary discovered .pth files across configs;
    only substitute likely matching fold*_avg_*.pth files when the configured paths are missing.
    """
    if any(Path(p).exists() for p in cfg.weights_path):
        return cfg

    guess_dir = MODEL1_DIR if cfg.img_size >= 4096 else MODEL2_DIR
    found = discover_pth_files(guess_dir)

    fold_found = [
        p for p in found if ("fold" in Path(p).name and "_avg_" in Path(p).name)
    ]
    if not fold_found:
        fold_found = [p for p in found if "fold" in Path(p).name]

    if not fold_found:
        found2 = discover_pth_files(MODEL1_DIR) + discover_pth_files(MODEL2_DIR)
        found2 = sorted(set(found2))
        fold_found = [
            p for p in found2 if ("fold" in Path(p).name and "_avg_" in Path(p).name)
        ]
        if not fold_found:
            fold_found = [p for p in found2 if "fold" in Path(p).name]

    if fold_found:
        print(
            f"[INFO] Replacing missing weights for encoder={cfg.encoder} with discovered {len(fold_found)} fold-like .pth files."
        )
        return ModelConfig(
            encoder=cfg.encoder,
            weights_path=fold_found,
            img_size=cfg.img_size,
            weight_blend=cfg.weight_blend,
        )
    return cfg


my_models = [patch_missing_weight_paths(cfg) for cfg in my_models]
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
def tiny_constant_rle(h: int, w: int, k: int = 16) -> str:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:k, :k] = 1
    return rle_encode_less_memory(mask)


if len(models_with_config) == 0:
    print(
        "[WARN] No model weights available -> writing tiny constant predictions for all test ids (avoid 0.0)."
    )
    preds = []
    for img_id in df_sub["id"].astype(str).tolist():
        if img_id in id2wh:
            w, h = id2wh[img_id]
        else:
            h = w = TRAIN_IMG_SIZE
        preds.append(tiny_constant_rle(h, w, k=16))
    df_sub["predicted"] = preds
else:
    for idx, row in df_sub.iterrows():
        img_id = str(row["id"])
        test_ds = SingleTiffDataset(
            img_id=img_id,
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
