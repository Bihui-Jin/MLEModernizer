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

0.9468367082589368

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed) by recreating the same UNet-style model using `torchvision`’s `efficientnet_b4` encoder and a lightweight decoder so the script runs in this environment. I also make model weight loading robust: if the expected Kaggle input checkpoint paths don’t exist, the code fall back to a safe baseline prediction (empty masks) and still write a valid `submission.csv` instead of crashing. Finally, I fix the cell numbering, keep all paths unchanged, and ensure `names/preds` are always defined so the submission file is always produced end-to-end.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the script never finds any checkpoints at the hardcoded `../input/skfoldalldata/...` paths, so it always falls back to empty masks. I keep the exact same tiling/inference/RLE logic, but make checkpoint discovery robust by also searching the provided dataset folders (`../input/` and `/kaggle/input/`) for matching `*model.pth` files and loading up to 5 of them. If no checkpoints are found, it still write a valid `submission.csv` (unchanged behavior), but if they exist anywhere in the environment the model now actually run and the Dice score should move upward toward your target. I also add a small safety fix to ensure we don’t accidentally double-resize (keep your existing semantics) and keep determinism intact.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with producing empty masks because no compatible checkpoints are found/loaded, so the smallest improvement is to ensure we can actually run inference with weights that exist in this environment. I keep your exact tiling/inference/RLE logic and the same EfficientNetB4-UNet architecture, but make checkpoint loading robust to common training-save formats (e.g., `{"state_dict": ...}` and `module.` prefixes) so more discovered `.pth` files become usable instead of being silently skipped. I also expand checkpoint discovery to include more generic `*.pth` patterns under the provided input roots, while still limiting to 5 models to stay within time. These changes should move the Dice score upward toward your target without changing evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from either (a) no checkpoints being found/loaded (so you always submit empty masks) or (b) a submission-format mismatch with the competition’s required column names. I keep your model/inference core logic unchanged, but I (1) enforce the exact sample_submission schema (`id` + `predicted`) and (2) make checkpoint discovery include common extensions (`.pt`, `.bin`) and prefer the original intended fold-named files if they exist. I also add a tiny safety fix to ensure we don’t accidentally write blank RLE strings as `NaN`-like values and to keep the output rows aligned to `sample_submission.csv` order. These minimal changes should move the score upward toward your target by producing non-empty masks when any usable weights exist and by ensuring Kaggle accepts/scoring the file correctly.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with either (1) the submission not matching the required sample order/IDs or (2) predictions being effectively empty/incorrectly merged due to tile aggregation. I make two minimal, score-relevant fixes while preserving your model/inference core: (a) force the final submission `id` order to exactly match `sample_submission.csv` (and only those IDs), and (b) change tile merging to use a per-pixel max over overlapping tiles (instead of summing then thresholding), which is a standard semantic-preserving fix that prevents overlaps from washing out predictions at the global threshold. Everything else (model, preprocessing, tiling, threshold, RLE) stays the same, and it still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script never actually reading the test images (so it either crashes in Kaggle or produces effectively empty/invalid masks), because the current `DATA="../input/..."` path is not guaranteed in your provided environment while `/kaggle/input/...` is. I keep your exact model/tiling/inference/RLE logic, but make the dataset root path robust by auto-selecting the first existing path among the common Kaggle locations (without changing filenames or downstream behavior). I also ensure the submission IDs come from the sample submission (already done) and keep the empty-mask fallback unchanged if no checkpoints are loadable. These minimal fixes should move the score upward toward your target by enabling real inference instead of silently failing or outputting blanks.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because no usable checkpoints exist in this environment, so the code always writes empty masks; to move toward the target, the smallest legitimate change is to use the provided training labels to fit a single global probability threshold (still Dice-aligned) and then apply that calibrated threshold at inference. This keeps your exact model, tiling, resizing, max-merge, and RLE logic unchanged; it only replaces the fixed `TH=0.5` with a data-driven `TH` found by scanning a small set of candidate thresholds on the training images. If checkpoints still cannot be loaded, the behavior remains unchanged (empty submission), but when checkpoints are available/loadable this typically yields a clear Dice increase with minimal risk. I also make the train/test CSV column handling robust (`id` vs `img`) so calibration can run without breaking submission formatting.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with Kaggle evaluating your submission as “all background” for every test image, which most often happens when the code never actually runs model inference on the real test TIFFs (missing/unread images → implicit empty outputs) or when the predicted masks are mis-sized/misaligned at RLE time. I make two minimal, score-relevant fixes while preserving your model/tiling/inference/RLE core: (1) ensure the dataset root points to a folder that actually contains the TIFFs by auto-resolving among common locations, and (2) fix the tile grid axis convention so x/y are not swapped (a silent but devastating alignment bug that yields near-zero Dice even with a good model). Everything else (model architecture, preprocessing, max-merge, threshold calibration, and submission schema) is kept the same, and it still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import glob
import warnings

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




## === cell 1
sz = 256  # tile size fed to model
reduce = 4

TH = 0.5

TRAIN = False


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def _resolve_comp_dir():
    candidates = [
        "../input/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation",
        "/kaggle/data/input/hubmap-kidney-segmentation",
        "../kaggle/data/input/hubmap-kidney-segmentation",
        "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
        "/kaggle/data/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
    ]
    for base in candidates:
        test_dir = os.path.join(base, "test")
        train_dir = os.path.join(base, "train")
        if (
            os.path.isdir(test_dir)
            and len(glob.glob(os.path.join(test_dir, "*.tiff"))) > 0
        ):
            return base
        if (
            os.path.isdir(train_dir)
            and len(glob.glob(os.path.join(train_dir, "*.tiff"))) > 0
        ):
            return base
    return _first_existing(candidates)


COMP_DIR = _resolve_comp_dir()

if TRAIN:
    DATA = os.path.join(COMP_DIR, "train") + "/"
    df_sample = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
else:
    DATA = os.path.join(COMP_DIR, "test") + "/"
    df_sample = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

MODELS = [
    f"../input/skfoldalldata/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]
bs = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
EXPAND = 4
minoverlap = 1 / 16
TTA = False

tile_size = int(sz * EXPAND)  # 1024
tile_resized = int(tile_size * reduce)  # 4096

print("Using COMP_DIR:", COMP_DIR)
print("Using DATA:", DATA)
print(
    "Found test TIFFs:",
    len(glob.glob(os.path.join(os.path.dirname(DATA.rstrip("/")), "*.tiff"))),
)




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
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40
p_th = 1000 * (tile_size // 256) ** 2


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array of shape (N,4) with x1,x2,y1,y2 in original (full-res) coordinates.
    shape: (H, W)
    """
    H, W = int(shape[0]), int(shape[1])

    nx = H // (window - min_overlap) + 1
    x1 = np.linspace(0, H, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = H - window
    x2 = (x1 + window).clip(0, H)

    ny = W // (window - min_overlap) + 1
    y1 = np.linspace(0, W, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = W - window
    y2 = (y1 + window).clip(0, W)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


class HuBMAPDataset(Dataset):
    """
    rasterio-free replacement:
    - Read full .tiff with tifffile (memory-mapped when possible).
    - Slice windows via NumPy.
    Preserves the original tiling / resizing / blank-tile filtering behavior.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Missing image: {self.path}")

        self.data = tiff.memmap(self.path)
        if self.data.ndim == 2:
            self.data = self.data[:, :, None]
        if self.data.shape[-1] == 3:
            self.hwc = self.data
        elif self.data.ndim == 3 and self.data.shape[0] == 3:
            self.hwc = np.moveaxis(self.data, 0, -1)
        else:
            if self.data.ndim == 3 and self.data.shape[-1] > 3:
                self.hwc = self.data[:, :, :3]
            elif self.data.ndim == 3 and self.data.shape[0] > 3:
                self.hwc = np.moveaxis(self.data[:3], 0, -1)
            else:
                raise ValueError(
                    f"Unexpected tiff shape for {self.path}: {self.data.shape}"
                )

        self.shape = self.hwc.shape[:2]  # (H, W)
        self.reduce = reduce
        self.sz = reduce * sz
        self.mask_grid = make_grid(
            self.shape, window=tile_resized, min_overlap=int(tile_resized * minoverlap)
        )

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, i):
        x1, x2, y1, y2 = self.mask_grid[i]
        img = np.array(self.hwc[x1:x2, y1:y2, :], copy=False)

        if self.reduce != 1:
            img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        if img.dtype != np.uint8:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_u8 = img

        hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)
        if (s > s_th).sum() <= p_th or img_u8.sum() <= p_th:
            return img2tensor((img_u8 / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img_u8 / 255.0 - mean) / std), vertices, i


class Model_pred:
    def __init__(self, models, dl, tta: bool = TTA, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, z, y in iter(self.dl):
                if (y >= 0).sum() > 0:
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
                        yield py[i], z[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 4
class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EfficientNetB4UNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = efficientnet_b4(weights=weights).features

        self.tap_idxs = [1, 2, 3, 4, 6, 7]
        self.tap_channels = [24, 32, 56, 112, 272, 448]

        self.center = nn.Sequential(
            ConvRelu(self.tap_channels[-1], 256), ConvRelu(256, 256)
        )
        self.up5 = UpBlock(256, self.tap_channels[-2], 256)
        self.up4 = UpBlock(256, self.tap_channels[-3], 128)
        self.up3 = UpBlock(128, self.tap_channels[-4], 64)
        self.up2 = UpBlock(64, self.tap_channels[-5], 32)
        self.up1 = UpBlock(32, self.tap_channels[-6], 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.tap_idxs:
                feats.append(x)
        while len(feats) < 6:
            feats.append(x)

        s1, s2, s3, s4, s5, s6 = feats
        x = self.center(s6)
        x = self.up5(x, s5)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)


def build_model():
    m = EfficientNetB4UNet(pretrained=False)
    return m




## === cell 5
def discover_checkpoints(preferred_paths, max_n=5):
    existing = [p for p in preferred_paths if os.path.exists(p)]
    if len(existing) > 0:
        return existing[:max_n]

    search_roots = ["../input", "/kaggle/input", "/kaggle/data/input"]
    patterns = [
        "**/*efficientnet-b4*FOLD*model.pth",
        "**/*efficientnet_b4*FOLD*model.pth",
        "**/*efficientnet*b4*unet*model.pth",
        "**/*hubmap*model*.pth",
        "**/*unet*model*.pth",
        "**/*FOLD-*-model.pth",
        "**/*model*.pth",
        "**/*model*.pt",
        "**/*model*.bin",
    ]
    found = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for pat in patterns:
            found.extend(glob.glob(os.path.join(root, pat), recursive=True))

    found = sorted(set(found))
    return found[:max_n]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def dice_coeff(
    pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-7
) -> float:
    pred = pred_mask.astype(np.uint8)
    tru = true_mask.astype(np.uint8)
    inter = (pred & tru).sum()
    union = pred.sum() + tru.sum()
    if union == 0:
        return 1.0
    return float((2.0 * inter + eps) / (union + eps))


def infer_full_prob(idx: str, models, bs: int):
    ds = HuBMAPDataset(idx)
    dl = DataLoader(
        ds,
        batch_size=bs,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=0,
    )
    mp = Model_pred(models, dl)
    prob = np.zeros(ds.shape, dtype=np.float32)
    for pred, vert, _i in iter(mp):
        x1, x2, y1, y2 = vert
        prob[x1:x2, y1:y2] = np.maximum(prob[x1:x2, y1:y2], pred.astype(np.float32))
    return prob, ds.shape


def calibrate_threshold(models, comp_dir: str, max_images: int = 12):
    train_csv_path = os.path.join(comp_dir, "train.csv")
    train_dir = os.path.join(comp_dir, "train") + "/"
    if (not os.path.exists(train_csv_path)) or (not os.path.exists(train_dir)):
        return None

    df_tr = pd.read_csv(train_csv_path)
    if "id" in df_tr.columns:
        id_col = "id"
    elif "img" in df_tr.columns:
        id_col = "img"
    else:
        return None
    if "rle" in df_tr.columns:
        rle_col = "rle"
    elif "encoding" in df_tr.columns:
        rle_col = "encoding"
    else:
        rle_candidates = [c for c in df_tr.columns if "rle" in c.lower()]
        if len(rle_candidates) == 0:
            return None
        rle_col = rle_candidates[0]

    ids = df_tr[id_col].astype(str).tolist()[:max_images]
    rles = df_tr[rle_col].tolist()[:max_images]

    global DATA
    old_data = DATA
    DATA = train_dir
    try:
        probs = []
        trues = []
        for img_id, rle in zip(ids, rles):
            prob, shape = infer_full_prob(img_id, models, bs=bs)
            true_mask = enc2mask([rle], shape).astype(np.uint8)
            true_mask = (true_mask > 0).astype(np.uint8)
            probs.append(prob)
            trues.append(true_mask)

        candidates = np.linspace(0.25, 0.75, 21)
        best_th, best_score = None, -1.0
        for th in candidates:
            scores = []
            for prob, tru in zip(probs, trues):
                pred = (prob > th).astype(np.uint8)
                scores.append(dice_coeff(pred, tru))
            mean_score = float(np.mean(scores)) if len(scores) else -1.0
            if mean_score > best_score:
                best_score = mean_score
                best_th = float(th)

        return best_th
    except Exception as e:
        print(f"Warning: threshold calibration failed: {repr(e)}")
        return None
    finally:
        DATA = old_data
        gc.collect()


ckpt_paths = discover_checkpoints(MODELS, max_n=5)

models = []
for path in ckpt_paths:
    try:
        raw = torch.load(path, map_location=torch.device("cpu"))
        state_dict = _extract_state_dict(raw)

        state_dict = _strip_prefix_if_present(state_dict, "module.")
        state_dict = _strip_prefix_if_present(state_dict, "model.")
        state_dict = _strip_prefix_if_present(state_dict, "net.")

        model = build_model()
        model.load_state_dict(state_dict, strict=False)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)

        del raw, state_dict
        gc.collect()
    except Exception as e:
        print(f"Warning: failed to load checkpoint {path}: {repr(e)}")
        continue

print(
    f"Discovered {len(ckpt_paths)} checkpoints; successfully loaded {len(models)} models."
)
if len(models) == 0:
    print(
        "No usable checkpoints found/loaded; will write empty-mask submission (valid but low score)."
    )
else:
    th_cal = calibrate_threshold(models, COMP_DIR, max_images=12)
    if th_cal is not None and np.isfinite(th_cal):
        TH = float(th_cal)
    print(f"Using threshold TH={TH:.4f}")

names, preds = [], []

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

sample_ids = df_sample["id"].astype(str).tolist()

if len(models) == 0:
    for idx in tqdm(sample_ids, total=len(sample_ids)):
        names.append(idx)
        preds.append("")  # empty RLE = no mask
else:
    for idx in tqdm(sample_ids, total=len(sample_ids)):
        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds,
            batch_size=bs,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            num_workers=0,
        )
        mp = Model_pred(models, dl)

        prob = np.zeros(ds.shape, dtype=np.float32)
        for pred, vert, _i in iter(mp):
            x1, x2, y1, y2 = vert
            prob[x1:x2, y1:y2] = np.maximum(prob[x1:x2, y1:y2], pred.astype(np.float32))

        mask = (prob > TH).astype(np.uint8)
        rle = rle_encode_less_memory(mask) if mask.sum() > 0 else ""
        if rle is None:
            rle = ""

        names.append(idx)
        preds.append(str(rle))

        del mask, prob, ds, dl
        gc.collect()




## === cell 6
sub = pd.DataFrame({"id": names, "predicted": preds})
sub = sub.set_index("id").reindex(sample_ids).reset_index()
sub["predicted"] = sub["predicted"].fillna("").astype(str)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
print(
    "Non-empty predictions:",
    int((sub["predicted"].astype(str).str.len() > 0).sum()),
    "/",
    len(sub),
)
