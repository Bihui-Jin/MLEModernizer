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

0.9370673044403018

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TIFF loading failure by adding a safe fallback that reads JPEG-compressed TIFFs via OpenCV (which is available) when `tifffile` can’t decode them due to missing `imagecodecs`. I also ensure the code always produces one prediction per `sample_submission` id by catching per-image read errors and outputting an empty RLE (valid “no mask”) rather than skipping the row. These changes keep the core tiling/prediction/RLE logic intact while making the pipeline run end-to-end and write a valid `submission.csv`. The heuristic fallback model path is preserved as-is, but the primary blocker (image IO) is removed so the actual model ensemble can run if weights are present.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an evaluation mismatch: this competition’s submission requires a **binary mask of all glomeruli merged**, but your current `rle_encode_less_memory` encodes the *count-accumulated* mask (values 0/1 are fine) yet it never forces strict binary for safety and can also emit an invalid/empty string in edge cases; additionally, the aggregation step `mask[x1:x2,y1:y2] += tile` with a final `>0.5` threshold is effectively “any-hit”, but using an explicit boolean OR avoids numeric accumulation artifacts and improves alignment with Dice on binary masks. I keep your tiling/model/fallback logic intact, but (1) switch the stitching to boolean OR, (2) ensure we always RLE-encode a strictly binary `uint8` mask (or empty string when truly empty), and (3) fix a subtle bug in `rle_encode_less_memory` where forcing endpoints to 0 on a view can corrupt the first/last pixel when the mask is all-ones/short runs. These are minimal changes aimed at moving from a broken/near-empty prediction behavior toward a reasonable Dice (toward your 0.937 target) without changing the model architecture or training.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with submission-format mismatch for this specific competition: it expects columns `img,pixels` (and the sample submission in your folder also differs), while your file uses `id,predicted`, which can yield an invalid submission and score 0. I keep your entire tiling/inference/thresholding/RLE logic intact, and only add a tiny adapter that writes the correct submission schema by renaming columns and using the sample submission’s column names when present. I also ensure row order matches the sample submission exactly (already true via iteration), and keep empty-mask predictions as an empty string which is valid for “no mask”. These are minimal changes aimed purely at moving from 0.0 toward your target by making the submission evaluable.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with a *submission schema mismatch* (this dataset’s `sample_submission.csv` uses `id,predicted`, while the competition page text you quoted shows `img,pixels`; if Kaggle expects `id,predicted`, sending different headers can score 0 even with good masks). I keep your entire inference/tiling/thresholding/RLE logic unchanged and only add a tiny, deterministic adapter that writes the submission with exactly the same column names as the provided sample submission (and in the same row order). I also harden the ID extraction to always use the sample’s first column (instead of sometimes forcing `"id"`) to avoid accidental misalignment if columns differ in other environments. These are minimal changes aimed purely at making your predictions actually evaluable (moving score upward toward your target).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “all-empty masks” (valid submission but Dice≈0), which can happen here because (a) `DATA` points to a folder that contains only JSONs (no `.tiff`), so every image read fails and you fall into the `except` that outputs empty RLE, and/or (b) your probability→mask threshold is too strict for the heuristic fallback. I keep your tiling/inference/RLE core logic intact, but minimally (1) fix test image discovery to read from the actual `.tiff` directory (auto-detect within the competition input), (2) remove the broad per-image exception that silently forces empty masks by logging the error once and still writing a row, and (3) slightly relax `TH` only when running the heuristic fallback (models missing), so you move upward toward the target rather than staying at 0.0. These changes do not alter the model architecture/training and still produce a valid `submission.csv` with the sample’s column names and row order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an **encoding mismatch**: this competition’s RLE expects pixels to be numbered in **column-major order** (top-to-bottom, then left-to-right), but your current `rle_encode_less_memory` transposes before flattening, which effectively switches to row-major and can yield near-total mismatch even if the mask is reasonable. I make the smallest possible change by fixing RLE encoding to flatten in the required order without changing any model/tiling/inference logic. I also keep the “empty mask -> empty string” behavior intact so the submission remains valid. This should move your score upward toward the target without altering architecture, training, or thresholds.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most likely coming from an RLE convention mismatch (flattening order) and/or silently producing all-empty masks due to TIFF read issues; your current script already addresses both, but it still risks emitting masks in the wrong coordinate convention because some helper functions transpose while the rest of the pipeline uses (H,W) indexing. I make the smallest change that aligns everything to the competition’s “top-to-bottom then left-to-right” (Fortran-order) convention end-to-end: remove the inconsistent transpose usage in the unused encode/decode helpers and harden `rle_encode_less_memory` to always operate on a strict (H,W) binary mask without accidental shape/order surprises. This preserves your model/tiling/inference/threshold logic exactly, but makes the produced RLE much more likely to match the evaluator, moving the score up toward your target. The submission still be written with exactly the sample submission’s column names and row order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with an RLE convention mismatch: the competition expects pixels counted left-to-right within each row, and rows top-to-bottom (row-major / C-order), but your encoder currently uses Fortran-order flattening, which can make otherwise reasonable masks score ~0. I make the smallest change by switching `rle_encode_less_memory` (and the encode/decode helpers for consistency) to row-major order while keeping the model/tiling/thresholding logic exactly the same. This should move the score sharply upward toward your 0.937 target without changing the core inference pipeline. I also keep the “empty mask -> empty string” behavior intact and preserve the sample submission column names/order.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

try:
    import segmentation_models_pytorch as smp  # noqa: F401

    _HAS_SMP = True
except ModuleNotFoundError:
    _HAS_SMP = False

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
sz = 256  # tile size after downsample
reduce = 4  # downsample factor
TH = 0.35  # threshold for positive predictions (kept for real models)
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
shift = True
minoverlap = 300

_CANDIDATE_TEST_DIRS = [
    "/kaggle/input/hubmap-kidney-segmentation/test",
    "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test",
]


def _find_test_dir_with_tiffs(candidates):
    for d in candidates:
        if os.path.isdir(d):
            try:
                for fn in os.listdir(d):
                    if fn.lower().endswith(".tiff") or fn.lower().endswith(".tif"):
                        return d if d.endswith("/") else d + "/"
            except Exception:
                pass
    return "/kaggle/input/hubmap-kidney-segmentation/test/"


DATA = _find_test_dir_with_tiffs(_CANDIDATE_TEST_DIRS)

MODELS = [
    f"/kaggle/input/all-data-medium-aug/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv(
    "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
)

ID_COL = df_sample.columns[0]
PRED_COL = df_sample.columns[1]

print("Using DATA:", DATA)
print("Sample columns:", ID_COL, PRED_COL)
print(
    "Found test tiffs:",
    (
        sum(1 for fn in os.listdir(DATA) if fn.lower().endswith((".tif", ".tiff")))
        if os.path.isdir(DATA)
        else 0
    ),
)




## === cell 2
def enc2mask(encs, shape):
    """Decode RLE into (H,W) mask using row-major order."""
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None:
            continue
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = str(enc).split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start : start + length] = 1 + m
    return img.reshape(shape, order="C")


def mask2enc(mask, n=1):
    """Encode (H,W) mask into RLE using row-major order."""
    pixels = mask.flatten(order="C")
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.uint8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(" ".join(str(x) for x in runs))
    return encs


def rle_encode_less_memory(img):
    """
    Competition RLE convention (per problem statement): pixels numbered top-to-bottom,
    then left-to-right. That corresponds to row-major flattening on an (H,W) array:
    flatten(order="C") after viewing as (row=y, col=x).

    Minimal hardening to avoid accidental non-binary / non-2D inputs:
    - coerce to 2D
    - coerce to strict binary uint8
    - return '' for empty masks (valid "no mask")
    """
    if img.ndim == 3:
        img = img[..., 0]
    img = (img > 0).astype(np.uint8, copy=False)

    pixels = img.flatten(order="C")
    if pixels.sum() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40
p_th = 1000 * (sz // 256) ** 2


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array of size (N,4), where N - number of tiles,
    2nd axis represents slices: x1,x2,y1,y2
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


def _open_tiff_array(path):
    """
    Kaggle image TIFFs are often JPEG-compressed and tifffile.imread can fail
    without 'imagecodecs'. Fall back to OpenCV TIFF decode.
    Normalize output to HxWx3 uint8.
    """
    arr = None

    try:
        arr = tiff.memmap(path)
    except Exception:
        try:
            arr = tiff.imread(path)
        except Exception:
            arr = None

    if arr is None:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or gray
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        arr = im  # HxWx3, uint8/uint16 depending on file

    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] < 3:
        pad = 3 - arr.shape[-1]
        arr = np.concatenate([arr] + [arr[..., :1]] * pad, axis=-1)
    arr = arr[..., :3]

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a -= a.min()
        denom = a.max() if a.max() > 0 else 1.0
        a = (a / denom * 255.0).clip(0, 255)
        arr = a.astype(np.uint8)

    return arr


def _read_window(arr, x1, x2, y1, y2, out_sz):
    """
    Read arr[x1:x2, y1:y2] into a fixed (out_sz, out_sz, 3) array (pad if at borders).
    """
    img = np.zeros((out_sz, out_sz, 3), dtype=np.uint8)
    xs = slice(int(x1), int(x2))
    ys = slice(int(y1), int(y2))
    crop = arr[xs, ys, :3]
    h, w = crop.shape[0], crop.shape[1]
    img[:h, :w] = crop.astype(np.uint8, copy=False)
    return img




## === cell 4
def _tile_prob_from_image(img_bgr_uint8):
    """
    Deterministic heuristic segmentation producing a probability map.
    Uses HSV + Otsu on Value channel, with morphology to smooth.
    Output shape: (h, w) float32 in [0,1].
    """
    hsv = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, th1 = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m1 = th1.astype(np.uint8)
    m2 = th2.astype(np.uint8)

    if m1.mean() < m2.mean():
        m = m1
    else:
        m = m2

    k = max(3, (img_bgr_uint8.shape[0] // 64) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    p = (m / 255.0).astype(np.float32)
    p = cv2.GaussianBlur(p, (0, 0), sigmaX=1.0)
    p = np.clip(p, 0.0, 1.0)
    return p


models = []
_can_load_models = _HAS_SMP and all(os.path.exists(p) for p in MODELS)
if _can_load_models:
    import segmentation_models_pytorch as smp  # noqa: F811

    for path in MODELS:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model.load_state_dict(state_dict)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)
    del state_dict
    gc.collect()
else:
    models = None  # indicates fallback path

TH_EFFECTIVE = TH if models is not None else 0.20
print("Models loaded:", models is not None, "| TH_EFFECTIVE:", TH_EFFECTIVE)



## === cell 5
if shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.arr = _open_tiff_array(os.path.join(DATA, idx + ".tiff"))
            self.shape = self.arr.shape[:2]  # (H, W)
            self.reduce = reduce
            self.sz = reduce * sz  # window size on original resolution
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            img = _read_window(self.arr, x1, x2, y1, y2, out_sz=self.sz)

            if self.reduce != 1:
                img = cv2.resize(
                    img,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            _, s, _ = cv2.split(hsv)
            vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

            if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                return img2tensor((img / 255.0 - mean) / std), vertices, -1
            else:
                return img2tensor((img / 255.0 - mean) / std), vertices, idx

    class Model_pred:
        def __init__(self, models, dl, tta: bool = False, half: bool = False):
            self.models = models  # can be None for fallback
            self.dl = dl
            self.tta = tta
            self.half = half

        def __iter__(self):
            with torch.no_grad():
                for x, z, y in iter(self.dl):
                    if (y >= 0).sum() == 0:
                        continue

                    x = x[y >= 0]
                    z = z[y >= 0]
                    y = y[y >= 0]

                    if self.models is None:
                        xb = x.permute(0, 2, 3, 1).numpy()
                        xb = (xb * std + mean) * 255.0
                        xb = np.clip(xb, 0, 255).astype(np.uint8)
                        py_list = []
                        for i in range(xb.shape[0]):
                            p = _tile_prob_from_image(xb[i])
                            py_list.append(p)
                        py = np.stack(py_list, axis=0)  # (B,h,w)
                        z = z.numpy()
                        for i in range(len(py)):
                            yield py[i], z[i], y[i]
                        continue

                    x = x.to(device)
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
                        py,
                        scale_factor=reduce,
                        mode="bilinear",
                        align_corners=False,
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()
                    py = py.squeeze(-1).numpy()
                    z = z.numpy()

                    for i in range(len(py)):
                        yield py[i], z[i], y[i]

        def __len__(self):
            return len(self.dl.dataset)

    names, preds = [], []
    _printed_first_error = False

    for _, row in df_sample.iterrows():
        idx = str(row[ID_COL])
        rle = ""
        try:
            ds = HuBMAPDataset(idx)
            dl = DataLoader(
                ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
            )
            mp = Model_pred(models, dl)

            mask = np.zeros(ds.shape, dtype=bool)

            for pred, vert, _ in iter(mp):
                x1, x2, y1, y2 = vert
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                tile = pred > TH_EFFECTIVE

                h = x2 - x1
                w = y2 - y1
                if tile.shape[0] != h or tile.shape[1] != w:
                    tile = tile[:h, :w]

                mask[x1:x2, y1:y2] |= tile

            rle = rle_encode_less_memory(mask)
            del mask, ds, dl
            gc.collect()
        except Exception as e:
            if not _printed_first_error:
                print("First image processing error:", idx, "|", repr(e))
                _printed_first_error = True
            rle = ""
            gc.collect()

        names.append(idx)
        preds.append(rle)



## === cell 6
sub_for_save = pd.DataFrame({ID_COL: names, PRED_COL: preds})
sub_for_save = sub_for_save.reindex(columns=list(df_sample.columns))

sub_for_save.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert list(sub_for_save.columns) == list(df_sample.columns)
assert len(sub_for_save) == len(df_sample)

print(sub_for_save.head())
print(
    "Non-empty predictions:",
    int((sub_for_save[PRED_COL].astype(str).str.len() > 0).sum()),
    "/",
    len(sub_for_save),
)
print("Saved:", os.path.abspath("submission.csv"), "rows:", len(sub_for_save))
