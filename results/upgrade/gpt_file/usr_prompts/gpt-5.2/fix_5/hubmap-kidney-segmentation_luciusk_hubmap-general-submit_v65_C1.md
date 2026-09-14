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

0.9442329012289872

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it runs end-to-end in this Kaggle environment and always produces a valid `submission.csv` with the correct number of rows and columns. The main blockers are (1) `segmentation_models_pytorch` is not installed, and (2) the referenced model `.pth` files are not present; I add a safe fallback that uses the provided anatomical-structure JSONs to create a reasonable binary mask and RLE-encode it. This keeps the core tiling/model logic intact when weights are available, but guarantees a valid submission when they are not. Finally, I also fix the submission column name mismatch (`img,pixels` vs `id,predicted`) by following the provided `sample_submission.csv` schema (`id,predicted`) so Kaggle accepts it.'
- What this solution (achieved 0.0) has done: 'You’re hitting a hard runtime blocker: `tifffile` can’t read the competition TIFFs because they’re JPEG-compressed and the `imagecodecs` dependency isn’t installed. I fix this by avoiding TIFF decoding entirely in the fallback path: we already have per-image width/height in `HuBMAP-20-dataset_information.csv`, so we can build masks using those dimensions without reading the `.tiff`. I also make the submission schema robust by aligning to `sample_submission.csv` (`id,predicted`) and ensuring every id gets a string (empty if mask is empty). This should run end-to-end and, compared to an all-empty submission (0.0), should increase the Dice score toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is very likely coming from an invalid/near-empty mask on the fallback path (since the real model path can’t run without `segmentation_models_pytorch`, weights, and TIFF decode support). I keep your fallback approach (anatomical-structure JSON → binary mask → RLE) but make two minimal fixes that directly increase Dice: (1) robustly include only anatomically plausible kidney structures (exclude cortex/medulla/capsule/background etc.) while prioritizing “glomerulus-like” regions if they exist, and (2) add a small morphology/open-close + hole filling step to reduce fragmented polygons and improve overlap. I also fix the submission schema mismatch by always writing exactly the sample’s columns (`id,predicted`) and ensure RLE encoding is correct for binary masks. These changes preserve your overall logic (fallback segmentation without TIFF/model) while producing more realistic masks than empty/all-wrong ones, moving the score upward toward the target.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import json
import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import warnings

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # type: ignore

    SMP_AVAILABLE = True
except Exception as e:
    SMP_AVAILABLE = False
    smp = None
    print(
        f"[WARN] segmentation_models_pytorch not available ({e}). Will use fallback segmentation."
    )



## === cell 1
sz = 256  # tile size at model input resolution
reduce = 4  # downscale factor for reading tiles (tiles are read at sz*reduce then resized to sz)
TH = 0.46  # threshold for positive predictions
DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [
    f"../input/skfold-without-test/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"  # efficientnet-b4, se_resnext50_32x4d
shift = True
minoverlap = 300




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
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
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    img = (img > 0).astype(np.uint8, copy=False)

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
p_th = 1000 * (sz // 256) ** 2  # threshold for minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return Array of size (N,4), where N - number of tiles,
    2nd axis represent slices: x1,x2,y1,y2
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


def _load_tiff_hwc(path):
    """
    Load a tiff image and return HWC uint8 array with 3 channels.
    NOTE: In this environment, JPEG-compressed TIFFs may fail without imagecodecs.
    This function is kept for the original model inference path (when available).
    """
    arr = tiff.imread(path)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            arr = np.moveaxis(arr[:3], 0, -1)
        else:
            if arr.shape[-1] >= 3:
                arr = arr[..., :3]
            else:
                arr = np.repeat(arr[..., :1], 3, axis=-1)
    elif arr.ndim == 4:
        arr = arr[0]
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            arr = np.moveaxis(arr[:3], 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[-1] >= 3:
                arr = arr[..., :3]
            else:
                arr = np.repeat(arr[..., :1], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    if arr.dtype != np.uint8:
        if arr.max() <= 255:
            arr = arr.astype(np.uint8)
        else:
            mx = float(arr.max()) if float(arr.max()) > 0 else 1.0
            arr = (arr / mx * 255.0).astype(np.uint8)
    return arr


def _polygon_to_mask(poly_xy, h, w):
    pts = np.asarray(poly_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return np.zeros((h, w), dtype=np.uint8)
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)  # x
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)  # y
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _postprocess_binary_mask(mask01: np.ndarray) -> np.ndarray:
    """
    Change rationale (score-improving, minimal): morphological cleanup and hole fill tends to
    increase Dice versus jagged/fragmented polygon unions from anatomical JSON.
    """
    mask01 = (mask01 > 0).astype(np.uint8, copy=False)
    if mask01.sum() == 0:
        return mask01

    k_open = np.ones((3, 3), np.uint8)
    k_close = np.ones((7, 7), np.uint8)
    mask01 = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, k_open, iterations=1)
    mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, k_close, iterations=1)

    h, w = mask01.shape
    inv = (1 - mask01).astype(np.uint8)
    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=(0, 0), newVal=0)
    holes = (ff > 0).astype(np.uint8)
    mask01 = (mask01 | holes).astype(np.uint8)

    return mask01


def anatomical_json_to_mask(idx, shape_hw, data_dir=DATA):
    """
    Fallback: create a binary mask from anatomical-structure polygons.
    Minimal change for higher Dice: prioritize "glomerulus-like" / "capillary" if present;
    otherwise include plausible small structures and exclude broad regions (cortex/medulla/etc.)
    that would create massive false positives.
    """
    h, w = shape_hw
    path = os.path.join(data_dir, f"{idx}-anatomical-structure.json")
    if not os.path.exists(path):
        return np.zeros((h, w), dtype=np.uint8)

    try:
        with open(path, "r") as f:
            feats = json.load(f)
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)

    feats = feats if isinstance(feats, list) else []
    if len(feats) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    exclude = {
        "background",
        "bg",
        "cortex",
        "medulla",
        "capsule",
        "fat",
        "adipose",
        "artifact",
        "unknown",
    }

    priority_tokens = ("glomer", "capillar", "tuft")
    secondary_tokens = ("vessel", "arter", "vein")  # weaker proxy

    priority_feats = []
    secondary_feats = []
    other_feats = []

    for feat in feats:
        name = feat.get("properties", {}).get("classification", {}).get("name", "")
        name_l = str(name).strip().lower()
        if name_l in exclude:
            continue
        if any(tok in name_l for tok in priority_tokens):
            priority_feats.append(feat)
        elif any(tok in name_l for tok in secondary_tokens):
            secondary_feats.append(feat)
        else:
            other_feats.append(feat)

    use_feats = (
        priority_feats
        if len(priority_feats)
        else (secondary_feats if len(secondary_feats) else other_feats)
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    for feat in use_feats:
        try:
            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates", None)
            if not coords or not isinstance(coords, list) or len(coords) == 0:
                continue
            ring = coords[0]
            poly_xy = [
                (p[0], p[1])
                for p in ring
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if len(poly_xy) < 3:
                continue
            mask |= _polygon_to_mask(poly_xy, h=h, w=w)
        except Exception:
            continue

    mask = _postprocess_binary_mask(mask)
    return mask.astype(np.uint8)




## === cell 4
names, preds = [], []

CAN_USE_MODELS = SMP_AVAILABLE and all(os.path.exists(p) for p in MODELS)

if not CAN_USE_MODELS:
    missing = [p for p in MODELS if not os.path.exists(p)]
    if not SMP_AVAILABLE:
        print(
            "[WARN] Model inference disabled because segmentation_models_pytorch is missing."
        )
    if missing:
        print(
            f"[WARN] Model inference disabled because {len(missing)} model file(s) are missing. Example: {missing[0]}"
        )

    info_path = "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
    df_info = pd.read_csv(info_path)

    df_info["id"] = (
        df_info["image_file"].astype(str).str.replace(".tiff", "", regex=False)
    )
    id2hw = {
        r["id"]: (int(r["height_pixels"]), int(r["width_pixels"]))
        for _, r in df_info.iterrows()
    }

    print(
        "[INFO] Using fallback anatomical-structure JSON masks (no TIFF decoding) to generate submission."
    )

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = row["id"]
        h, w = id2hw.get(idx, (0, 0))
        if h <= 0 or w <= 0:
            names.append(idx)
            preds.append("")
            continue

        mask = anatomical_json_to_mask(idx, (h, w), data_dir=DATA)
        rle = rle_encode_less_memory(mask) if mask.sum() > 0 else ""
        names.append(idx)
        preds.append(rle)
        del mask
        gc.collect()

else:
    if not shift:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.img = _load_tiff_hwc(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.img.shape[:2]  # (H,W)
                self.reduce = reduce
                self.sz = reduce * sz
                self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
                self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
                self.n0max = (self.shape[0] + self.pad0) // self.sz
                self.n1max = (self.shape[1] + self.pad1) // self.sz

            def __len__(self):
                return self.n0max * self.n1max

            def __getitem__(self, idx):
                n0, n1 = idx // self.n1max, idx % self.n1max
                x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
                p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
                p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

                img = np.zeros((self.sz, self.sz, 3), np.uint8)
                img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self.img[
                    p00:p01, p10:p11
                ]

                if self.reduce != 1:
                    img = cv2.resize(
                        img,
                        (self.sz // reduce, self.sz // reduce),
                        interpolation=cv2.INTER_AREA,
                    )

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                _, s, _ = cv2.split(hsv)
                if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                    return img2tensor((img / 255.0 - mean) / std), -1
                else:
                    return img2tensor((img / 255.0 - mean) / std), idx

        class Model_pred:
            def __init__(self, models, dl, tta: bool = False, half: bool = False):
                self.models = models
                self.dl = dl
                self.tta = tta
                self.half = half

            def __iter__(self):
                with torch.no_grad():
                    for x, y in iter(self.dl):
                        if (y >= 0).sum() > 0:
                            x = x[y >= 0].to(device)
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
                                py,
                                scale_factor=reduce,
                                mode="bilinear",
                                align_corners=False,
                            )
                            py = py.permute(0, 2, 3, 1).float().cpu()

                            for i in range(len(py)):
                                yield py[i], y[i]

            def __len__(self):
                return len(self.dl.dataset)

        models = []
        for path in MODELS:
            state_dict = torch.load(path, map_location=torch.device("cpu"))
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict)
            model.float()
            model.eval()
            model.to(device)
            models.append(model)
        del state_dict

        for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            idx = row["id"]
            ds = HuBMAPDataset(idx)
            dl = DataLoader(
                ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
            )
            mp = Model_pred(models, dl)

            mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
            for p, i in iter(mp):
                mask[i.item()] = p.squeeze(-1) > TH

            mask = (
                mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
            )
            mask = mask[
                ds.pad0
                // 2 : -(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
                ds.pad1
                // 2 : -(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz,
            ]

            rle = rle_encode_less_memory(mask.numpy())
            names.append(idx)
            preds.append(rle)
            del mask, ds, dl
            gc.collect()

    if shift:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.img = _load_tiff_hwc(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.img.shape[:2]  # (H,W)
                self.reduce = reduce
                self.sz = reduce * sz
                self.mask_grid = make_grid(
                    self.shape, window=self.sz, min_overlap=minoverlap
                )

            def __len__(self):
                return len(self.mask_grid)

            def __getitem__(self, idx):
                x1, x2, y1, y2 = self.mask_grid[idx]
                img = self.img[x1:x2, y1:y2]

                if self.reduce != 1:
                    img = cv2.resize(
                        img,
                        (self.sz // reduce, self.sz // reduce),
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

        models = []
        for path in MODELS:
            state_dict = torch.load(path, map_location=torch.device("cpu"))
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict)
            model.float()
            model.eval()
            model.to(device)
            models.append(model)
        del state_dict

        for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            idx = row["id"]
            ds = HuBMAPDataset(idx)
            dl = DataLoader(
                ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
            )
            mp = Model_pred(models, dl)

            mask = np.zeros(ds.shape, dtype=np.uint8)
            for pred, vert, i in iter(mp):
                x1, x2, y1, y2 = vert
                mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

            mask = (mask > 0.5).astype(np.uint8)

            rle = rle_encode_less_memory(mask)
            names.append(idx)
            preds.append(rle)

            del mask, ds, dl
            gc.collect()



## === cell 5
sub = df_sample.copy()
pred_map = {k: v for k, v in zip(names, preds)}
sub["predicted"] = sub["id"].map(pred_map).fillna("").astype(str)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print(
    f"Saved submission.csv with shape={sub.shape} at {os.path.abspath('submission.csv')}"
)
