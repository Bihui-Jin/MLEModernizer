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

0.9329997292765037

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the unavailable `segmentation_models_pytorch` dependency and replacing it with a tiny compatibility `smp.Unet` wrapper that can still load the provided checkpoints when they contain a plain state_dict. I also fix the missing model files issue by automatically falling back to a safe baseline (empty masks) when the external checkpoint paths don’t exist in this environment, so a valid `submission.csv` is always produced end-to-end. Finally, I correct the `DATA` path to point at the actual Kaggle input directory and harden the inference loop so `names/preds` are always defined, avoiding the downstream `NameError`. These changes are focused on correctness and producing a valid submission; score improvement beyond “valid submission” isn’t possible here without the missing model weights.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from always falling back to empty masks because the configured checkpoint paths don’t exist, so the model branch never runs. The smallest score-improving change is to point `MODELS` at checkpoints that actually exist in this dataset (there are none in your current inputs), and if none exist, switch to a legitimate, non-leaky heuristic using only the provided anatomical-structure JSON to generate a coarse kidney-region mask instead of predicting nothing. This keeps the overall pipeline (tiling/inference/RLE writing) intact while ensuring non-empty predictions that should move Dice upward from 0.0. I also fix the `DATA` root to the dataset directory (not `.../test/`) so both `.tiff` and JSON paths resolve consistently.'
- What this solution (achieved 0.08157) has done: 'I fix the immediate runtime error caused by JPEG-compressed TIFFs requiring the missing `imagecodecs` package by avoiding TIFF decoding entirely in the fallback branch (which doesn’t actually need pixel data). Instead, I derive each image’s `(height, width)` from the provided `HuBMAP-20-dataset_information.csv`, so we can rasterize the anatomical-structure polygons safely and still output valid RLE. I also make the RLE encoder robust to empty masks (returning an empty string rather than an invalid “2 -2” run), which prevents accidental malformed submissions. These changes keep the core modeling/inference logic intact and should move the score up from 0.0 by producing non-empty, correctly-formatted predictions based on the available anatomical-structure annotations.'
- What this solution (achieved 0.08384) has done: 'Your current 0.08157 score comes from the fallback branch producing a very coarse “any anatomical structure” mask, which doesn’t match the glomeruli target well. To move toward the 0.933 target without changing the core inference/training logic, I keep the model path exactly as-is and improve only the fallback mask generation by (1) filtering anatomical polygons by likely kidney structure names (case-insensitive keywords), (2) using a conservative “interior” mask via erosion to reduce over-coverage, and (3) removing small connected components so we don’t predict lots of tiny false positives. This should increase Dice materially versus the current fallback while still being a legitimate non-leaky heuristic using only provided test metadata. The submission writing remains identical and still guarantees a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current low Dice is dominated by the fallback (no checkpoints) producing an overly broad “any structure” mask; to move toward the 0.933 target with minimal change, we make that fallback more glomerulus-like rather than changing any model/inference core logic. Concretely, we (1) tighten which anatomical polygons are used (prioritize cortex/medulla/tubules and explicitly de-prioritize vessels/arteries/veins that create huge false positives), (2) convert the filled regions into a *ring/edge band* (morphological gradient) which better matches small rounded targets than full-area fills, and (3) add a light open + component filtering tuned for small-ish blobs to reduce noise without collapsing everything to empty. The model branch is left untouched; only the no-checkpoint fallback mask generator changes, and submission formatting remains the same.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score indicates the submission is effectively predicting nothing useful (empty/near-empty masks), which most commonly happens when the fallback mask becomes too sparse after the “ring/edge band + component filtering” steps. To move the Dice upward toward the 0.933 target with minimal changes and without touching the model/inference core logic, I only adjust the fallback post-processing to be less aggressive: widen the band (stronger dilate/erode gap), add a small closing to connect broken rings, and relax the connected-component area limits so real blobs aren’t discarded. I also keep the RLE encoder and submission formatting unchanged to ensure a valid `submission.csv` is produced end-to-end. These tweaks should increase recall versus the current fallback, which is the key to moving off a 0.0 Dice.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import warnings
import json

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


class _SMPStub:
    class Unet(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, 16, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, 1, 1),
            )

        def forward(self, x):
            return self.net(x)


smp = _SMPStub()



## === cell 1
sz = 256  # tile size at model input resolution
reduce = 4  # original image reduction factor (model sees sz, we tile at reduce*sz on original)
TH = 0.3  # threshold for positive predictions

DATA_ROOT = "/kaggle/input/hubmap-kidney-segmentation"
TEST_DIR = os.path.join(DATA_ROOT, "test")

MODELS = [
    f"/kaggle/input/all-data-medium-aug/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "efficientnet-b4"
shift = True
minoverlap = 300




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if (isinstance(enc, (float, np.floating)) and np.isnan(enc)) or (enc is None):
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
    if pixels.size == 0 or pixels.max() == 0:
        return ""
    pixels = pixels.astype(np.uint8, copy=False)

    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    if len(runs) == 0:
        return ""
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


def read_tiff_rgb(path):
    img = tiff.imread(path)
    if img.ndim == 3:
        if img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
            img = np.moveaxis(img[:3], 0, -1)
        elif img.shape[2] in (3, 4):
            img = img[..., :3]
        else:
            if img.shape[-1] >= 3:
                img = img[..., :3]
            elif img.shape[0] >= 3:
                img = np.moveaxis(img[:3], 0, -1)
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        raise ValueError(f"Unexpected TIFF shape {img.shape} for {path}")
    return img


def _keep_large_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8
    out = np.zeros_like(mask_u8)
    for k in range(1, num):
        area = stats[k, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[lab == k] = 1
    return out


def _keep_components_in_range(
    mask_u8: np.ndarray, min_area: int, max_area: int
) -> np.ndarray:
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8
    out = np.zeros_like(mask_u8)
    for k in range(1, num):
        area = stats[k, cv2.CC_STAT_AREA]
        if area >= min_area and area <= max_area:
            out[lab == k] = 1
    return out


def anatomical_json_to_mask(idx: str, shape_hw):
    h, w = shape_hw
    path = os.path.join(TEST_DIR, f"{idx}-anatomical-structure.json")
    if not os.path.exists(path):
        return np.zeros((h, w), dtype=np.uint8)

    with open(path, "r") as f:
        data = json.load(f)

    allow_kw = (
        "glomer",  # if present
        "cortex",
        "medulla",
        "tubul",
        "interstit",
    )
    deny_kw = (
        "arter",
        "vein",
        "vessel",
        "capillar",
        "pelvis",
        "nerve",
        "fat",
        "adip",
    )

    base = np.zeros((h, w), dtype=np.uint8)

    for feat in data:
        props = feat.get("properties", {})
        cls = props.get("classification", {}) if isinstance(props, dict) else {}
        name = str(cls.get("name", "")).lower()

        if name:
            if any(k in name for k in deny_kw):
                continue
            if not any(k in name for k in allow_kw):
                continue

        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = coords[0]
        if not ring or len(ring) < 3:
            continue

        pts = []
        for xy in ring:
            if len(xy) < 2:
                continue
            x, y = xy[0], xy[1]
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            pts.append([xi, yi])

        if len(pts) >= 3:
            pts = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(base, [pts], 1)

    if base.sum() == 0:
        return base

    k = max(5, (min(h, w) // 750) * 2 + 5)  # slightly larger than before -> wider band
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(base, kernel, iterations=2)
    ero = cv2.erode(base, kernel, iterations=1)
    band = cv2.subtract(dil, ero)

    k_close = max(3, (min(h, w) // 1200) * 2 + 3)
    kernel_close = np.ones((k_close, k_close), np.uint8)
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    k_open = max(3, (min(h, w) // 1600) * 2 + 3)
    kernel_open = np.ones((k_open, k_open), np.uint8)
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel_open, iterations=1)

    min_area = max(120, (h * w) // 400000)
    max_area = max(
        20000, (h * w) // 800
    )  # allow larger blobs; still prevents full-image masks
    band = _keep_components_in_range(
        band.astype(np.uint8), min_area=min_area, max_area=max_area
    )

    return band.astype(np.uint8)




## === cell 4
names, preds = [], []

existing_model_paths = [p for p in MODELS if os.path.exists(p)]
can_run_models = len(existing_model_paths) > 0

info_path = os.path.join(DATA_ROOT, "HuBMAP-20-dataset_information.csv")
df_info = pd.read_csv(info_path)
df_info["id"] = df_info["image_file"].astype(str).str.replace(".tiff", "", regex=False)
size_map = df_info.set_index("id")[["height_pixels", "width_pixels"]].to_dict("index")

if not can_run_models:
    for idx in tqdm(df_sample["id"].tolist(), total=len(df_sample)):
        if idx not in size_map:
            h, w = 512, 512
        else:
            h = int(size_map[idx]["height_pixels"])
            w = int(size_map[idx]["width_pixels"])

        mask = anatomical_json_to_mask(idx, (h, w))
        rle = rle_encode_less_memory(mask)
        names.append(idx)
        preds.append(rle)
        del mask
        gc.collect()
else:
    if not shift:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.img = read_tiff_rgb(os.path.join(TEST_DIR, idx + ".tiff"))
                self.shape = self.img.shape[:2]  # (H, W)

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
                patch = self.img[p00:p01, p10:p11]
                if patch.dtype != np.uint8:
                    patch = np.clip(patch, 0, 255).astype(np.uint8)
                img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = patch

                if self.reduce != 1:
                    img = cv2.resize(
                        img,
                        (self.sz // reduce, self.sz // reduce),
                        interpolation=cv2.INTER_AREA,
                    )

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
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
        for path in existing_model_paths:
            state = torch.load(path, map_location=torch.device("cpu"))
            if isinstance(state, dict) and "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state

            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict, strict=False)
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

            rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
            names.append(idx)
            preds.append(rle)

            del mask, ds, dl
            gc.collect()

    else:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.img = read_tiff_rgb(os.path.join(TEST_DIR, idx + ".tiff"))
                self.shape = self.img.shape[:2]  # (H, W)
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
                if img.shape[0] != self.sz or img.shape[1] != self.sz:
                    pad_h = self.sz - img.shape[0]
                    pad_w = self.sz - img.shape[1]
                    img = np.pad(
                        img,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                if self.reduce != 1:
                    img = cv2.resize(
                        img,
                        (self.sz // reduce, self.sz // reduce),
                        interpolation=cv2.INTER_AREA,
                    )

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
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
        for path in existing_model_paths:
            state = torch.load(path, map_location=torch.device("cpu"))
            if isinstance(state, dict) and "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state

            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict, strict=False)
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
df = pd.DataFrame({"id": names, "predicted": preds})

df = df_sample[["id"]].merge(df, on="id", how="left")
df["predicted"] = df["predicted"].fillna("")

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Used checkpoints:", len(existing_model_paths), "of", len(MODELS))
print(
    "Inference mode:",
    "models" if len(existing_model_paths) > 0 else "anatomical-structure fallback",
)
