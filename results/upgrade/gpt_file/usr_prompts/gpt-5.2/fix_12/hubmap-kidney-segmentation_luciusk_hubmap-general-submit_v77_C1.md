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

0.03657

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the crash by replacing `tifffile.memmap()` with a safe windowed TIFF reader that uses `tifffile.TiffFile(...).pages[0].asarray()` (with a small fallback), since the competition TIFFs are not always memory-mappable. I also fix the submission row mismatch by using the correct test id column name from `sample_submission.csv` (`id`, not `img`) and by ensuring we iterate over all rows and always append exactly one RLE per id. These changes are execution/stability fixes (score-neutral by intent), and they produce a valid `submission.csv` with the required `id,predicted` columns.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash by adding a TIFF reader fallback that can decode JPEG-compressed TIFFs without `imagecodecs` by using OpenCV (`cv2.imread`) and PIL when tifffile fails. I keep the existing tiling/inference core logic intact and only change the image-loading portion plus a small safety fallback to still produce a valid submission even if an image can’t be decoded. Since your current score is 0.0 due to the crash (no valid submission), these fixes should move the score upward by allowing the pipeline to run end-to-end and generate `submission.csv` in the required format. I also keep the `id,predicted` column handling and row alignment as-is to avoid submission mismatches.'
- What this solution (achieved 0.02731) has done: 'Your current 0.0 score is coming from producing essentially-all-empty masks: the checkpoints are not available/loadable in this environment, so the code falls back to `DummySegModel` (all-negative logits), which yields empty RLEs and a Dice near 0. To move the score up toward the target with minimal changes and without changing the tiling/inference core, I switch the “no-checkpoint” fallback to a simple, deterministic tissue-based segmentation computed from the already-loaded tiles (using saturation/value thresholds) instead of an all-empty mask. This keeps the same data flow (iterate ids → tile over image → build binary mask → RLE) and should yield a non-zero Dice by at least masking tissue regions where glomeruli exist. I also fix one small correctness bug: `cv2.cvtColor` should use `RGB2HSV` because the tiles are in RGB, which improves both the blank-tile filter and the fallback tissue segmentation.'
- What this solution (achieved 0.00845) has done: 'Your current score is far below target, and the biggest likely contributor (without changing the model/tiling core) is that the “no-checkpoint” fallback is segmenting generic tissue rather than glomeruli, producing many false positives and hurting Dice. I keep your pipeline intact but make the fallback more glomerulus-shaped by (1) restricting detections to cortex-like areas using the provided anatomical-structure JSON (when available) and (2) tightening the fallback morphology to favor compact blob-like regions instead of broad tissue. I also set the fallback threshold `TH` more conservatively only for the fallback path (leaving the model path unchanged) to reduce overmasking. These are minimal changes aimed at increasing Dice toward the target while preserving the overall inference flow and output format.'
- What this solution (achieved 0.00845) has done: 'Your score is far below target, so we should improve Dice by reducing systematic false positives/negatives in the fallback path while keeping your overall tiling→mask→RLE pipeline unchanged. The biggest minimal win is to make the anatomy restriction actually work: right now the polygon coordinates are treated as (x,y) but are applied to an (h,w) mask as if (row,col), which misplaces the cortex mask and harms predictions. I fix the coordinate order when rasterizing polygons, and I also expand the allowed anatomical keywords slightly (to catch “renal cortex”/“cortex of kidney” variants) while keeping the same fallback logic. Finally, I add one small safety: ensure the generated mask is C-contiguous uint8 before RLE encoding to avoid any unexpected stride issues (score-neutral but stabilizes output).'
- What this solution (achieved 0.03942) has done: 'Your current score is far below the target, so we should make small, low-risk corrections that reduce obvious false positives in the fallback path (since checkpoints aren’t loadable) while keeping your tiling→mask→RLE pipeline intact. I (1) fix a key orientation mismatch: the anatomy mask is built at full resolution but is applied to downscaled tiles without resizing, which can mis-restrict or fail to restrict predictions; (2) make the fallback component filtering scale-aware so it behaves consistently across different `reduce` values; and (3) apply a very light “keep only plausible blob sizes” post-filter on the final mask to cut spurious regions. These are minimal changes that preserve your core logic and should improve Dice by reducing systematic overmasking.'
- What this solution (achieved 0.04727) has done: 'Your current score is far below the target, so the biggest minimal, low-risk gain is to reduce the systematic false positives produced by the heuristic fallback while keeping your tiling→stitching→RLE pipeline unchanged. I make the fallback stricter and more glomerulus-shaped by adding a compactness filter (circularity) and rejecting very elongated components, which tends to remove vessel/tissue streaks that hurt Dice. I also ensure the anatomy restriction is applied with correct tile-to-small-mask coordinate mapping (using consistent integer scaling), which prevents misaligned masking when sizes aren’t perfectly divisible by `reduce`. Finally, I keep the model path untouched and still write a valid `submission.csv` with `id,predicted` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.037) has done: 'Your score gap to the target is large (0.047 → 0.947), so we should improve Dice by making the fallback segmentation less noisy and better aligned with glomerulus appearance, while keeping your tiling→stitching→RLE core unchanged. The smallest, highest-leverage change is to restrict fallback candidates to purple-ish glomerular hues in HSV (in addition to your existing v/s darkness logic) and then slightly relax the circularity/aspect thresholds to recover true positives that are currently being filtered out. I also make the final connected-component filter a bit less aggressive on max area (to reduce false negatives), while keeping the model-path completely untouched. The script still runs end-to-end and writes a valid `submission.csv` with `id,predicted` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.03657) has done: 'Your current score (0.037) is far below the target (0.9468), and since checkpoints aren’t being loaded, almost all performance comes from the heuristic fallback; the smallest way to move Dice upward is to reduce false positives while recovering some true positives. I keep your tiling→stitching→RLE pipeline intact and only adjust fallback post-processing: (1) apply a gentle distance-transform-based “core emphasis” inside candidates to prefer compact glomerulus-like cores over broad tissue, and (2) add a light hole-fill step so true glomeruli aren’t fragmented into low-Dice speckles. I also fix a subtle bug in `rle_encode_less_memory` (it mutates the `pixels` view in-place) by working on a padded copy; this is score-stabilizing and avoids edge artifacts without changing evaluation semantics. Everything still runs end-to-end and writes `submission.csv` with `id,predicted` aligned to `sample_submission.csv`.'

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
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import warnings
import json

warnings.filterwarnings("ignore")


class DummySegModel(nn.Module):
    """Deterministic fallback: outputs very negative logits => sigmoid ~ 0 => empty mask."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3]),
            -20.0,
            device=x.device,
            dtype=x.dtype,
        )


torch.manual_seed(0)
np.random.seed(0)



## === cell 1
sz = 256  # the size of tiles (model input resolution)
reduce = (
    4  # downsample original image by 4 times before model (via cv2.resize to tile_size)
)
TH = 0.5  # threshold for positive predictions (MODEL PATH ONLY; fallback uses its own)
TRAIN = False

if TRAIN:
    DATA = "../input/hubmap-kidney-segmentation/train/"
    df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/train.csv")
else:
    DATA = "../input/hubmap-kidney-segmentation/test/"
    df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

MODELS = [
    f"../input/skfoldalldata/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]
bs = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = (
    "efficientnet-b4"  # must match checkpoint architecture (kept for compatibility)
)
EXPAND = 4
minoverlap = 1 / 16
TTA = False

tile_size = int(sz * EXPAND)  # 1024
tile_resized = int(
    tile_size * reduce
)  # 4096 (original crop size before downscale to tile_size)

ID_COL = "id" if "id" in df_sample.columns else df_sample.columns[0]




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None:
            continue
        if isinstance(enc, (float, np.floating)) and np.isnan(enc):
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
    img = np.ascontiguousarray(img.astype(np.uint8, copy=False))
    pixels = img.T.flatten()
    pixels = np.concatenate(
        [np.array([0], dtype=pixels.dtype), pixels, np.array([0], dtype=pixels.dtype)]
    )
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
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
    Return array of size (N,4), where N - number of tiles,
    axis=1 slices: x1, x2, y1, y2

    Note: 'shape' is (H, W).
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




## === cell 4
def _read_tiff_opencv_fallback(path: str) -> np.ndarray:
    """
    Bugfix: Some competition TIFFs use JPEG compression requiring imagecodecs for tifffile.
    OpenCV can often decode these TIFFs without imagecodecs. Returns RGB array.
    """
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("cv2.imread returned None")
    if bgr.ndim == 2:
        arr = bgr[..., None]
    else:
        if bgr.shape[2] == 4:
            bgr = bgr[..., :3]
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return arr


def _read_tiff_pil_fallback(path: str) -> np.ndarray:
    """
    Another fallback for TIFF decoding when tifffile/OpenCV fail.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im)


def _read_tiff_first_page(path: str) -> np.ndarray:
    """
    Robust TIFF reader:
    - First try tifffile (fast when supported)
    - If it fails due to missing imagecodecs/JPEG compression, fall back to OpenCV then PIL
    """
    try:
        with tiff.TiffFile(path) as tif:
            page = tif.pages[0]
            arr = page.asarray()
        return arr
    except Exception:
        pass

    try:
        return tiff.imread(path)
    except Exception:
        pass

    try:
        return _read_tiff_opencv_fallback(path)
    except Exception:
        pass

    return _read_tiff_pil_fallback(path)


def _load_anatomical_structure_mask(idx: str, shape_hw: tuple) -> np.ndarray:
    """
    Rasterize cortex-like polygons correctly (coords are [x,y]=[col,row]),
    producing a mask aligned with the full-resolution image (H,W).
    """
    h, w = shape_hw
    path = os.path.join(DATA, idx + "-anatomical-structure.json")
    if not os.path.exists(path):
        return np.ones((h, w), dtype=np.uint8)

    try:
        with open(path, "r") as f:
            feats = json.load(f)
    except Exception:
        return np.ones((h, w), dtype=np.uint8)

    keep_keywords = ("cortex", "renal cortex", "cortex of kidney", "glomer")
    mask = np.zeros((h, w), dtype=np.uint8)
    found = False

    for feat in feats if isinstance(feats, list) else []:
        try:
            name = (
                feat.get("properties", {})
                .get("classification", {})
                .get("name", "")
                .strip()
                .lower()
            )
            coords = feat.get("geometry", {}).get("coordinates", None)
            gtype = feat.get("geometry", {}).get("type", "")
            if coords is None or gtype.lower() != "polygon":
                continue

            if not any(k in name for k in keep_keywords):
                continue

            ring = coords[0]
            if not ring or len(ring) < 3:
                continue

            pts = np.asarray(ring, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue

            x = np.clip(pts[:, 0], 0, w - 1)
            y = np.clip(pts[:, 1], 0, h - 1)
            poly = np.stack([x, y], axis=1).astype(np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(mask, [poly], 1)
            found = True
        except Exception:
            continue

    if not found:
        return np.ones((h, w), dtype=np.uint8)
    return mask


class HuBMAPDataset(Dataset):
    """
    Windowed reads are done via slicing on a loaded NumPy array.
    This keeps the original tiling logic intact while avoiding tifffile.memmap() failures.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")

        arr = _read_tiff_first_page(self.path)

        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3:
            if arr.shape[0] <= 4 and arr.shape[1] > 256 and arr.shape[2] > 256:
                arr = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError(f"Unexpected TIFF ndim={arr.ndim} for {self.path}")

        if arr.shape[-1] < 3:
            pad = 3 - arr.shape[-1]
            arr = np.concatenate([arr] + [arr[..., :1]] * pad, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

        self.arr = arr
        self.shape = (arr.shape[0], arr.shape[1])  # H, W

        self.reduce = reduce
        self.sz = reduce * sz
        self.mask_grid = make_grid(
            self.shape, window=tile_resized, min_overlap=int(tile_resized * minoverlap)
        )

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = np.array(self.arr[x1:x2, y1:y2, :], copy=False)

        if img.dtype != np.uint8:
            if img.dtype == np.uint16:
                img = (img / 257).astype(np.uint8)  # 65535/257 ~= 255
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        if self.reduce != 1:
            img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img / 255.0 - mean) / std), vertices, idx




## === cell 5
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
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    py = py.squeeze(-1).numpy()
                    z = z.numpy()

                    for i in range(len(py)):
                        yield py[i], z[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 6
def _fallback_glom_like_mask_from_tile_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Score change (fallback-only, same core pipeline):
    - Keep your hue-based candidate generation (purple-ish + dark tissue).
    - Add a compact "core emphasis" using distance transform to suppress broad stains and
      favor blob-like regions, which tends to increase Dice by reducing false positives.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    purple = ((h >= 115) & (h <= 170) & (s >= 20) & (v <= 235)) | (
        (h <= 10) & (s >= 20) & (v <= 235)
    )

    cand_base = ((v < 210) & (s > 25)) | (v < 185)
    cand = (cand_base & (purple | (v < 175))).astype(np.uint8)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k3, iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k7, iterations=1)

    dt = cv2.distanceTransform(cand, distanceType=cv2.DIST_L2, maskSize=3)
    core = (dt >= 2.0).astype(np.uint8)  # gentle: keep >=2px from boundary
    if core.sum() > 0:
        core = cv2.dilate(core, k3, iterations=2)
        cand = (cand & core).astype(np.uint8)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(cand)

    tile_area = float(cand.shape[0] * cand.shape[1])
    min_area = int(0.0020 * tile_area)
    max_area = int(0.40 * tile_area)

    min_circ = 0.14
    max_aspect = 3.8

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h_ = int(stats[i, cv2.CC_STAT_HEIGHT])

        aspect = max(w, h_) / max(1.0, min(w, h_))
        if aspect > max_aspect:
            continue

        comp = (lab[y : y + h_, x : x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        per = float(cv2.arcLength(contours[0], True))
        if per <= 1e-6:
            continue
        circ = float(4.0 * np.pi * area / (per * per))
        if circ < min_circ:
            continue

        out[lab == i] = 1

    return out


def _filter_final_mask_components(mask: np.ndarray) -> np.ndarray:
    """
    Keep your component-size sanity filter; add a tiny hole-fill to improve Dice on
    ring-like predictions without expanding to new areas.
    """
    m = (mask > 0).astype(np.uint8)

    m = cv2.morphologyEx(
        m,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 2:
        return m

    h, w = m.shape
    img_area = float(h * w)
    min_area = int(0.00002 * img_area)
    max_area = int(0.02000 * img_area)  # less aggressive than 0.0125 to improve recall

    out = np.zeros_like(m)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            out[lab == i] = 1
    return out


valid_model_paths = [p for p in MODELS if os.path.exists(p)]

models = []
use_tissue_fallback = False
if len(valid_model_paths) == 0:
    print(
        "WARNING: No model checkpoints found. Using anatomy-restricted glomerulus-like fallback segmentation."
    )
    use_tissue_fallback = True
else:
    loaded_any = False
    for path in valid_model_paths:
        try:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, nn.Module):
                model = obj
            elif (
                isinstance(obj, dict)
                and "state_dict" in obj
                and isinstance(obj["state_dict"], dict)
            ):
                raise RuntimeError(
                    "Checkpoint provides state_dict but model definition (smp.Unet) is unavailable."
                )
            elif isinstance(obj, dict):
                raise RuntimeError(
                    "Unsupported checkpoint dict without model definition."
                )
            else:
                raise RuntimeError("Unsupported checkpoint format.")
            model = model.to(device).eval()
            models.append(model)
            loaded_any = True
        except Exception as e:
            print(f"WARNING: Failed to load {path}: {e}")

    if not loaded_any:
        print(
            "WARNING: Could not load any checkpoints. Using anatomy-restricted glomerulus-like fallback segmentation."
        )
        use_tissue_fallback = True

gc.collect()

names, preds = [], []

FALLBACK_POST_TH = 0.50  # binary already; kept for clarity/possible tuning

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = str(row[ID_COL])

    try:
        ds = HuBMAPDataset(idx)
    except Exception as e:
        print(f"WARNING: Failed to build dataset for {idx}: {e}. Using empty mask.")
        names.append(idx)
        preds.append("")
        continue

    if use_tissue_fallback:
        anatomy_mask_full = _load_anatomical_structure_mask(idx, ds.shape)

        if reduce != 1:
            small_w = int(round(ds.shape[1] / reduce))
            small_h = int(round(ds.shape[0] / reduce))
            anatomy_mask_small = cv2.resize(
                anatomy_mask_full,
                (small_w, small_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        else:
            anatomy_mask_small = anatomy_mask_full

        mask = np.zeros(ds.shape, dtype=np.uint8)

        for x1, x2, y1, y2 in ds.mask_grid:
            tile = np.array(ds.arr[x1:x2, y1:y2, :], copy=False)

            if tile.dtype != np.uint8:
                if tile.dtype == np.uint16:
                    tile = (tile / 257).astype(np.uint8)
                else:
                    tile = np.clip(tile, 0, 255).astype(np.uint8)

            if reduce != 1:
                tile_small = cv2.resize(
                    tile, (tile_size, tile_size), interpolation=cv2.INTER_AREA
                )
            else:
                tile_small = tile

            blob_small = _fallback_glom_like_mask_from_tile_rgb(tile_small)

            if reduce != 1:
                xs1 = int(round(x1 / reduce))
                xs2 = int(round(x2 / reduce))
                ys1 = int(round(y1 / reduce))
                ys2 = int(round(y2 / reduce))
                xs1 = max(0, min(xs1, anatomy_mask_small.shape[0]))
                xs2 = max(0, min(xs2, anatomy_mask_small.shape[0]))
                ys1 = max(0, min(ys1, anatomy_mask_small.shape[1]))
                ys2 = max(0, min(ys2, anatomy_mask_small.shape[1]))

                am = anatomy_mask_small[xs1:xs2, ys1:ys2]
                if am.shape == blob_small.shape:
                    blob_small = (blob_small & am).astype(np.uint8)
            else:
                blob_small = (blob_small & anatomy_mask_small[x1:x2, y1:y2]).astype(
                    np.uint8
                )

            if reduce != 1:
                blob = cv2.resize(
                    blob_small,
                    (y2 - y1, x2 - x1),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                blob = blob_small

            mask[x1:x2, y1:y2] = np.maximum(
                mask[x1:x2, y1:y2], (blob > 0).astype(np.uint8)
            )

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = (mask > FALLBACK_POST_TH).astype(np.uint8)

        mask = _filter_final_mask_components(mask)

        rle = rle_encode_less_memory(mask)
        names.append(idx)
        preds.append(rle)

        del mask, ds, anatomy_mask_full, anatomy_mask_small
        gc.collect()
        continue

    dl = DataLoader(
        ds,
        batch_size=bs,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=0,
    )
    mp = Model_pred(models, dl)

    mask = np.zeros(ds.shape, dtype=np.uint8)
    for pred, vert, _i in iter(mp):
        x1, x2, y1, y2 = vert
        mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

    mask = (mask > 0.5).astype(np.uint8)
    rle = rle_encode_less_memory(mask)

    names.append(idx)
    preds.append(rle)

    del mask, ds, dl
    gc.collect()



## === cell 7
df = pd.DataFrame({"id": names, "predicted": preds})
df = df.set_index("id").reindex(df_sample[ID_COL].astype(str)).reset_index()
df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
