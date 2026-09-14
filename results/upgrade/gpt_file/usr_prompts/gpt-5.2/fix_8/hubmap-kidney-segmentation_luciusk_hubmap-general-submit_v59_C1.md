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

0.9240654772247268

# 6. Current score

0.00021

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the two blockers preventing any valid submission: `segmentation_models_pytorch` is not installed in this environment, and the referenced pretrained model `.pth` files are not present. To preserve the core inference/tiling/RLE logic while ensuring end-to-end execution, I add a safe fallback that uses the provided `anatomical-structure.json` polygons to build a reasonable binary mask when model inference cannot run. This produces a correctly formatted `submission.csv` (columns `id,predicted`) and avoids NameErrors by always defining `names/preds`. The fallback is score-oriented (not random): it leverages available anatomical structure annotations to approximate cortex/tissue regions, which should score better than empty masks.'
- What this solution (achieved 0.0) has done: 'I fix the fallback path crash by making TIFF loading robust when `tifffile.memmap` is not possible (common for compressed/striped TIFFs), falling back to reading only the first page/series to get image shape without loading everything unnecessarily. I also make `rle_encode_less_memory` safe on empty/all-zero masks (so it returns an empty string instead of an invalid RLE). These changes preserve your core tiling/inference logic and only affect the fallback path so the notebook runs end-to-end and produces a valid `submission.csv`. With the fallback actually running (instead of crashing), your score should move up from 0.0 toward the target by producing non-empty, reasonable masks derived from anatomical structures.'
- What this solution (achieved 0.09744) has done: 'I fix the crash in the fallback path by avoiding full TIFF decoding (which currently fails because JPEG-compressed TIFFs need `imagecodecs`, not available here). Instead, the fallback infer image `height,width` from the provided `HuBMAP-20-dataset_information.csv` (and only if missing it try a minimal TIFF read). I also correct the RLE encoder to not forcibly zero out the first/last pixel (that can silently drop true positives) by padding with sentinel zeros instead, which is score-positive and still consistent with the competition’s RLE definition. These changes keep the original model/tiling logic intact and only adjust robustness + encoding so you get a valid, non-empty submission and move the score up from 0.0 toward the target.'
- What this solution (achieved 0.00021) has done: 'Your current score (0.09744) is far below the target (0.92406), so we should improve the fallback mask quality without changing the core model/tiling logic (which can’t run here due to missing SMP/models and TIFF decode constraints). I keep your existing anatomical-structure fallback, but make it more score-aligned by (1) using the *training* glomerulus masks to estimate a better prior for glomeruli size and coverage, then (2) converting the anatomical tissue region into plausible “glomeruli-like” blobs via distance-transform peak seeding and controlled dilation. This keeps everything deterministic, fast, and uses only provided data (no leakage from test labels), while producing masks that should match the expected object scale better than filling large tissue regions. Submission writing stays identical (`id,predicted` to `submission.csv`).'
- What this solution (achieved 0.00021) has done: 'Your current score (0.00021) is far below the target (0.92406), so we should improve mask quality while keeping your core fallback approach (anatomical-structure polygon → tissue mask → glomeruli-like blobs) intact. The biggest score-killer is likely **RLE orientation/ordering mismatch**: your encoder uses `.T.flatten()` while your fallback masks are already in `(H,W)`; this can produce near-random masks vs the evaluator and collapse Dice. I make encoding consistent with the competition’s “top-to-bottom then left-to-right” definition by using **Fortran-order flattening without transposing**, and I update both encoder/decoder helpers accordingly to preserve semantics. I also make the fallback a bit more conservative by restricting blobs to the largest connected tissue component to reduce false positives (minimal change, same logic).'
- What this solution (achieved 0.00021) has done: 'Your score is far below the target (0.00021 vs 0.924), so we should improve the *fallback* predictions (since SMP/models/TIFF decode are unavailable) while keeping your overall pipeline and RLE semantics intact. The main likely issue is that the anatomical-structure polygons don’t reliably represent the tissue extent, so your current fallback can produce near-empty or badly shaped “glomeruli-like blobs.” I minimally improve the fallback by (1) using the test `*-anatomical-structure.json` to estimate a tissue bounding box, (2) filling that box as a tissue proxy when polygons are sparse/broken, and (3) generating blobs inside that proxy with a slightly safer prior calibrated from train masks. This preserves your existing flow (anatomy → tissue → blobs → RLE) but should move Dice up meaningfully from ~0.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # type: ignore

    HAS_SMP = True
except Exception:
    smp = None
    HAS_SMP = False



## === cell 1
sz = 256  # the size of tiles after reduction
reduce = 4  # reduce the original images by 4 times
TH = 0.35  # threshold for positive predictions

DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [
    f"../input/b2alldata/efficientnet-b2-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b2"  # efficientnet-b4, se_resnext50_32x4d
shift = True
minoverlap = 300




## === cell 2
def enc2mask(encs, shape_hw):
    """
    Decode RLE to mask with competition's pixel ordering:
    pixels are numbered top-to-bottom then left-to-right => Fortran order (column-major).
    shape_hw is (H,W).
    """
    h, w = shape_hw
    img = np.zeros(h * w, dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        if enc is None or (isinstance(enc, str) and enc.strip() == ""):
            continue
        s = str(enc).split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start : start + length] = 1 + m
    return img.reshape((h, w), order="F")


def mask2enc(mask, n=1):
    """
    Encode mask with competition ordering (Fortran order).
    """
    pixels = mask.flatten(order="F")
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
    Encode using competition pixel ordering (Fortran order),
    and use sentinel padding (don't overwrite boundary pixels).
    """
    pixels = img.flatten(order="F")
    if pixels.size == 0:
        return ""
    pixels = pixels.astype(np.uint8, copy=False)
    if pixels.max() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if runs.size == 0:
        return ""
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


def _open_tiff_memmap(path):
    """
    Keep original behavior for model path (needs actual pixels),
    but be robust to a broader set of TIFF layouts when possible.
    Note: JPEG-compressed TIFFs may still fail without imagecodecs.
    """
    try:
        arr = tiff.memmap(path)
    except Exception:
        with tiff.TiffFile(path) as tif:
            arr = tif.pages[0].asarray()

    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


def _models_available():
    if not HAS_SMP:
        return False
    for p in MODELS:
        if not os.path.exists(p):
            return False
    return True


def _load_models():
    models = []
    for path in MODELS:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model.load_state_dict(state_dict)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)
    return models


def _poly_to_mask(coords, shape_hw):
    """Rasterize (multi)polygon coordinates to a binary mask. shape_hw is (H, W)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    def _fill_ring(ring):
        if ring is None or len(ring) < 3:
            return
        pts = np.asarray(ring, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts_i], 1)

    if not isinstance(coords, list) or len(coords) == 0:
        return mask

    if (
        isinstance(coords[0], list)
        and len(coords[0]) > 0
        and isinstance(coords[0][0], (list, tuple))
    ):
        if (
            len(coords[0]) > 0
            and isinstance(coords[0][0], (list, tuple))
            and len(coords[0][0]) > 0
            and isinstance(coords[0][0][0], (int, float))
        ):
            for ring in coords:
                _fill_ring(ring)
        else:
            for poly in coords:
                if not isinstance(poly, list):
                    continue
                for ring in poly:
                    _fill_ring(ring)
    return mask


def _largest_connected_component(mask_u8: np.ndarray):
    """
    Minimal score-oriented postprocess: keep only largest component of tissue proxy
    to reduce false positives from stray polygons.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    j = 1 + int(np.argmax(areas))
    return (labels == j).astype(np.uint8)


_INFO_PATHS = [
    "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv",
    "../input/HuBMAP-20-dataset_information.csv",
]
_info_df = None
for _p in _INFO_PATHS:
    if os.path.exists(_p):
        _info_df = pd.read_csv(_p)
        break

_shape_map = {}
if _info_df is not None:
    for _, r in _info_df.iterrows():
        try:
            img_file = str(r["image_file"])
            _id = os.path.splitext(os.path.basename(img_file))[0]
            w = int(r["width_pixels"])
            h = int(r["height_pixels"])
            _shape_map[_id] = (h, w)
        except Exception:
            continue


def _get_hw_for_id(idx):
    hw = _shape_map.get(idx, None)
    if hw is not None:
        return hw
    tiff_path = os.path.join(DATA, f"{idx}.tiff")
    arr = _open_tiff_memmap(tiff_path)
    return arr.shape[:2]


def _estimate_glom_prior_from_train(max_imgs: int = 6):
    train_csv = "../input/hubmap-kidney-segmentation/train.csv"
    info_csv = "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
    if not (os.path.exists(train_csv) and os.path.exists(info_csv)):
        return {
            "area_frac_q": 0.006,  # safe small default
            "radius_px": 18,  # moderate blob radius default
            "seed_spacing": 220,  # moderate spacing default
        }

    tr = pd.read_csv(train_csv)
    info = pd.read_csv(info_csv)
    shape_map_tr = {}
    for _, r in info.iterrows():
        try:
            _id = os.path.splitext(os.path.basename(str(r["image_file"])))[0]
            shape_map_tr[_id] = (int(r["height_pixels"]), int(r["width_pixels"]))
        except Exception:
            continue

    area_fracs = []
    for _, r in tr.head(max_imgs).iterrows():
        _id = str(r["id"])
        rle = r.get("encoding", None)
        if _id not in shape_map_tr:
            continue
        h, w = shape_map_tr[_id]
        if (
            rle is None
            or (isinstance(rle, float) and np.isnan(rle))
            or str(rle).strip() == ""
        ):
            area_fracs.append(0.0)
            continue
        m = enc2mask([str(rle)], (h, w))
        area_fracs.append(float((m > 0).sum()) / float(h * w))

    if len(area_fracs) == 0:
        area_frac_q = 0.006
    else:
        area_frac_q = float(np.clip(np.quantile(area_fracs, 0.55), 0.0025, 0.04))

    radius_px = int(np.clip(10 + 520 * area_frac_q, 12, 44))
    seed_spacing = int(np.clip(110 + 7000 * area_frac_q, 140, 420))

    return {
        "area_frac_q": area_frac_q,
        "radius_px": radius_px,
        "seed_spacing": seed_spacing,
    }


_GLOM_PRIOR = _estimate_glom_prior_from_train(max_imgs=6)


def _make_glom_like_mask(
    tissue_mask: np.ndarray, area_target: float, radius_px: int, seed_spacing: int
):
    h, w = tissue_mask.shape
    if tissue_mask.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    tissue = (tissue_mask > 0).astype(np.uint8)

    dt = cv2.distanceTransform(tissue, cv2.DIST_L2, 5)
    if not np.isfinite(dt).all():
        dt = np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)

    ksz = int(np.clip(radius_px * 2 + 1, 15, 81))
    if ksz % 2 == 0:
        ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    dt_dil = cv2.dilate(dt, kernel)
    peaks = (dt == dt_dil) & (dt >= max(2.0, radius_px * 0.55))

    coords = np.column_stack(np.where(peaks))
    if coords.shape[0] == 0:
        ys, xs = np.where(tissue > 0)
        if ys.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
        step = max(1, int((ys.size / 2000) ** 0.5))
        coords = np.column_stack([ys[::step], xs[::step]])

    order = np.argsort(dt[coords[:, 0], coords[:, 1]])[::-1]
    coords = coords[order]

    selected = []
    min_dist2 = float(seed_spacing * seed_spacing)
    for y, x in coords:
        ok = True
        for sy, sx in selected[-64:]:
            dy = float(y - sy)
            dx = float(x - sx)
            if dy * dy + dx * dx < min_dist2:
                ok = False
                break
        if ok:
            selected.append((int(y), int(x)))
        if len(selected) >= 2500:
            break

    seeds = np.zeros((h, w), dtype=np.uint8)
    for y, x in selected:
        cv2.circle(seeds, (x, y), 1, 1, -1)

    r = int(np.clip(radius_px, 8, 50))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    blobs = cv2.dilate(seeds, k2, iterations=1)
    blobs = (blobs & tissue).astype(np.uint8)

    target_area = int(area_target * h * w)
    cur = int(blobs.sum())
    if target_area > 0:
        if cur > int(1.35 * target_area):
            k3 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, r // 2 * 2 + 1),) * 2
            )
            blobs = cv2.erode(blobs, k3, iterations=1)
            blobs = (blobs & tissue).astype(np.uint8)
        elif cur < int(0.75 * target_area):
            k3 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, r // 3 * 2 + 1),) * 2
            )
            blobs = cv2.dilate(blobs, k3, iterations=1)
            blobs = (blobs & tissue).astype(np.uint8)

    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, k4, iterations=1).astype(np.uint8)
    return blobs


def _anatomy_bbox_from_json(feats, shape_hw):
    """
    Score-oriented robustness: if anatomy polygons are incomplete/sparse, use their
    combined bounds to create a tissue proxy ROI (prevents near-empty masks).
    """
    h, w = shape_hw
    xs = []
    ys = []
    for feat in feats:
        try:
            geom = feat["geometry"]
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", None)
        except Exception:
            continue
        if gtype not in ("Polygon", "MultiPolygon"):
            continue
        if not isinstance(coords, list) or len(coords) == 0:
            continue

        def _collect_ring(ring):
            if ring is None or len(ring) < 3:
                return
            arr = np.asarray(ring, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                return
            xs.extend(arr[:, 0].tolist())
            ys.extend(arr[:, 1].tolist())

        if (
            isinstance(coords[0], list)
            and len(coords[0]) > 0
            and isinstance(coords[0][0], (list, tuple))
        ):
            if (
                len(coords[0]) > 0
                and isinstance(coords[0][0], (list, tuple))
                and len(coords[0][0]) > 0
                and isinstance(coords[0][0][0], (int, float))
            ):
                for ring in coords:
                    _collect_ring(ring)
            else:
                for poly in coords:
                    if not isinstance(poly, list):
                        continue
                    for ring in poly:
                        _collect_ring(ring)

    if len(xs) < 10 or len(ys) < 10:
        return None

    x0 = int(np.clip(np.floor(np.percentile(xs, 2.0)), 0, w - 1))
    x1 = int(np.clip(np.ceil(np.percentile(xs, 98.0)), 0, w - 1))
    y0 = int(np.clip(np.floor(np.percentile(ys, 2.0)), 0, h - 1))
    y1 = int(np.clip(np.ceil(np.percentile(ys, 98.0)), 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return None

    padx = max(10, int(0.03 * (x1 - x0)))
    pady = max(10, int(0.03 * (y1 - y0)))
    x0 = max(0, x0 - padx)
    x1 = min(w - 1, x1 + padx)
    y0 = max(0, y0 - pady)
    y1 = min(h - 1, y1 + pady)
    return (y0, y1, x0, x1)


def fallback_predict_from_anatomical_json(idx):
    """
    Score-oriented fallback when SMP/models are unavailable:
    - Build a tissue-region proxy from anatomical structure polygons.
    - If polygons are sparse/broken, fall back to a bbox-derived ROI proxy.
    - Keep only the largest connected tissue component (reduces false positives).
    - Convert tissue region into a more glomeruli-like mask using a size prior from train.csv.
    """
    json_path = os.path.join(DATA, f"{idx}-anatomical-structure.json")
    h, w = _get_hw_for_id(idx)

    if not os.path.exists(json_path):
        return np.zeros((h, w), dtype=np.uint8)

    with open(json_path, "r") as f:
        feats = json.load(f)

    tissue = np.zeros((h, w), dtype=np.uint8)

    preferred = []
    others = []
    for feat in feats:
        try:
            name = feat["properties"]["classification"]["name"]
            geom = feat["geometry"]
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", None)
        except Exception:
            continue

        if gtype not in ("Polygon", "MultiPolygon"):
            continue

        lname = str(name).lower()
        target_list = (
            preferred
            if any(k in lname for k in ["cortex", "tissue", "kidney", "parenchyma"])
            else others
        )
        target_list.append(coords)

    use_coords = preferred if len(preferred) > 0 else others
    for coords in use_coords:
        tissue |= _poly_to_mask(coords, (h, w))

    area = int(tissue.sum())
    if area < int(0.02 * h * w):
        bb = _anatomy_bbox_from_json(feats, (h, w))
        if bb is not None:
            y0, y1, x0, x1 = bb
            tissue2 = np.zeros((h, w), dtype=np.uint8)
            tissue2[y0 : y1 + 1, x0 : x1 + 1] = 1
            tissue = np.maximum(tissue, tissue2)

    if tissue.sum() == 0:
        return tissue

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, k, iterations=1).astype(np.uint8)

    tissue = _largest_connected_component(tissue)

    return _make_glom_like_mask(
        tissue_mask=tissue,
        area_target=float(_GLOM_PRIOR["area_frac_q"]),
        radius_px=int(_GLOM_PRIOR["radius_px"]),
        seed_spacing=int(_GLOM_PRIOR["seed_spacing"]),
    )




## === cell 4
names, preds = [], []

if _models_available():
    if not shift:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.data = _open_tiff_memmap(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.data.shape[:2]  # (H,W)
                self.reduce = reduce
                self.sz = reduce * sz  # window in original resolution

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
                crop = self.data[p00:p01, p10:p11]
                if crop.shape[-1] == 1:
                    crop = np.repeat(crop, 3, axis=-1)
                img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = crop.astype(
                    np.uint8, copy=False
                )

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

        models = _load_models()

        for _, row in df_sample.iterrows():
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
                self.data = _open_tiff_memmap(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.data.shape[:2]  # (H,W)
                self.reduce = reduce
                self.sz = reduce * sz
                self.mask_grid = make_grid(
                    self.shape, window=self.sz, min_overlap=minoverlap
                )

            def __len__(self):
                return len(self.mask_grid)

            def __getitem__(self, idx):
                x1, x2, y1, y2 = self.mask_grid[idx]
                crop = self.data[x1:x2, y1:y2]
                if crop.ndim == 2:
                    crop = crop[..., None]
                if crop.shape[-1] == 1:
                    crop = np.repeat(crop, 3, axis=-1)
                img = crop.astype(np.uint8, copy=False)

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

        models = _load_models()

        for _, row in df_sample.iterrows():
            idx = row["id"]
            ds = HuBMAPDataset(idx)
            dl = DataLoader(
                ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
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
else:
    for _, row in df_sample.iterrows():
        idx = row["id"]
        mask = fallback_predict_from_anatomical_json(idx)
        rle = rle_encode_less_memory(mask)
        names.append(idx)
        preds.append(rle)
        del mask
        gc.collect()



## === cell 5
pred_map = {k: v for k, v in zip(names, preds)}
sub = df_sample.copy()
sub["predicted"] = sub["id"].map(pred_map).fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(f"Saved submission.csv with {len(sub)} rows")
print("smp_available:", HAS_SMP, "models_available:", _models_available())
print("fallback_shape_map_size:", len(_shape_map))
print("glom_prior:", _GLOM_PRIOR)
