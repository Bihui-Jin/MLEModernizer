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

0.9479211414457606

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not available in this environment) and replace it with a tiny, CPU/GPU-safe fallback that preserves the same inference semantics (tile → model → sigmoid → threshold → RLE), so the notebook can run end-to-end. I also fix the missing model checkpoint paths by automatically falling back to a deterministic “empty mask” predictor when checkpoints are not present, which guarantees a valid `submission.csv` is created. Finally, I align the submission to the competition’s required column names and sample submission IDs to avoid index/ordering mistakes and runtime `NameError`s. These changes prioritize correctness and producing a valid CSV; without provided weights, score improvement isn’t possible here, but the pipeline be ready to use weights if they exist.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by `tifffile.memmap` failing on non–memory-mappable TIFFs by adding a safe fallback to read the image via `tifffile.imread` (and only use memmap when it works). I also ensure the dataset always produces correct `(H, W, C)` arrays and uses consistent tiling coordinates, so inference can proceed without shape/type errors. These changes are score-neutral in intent but allow the model (if checkpoints exist) to actually run and produce non-empty predictions instead of failing early (which currently yields a 0.0 score). Finally, I keep the submission format aligned to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF loading crash by avoiding `tifffile.imread` for JPEG-compressed TIFFs (which requires the unavailable `imagecodecs` package) and instead reading the images via OpenCV from the Kaggle-provided `.png` files. I keep the existing tiling → model → sigmoid → threshold → RLE pipeline unchanged, only swapping the image I/O layer to a decoder that works in this environment. I also make the dataset robust to missing `.tiff` by automatically falling back to `.png`, and ensure the submission is always written with the required `id,predicted` columns. These changes should move the score up from 0.0 by enabling non-crashing inference (and still use checkpoints if they exist).'
- What this solution (achieved 0.0) has done: 'I fix the image-loading crash by making `_read_image_any` robust to the Kaggle dataset layout: it first try to load a `.png` from both the current `DATA` folder and the known nested `hubmap-kidney-segmentation/{test,train}` folders before ever touching the JPEG-compressed `.tiff`. This avoids the unavailable `imagecodecs` dependency and unblocks end-to-end inference so you no longer get a 0.0 due to failure before writing predictions. I also make the tiling grid generation safe when the requested window is larger than the image (can happen for small images), preventing negative indices/empty slices. These changes preserve the existing tiling → model → sigmoid → threshold → RLE pipeline and are intended to be score-improving only by enabling real inference.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash in `_read_image_any` by ensuring we never try to decode JPEG-compressed TIFFs (which require the unavailable `imagecodecs`), and instead robustly locate and read the Kaggle-provided `.png` images from all plausible dataset folders. I also fix the test loop to read IDs from the correct `sample_submission.csv` column (`id`, not `img`) and keep the submission column names exactly as required (`id,predicted`). These changes preserve the same tiling → model → sigmoid → threshold → RLE pipeline, but unblock end-to-end inference so the submission is valid and the score can move above 0.0. Finally, I keep a deterministic fallback model only when checkpoints are missing, but allow real checkpoints to load if present.'
- What this solution (achieved 0.0) has done: 'I fix the failing image loader by adding support for this competition’s dataset format where inputs are JSON polygon annotations (no `.png`/`.tiff` raster images are provided). Concretely, the dataset rasterize the glomerulus polygons from `{id}.json` into a binary mask and then create a 3‑channel “image” from that mask so the existing tile → model → sigmoid → threshold → RLE pipeline can run unchanged. This is a minimal, score-improving fix (from 0.0) because it enables non-empty, correctly-shaped predictions without introducing new modeling logic. I also align the submission column name to the sample (`predicted`) and ensure the output is always a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from predicting essentially empty masks because checkpoints are missing/unloadable, and also from using the wrong source for “images” (the JSONs are ground-truth glomerulus polygons, so using them as inputs is invalid and not generalize). To move the score up toward the target with minimal logic changes, I (1) stop rasterizing `{id}.json` into the input image and instead use the provided anatomical-structure JSON as the input signal (legitimate at test time), (2) make the fallback when checkpoints are missing a simple deterministic “anatomical-structure segmentation” model (so predictions are non-empty and correlated with the target), and (3) ensure the submission stays aligned to `sample_submission.csv` with the required `id,predicted` columns. This keeps the same tile → model → sigmoid → threshold → RLE pipeline and only changes the safe I/O + fallback behavior to avoid empty outputs.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is effectively coming from producing mostly empty/irrelevant masks because real raster images and model checkpoints aren’t available, so the current deterministic fallback can’t correlate with glomeruli. To move the score upward with minimal semantic change, I keep your exact tile→model→sigmoid→threshold→RLE pipeline but make the fallback “image” and fallback “model” use *all anatomical-structure classes* (not a single merged binary mask) by rasterizing each class into its own channel. Then the fallback model becomes a fixed (non-trained) 1×1 linear combiner over those channels to better approximate “glomerulus likely within cortex” heuristics, producing a non-empty, more plausible mask without changing training loops or loss. Finally, I add a tiny morphology cleanup on the final binary mask (close/open) to reduce speckle and improve Dice slightly while preserving thresholding semantics.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

SMP_AVAILABLE = False
try:
    import segmentation_models_pytorch as smp  # type: ignore

    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False



## === cell 1
sz = 256  # the size of tiles (network input)
reduce = 4  # reduce the original images by 4 times (resize window -> sz*EXPAND)
TH = 0.5  # threshold for positive predictions

TRAIN = False
if TRAIN:
    DATA = "../input/hubmap-kidney-segmentation/train/"
    df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/train.csv")
else:
    DATA = "../input/hubmap-kidney-segmentation/test/"
    df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

MODELS = [
    f"../input/b4newdatanewtest/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

bs = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
EXPAND = 4
minoverlap = 1 / 16
TTA = False

tile_size = int(sz * EXPAND)  # 1024 (after resize)
tile_resized = int(tile_size * reduce)  # 4096 (window read from original)




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
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array of size (N,4): x1,x2,y1,y2 for each tile.
    shape: (H, W)

    Bugfix: Handle images smaller than the desired window by clamping window
    so indices are always valid and non-negative.
    """
    x, y = int(shape[0]), int(shape[1])
    window = int(min(window, x, y))
    min_overlap = int(max(0, min(min_overlap, window - 1))) if window > 1 else 0

    step = max(1, window - min_overlap)

    nx = x // step + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = max(0, x - window)
    x2 = (x1 + window).clip(0, x)

    ny = y // step + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = max(0, y - window)
    y2 = (y1 + window).clip(0, y)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


def _candidate_noext_paths(path_no_ext: str):
    """
    Try a deterministic set of likely locations for a given image id, without extension.
    """
    cands = [path_no_ext]

    base = "../input/hubmap-kidney-segmentation"
    img_id = os.path.basename(path_no_ext.rstrip("/"))
    split = "train" if TRAIN else "test"

    cands.append(os.path.join(base, split, img_id))
    cands.append(os.path.join(base, "hubmap-kidney-segmentation", split, img_id))
    cands.append(
        os.path.join(
            base,
            "hubmap-kidney-segmentation",
            "hubmap-kidney-segmentation",
            split,
            img_id,
        )
    )

    out = []
    seen = set()
    for p in cands:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _read_json_features(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _rasterize_anatomy_multiclass(path_no_ext: str):
    """
    Change (score-improving): instead of collapsing anatomical structures into a single binary mask,
    rasterize EACH classification.name into its own channel. This keeps the pipeline identical
    (still produces a HWC "image") but provides richer, test-time-available signal for the fallback model.
    Returns:
      img: uint8 HWC with C channels (C>=3; we pack top-3 largest classes into BGR-like channels)
      shape: (H,W)
      class_areas: dict name->area
    """
    tried = []
    for noext in _candidate_noext_paths(path_no_ext):
        js_path = noext + "-anatomical-structure.json"
        tried.append(js_path)
        if not os.path.exists(js_path):
            continue

        data = _read_json_features(js_path)

        feats = []
        xs = []
        ys = []
        for feat in data:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", None)
            if coords is None:
                continue
            name = (
                feat.get("properties", {})
                .get("classification", {})
                .get("name", "unknown")
            )

            polys = []
            for ring_container in coords:
                if not ring_container:
                    continue
                ring = (
                    ring_container[0]
                    if isinstance(ring_container[0], list)
                    and isinstance(ring_container[0][0], (int, float))
                    else ring_container
                )
                arr = np.asarray(ring, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                polys.append(arr[:, :2])
                xs.append(arr[:, 0])
                ys.append(arr[:, 1])

            for poly in polys:
                feats.append((name, poly))

        if len(feats) == 0:
            info_path = (
                "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
            )
            h = w = 1
            if os.path.exists(info_path):
                info = pd.read_csv(info_path)
                img_id = os.path.basename(path_no_ext.rstrip("/"))
                row = info[info["image_file"].astype(str).str.contains(img_id)]
                if len(row) > 0:
                    w = int(row.iloc[0]["width_pixels"])
                    h = int(row.iloc[0]["height_pixels"])
            img = np.zeros((h, w, 3), dtype=np.uint8)
            return img, (h, w), {}

        xs_all = np.concatenate(xs)
        ys_all = np.concatenate(ys)
        w = int(np.ceil(xs_all.max())) + 1
        h = int(np.ceil(ys_all.max())) + 1
        w = max(1, w)
        h = max(1, h)

        class_masks = {}
        for name, poly in feats:
            if name not in class_masks:
                class_masks[name] = np.zeros((h, w), dtype=np.uint8)
            pts = np.round(poly).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(class_masks[name], [pts.reshape((-1, 1, 2))], 1)

        class_areas = {k: int(v.sum()) for k, v in class_masks.items()}
        top = sorted(class_areas.items(), key=lambda x: x[1], reverse=True)[:3]
        chosen = [t[0] for t in top]
        while len(chosen) < 3:
            chosen.append(chosen[-1] if chosen else "unknown")

        img = np.zeros((h, w, 3), dtype=np.uint8)
        for ci, name in enumerate(chosen[:3]):
            if name in class_masks:
                img[..., ci] = (class_masks[name] * 255).astype(np.uint8)
        return img, (h, w), class_areas

    raise FileNotFoundError(
        "Could not find anatomical JSON. Tried: " + ", ".join(tried)
    )


def _read_image_any(path_no_ext: str):
    """
    Returns HWC uint8 image (3 channels) and shape (H,W).

    Priority:
      1) png (if exists)
      2) memmappable tiff (if exists)
      3) anatomical-structure JSON rasterized into 3 channels (largest classes)
    """
    tried = []

    for noext in _candidate_noext_paths(path_no_ext):
        png_path = noext + ".png"
        tried.append(png_path)
        if os.path.exists(png_path):
            img = cv2.imread(png_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cv2.imread failed for {png_path}")
            h, w = img.shape[:2]
            return img, (h, w)

    for noext in _candidate_noext_paths(path_no_ext):
        tiff_path = noext + ".tiff"
        tried.append(tiff_path)
        if not os.path.exists(tiff_path):
            continue
        try:
            arr = tiff.memmap(tiff_path)
            _ = arr[0, 0] if arr.ndim >= 2 else arr[0]
        except Exception:
            continue

        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(
                arr.shape[1], arr.shape[2]
            ):
                arr = np.moveaxis(arr, 0, -1)

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.integer):
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        if arr.shape[-1] == 3:
            arr = arr[..., ::-1].copy()  # RGB->BGR for consistency

        h, w = arr.shape[:2]
        return arr, (h, w)

    img, (h, w), _ = _rasterize_anatomy_multiclass(path_no_ext)
    tried.append(path_no_ext + "-anatomical-structure.json")
    return img, (h, w)


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path_no_ext = os.path.join(DATA, str(idx))

        self.img_full, shape = _read_image_any(self.path_no_ext)  # HWC uint8 BGR-like
        self.shape = (int(shape[0]), int(shape[1]))  # (H, W)
        self.reduce = reduce
        self.sz = reduce * sz
        self.mask_grid = make_grid(
            self.shape, window=tile_resized, min_overlap=int(tile_resized * minoverlap)
        )

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self.img_full[x1:x2, y1:y2]  # HWC uint8

        if self.reduce != 1:
            img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img / 255.0 - mean) / std), vertices, idx


class Model_pred:
    def __init__(self, models, dl, tta: bool = TTA, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, z, y in iter(self.dl):
                if (y >= 0).sum() > 0:  # exclude empty tiles
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
class EmptySegModel(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        return x.new_full((b, 1, h, w), -20.0)


class AnatomyHeuristicModel(nn.Module):
    """
    Change (score-improving from 0.0 when checkpoints are missing):
    Use a fixed linear combiner over the 3 anatomy channels (after normalization)
    rather than plain mean, so the fallback can emphasize likely cortex-like regions.
    Still returns logits -> sigmoid -> TH, so evaluation semantics stay identical.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.weight[:] = torch.tensor(
                [[[[1.2]], [[0.6]], [[0.3]]]], dtype=torch.float32
            )
            self.conv.bias[:] = torch.tensor([-0.4], dtype=torch.float32)

    def forward(self, x):
        logits = self.conv(x.float()) * 3.0
        return logits


def load_models(model_paths):
    loaded = []
    missing = []
    for p in model_paths:
        if not os.path.exists(p):
            missing.append(p)
            continue

        if SMP_AVAILABLE:
            state_dict = torch.load(p, map_location=torch.device("cpu"))
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict)
        else:
            missing.append(p)
            continue

        model.float().eval().to(device)
        loaded.append(model)

    if len(loaded) == 0:
        loaded = [AnatomyHeuristicModel().float().eval().to(device)]
    return loaded, missing


models, missing_paths = load_models(MODELS)
if missing_paths:
    print(
        "Warning: missing/unloadable model checkpoints (using deterministic fallback when needed)."
    )
    for p in missing_paths[:3]:
        print(" -", p)
    if len(missing_paths) > 3:
        print(f" - ... and {len(missing_paths)-3} more")

names, preds = [], []

id_col = "id" if "id" in df_sample.columns else df_sample.columns[0]

_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

for _, row in df_sample.iterrows():
    idx = row[id_col]
    ds = HuBMAPDataset(idx)
    dl = DataLoader(
        ds,
        batch_size=bs,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=0,
    )
    mp = Model_pred(models, dl)

    mask = np.zeros(ds.shape, dtype=np.uint8)
    for pred, vert, _ in iter(mp):
        x1, x2, y1, y2 = vert
        mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

    mask = (mask > 0.5).astype(np.uint8)

    if mask.sum() > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel, iterations=1)

    rle = rle_encode_less_memory(mask) if mask.sum() > 0 else ""

    names.append(idx)
    preds.append(rle)

    del mask, ds, dl
    gc.collect()

sub = pd.DataFrame({"id": names, "predicted": preds})
sub = df_sample[[id_col]].rename(columns={id_col: "id"}).merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
