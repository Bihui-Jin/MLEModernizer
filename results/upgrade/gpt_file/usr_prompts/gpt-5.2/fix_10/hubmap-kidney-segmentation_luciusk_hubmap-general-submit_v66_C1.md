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

0.9454614097114448

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the crash by making TIFF loading robust: try `tifffile.memmap`, but if the file is not memory-mappable (as in this dataset), fall back to reading it normally via `tifffile.imread` while keeping the rest of the tiling/inference logic unchanged. I also correct the submission schema to exactly match the competition’s required columns (`id`, `predicted`) instead of (`img`, `pixels`), while preserving the same RLE encoding function and mask creation. Finally, I make the test-data path resolution safer by auto-detecting the correct `/kaggle/input/.../test` directory so it works in Kaggle without manual edits. These changes are score-neutral logically (they don’t change the model math) but unblock end-to-end execution and produce a valid `.csv` submission.'
- What this solution (achieved 0.0) has done: 'We need to fix the TIFF reader crash: `tifffile.imread` can’t decode JPEG-compressed TIFFs without `imagecodecs`, so the current fallback still fails. The minimal robust fix is to read each requested window (tile) via OpenCV’s `imreadmulti` (which can decode these TIFFs in Kaggle) instead of decoding the entire TIFF through tifffile, while keeping the same tiling/inference/RLE logic. I adjust `HuBMAPDataset` to lazily load the TIFF pages once per image using OpenCV and return tiles from that array, preserving all normalization, blank-tile skipping, and model prediction semantics. This unblock end-to-end execution and produce a valid `submission.csv`; score should increase from 0.0 because predictions actually be generated instead of crashing.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF decoding crash by making `open_tiff_any` use OpenCV’s `imread` (which reliably decodes these JPEG-compressed TIFFs in Kaggle) and only fall back to tifffile for non-JPEG cases, avoiding the `imagecodecs` dependency. I also correct a small logic bug in `Model_pred` where TTA averaging was off (it was dividing twice), which should improve Dice without changing the overall inference approach. Finally, I keep the same tiling/masking/RLE logic but ensure the pipeline completes and always writes a valid `submission.csv` with the required `id,predicted` columns.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF decoding failure by making `open_tiff_any()` never fall through to `tifffile.imread()` for JPEG-compressed TIFFs (which requires `imagecodecs`), and instead decode reliably via OpenCV. To keep memory/time reasonable and preserve the existing tiling/inference logic, I load the TIFF once using `cv2.imdecode` (with a safe fallback to `cv2.imread`) and keep the same HWC/uint8 formatting. This unblocks dataset creation, runs inference end-to-end, and writes a valid `submission.csv` with the required `id,predicted` columns. The model/tiling/RLE logic stays unchanged; the score should increase from 0.0 because predictions actually be produced instead of crashing.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF decoding so it works for all HuBMAP test TIFF variants in this environment by adding a robust OpenCV multi-page read fallback (some files fail `cv2.imdecode` and are not memmappable). I keep the tiling/inference/RLE logic the same, only changing how the TIFF is loaded into an HWC uint8 array. I also ensure the test directory detection includes the correct nested `/kaggle/input/.../hubmap-kidney-segmentation/test` path so the `.tiff` files are actually found. These changes unblock end-to-end inference so you get a non-empty, valid `submission.csv`, which should raise the score from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the crash by making TIFF loading robust for the HuBMAP JPEG-compressed, non-memmappable TIFFs: keep your OpenCV attempts, but add a safe `tifffile.TiffFile(...).asarray()` fallback (and finally `tifffile.imread`) to actually decode when memmap fails. This is a minimal change isolated to `open_tiff_any()` and does not alter the tiling/inference/RLE logic, but it unblocks end-to-end execution so predictions are produced (score should increase from 0.0). I also ensure the decoded array is always converted into HWC uint8 3-channel format so downstream slicing/resizing is stable. The submission-writing cell remain the same and still produce `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash by making TIFF decoding robust for JPEG-compressed TIFFs without `imagecodecs`, using OpenCV’s `imreadmulti`/`imdecode` to decode pages and then composing them into an HWC image (instead of falling through to `tifffile.imread`). I keep all tiling/inference/RLE logic unchanged and only touch `open_tiff_any()` plus a small guard to ensure the test image path exists before loading. This should move the score up from 0.0 because the pipeline complete and produce non-empty predictions. The submission writing remain the same (`id,predicted`) and always produce `submission.csv`.'

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
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True



## === cell 1
sz = 256  # the size of tiles
reduce = 4  # reduce the original images by 4 times
TH = 0.5  # threshold for positive predictions

_CANDIDATE_TEST_DIRS = [
    "../input/hubmap-kidney-segmentation/test/",
    "/kaggle/input/hubmap-kidney-segmentation/test/",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
    "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
    "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
]
DATA = None
for _p in _CANDIDATE_TEST_DIRS:
    if os.path.isdir(_p):
        DATA = _p
        break
if DATA is None:
    DATA = "../input/hubmap-kidney-segmentation/test/"

MODELS = [
    f"../input/skfold-without-test/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

_SAMPLE_CANDS = [
    "../input/hubmap-kidney-segmentation/sample_submission.csv",
    "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/sample_submission.csv",
    "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/sample_submission.csv",
    "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/hubmap-kidney-segmentation/sample_submission.csv",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/hubmap-kidney-segmentation/sample_submission.csv",
]
_sample_path = None
for _p in _SAMPLE_CANDS:
    if os.path.exists(_p):
        _sample_path = _p
        break
if _sample_path is None:
    _sample_path = "../input/hubmap-kidney-segmentation/sample_submission.csv"

df_sample = pd.read_csv(_sample_path)

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"  # kept for compatibility with original config (not used by fallback model)
shift = True
minoverlap = 300




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None or (isinstance(enc, (float, np.floating)) and np.isnan(enc)):
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

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return Array of size (N,4), where N - number of tiles,
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


def _ensure_hwc_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert various layouts to HWC with 3 channels uint8."""
    if arr is None:
        raise ValueError("Failed to decode image (arr is None).")

    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.integer):
            if arr.max() > 255:
                arr = (arr / (arr.max() + 1e-6) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8, copy=False)

    return arr


def open_tiff_any(tiff_path: str):
    """
    Bugfix: HuBMAP test TIFFs can be JPEG-compressed and fail tifffile decoding without imagecodecs.
    Minimal robust approach: decode via OpenCV first (supports these TIFFs in Kaggle),
    including multi-page TIFFs. Only if OpenCV fails, fall back to tifffile.
    Returns HWC uint8 3-channel image in OpenCV BGR order (consistent with downstream code).
    """
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"TIFF not found: {tiff_path}")

    try:
        ok, pages = cv2.imreadmulti(tiff_path, flags=cv2.IMREAD_ANYCOLOR)
        if ok and pages is not None and len(pages) > 0:
            if (
                len(pages) >= 3
                and pages[0].ndim == 2
                and pages[1].ndim == 2
                and pages[2].ndim == 2
            ):
                h = min(p.shape[0] for p in pages[:3])
                w = min(p.shape[1] for p in pages[:3])
                b = pages[2][:h, :w]
                g = pages[1][:h, :w]
                r = pages[0][:h, :w]
                arr = np.stack([b, g, r], axis=-1)
                return _ensure_hwc_rgb_uint8(arr)
            else:
                page0 = pages[0]
                if page0.ndim == 2:
                    page0 = cv2.cvtColor(page0, cv2.COLOR_GRAY2BGR)
                elif page0.ndim == 3 and page0.shape[2] == 4:
                    page0 = page0[:, :, :3]
                return _ensure_hwc_rgb_uint8(page0)
    except Exception:
        pass

    try:
        with open(tiff_path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if arr is not None:
            return _ensure_hwc_rgb_uint8(arr)
    except Exception:
        pass

    try:
        arr = cv2.imread(tiff_path, cv2.IMREAD_COLOR)
        if arr is not None:
            return _ensure_hwc_rgb_uint8(arr)
    except Exception:
        pass

    try:
        with tiff.TiffFile(tiff_path) as tf:
            arr = tf.asarray()
        arr = _ensure_hwc_rgb_uint8(arr)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[..., ::-1].copy()  # RGB -> BGR
        return arr
    except Exception:
        pass

    try:
        arr = tiff.imread(tiff_path)
        arr = _ensure_hwc_rgb_uint8(arr)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[..., ::-1].copy()
        return arr
    except Exception as e:
        raise ValueError(f"Could not decode TIFF at {tiff_path}: {e}")




## === cell 4
class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.tiff_path = os.path.join(DATA, idx + ".tiff")
        self.data = open_tiff_any(self.tiff_path)
        self.shape = self.data.shape[:2]  # (H, W)
        self.reduce = reduce
        self.sz = reduce * sz  # window on original scale
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self.data[x1:x2, y1:y2, :3]

        if img.shape[0] != self.sz or img.shape[1] != self.sz:
            pad_h = self.sz - img.shape[0]
            pad_w = self.sz - img.shape[1]
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

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


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNAct(in_channels, base), ConvBNAct(base, base))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            ConvBNAct(base, base * 2), ConvBNAct(base * 2, base * 2)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            ConvBNAct(base * 2, base * 4), ConvBNAct(base * 4, base * 4)
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBNAct(base * 4, base * 2), ConvBNAct(base * 2, base * 2)
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvBNAct(base * 2, base), ConvBNAct(base, base))

        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
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
                    py = py.permute(0, 2, 3, 1).float().cpu()
                    py = py.squeeze(-1).numpy()
                    z = z.numpy()

                    for i in range(len(py)):
                        yield py[i], z[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
models = []
existing_paths = [p for p in MODELS if os.path.exists(p)]

if len(existing_paths) == 0:
    model = TinyUNet(in_channels=3, base=32).to(device)
    model.eval()
    models = [model]
else:
    for path in existing_paths:
        state_dict = torch.load(path, map_location="cpu")
        model = TinyUNet(in_channels=3, base=32)
        try:
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.float()
            model.eval()
            models.append(model)
        except Exception:
            continue
    if len(models) == 0:
        model = TinyUNet(in_channels=3, base=32).to(device)
        model.eval()
        models = [model]

gc.collect()

names, preds = [], []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl)

    mask = np.zeros(ds.shape, dtype=np.uint8)
    for pred, vert, _tile_i in iter(mp):
        x1, x2, y1, y2 = vert
        ph, pw = (x2 - x1), (y2 - y1)
        pred = pred[:ph, :pw]
        mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

    mask = (mask > 0.5).astype(np.uint8)
    rle = rle_encode_less_memory(mask)
    if isinstance(rle, str) and len(rle.strip()) == 0:
        rle = ""

    names.append(idx)
    preds.append(rle)

    del mask, ds, dl
    gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/464294122.py in open_tiff_any(tiff_path)
    132     try:
--> 133         arr = tiff.imread(tiff_path)
    134         arr = _ensure_hwc_rgb_uint8(arr)

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in imread(files, selection, aszarr, key, series, level, squeeze, maxworkers, buffersize, mode, name, offset, size, pattern, axesorder, categories, imread, imreadargs, sort, container, chunkshape, chunkdtype, axestiled, ioworkers, chunkmode, fillvalue, zattrs, multiscales, omexml, out, out_inplace, _multifile, _useframes, **kwargs)
   1237                     return zarr_selection(store, selection, out=out)
-> 1238                 return tif.asarray(
   1239                     key=key,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, key, series, level, squeeze, out, maxworkers, buffersize)
   4541                 raise ValueError('page is None')
-> 4542             result = page0.asarray(
   4543                 out=out, maxworkers=maxworkers, buffersize=buffersize

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, out, squeeze, lock, maxworkers, buffersize)
   8884 
-> 8885             for _ in self.segments(
   8886                 func=func,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in segments(self, lock, maxworkers, func, sort, buffersize, _fullsize)
   8698                 ):
-> 8699                     yield from executor.map(decode, segments)
   8700 

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    316         try:
--> 317             return fut.result(timeout)
    318         finally:

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    400             try:
--> 401                 raise self._exception
    402             finally:

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode(args, decodeargs, decode)
   8671             def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
-> 8672                 return func(decode(*args, **decodeargs))
   8673 

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode_raise_compression(exc, *args, **kwargs)
   8093             def decode_raise_compression(*args, exc=str(exc)[1:-1], **kwargs):
-> 8094                 raise ValueError(f'{exc}')
   8095 

ValueError: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/758904640.py in <cell line: 0>()
     29 for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
     30     idx = row["id"]
---> 31     ds = HuBMAPDataset(idx)
     32     dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
     33     mp = Model_pred(models, dl)

/tmp/ipykernel_55/1804082611.py in __init__(self, idx, sz, reduce)
      3         self.idx = idx
      4         self.tiff_path = os.path.join(DATA, idx + ".tiff")
----> 5         self.data = open_tiff_any(self.tiff_path)
      6         self.shape = self.data.shape[:2]  # (H, W)
      7         self.reduce = reduce

/tmp/ipykernel_55/464294122.py in open_tiff_any(tiff_path)
    137         return arr
    138     except Exception as e:
--> 139         raise ValueError(f"Could not decode TIFF at {tiff_path}: {e}")
    140 
    141 

ValueError: Could not decode TIFF at ../input/hubmap-kidney-segmentation/test/8242609fa.tiff: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package

## === cell 6
sub = pd.DataFrame({"id": names, "predicted": preds})

if len(sub) != len(df_sample) or set(sub["id"]) != set(df_sample["id"]):
    sub = df_sample[["id"]].merge(sub, on="id", how="left")
    sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)
sub
