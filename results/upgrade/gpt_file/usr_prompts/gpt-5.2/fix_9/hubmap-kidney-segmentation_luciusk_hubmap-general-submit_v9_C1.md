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

0.92771180548043

# 6. Current score

0.049

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by implementing a small TIFF reader based on `tifffile`, while keeping the exact tiling/padding/reduction and model inference logic unchanged. I also fix a few runtime issues: `np.float` deprecation checks, missing imports due to the first-cell crash, and a `tqdm.notebook` import that can fail in non-notebook execution. Finally, I make the submission-writing robust so it always outputs exactly the sample_submission IDs (3 rows here) and produces a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed) by providing a tiny Unet-like fallback that preserves the “CNN → logits → sigmoid → threshold → RLE” inference semantics so the pipeline runs end-to-end. I also fix the test image loading crash by using OpenCV’s TIFF reader (works without `imagecodecs`) and only fall back to `tifffile` if needed, which resolves the JPEG-compressed TIFF error. Since the provided pretrained model paths don’t exist, I make model loading robust: if weights are missing, the code still run and output a valid submission (this won’t reach the target score but avoid the 0.0-from-crash/invalid-submission failure mode). Finally, I keep submission formatting aligned with `sample_submission.csv` and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by OpenCV refusing to load extremely large TIFFs by bypassing `cv2.imread` and instead using `tifffile` with a safe fallback that loads a downsampled pyramid level when available. This keeps your existing tiling/downsample (`reduce`), model inference, thresholding, and RLE logic unchanged, but ensures the dataset can actually be read end-to-end in the Kaggle runtime. I also add a small path fallback for `DATA` so it works whether the dataset is mounted under `../input/...` or `/kaggle/input/...`. With these fixes, the pipeline produce a valid `submission.csv` (and should score > 0.0 instead of failing), without altering the model core logic.'
- What this solution (achieved 0.01674) has done: 'I fix the runtime crash caused by JPEG-compressed TIFF pages requiring `imagecodecs` by changing the TIFF loader to avoid decoding those pages: first try OpenCV (which can read these TIFFs in this environment), and if that fails, use `tifffile` but explicitly select a non-JPEG-compressed page/series level. I also add a safe final fallback that returns a blank image with the correct shape (from the dataset information CSV) so the pipeline always completes and writes a valid `submission.csv` instead of crashing. These changes are strictly limited to input I/O and robustness; the tiling, normalization, model inference, thresholding, and RLE encoding logic remain unchanged. This should move the score above 0.0 by producing a non-empty valid submission.'
- What this solution (achieved 0.01674) has done: 'Your current low score is mainly because the code is not actually using the intended pretrained EfficientNet-B4 weights (they’re missing), so you’re effectively predicting with random weights. To move the score toward the target with minimal semantic change, I keep your exact inference/tiling/RLE pipeline but restore the original backbone by loading `segmentation_models_pytorch` when available (it typically is in this competition’s Kaggle image), and only fall back to `TinyUNet` if SMP truly isn’t present. I also make weight-loading stricter to ensure the checkpoint keys map onto the real model correctly (and error loudly only if a checkpoint exists but can’t be loaded), because silently running random weights is the biggest score killer. No thresholding/metric logic changes are made, and it still writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.049) has done: 'Your current score suggests the inference pipeline is often producing near-empty or mis-scaled masks; the highest-impact minimal fix (without touching model architecture/inference semantics) is to ensure the test TIFFs are read at full resolution and only then downsampled by the existing `reduce` logic. Right now the loader can pick a pyramid level close to the `reduce` factor, and then the dataset downsamples again, effectively double-reducing and destroying detail (which collapses Dice). I change the TIFF reading fallback to always prefer the highest-resolution non-JPEG page/level when OpenCV can’t read the file, while keeping the same downstream tiling, normalization, interpolation, thresholding, and RLE encoding unchanged. This is a strictly I/O-scale correction and should move the score substantially toward the target without altering the model core logic.'
- What this solution (achieved 0.049) has done: 'Your current score gap is large (0.049 vs target 0.9277), and the biggest likely cause is that the pretrained weights you intend to use are not being found, so inference is effectively random. I keep your exact dataset tiling/downsample/normalization, model forward/sigmoid/threshold, and RLE logic unchanged, but make the model weight discovery robust by automatically locating the `efficientnet-b4-FOLD-*-model.pth` files anywhere under `/kaggle/input` (and falling back to your original paths if found). I also ensure we don’t accidentally ensemble multiple untrained models: if no weights are found, we run a single model (same core inference semantics, just avoids diluting predictions). These are minimal, score-relevant changes that should move Dice substantially upward toward the target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.049) has done: 'The score gap to the target is very large, and the most likely cause is that your model is still running without the intended pretrained checkpoints (or loading them into a mismatched architecture), which yields near-random masks. I make a minimal, score-relevant change to ensure `segmentation_models_pytorch` is used when available and that checkpoint key-remapping prefers the common `model.` prefix used in many SMP training scripts (instead of forcing everything under `cnn_model.`), so weights actually land on the right layers. I also change the behavior to *not* include randomly initialized models in the ensemble when some folds are missing (keep only successfully loaded models), since averaging random logits with good ones can destroy Dice. Everything else (tiling, downsample `reduce`, normalization, sigmoid+threshold, and RLE encoding/submission format) stays unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import sys
import warnings

import numpy as np
import pandas as pd

import cv2
import tifffile as tiff

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
sz = 256  # the size of tiles
reduce = 4  # reduce the original images by 4 times
TH = 0.45  # threshold for positive predictions

DATA = "../input/hubmap-kidney-segmentation/test/"
if not os.path.isdir(DATA):
    DATA = "/kaggle/input/hubmap-kidney-segmentation/test/"


def _discover_model_paths():
    original = [
        f"../input/b4-test/result/efficientnet-b4-FOLD-{i}-model.pth" for i in range(5)
    ]
    if all(os.path.exists(p) for p in original):
        return original

    found = []
    search_roots = [
        "/kaggle/input",
        "../input",
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.startswith("efficientnet-b4-FOLD-") and fn.endswith("-model.pth"):
                    found.append(os.path.join(dirpath, fn))

    def _fold_key(p):
        base = os.path.basename(p)
        try:
            fold = int(base.split("efficientnet-b4-FOLD-")[1].split("-model.pth")[0])
        except Exception:
            fold = 999
        return (fold, p)

    found = sorted(set(found), key=_fold_key)

    per_fold = {}
    for p in found:
        f = _fold_key(p)[0]
        if f not in per_fold and 0 <= f <= 4:
            per_fold[f] = p
    if len(per_fold) > 0:
        return [per_fold[f] for f in sorted(per_fold.keys())]

    return original  # fallback (may be missing)


MODELS = _discover_model_paths()

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFO_PATH = "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
if not os.path.exists(INFO_PATH):
    INFO_PATH = (
        "/kaggle/input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
    )
df_info = pd.read_csv(INFO_PATH) if os.path.exists(INFO_PATH) else None
_id2shape = {}
if df_info is not None and {"image_file", "height_pixels", "width_pixels"}.issubset(
    df_info.columns
):
    for _, r in df_info.iterrows():
        img_file = str(r["image_file"])
        img_id = os.path.splitext(os.path.basename(img_file))[0]
        _id2shape[img_id] = (int(r["height_pixels"]), int(r["width_pixels"]))

print("Using model checkpoints:")
for p in MODELS:
    print(" ", p, "(exists)" if os.path.exists(p) else "(missing)")




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

s_th = 40  # saturation blancking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def _to_uint8_hwc3(img: np.ndarray) -> np.ndarray:
    """Ensure HWC with 3 channels, uint8."""
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3:
        if img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
            img = np.moveaxis(img[:3], 0, -1)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")

    if img.dtype != np.uint8:
        mx = float(np.max(img)) if img.size else 0.0
        if mx <= 255.0:
            img = img.astype(np.uint8, copy=False)
        else:
            img = (
                (img.astype(np.float32) / (mx + 1e-8) * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
    return img


def _safe_blank_image_for_id(img_id: str) -> np.ndarray:
    h, w = _id2shape.get(img_id, (sz * reduce, sz * reduce))
    return np.zeros((h, w, 3), dtype=np.uint8)


def _read_tiff_no_imagecodecs(path: str, img_id: str = None) -> np.ndarray:
    """
    Loader returns FULL-res when possible (or the highest-res non-JPEG page/level).
    Downstream Dataset downsamples by `reduce`.
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = _to_uint8_hwc3(img)
            return img
    except Exception:
        pass

    try:
        with tiff.TiffFile(path) as tif:
            best_arr = None
            best_area = None

            if tif.series:
                ser = tif.series[0]
                levels = getattr(ser, "levels", None)
                if levels:
                    for lvl in levels:
                        page0 = (
                            lvl.pages[0]
                            if hasattr(lvl, "pages") and len(lvl.pages)
                            else None
                        )
                        if page0 is None:
                            continue
                        try:
                            comp = page0.compression
                            if hasattr(comp, "name") and comp.name.upper() == "JPEG":
                                continue
                        except Exception:
                            pass
                        h, w = lvl.shape[:2]
                        area = int(h) * int(w)
                        if best_area is None or area > best_area:
                            best_area = area
                            best_arr = lvl.asarray()

            if best_arr is None:
                for page in tif.pages:
                    try:
                        comp = page.compression
                        if hasattr(comp, "name") and comp.name.upper() == "JPEG":
                            continue
                    except Exception:
                        pass
                    h, w = page.shape[:2]
                    area = int(h) * int(w)
                    if best_area is None or area > best_area:
                        best_area = area
                        best_arr = page.asarray()

            if best_arr is None:
                raise ValueError(
                    "All pages/levels appear JPEG-compressed; cannot decode without imagecodecs."
                )

            img = _to_uint8_hwc3(best_arr)
            img = img[:, :, ::-1].copy()
            return img
    except Exception:
        return _safe_blank_image_for_id(
            img_id or os.path.splitext(os.path.basename(path))[0]
        )


class HuBMAPDataset(Dataset):
    """
    Core logic preserved: same tiling/padding scheme, same reduce/downsample,
    same blank-tile filtering and normalization.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")

        self.full = _read_tiff_no_imagecodecs(self.path, img_id=idx)
        self.shape = self.full.shape[:2]  # (H, W)

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
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self.full[
            p00:p01, p10:p11, :
        ]

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




## === cell 4
class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, y in iter(self.dl):
                if (y >= 0).sum() > 0:  # exclude empty images
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
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    for i in range(len(py)):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
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


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base=32, out_ch=1):
        super().__init__()
        self.enc1 = nn.Sequential(ConvRelu(in_ch, base), ConvRelu(base, base))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            ConvRelu(base, base * 2), ConvRelu(base * 2, base * 2)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ConvRelu(base * 2, base * 4), ConvRelu(base * 4, base * 4)
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ConvRelu(base * 4, base * 2), ConvRelu(base * 2, base * 2)
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(ConvRelu(base * 2, base), ConvRelu(base, base))
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def _build_model():
    try:
        import segmentation_models_pytorch as smp  # type: ignore

        return smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    except Exception:
        return TinyUNet(in_ch=3, base=32, out_ch=1)


class HuBMAP(nn.Module):
    def __init__(self):
        super(HuBMAP, self).__init__()
        self.cnn_model = _build_model()

    def forward(self, imgs):
        return self.cnn_model(imgs)




## === cell 6
def _remap_state_dict_for_hubmap(model: nn.Module, state_dict: dict) -> dict:
    sd = state_dict
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]

    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        out[nk] = v

    model_keys = set(model.state_dict().keys())

    remapped = {}
    for k, v in out.items():
        candidates = [k]

        if k.startswith("model."):
            candidates.append("cnn_model." + k[len("model.") :])
        candidates.append("cnn_model." + k)
        chosen = None
        for c in candidates:
            if c in model_keys:
                chosen = c
                break
        if chosen is None:
            chosen = candidates[0]
        remapped[chosen] = v

    return remapped


models = []
missing = []
loaded_models = []

for path in MODELS:
    model = HuBMAP()
    if os.path.exists(path):
        raw = torch.load(path, map_location=torch.device("cpu"))
        new_sd = _remap_state_dict_for_hubmap(model, raw)

        missing_keys, unexpected_keys = model.load_state_dict(new_sd, strict=False)

        matched = len(new_sd) - len(unexpected_keys)
        if matched < 50:
            raise RuntimeError(
                f"Checkpoint at {path} appears incompatible (matched params={matched}). "
                f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )

        model.float()
        model.eval()
        model.to(device)
        loaded_models.append(model)
    else:
        missing.append(path)

if len(loaded_models) == 0:
    print(
        "WARNING: No pretrained weights were found/loaded; using a single randomly initialized model."
    )
    model = HuBMAP().float().eval().to(device)
    models = [model]
else:
    models = loaded_models

if len(missing) > 0:
    print("WARNING: Some pretrained weights are missing; using only loaded models.")
    for p in missing[:3]:
        print("  missing:", p)
    if len(missing) > 3:
        print(f"  ... and {len(missing)-3} more")

print(f"Using {len(models)} model(s) for inference.")

gc.collect()



## === cell 7
names, preds = [], []
for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)
    dl = DataLoader(
        ds,
        batch_size=bs,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=0,
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
        ds.pad0 // 2 : -(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
        ds.pad1 // 2 : -(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz,
    ]

    rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
    names.append(idx)
    preds.append(rle)

    del mask, ds, dl
    gc.collect()



## === cell 8
sub = df_sample[["id"]].copy()
pred_map = dict(zip(names, preds))
sub["predicted"] = sub["id"].map(pred_map).fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(f"Wrote submission.csv with {len(sub)} rows")
print("submission.csv saved in current working directory.")
