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

0.9446739630779464

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by switching TIFF tile reading to `tifffile.memmap`, which keeps the same tiling/inference logic but avoids the import/runtime crash. I also fix a couple of compatibility bugs that currently prevent end-to-end execution: `np.float` deprecation, missing `torch` due to the failed first cell, and use of `F.upsample` (deprecated) by replacing with `F.interpolate` (same behavior). Finally, I make the script robust to missing external model weight paths by falling back to a valid “empty mask” submission (still producing a correct CSV), while preserving the original ensemble inference path when weights exist.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid submission format (this competition expects the column name `predicted`, but the RLE must be `NaN` for empty masks; submitting empty strings often scores as all-wrong). I make the smallest change to ensure “no-mask” cases are encoded as `NaN` exactly like the competition’s baseline, while keeping your inference logic/thresholding untouched when model weights exist. I also make one safety fix to avoid producing an invalid all-zero RLE string (`""`) when a mask is empty, again matching the expected semantics. These changes should move the score up from 0.0 without altering your model architecture or tiling approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the “fallback” path being taken (no model weights found / `segmentation_models_pytorch` missing), which yields an all-empty submission that score ~0 on Dice. The smallest legitimate improvement toward your target is to ensure the ensemble inference path actually runs by (1) loading `segmentation_models_pytorch` from the correct Kaggle dataset path (instead of a non-existent `../input/segmentation-models-pytorch-install`), and (2) pointing `MODELS` at a real weights location if present; otherwise we still produce a valid empty-mask submission. I’m keeping the exact tiling, model architecture, thresholding, and RLE logic unchanged; the only changes are to make your intended models discoverable and loadable so predictions are non-empty. I also add a tiny safety print to confirm whether models were found/used so you can verify why a run would still fall back to empty masks.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with your “use_models” branch never running (missing `segmentation_models_pytorch` import path and/or missing weight files), causing an all-empty submission. I make the smallest execution-relevant change: broaden the search paths for `segmentation_models_pytorch` and for the model weight files under `../input/` so your existing ensemble inference can actually run when the datasets are present, while preserving the exact architecture, tiling, thresholding, and RLE logic. I also add a hard check that at least one non-empty mask is produced when models are supposedly enabled, so you don’t silently submit all-NaN again. If weights truly are not available, the code still produce a valid `submission.csv` (but that likely remain near-0 Dice).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from submitting all-empty masks because the model branch is never reached (missing `segmentation_models_pytorch` at runtime and/or missing weights), not from the RLE code itself. To move the score upward toward your target while keeping the same inference logic, I’m making the smallest change that actually enables your intended ensemble: install a compatible `segmentation_models_pytorch` package at runtime (only if import fails) and then re-check `use_models`. I’m also broadening weight discovery slightly (still only under `../input`) and adding a strict sanity check that prevents silently writing an all-NaN submission when models were expected to run. Everything else—tiling, normalization, model architecture (Unet + efficientnet-b4), thresholding, and RLE encoding—remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the code falling back to the all-NaN (empty mask) submission because it never finds the model weight files and/or can’t import `segmentation_models_pytorch` in the Kaggle runtime. To move the score upward toward your target while preserving the exact inference/tiling/threshold/RLE logic, I only (1) broaden `DATA` to the actually-present test folder path, (2) broaden and validate model-weight discovery under the real `/kaggle/input/...` locations, and (3) make `_ensure_smp_importable()` also search the local competition dataset folder you already have. If models still can’t be loaded, the script still produce a valid `submission.csv`, but now it’s much more likely to execute the intended model branch and generate non-empty masks (hence non-zero Dice).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing an all-empty (all-NaN) submission because the model inference branch is never entered (weights not found and/or test TIFFs not located). I make minimal changes that only improve “discoverability”: broaden the test TIFF directory detection to also handle `/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test`, and broaden the weight search to pick up `.pth` files anywhere under `/kaggle/input/**` while keeping the same filenames and loading logic. I also add a small safety check that confirms the chosen `DATA` directory actually contains `.tiff` files before running, so we don’t silently fall back. Model architecture, tiling, thresholding, and RLE encoding are unchanged; this is purely to ensure your intended ensemble path actually executes and produces non-empty masks.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import warnings
import subprocess

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

warnings.filterwarnings("ignore")


def _ensure_smp_importable():
    try:
        import segmentation_models_pytorch as smp  # noqa: F401

        return True
    except Exception:
        pass

    _smp_candidates = [
        "/kaggle/input/segmentation-models-pytorch/segmentation_models_pytorch",
        "/kaggle/input/segmentation-models-pytorch-install/segmentation_models_pytorch",
        "/kaggle/input/segmentation-models-pytorch",
        "/kaggle/input/segmentation-models-pytorch-install",
        "../input/segmentation-models-pytorch/segmentation_models_pytorch",
        "../input/segmentation-models-pytorch-install/segmentation_models_pytorch",
        "../input/segmentation-models-pytorch",
        "../input/segmentation-models-pytorch-install",
        "/kaggle/input/hubmap-kidney-segmentation/segmentation_models_pytorch",
        "../input/hubmap-kidney-segmentation/segmentation_models_pytorch",
    ]
    for p in _smp_candidates:
        if os.path.exists(p):
            sys.path.append(
                os.path.dirname(p) if p.endswith("segmentation_models_pytorch") else p
            )

    try:
        import segmentation_models_pytorch as smp  # noqa: F401

        return True
    except Exception:
        pass

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "segmentation-models-pytorch==0.3.3",
            ]
        )
        import segmentation_models_pytorch as smp  # noqa: F401

        return True
    except Exception:
        return False


_smp_ok = _ensure_smp_importable()
try:
    import segmentation_models_pytorch as smp  # noqa: E402
except Exception:
    smp = None



## === cell 1
sz = 256  # tile size at network input resolution
reduce = 4  # downscale factor used by the pipeline
TH = 0.52  # threshold for positive predictions

_DATA_CANDIDATES = [
    "/kaggle/input/hubmap-kidney-segmentation/test/",
    "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
    "../input/hubmap-kidney-segmentation/test/",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
]
DATA = None
for p in _DATA_CANDIDATES:
    if os.path.isdir(p):
        DATA = p
        break
if DATA is None:
    DATA = "/kaggle/input/hubmap-kidney-segmentation/test/"


def _find_model_weights():
    pat = "efficientnet-b4-unet-BCELoss-256-FOLD-{}-model.pth"
    needed = [pat.format(i) for i in range(5)]

    explicit_roots = [
        "/kaggle/input/skfoldalldata",
        "/kaggle/input/hubmap-models",
        "/kaggle/input/hubmap-weights",
        "/kaggle/input/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
        "../input/skfoldalldata",
        "../input/hubmap-models",
        "../input/hubmap-weights",
        "../input/hubmap-kidney-segmentation",
        "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
    ]
    for r in explicit_roots:
        cand = [os.path.join(r, f) for f in needed]
        if all(os.path.exists(p) for p in cand):
            return cand

    hits = {}
    for base in ["/kaggle/input", "../input"]:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if fn in needed:
                    hits[fn] = os.path.join(root, fn)
            if len(hits) == len(needed):
                break
        if len(hits) == len(needed):
            break

    if len(hits) == len(needed):
        return [hits[pat.format(i)] for i in range(5)]

    return [os.path.join("../input/skfoldalldata", pat.format(i)) for i in range(5)]


MODELS = _find_model_weights()

df_sample = pd.read_csv(
    "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
)

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
        s = enc.split()
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
    if img is None or img.size == 0 or img.max() == 0:
        return np.nan
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
p_th = 1000 * (sz // 256) ** 2  # min pixel threshold


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


def open_tiff_memmap(path):
    arr = tiff.memmap(path)
    return arr


def read_window(arr, x1, x2, y1, y2):
    if arr.ndim == 3 and arr.shape[-1] == 3:
        patch = np.asarray(arr[x1:x2, y1:y2, :])
    elif arr.ndim == 3 and arr.shape[0] == 3:
        patch = np.moveaxis(np.asarray(arr[:, x1:x2, y1:y2]), 0, -1)
    else:
        raise ValueError(f"Unexpected TIFF shape {arr.shape} for windowed RGB read")
    return patch




## === cell 4
use_models = (smp is not None) and all(os.path.exists(p) for p in MODELS)

print("segmentation_models_pytorch available:", smp is not None)
print("Test data dir:", DATA, "exists:", os.path.isdir(DATA))
if os.path.isdir(DATA):
    _tiff_ct = sum(1 for f in os.listdir(DATA) if f.endswith(".tiff"))
else:
    _tiff_ct = 0
print("TIFF files in DATA:", _tiff_ct)
print("Found all model weight files:", all(os.path.exists(p) for p in MODELS))
if not all(os.path.exists(p) for p in MODELS):
    missing = [p for p in MODELS if not os.path.exists(p)]
    print("Missing weights (showing up to 5):", missing[:5])

if _tiff_ct == 0:
    use_models = False

if use_models and shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.path = os.path.join(DATA, idx + ".tiff")
            self.data = open_tiff_memmap(self.path)

            if self.data.ndim == 3 and self.data.shape[-1] == 3:
                self.shape = self.data.shape[:2]
            elif self.data.ndim == 3 and self.data.shape[0] == 3:
                self.shape = (self.data.shape[1], self.data.shape[2])
            else:
                raise ValueError(f"Unexpected TIFF shape {self.data.shape}")

            self.reduce = reduce
            self.sz = reduce * sz
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            img = read_window(self.data, x1, x2, y1, y2)

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

    names, preds = [], []
    non_empty_count = 0

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
        if mask.max() > 0:
            non_empty_count += 1

        rle = rle_encode_less_memory(mask)
        names.append(idx)
        preds.append(rle)

        del mask, ds, dl
        gc.collect()

    if non_empty_count == 0:
        raise RuntimeError(
            "Models were enabled and ran, but all predicted masks are empty. "
            "This would likely score ~0; investigate weights/normalization/TH."
        )

else:
    names = df_sample["id"].tolist()
    preds = [np.nan] * len(names)


df = pd.DataFrame({"id": names, "predicted": preds})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows")
print("Non-NaN predictions:", int(pd.Series(preds).notna().sum()))
