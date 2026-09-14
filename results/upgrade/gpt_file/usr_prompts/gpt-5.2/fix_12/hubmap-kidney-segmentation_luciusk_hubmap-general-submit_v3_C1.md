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

0.9273286710234798

# 6. Current score

0.04763

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed in this environment) by implementing equivalent TIFF window/tiling reads using `tifffile` and NumPy, while keeping the same tiling, padding, resizing, and model inference logic. I also fix compatibility bugs (`np.float` deprecation, missing imports due to earlier crash, and deprecated `F.upsample`) so all cells execute in order. Finally, I make model loading robust to missing weight files (so the notebook still produces a valid 3-row submission) and ensure the submission is aligned to `sample_submission.csv` ids and written as `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most likely from producing effectively empty masks (either because inference was disabled due to missing `segmentation_models_pytorch`/weights, or because the blank-tile filter is mistakenly flagging most tiles as empty). I keep the exact model/tiling/inference logic, but fix two minimal issues that commonly cause “all empty” predictions: (1) use correct RGB↔BGR handling before HSV saturation filtering (your TIFFs are RGB but OpenCV assumed BGR), and (2) write truly empty predictions as NaN (as used in this competition’s common baselines) rather than an empty string, which can be parsed inconsistently by the scorer. These changes should move the score upward toward your target without changing the core approach. The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with “all-empty” masks reaching the scorer, which can happen if the tile blank-filter is too aggressive or if the RLE encoder always emits a non-empty string even for an all-zero mask. I keep your exact tiling/model/inference approach, but (1) relax the blank-tile filter slightly so non-empty tissue tiles aren’t mistakenly skipped (while still skipping true background), and (2) make RLE encoding return `NaN` for truly empty masks (the competition’s scorer/baselines handle this reliably) instead of producing an invalid non-empty run. These two minimal changes should move the score upward toward your target without altering the model or training semantics, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is almost certainly because `can_infer` becomes `False` (missing `segmentation_models_pytorch` and/or missing weight files), which makes the notebook submit all-NaN masks. The smallest score-moving change is to ensure inference can actually run by (1) installing `segmentation_models_pytorch` is not possible here, so instead (2) load weights from a path that actually exists in your provided dataset tree and (3) add a clear hard-fail if no weights are found (so you don’t accidentally submit all-empty again). I keep the exact same tiling, blank-tile filter, model (Unet se_resnext50_32x4d), thresholding, upsampling, and RLE encoding; the only functional change is making `MODELS` resolve to real files so predictions are non-empty. The script still always write `submission.csv`, but now refuse to silently produce an all-empty submission.'
- What this solution (achieved 0.0) has done: 'I (1) remove the hard failure when no `.pth` weights are found and instead fall back to a safe all-empty submission so the notebook always finishes and writes `submission.csv`. I also fix the TIFF loading crash by avoiding full `tifffile.imread()` (which requires `imagecodecs` for JPEG-compressed tiles) and instead read tiles on-demand via `tifffile.TiffFile(...).pages[0].asarray(region=...)`, which works in this environment without extra deps. Finally, I ensure `names` and `preds` always stay aligned even if any single image inference fails (so the DataFrame creation cannot error), and keep the rest of the tiling/inference/RLE logic unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with submitting mostly/all-empty masks (NaNs) because the script never actually performs inference: it can’t import `segmentation_models_pytorch` and also can’t find any `.pth` weights, so `can_infer=False`. The smallest score-moving fix is therefore to (1) make `segmentation_models_pytorch` available via a lightweight pip install (common/allowed in Kaggle notebooks) so the intended architecture can be instantiated, and (2) hard-fail if no weight files are discovered (instead of silently producing an all-empty submission), so you don’t accidentally submit another 0.0. Everything else (tiling, blank-tile filter, thresholding, upsampling, and RLE encoding) is kept the same to preserve core logic and evaluation semantics. The code still writes `submission.csv` when it can run inference end-to-end.'
- What this solution (achieved 0.08161) has done: 'I fix the immediate runtime failure by removing the hard error when no `.pth` weights are found and instead running a deterministic fallback inference path that produces non-empty masks from the provided anatomical-structure JSONs (so you don’t submit all-NaNs/empties and get 0.0). This keeps your core tiling/inference/RLE pipeline intact when weights exist, but guarantees end-to-end execution and a valid `submission.csv` even in this environment where weights are absent. I also fix the submission column name to match `sample_submission.csv` (`predicted`) to avoid format-related scoring issues. The fallback is deliberately conservative (simple filled polygons from the anatomical structure file) to move the score upward from 0.0 without introducing heavy new modeling/training.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target (0.08161 vs 0.9273), and the most likely cause is that your submission is being generated from the anatomical-structure JSON fallback (or from near-empty masks), which is not aligned to the glomerulus target and yields very low Dice. The minimal, score-moving fix is to make inference actually run by (1) finding and loading a real model `.pth` if one exists in the Kaggle input tree, and (2) if no weights exist, **hard-stop before writing a submission** to prevent another low-scoring fallback submission. Additionally, a common low-score bug in this competition is wrong RLE orientation; we keep your existing mask construction but make the RLE encoder consistent with the competition’s “top-to-bottom then left-to-right” convention by removing the extra transpose during encoding (your pipeline currently transposes twice: once in mask building and again in RLE). These changes keep your core tiling/model/thresholding logic intact while addressing the most direct causes of the low Dice.'
- What this solution (achieved 0.04763) has done: 'I remove the hard-stop that prevents the notebook from running when no `.pth` weights are found, because right now it crashes before producing any submission (hence the 0.0 score). To still move the score upward without changing your tiling/inference/RLE core logic, I add a minimal, deterministic fallback that constructs a binary mask directly from each test image’s `*-anatomical-structure.json` polygons and RLE-encodes it in the required order. This keeps your existing model path intact when weights exist, but guarantees non-empty (and likely non-zero Dice) submissions in this environment where weights are absent. I also make the fallback robust to JSON coordinate formats and ensure the submission stays aligned to `sample_submission.csv` with the correct `predicted` column.'
- What this solution (achieved 0.04763) has done: 'Your current score (0.04763) is far below the target (0.9273), and the biggest score-limiting issue in this script is that it likely never runs the intended glomerulus model inference (no usable `.pth` weights found), so it falls back to anatomical-structure polygons which are not the target mask. I keep your tiling/inference/RLE pipeline identical, but make weight discovery actually work for this competition by (1) looking for common HuBMAP kidney baseline weight filenames under `/kaggle/input` and (2) supporting both “raw state_dict” and “checkpoint dict with `state_dict` key” formats (a very common cause of silent mis-loading). If weights still aren’t present, the code still produce a valid submission (as required), but with a clear warning; if they are present, this should move your score sharply upward toward the target without changing the model/thresholding logic.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

warnings.filterwarnings("ignore")

smp = None
try:
    import segmentation_models_pytorch as smp  # noqa: F401
except Exception as e:
    print(
        "segmentation_models_pytorch import failed; attempting pip install. Error:",
        repr(e),
    )
    try:
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "segmentation-models-pytorch",
            ]
        )
        import segmentation_models_pytorch as smp  # noqa: F401

        print("segmentation_models_pytorch installed successfully.")
    except Exception as e2:
        smp = None
        print(
            "WARNING: segmentation_models_pytorch still unavailable after pip install. "
            "Will use local Unet implementation instead.\n",
            repr(e2),
        )



## === cell 1
sz = 256  # the size of tiles (after reduce)
reduce = 4  # reduce the original images by 4 times
TH = 0.45  # threshold for positive predictions
DATA = "../input/hubmap-kidney-segmentation/test/"
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_model_paths():
    roots = [
        "../input",
        "/kaggle/input",
        "../input/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation",
    ]
    exts = (".pth", ".pt", ".bin")
    found = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    found.append(os.path.join(dirpath, fn))

    prefer_tokens = [
        "hubmap",
        "kidney",
        "glomer",
        "unet",
        "se_resnext50",
        "seresnext50",
        "resnext50",
        "fold",
        "best",
        "model",
        "weight",
    ]

    def _score(p):
        bn = os.path.basename(p).lower()
        return sum(tok in bn for tok in prefer_tokens)

    found = sorted(found, key=lambda p: (-_score(p), p))
    return found


MODELS = _find_model_paths()
print("Discovered candidate model files:", len(MODELS))
for p in MODELS[:30]:
    print(" ", p)




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, (float, np.floating)) and np.isnan(enc):
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
    pixels = img.flatten(order="C")
    if pixels.sum() == 0:
        return np.nan
    pixels = pixels.astype(np.uint8, copy=False)
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 30  # was 40
p_th = 800 * (sz // 256) ** 2  # was 1000 * ...


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    """
    Avoid full TIFF decompression (JPEG-compressed TIFFs may need imagecodecs).
    Reads windows via TiffFile.page.asarray(region=...).
    Core tiling/padding/reduce/HSV blank filter unchanged.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Missing tiff: {self.path}")

        self.tf = tiff.TiffFile(self.path)
        self.page = self.tf.pages[0]

        shp = self.page.shape
        if len(shp) == 2:
            self.h, self.w = int(shp[0]), int(shp[1])
            self._tiff_layout = "HW"
        elif len(shp) == 3:
            if shp[0] in (3, 4) and shp[1] > 16 and shp[2] > 16:
                self.h, self.w = int(shp[1]), int(shp[2])
                self._tiff_layout = "CHW"
            else:
                self.h, self.w = int(shp[0]), int(shp[1])
                self._tiff_layout = "HWC"
        else:
            raise ValueError(f"Unexpected tiff shape for {self.path}: {shp}")

        self.shape = (self.h, self.w)
        self.reduce = reduce
        self.sz0 = reduce * sz
        self.sz = self.sz0

        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def _read_region_rgb_uint8(self, x0, y0, x1, y1):
        h = max(0, x1 - x0)
        w = max(0, y1 - y0)
        if h == 0 or w == 0:
            return np.zeros((0, 0, 3), np.uint8)

        arr = self.page.asarray(region=(int(x0), int(y0), int(h), int(w)))

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3:
            if self._tiff_layout == "CHW":
                arr = np.moveaxis(arr[:3], 0, -1)
            else:
                arr = arr[..., :3]
        else:
            raise ValueError(f"Unexpected region array shape: {arr.shape}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0 = -self.pad0 // 2 + n0 * self.sz
        y0 = -self.pad1 // 2 + n1 * self.sz

        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        region = self._read_region_rgb_uint8(p00, p10, p01, p11)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = region

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, s, _ = cv2.split(hsv)
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor(img / 255.0), torch.tensor(-1, dtype=torch.int64)
        else:
            return img2tensor(img / 255.0), torch.tensor(idx, dtype=torch.int64)

    def __del__(self):
        try:
            if hasattr(self, "tf") and self.tf is not None:
                self.tf.close()
        except Exception:
            pass




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
                if (y >= 0).sum() > 0:  # exclude empty tiles
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()

                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        py = p if py is None else (py + p)
                    py /= max(1, len(self.models))

                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    for i in range(len(py)):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 8)
        self.up1 = Up(base * 16, base * 4)
        self.up2 = Up(base * 8, base * 2)
        self.up3 = Up(base * 4, base)
        self.up4 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class HuBMAP(nn.Module):
    def __init__(self):
        super(HuBMAP, self).__init__()
        if smp is not None:
            self.cnn_model = smp.Unet(
                "se_resnext50_32x4d", encoder_weights=None, classes=1
            )
        else:
            self.cnn_model = SimpleUNet(in_channels=3, out_channels=1, base=64)

    def forward(self, imgs):
        return self.cnn_model(imgs)




## === cell 6
models = []
can_infer = True


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj


def _strip_prefix_from_state_dict(sd, prefixes=("module.", "model.", "cnn_model.")):
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for pfx in prefixes:
            if nk.startswith(pfx):
                nk = nk[len(pfx) :]
        out[nk] = v
    return out


if MODELS is None or len(MODELS) == 0:
    can_infer = False
    print(
        "WARNING: No model weights found under /kaggle/input or ../input. "
        "Will use fallback polygon-based mask from *-anatomical-structure.json."
    )
else:
    MODELS = MODELS[:5]
    for path in MODELS:
        raw = torch.load(path, map_location=torch.device("cpu"))
        state_dict = _unwrap_state_dict(raw)
        state_dict = _strip_prefix_from_state_dict(state_dict)

        model = HuBMAP()

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(
                f"WARNING: strict load failed for {os.path.basename(path)}; retrying strict=False. Error: {repr(e)}"
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("  missing_keys:", len(missing), "unexpected_keys:", len(unexpected))

        model.float()
        model.eval()
        model.to(device)
        models.append(model)

    del raw, state_dict
    gc.collect()


def _json_to_polygons(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    polys = []
    if not isinstance(data, list):
        return polys

    for feat in data:
        try:
            coords = feat["geometry"]["coordinates"]
        except Exception:
            continue
        if not coords:
            continue

        rings = []
        if (
            isinstance(coords, list)
            and len(coords) > 0
            and isinstance(coords[0], list)
            and len(coords[0]) > 0
            and isinstance(coords[0][0], (list, tuple))
            and len(coords[0][0]) >= 2
        ):
            if isinstance(coords[0][0][0], (int, float)):
                rings = [coords[0]]
            else:
                rings = coords[0]  # list of rings
        else:
            continue

        for ring in rings:
            if not ring or len(ring) < 3:
                continue
            arr = np.asarray(ring, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            polys.append(arr[:, :2])
    return polys


def fallback_anatomical_rle(idx):
    tiff_path = os.path.join(DATA, idx + ".tiff")
    json_path = os.path.join(DATA, idx + "-anatomical-structure.json")
    if (not os.path.exists(tiff_path)) or (not os.path.exists(json_path)):
        return np.nan

    with tiff.TiffFile(tiff_path) as tf:
        page = tf.pages[0]
        shp = page.shape
        if len(shp) == 2:
            h, w = int(shp[0]), int(shp[1])
        elif len(shp) == 3:
            if shp[0] in (3, 4) and shp[1] > 16 and shp[2] > 16:
                h, w = int(shp[1]), int(shp[2])
            else:
                h, w = int(shp[0]), int(shp[1])
        else:
            return np.nan

    polys = _json_to_polygons(json_path)
    if len(polys) == 0:
        return np.nan

    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = np.round(poly).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 1)

    return rle_encode_less_memory(mask)




## === cell 7
names, preds = [], []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    names.append(idx)

    if can_infer:
        try:
            ds = HuBMAPDataset(idx)
            dl = DataLoader(
                ds,
                batch_size=bs,
                pin_memory=torch.cuda.is_available(),
                shuffle=False,
                num_workers=0,
            )
            mp = Model_pred(models, dl)

            mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.uint8)
            for p, i in iter(mp):
                mask[i.item()] = (p.squeeze(-1) > TH).to(torch.uint8)

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
            preds.append(rle if isinstance(rle, str) and len(rle) > 0 else np.nan)

            del mask, ds, dl
            gc.collect()

        except Exception as e:
            print(f"WARNING: inference failed for id={idx}: {repr(e)}")
            preds.append(np.nan)
    else:
        try:
            rle = fallback_anatomical_rle(idx)
            preds.append(rle if isinstance(rle, str) and len(rle) > 0 else np.nan)
        except Exception as e:
            print(f"WARNING: fallback failed for id={idx}: {repr(e)}")
            preds.append(np.nan)



## === cell 8
df = pd.DataFrame({"id": names, "predicted": preds})
df = df.set_index("id").reindex(df_sample["id"]).reset_index()

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with rows:", len(df), "expected:", len(df_sample))
print("Empty predictions:", int(pd.isna(df["predicted"]).sum()))
print("can_infer:", can_infer, "num_models:", len(models))
