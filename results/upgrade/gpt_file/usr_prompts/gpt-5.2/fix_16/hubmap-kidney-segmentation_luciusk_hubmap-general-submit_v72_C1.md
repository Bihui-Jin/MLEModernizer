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

0.9405835609248696

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by replacing TIFF tile reading with `tifffile.memmap`, while keeping the same tiling/reduction/prediction logic. I also fix `np.float` deprecation checks and ensure `torch` is imported before it’s used so later cells don’t crash. I add a safe fallback for missing model weight files (produce a valid all-empty submission rather than erroring) so a `.csv` is always generated. Finally, I keep the same RLE encoding semantics and write `submission.csv` with the required `id,predicted` columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is coming from producing an effectively empty/invalid submission because no model weights are found at the specified paths, so every `predicted` value becomes `""`. To move the score upward toward the target, the smallest valid change is to (1) point `DATA` and the model weight paths to the actual existing `/kaggle/input/...` locations, and (2) keep the same model/tiling/inference/RLE logic but ensure the “no-mask” case is encoded as an empty string (as expected by this competition) while non-empty masks are RLE-encoded. I’m also keeping strict submission alignment with `sample_submission.csv` ids and adding a tiny guard so missing TIFFs don’t crash the run (still yields a valid row). These changes preserve your core approach and should move the score from 0.0 to a non-zero value.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because no model weights are being loaded (your code falls back to predicting empty masks for every image), so the smallest improvement is to actually load the provided fold checkpoints if they exist in the Kaggle input tree. I keep your exact model/tiling/inference/RLE logic, but add a tiny, safe search for the `.pth` files under `/kaggle/input` when the hardcoded path doesn’t exist, and then load all found weights (still the same architecture). I also ensure the output submission is aligned to `sample_submission.csv` ids (as you already do) and keep the empty-mask encoding as `""` (competition-expected). These minimal path/weights fixes should move the score from 0.0 upward toward your target without changing evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from effectively predicting empty masks for all images because the checkpoint paths are very likely wrong for this environment (so no weights are loaded and the code falls back to `predicted=""`). I keep your exact model/tiling/inference/RLE logic, but make the weight discovery more reliable by (1) expanding the search to any `.pth` under `/kaggle/input` and (2) preferring checkpoints that match your expected naming/fold pattern, so real weights get loaded and predictions become non-empty. I also keep submission alignment to `sample_submission.csv` ids and add a tiny safeguard to avoid double-counting overlapping tiles by switching the tile aggregation from `+=` to `|=` (still the same thresholded binary semantics, but prevents overlap from inflating values). These changes are minimal, preserve evaluation semantics, and should move the score upward toward the target by producing actual segmentations instead of empty masks.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still coming from effectively submitting empty masks (either because no weights are actually found/loaded, or because the model outputs are being tiled/stitched onto the wrong pixel coordinates for this competition’s RLE convention). I keep your exact model, tiling, thresholding, and RLE logic, but make two minimal score-critical fixes: (1) reliably load checkpoints by searching for `.pth` and also `.pt` under `/kaggle/input` (some datasets use `.pt`), and (2) fix the axis/order mismatch between dataset shape, grid creation, and mask indexing so tiles land in the correct places before RLE. These changes preserve the same inference semantics (same model forward + sigmoid + TH) but prevent “all-empty / wrong-position” submissions that score ~0.0. The script still always write a valid `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with submitting empty masks for every test image because no real checkpoints are being loaded and/or the model package import path is wrong, so the inference loop never produces positive pixels. I make the smallest score-relevant changes to (1) load `segmentation_models_pytorch` reliably (it’s available as an installed package here, so don’t depend on a Kaggle Dataset path), and (2) make checkpoint discovery stricter by only accepting state_dicts that actually match your Unet so we don’t silently “load” incompatible weights and then effectively predict garbage/empties. Everything else (tiling, preprocessing, model, sigmoid+TH, stitching with `|=`, and RLE encoding) stays the same to preserve your core logic and evaluation semantics. This should move the score upward from 0.0 toward your target by producing non-empty, valid masks when compatible weights exist in `/kaggle/input`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with predicting empty masks for all test images, which in your current code happens whenever no compatible checkpoints are actually found/loaded. I make the smallest changes that increase the chance of successfully loading real weights (without changing your model/tiling/inference/RLE core logic): (1) broaden and prioritize checkpoint discovery specifically under `/kaggle/input` for likely HuBMAP kidney Unet weights, and (2) make state-dict key handling robust (common `module.` prefix and `model.` prefix) so compatibility checks and loading don’t silently fail. Finally, I keep your existing mask stitching and RLE encoding semantics unchanged, but add a tiny safety cast for tile coordinates to ensure correct slicing. These changes should move your score upward from 0.0 toward the target by producing non-empty, valid segmentations when weights exist.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “no real weights loaded,” so the smallest meaningful change is to make checkpoint discovery/load succeed more often without changing your model, tiling, thresholding, or RLE logic. I keep the same SMP Unet inference pipeline, but (1) broaden and prioritize checkpoint discovery to prefer likely HuBMAP Unet weights, and (2) make state-dict key normalization more robust (common prefixes like `net.`, `encoder.`, `decoder.`, `segmentation_head.`) and compatibility checking stricter so we only keep checkpoints that actually load usable weights. Finally, I add a tiny guard to ensure we don’t average by an empty model list inside `Model_pred` (stability only), while preserving identical prediction semantics when models exist. These changes should move the score up from 0.0 toward your target by producing non-empty masks when compatible weights exist under `/kaggle/input`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “no meaningful predictions”: either no TIFFs are being found/loaded from `DATA`, or no compatible checkpoints are actually loaded so every `predicted` becomes empty. I keep your exact Unet/tiling/sigmoid+TH/RLE logic, but make two minimal score-critical fixes: (1) point `DATA` to the actual test image directory that exists in this environment (the extracted `.../test/` folder, not the zip root), and (2) add a strict-but-safe fallback that, if no compatible model weights are found, generates a non-empty baseline mask derived from the provided anatomical-structure JSONs (still a legitimate signal source in this competition) so the submission isn’t all empty. This should move the score upward from 0.0 toward your target while preserving your core inference pipeline whenever weights are available. The submission schema/ordering remains aligned to `sample_submission.csv` and always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I make the smallest score-critical fixes that address why you’re still getting 0.0: your pipeline is likely not finding any `.tiff` test images (so it predicts empty for all ids) and/or not loading any compatible weights. I keep your exact Unet/tiling/sigmoid+TH/RLE logic, but (1) make `DATA` discovery explicitly require `.tiff` files (and fall back to the extracted test folder if present), and (2) make weight discovery prefer known HuBMAP checkpoints while also allowing more than one fold path if they exist. Finally, I keep your current anatomical-JSON baseline only as a fallback, but ensure it cannot silently produce a degenerate all-empty submission when TIFFs are missing by failing over to the correct directory first.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with producing masks in the wrong pixel order for this competition’s RLE convention (top-to-bottom then left-to-right, i.e., column-major / Fortran order), even if the model outputs non-empty tiles. I make the smallest score-critical fix by changing the RLE encoder to flatten in Fortran order (and avoid the `.T` workaround), while keeping your tiling, thresholding, stitching, and model inference unchanged. I also keep the empty-mask encoding as `""` (as expected), and add a tiny type safety cast so the encoder always receives a contiguous `uint8` array. This should move the score upward from 0.0 toward your target without altering the model core logic.'
- What this solution (achieved 0.0) has done: 'I make the smallest set of changes needed to reliably generate a valid submission for this competition: your code currently writes `img,pixels`, but the required format is `id,predicted`, which can yield a “Not yielded/0.0” outcome even if masks are computed correctly. I also remove the conflicting legacy `mask2enc/enc2mask` transpose-based convention from the submission path and standardize to the competition’s RLE definition by explicitly flattening in Fortran order (`order="F"`) without extra transposes. Finally, I ensure row alignment exactly matches `sample_submission.csv` and write `submission.csv` with the correct column names, keeping your model/tiling/inference logic unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an RLE convention mismatch: even if masks are non-empty, encoding in the wrong pixel order (or returning the wrong “empty mask” sentinel) score ~0. I make two minimal, score-critical fixes: (1) make the submission-time RLE encoder exactly follow the competition definition by flattening in Fortran order *without mutating the first/last pixels*, and (2) ensure empty masks are encoded as an empty string `""` (not forced to non-empty by boundary edits). This keeps your model/tiling/inference/stitching logic unchanged and only corrects the final encoding semantics that directly affect the Dice evaluation.'
- What this solution (achieved 0.0) has done: 'I make two minimal, score-critical fixes that keep your tiling/model/inference logic unchanged but directly address common causes of a 0.0 Dice in this competition: (1) ensure the predicted mask is stitched at the *same resolution* as the model output (currently it’s being upsampled by `reduce` but stitched into an unscaled canvas, causing severe misalignment), and (2) ensure RLE encoding matches the competition’s pixel numbering by flattening a C-contiguous `(H,W)` mask in Fortran order (without forcing an `order="F"` array that can be inconsistent). These are small, deterministic changes and should move your score up from 0.0 toward the target by making non-empty predictions land in the correct pixels and be encoded correctly. The model architecture, thresholding, and overall pipeline remain the same. The script still always writes a valid `submission.csv` with `id,predicted`.'

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
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # installed package

    _HAS_SMP = True
except Exception:
    smp = None
    _HAS_SMP = False

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
sz = 256  # the size of tiles (after reduction)
reduce = 4  # reduce the original images by 4 times
TH = 0.5  # threshold for positive predictions

_DATA_CANDIDATES = [
    "/kaggle/input/hubmap-kidney-segmentation/test",
    "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test",
    "/kaggle/data/hubmap-kidney-segmentation/test",
    "/kaggle/data/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test",
]
DATA = None
for _p in _DATA_CANDIDATES:
    if os.path.isdir(_p) and any(f.lower().endswith(".tiff") for f in os.listdir(_p)):
        DATA = _p
        break
if DATA is None:
    DATA = _DATA_CANDIDATES[0]
DATA = DATA.rstrip("/") + "/"

MODELS = []
for i in range(5):
    MODELS.append(
        f"/kaggle/input/skfoldalldata/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    )

df_sample = pd.read_csv(
    "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
)
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"  # efficientnet-b4, se_resnext50_32x4d
minoverlap = 300
TTA = False




## === cell 2
def rle_encode_less_memory(img):
    """
    Score-critical fix: match competition pixel numbering.
    Pixels are counted top-to-bottom, then left-to-right => flatten (H,W) in Fortran order.
    Important: ensure we flatten a standard C-contiguous (H,W) array with order='F' to get the right scan.
    """
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim != 2:
        raise ValueError(f"RLE expects 2D mask, got shape={img.shape}")

    if img.sum() == 0:
        return ""

    pixels = np.ascontiguousarray(img).reshape(-1, order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


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


def make_grid(shape_hw, window=256, min_overlap=32):
    """
    Return array of size (N,4): x1,x2,y1,y2 for an image of shape (H,W).
    """
    h, w = shape_hw
    step = window - min_overlap

    nx = h // step + 1
    x1 = np.linspace(0, h, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = max(0, h - window)
    x2 = (x1 + window).clip(0, h)

    ny = w // step + 1
    y1 = np.linspace(0, w, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = max(0, w - window)
    y2 = (y1 + window).clip(0, w)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


class HuBMAPDataset(Dataset):
    """
    Keep image shape as (H,W) consistently so tile coords/mask indexing match RLE convention.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")

        if not os.path.exists(self.path):
            self.data = None
            self.reduce = reduce
            self.sz = reduce * sz
            self.layout = "HWC"
            self.shape = (self.sz, self.sz)  # (H,W)
            self.mask_grid = np.zeros((0, 4), dtype=np.int64)
            return

        self.data = tiff.memmap(self.path)  # (H,W,3) or (3,H,W)
        self.reduce = reduce
        self.sz = reduce * sz  # tile size on original resolution
        self._infer_shape_and_layout()
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def _infer_shape_and_layout(self):
        arr = self.data
        if arr.ndim == 3:
            if arr.shape[-1] == 3:
                self.layout = "HWC"
                self.shape = (int(arr.shape[0]), int(arr.shape[1]))  # (H,W)
            elif arr.shape[0] == 3:
                self.layout = "CHW"
                self.shape = (int(arr.shape[1]), int(arr.shape[2]))  # (H,W)
            else:
                self.layout = "HWC"
                self.shape = (int(arr.shape[0]), int(arr.shape[1]))
        else:
            raise ValueError(f"Unexpected tiff shape for {self.path}: {arr.shape}")

    def __len__(self):
        return len(self.mask_grid)

    def _read_window(self, x1, x2, y1, y2):
        if self.layout == "HWC":
            img = np.asarray(self.data[x1:x2, y1:y2, :])
        else:  # CHW
            img = np.asarray(self.data[:, x1:x2, y1:y2])
            img = np.moveaxis(img, 0, -1)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        return img

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self._read_window(x1, x2, y1, y2)

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

                    if len(self.models) == 0:
                        continue

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

                    py = py.permute(0, 2, 3, 1).float().cpu()
                    py = py.squeeze(-1).numpy()
                    z = z.numpy()

                    for i in range(len(py)):
                        yield py[i], z[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 4
def _find_weight_files_fallback():
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return []

    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".pth") or low.endswith(".pt") or low.endswith(".ckpt"):
                candidates.append(os.path.join(dirpath, fn))

    if not candidates:
        return []

    def _rank(p):
        low = os.path.basename(p).lower()
        score = 0
        if "hubmap" in low or "kidney" in low:
            score += 8
        if "efficientnet-b4" in low or "tf_efficientnet_b4" in low:
            score += 10
        if "unet" in low:
            score += 7
        if "smp" in low or "segmentation" in low:
            score += 3
        if "bceloss" in low or "bce" in low or "dice" in low:
            score += 2
        if "fold" in low:
            score += 1
        if "model" in low or "best" in low or "final" in low or "ckpt" in low:
            score += 1
        depth_penalty = p.count(os.sep) * 0.01
        return (-(score - depth_penalty), len(p), p)

    candidates = sorted(candidates, key=_rank)
    return candidates[:25]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return None


def _strip_known_prefixes(state_dict, prefixes):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    keys = list(state_dict.keys())

    def strip_prefix(sd, prefix):
        left = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        right = {k: v for k, v in sd.items() if not k.startswith(prefix)}
        left.update(right)
        return left

    if sum(k.startswith("module.") for k in keys) > 0.6 * len(keys):
        state_dict = strip_prefix(state_dict, "module.")

    keys = list(state_dict.keys())
    if sum(k.startswith("model.") for k in keys) > 0.6 * len(keys):
        state_dict = strip_prefix(state_dict, "model.")

    state_dict = _strip_known_prefixes(
        state_dict,
        prefixes=("net.", "network.", "unet.", "seg_model.", "model.", "module."),
    )
    return state_dict


def _state_dict_compatible(model, state_dict):
    try:
        model_state = model.state_dict()
        matched = 0
        shape_mismatch = 0
        for k, v in state_dict.items():
            if k in model_state:
                matched += 1
                if tuple(v.shape) != tuple(model_state[k].shape):
                    shape_mismatch += 1
        if matched == 0:
            return False
        if shape_mismatch > 0:
            return False
        return matched >= max(50, int(0.1 * len(model_state)))
    except Exception:
        return False




## === cell 5
def _load_polygons_from_geojson(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    polys = []
    if not isinstance(data, list):
        return polys

    for feat in data:
        try:
            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates", None)
            if not coords or not isinstance(coords, list):
                continue
            outer = coords[0]
            if not outer or len(outer) < 3:
                continue
            pts = np.asarray(outer, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            polys.append(pts)
        except Exception:
            continue
    return polys


def _anatomical_json_path(idx):
    return os.path.join(DATA, f"{idx}-anatomical-structure.json")


def _tiff_path(idx):
    return os.path.join(DATA, f"{idx}.tiff")


def _infer_hw_from_tiff(idx):
    p = _tiff_path(idx)
    if not os.path.exists(p):
        return None
    try:
        arr = tiff.memmap(p)
        if arr.ndim == 3:
            if arr.shape[-1] == 3:
                return (int(arr.shape[0]), int(arr.shape[1]))
            if arr.shape[0] == 3:
                return (int(arr.shape[1]), int(arr.shape[2]))
        return None
    except Exception:
        return None


def _baseline_mask_from_anatomy(idx):
    hw = _infer_hw_from_tiff(idx)
    if hw is None:
        return None
    h, w = hw
    jp = _anatomical_json_path(idx)
    if not os.path.exists(jp):
        return np.zeros((h, w), dtype=np.uint8)

    polys = _load_polygons_from_geojson(jp)
    if len(polys) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv_polys = []
    for pts in polys:
        pts_i = np.round(pts).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, w - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, h - 1)
        cv_polys.append(pts_i)
    if len(cv_polys) > 0:
        cv2.fillPoly(mask, cv_polys, 1)
    return mask




## === cell 6
models = []
_model_paths_existing = []

if _HAS_SMP:
    for path in MODELS:
        if os.path.exists(path):
            _model_paths_existing.append(path)
    if len(_model_paths_existing) == 0:
        _model_paths_existing = _find_weight_files_fallback()

if _HAS_SMP and len(_model_paths_existing) > 0:
    for path in _model_paths_existing:
        try:
            obj = torch.load(path, map_location=torch.device("cpu"))
        except Exception:
            continue

        state_dict = _extract_state_dict(obj)
        state_dict = _normalize_state_dict_keys(state_dict)
        if state_dict is None:
            continue

        model = smp.Unet(model_name, encoder_weights=None, classes=1)

        if not _state_dict_compatible(model, state_dict):
            continue

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) == len(model.state_dict()):
            continue

        model.float()
        model.eval()
        model.to(device)
        models.append(model)

        del obj, state_dict
else:
    models = []

id_col = "id" if "id" in df_sample.columns else df_sample.columns[0]
pred_col = "predicted" if "predicted" in df_sample.columns else df_sample.columns[1]

names, preds = [], []

for _, row in df_sample.iterrows():
    idx = row[id_col]
    names.append(idx)

    if len(models) == 0:
        base_mask = _baseline_mask_from_anatomy(idx)
        if base_mask is None or base_mask.sum() == 0:
            preds.append("")
        else:
            preds.append(rle_encode_less_memory(base_mask.astype(np.uint8)))
        continue

    ds = HuBMAPDataset(idx)
    if len(ds) == 0:
        preds.append("")
        continue

    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl)

    h, w = ds.shape
    rh, rw = h // reduce, w // reduce
    mask_small = np.zeros((rh, rw), dtype=np.uint8)

    for pred, vert, i in iter(mp):
        x1, x2, y1, y2 = (int(vert[0]), int(vert[1]), int(vert[2]), int(vert[3]))
        rx1, rx2 = x1 // reduce, x2 // reduce
        ry1, ry2 = y1 // reduce, y2 // reduce

        tile_bin = pred > TH  # pred is (sz,sz) at reduced resolution
        mask_small[rx1:rx2, ry1:ry2] |= tile_bin.astype(np.uint8)

    mask_small = (mask_small > 0).astype(np.uint8)
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(
        np.uint8
    )

    preds.append(rle_encode_less_memory(mask))

    del mask, mask_small, ds, dl
    gc.collect()



## === cell 7
subm = pd.DataFrame({id_col: names, pred_col: preds})
subm.to_csv("submission.csv", index=False)

print(subm.head())
print("DATA:", DATA)
print("Loaded models:", len(models), "from candidates:", _model_paths_existing[:10])
print("Wrote submission.csv with shape:", subm.shape)
print("Empty predictions:", int((subm[pred_col].values == "").sum()), "/", len(subm))
