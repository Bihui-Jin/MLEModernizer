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

0.9270807336267948

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the dependency on `segmentation_models_pytorch` (not available in this environment) and also make the model-weight loading robust (your referenced `../input/b4256shiftfreezebce/*.pth` files are not present). To keep the pipeline running end-to-end and produce a valid submission, I fall back to a simple deterministic baseline that uses the provided anatomical-structure polygons to build a binary mask and then RLE-encode it in the correct Kaggle format. This fixes the runtime errors (missing module, missing weight files, and undefined `names/preds`) while preserving the same RLE encoding/evaluation semantics. The output be a `submission.csv` with `id,predicted` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the crash by making TIFF reading robust: `tifffile.memmap` fails on these files because the image data aren’t memory-mappable, so we fall back to `tifffile.imread` when needed (keeping the same array layout handling). I also correct the anatomical-structure JSON polygon parsing to handle the common nested `coordinates` structure (`[[[x,y],...]]`) so masks aren’t accidentally empty. These changes are execution-unblocking and should move the score up from 0.0 by producing non-empty, correctly sized masks and valid RLE. The submission writing remains the same and always output `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.08161) has done: 'Your current pipeline often falls back to the anatomical-structure mask (or produces masks with wrong orientation), which can yield near-zero Dice; we keep the same tiling/inference/RLE core logic but fix the most likely correctness bugs that prevent any reasonable score. Specifically, we (1) correct the tile-grid generation to use (H,W) consistently (it currently swaps axes), (2) fix BGR/RGB confusion before HSV saturation blanking (OpenCV expects BGR, TIFF is typically RGB), and (3) output the exact submission columns (`id,predicted`) matching `sample_submission.csv` to avoid any format-related scoring issues. These are minimal changes that preserve the model, thresholds, and aggregation, but should move the score upward toward your target by producing non-empty, spatially aligned predictions.'
- What this solution (achieved 0.08161) has done: 'Your current score gap is large (0.08161 vs target 0.92708), so we need a real-but-minimal boost while keeping your tiling+Unet inference+RLE core intact. The main likely score-killer now is RLE orientation: your encoder/decoder mix transposes, and for this competition the canonical encoding is column-major (flatten after transpose), so we standardize all RLE functions to the same convention and remove the extra transpose in `enc2mask`/`mask2enc` to prevent rotated predictions. Second, we fix the TTA averaging bug (you currently divide by `1+len(flips)` but you’re also summing across models, which mis-calibrates probabilities and can push them below TH), by correctly normalizing by the number of model*augment evaluations. These are small, metric-aligned correctness fixes that should materially improve Dice without changing the model, thresholds, tiling, or aggregation logic.'
- What this solution (achieved 0.08161) has done: 'Your current score (0.08161) is far below the target (0.92708), so we need correctness fixes that keep your exact tiling + model inference + thresholding core intact but remove avoidable mask/coordinate mistakes that destroy Dice. I (1) fix the grid step computation so tiles actually overlap by `minoverlap` (your current `nx/ny` formula produces huge steps and leaves most pixels uncovered), (2) prevent uint8 overflow in the “blank tile” check by computing intensity on a wider dtype (otherwise many valid tiles get incorrectly dropped), and (3) switch the aggregation from “any tile positive” to a minimal-vote scheme (>=2 hits) to cut isolated false positives without changing the model or TH. These are small, local changes that should move Dice substantially upward while preserving your architecture, inference loop, and RLE semantics.'
- What this solution (achieved 0.08161) has done: 'Your current score is far below the target, so the most likely “minimal but big impact” fixes are correctness issues that can collapse Dice even when inference runs: (1) ensure we’re predicting at the correct spatial size by reading the image shape from the TIFF itself when it’s not in `HuBMAP-20-dataset_information.csv`, and (2) use the correct ID for this competition (`img` column) in the submission rather than the `id`/`predicted` sample from a different variant of the dataset, which can silently misalign predictions to ground truth. These changes don’t alter your model, tiling, thresholding, voting, or RLE convention; they only fix ID alignment and mask sizing so the same core predictions are scored on the right images. I also keep the existing RLE (column-major via transpose) and your inference/vote logic unchanged to avoid destabilizing behavior. The script still runs end-to-end and always writes `submission.csv`.'
- What this solution (achieved 0.08161) has done: 'We keep your tiling + Unet inference + thresholding + vote aggregation intact, but fix two high-impact correctness issues that can crush Dice: (1) you currently skip inference on “informative” tiles due to an inverted blank-tile condition, and (2) the per-tile intensity threshold (`p_th`) is scaled incorrectly (it explodes for your 1024px tiles, so almost everything is treated as blank). These minimal fixes should substantially increase the amount of valid model predictions, moving the score upward toward your target without changing the model, loss, or postprocessing semantics. We also make the “blank tile” check compare against the correct pixel count for the resized (model-input) tile, so it behaves consistently across `reduce`. The submission format/path stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'We keep your tiling + Unet inference + threshold + voting exactly as-is, but fix two correctness issues that can easily collapse Dice to ~0 even when the code runs. First, the pipeline is currently looking for `.tiff` files in the `test/` folder, but this dataset provides `test/*.json` (no TIFFs), so we derive `TEST_IDS` from JSON filenames and always use the correct `sample_submission.csv` ID column (`id,predicted`). Second, in this competition the test JSONs are *not* ground-truth masks; using them as fallback predictions is invalid and score near-random, so we switch the fallback to a deterministic empty-mask submission (still valid format) when model checkpoints aren’t available. These are minimal, execution-safe changes that should move you upward from “Not yielded/very low” by producing a correctly aligned, valid submission for the real test set, and improve further automatically if the model weight files are actually present.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with always submitting empty masks because this environment’s `test/` folder has JSONs but no TIFFs and your inference is gated on TIFF existence, so the model never runs. I keep your Unet+tiling+TTA+threshold+vote core unchanged, but (1) point `DATA` to the correct test folder that contains TIFFs when available, (2) fix `TEST_IDS` to always come from `sample_submission.csv` (the scoring alignment source), and (3) add a safe TIFF existence check fallback that searches both possible test directories before giving up to empty RLE. These are minimal, correctness-focused path/alignment fixes that should move the score substantially upward toward your target without changing model logic or postprocessing semantics. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that’s perfectly valid CSV-wise but contains empty (or effectively empty) masks for every test image because the code never finds any test TIFFs and therefore always takes the empty-mask fallback. To move the score sharply upward toward the target while keeping the exact inference/tiling/thresholding/RLE logic intact, I (1) ensure `DATA` points to a directory that actually contains the test `.tiff` files by auto-extracting `test.zip` into `/kaggle/working/test/` when needed, and (2) update the candidate search paths to include the extracted working directory. This unblocks real model inference (when checkpoints exist) instead of producing empty RLEs, which is the minimal change likely to fix the “always empty” failure mode. Submission schema and RLE convention remain unchanged.'

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
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import segmentation_models_pytorch as smp  # type: ignore
except Exception as e:
    smp = None
    SMP_IMPORT_ERROR = repr(e)
else:
    SMP_IMPORT_ERROR = None



## === cell 1
sz = 256  # tile size at model input
reduce = 4  # downscale factor applied before inference (tile read is reduce*sz)
TH = 0.3  # threshold for positive predictions


def _ensure_test_tiffs_available():
    zip_candidates = [
        "../input/hubmap-kidney-segmentation/test.zip",
        "../input/test.zip",
        "/kaggle/input/hubmap-kidney-segmentation/test.zip",
        "/kaggle/input/test.zip",
    ]
    work_test_dir = "/kaggle/working/test"
    os.makedirs(work_test_dir, exist_ok=True)

    try:
        if any(
            fn.lower().endswith((".tif", ".tiff")) for fn in os.listdir(work_test_dir)
        ):
            return work_test_dir
    except Exception:
        pass

    zip_path = None
    for zp in zip_candidates:
        if os.path.exists(zp):
            zip_path = zp
            break
    if zip_path is None:
        return work_test_dir

    try:
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [
                m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff"))
            ]
            if len(members) > 0:
                zf.extractall(work_test_dir, members)
    except Exception:
        pass

    return work_test_dir


EXTRACTED_TEST_DIR = _ensure_test_tiffs_available()

DATA_CANDIDATES = [
    EXTRACTED_TEST_DIR,  # extracted TIFFs (if present)
    "/kaggle/working/test",  # explicit
    "../input/hubmap-kidney-segmentation/test/",
    "../input/test/",
    "/kaggle/input/hubmap-kidney-segmentation/test/",
    "/kaggle/input/test/",
]


def _pick_test_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            try:
                if any(fn.lower().endswith((".tif", ".tiff")) for fn in os.listdir(d)):
                    return d
            except Exception:
                pass
    for d in candidates:
        if os.path.isdir(d):
            return d
    return candidates[0]


DATA = _pick_test_dir(DATA_CANDIDATES)
print("Using DATA dir:", DATA)

MODELS = [
    f"../input/b4256shiftfreezebce/efficientnet-b4-256-FOLD-{i}-model.pth"
    for i in range(5)
]
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"  # efficientnet-b4, se_resnext50_32x4d
shift = True
minoverlap = 300

INFO_CSV = "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
info_df = pd.read_csv(INFO_CSV)
info_df["_id"] = info_df["image_file"].astype(str).str.replace(".tiff", "", regex=False)
ID2HW = {
    r["_id"]: (int(r["height_pixels"]), int(r["width_pixels"]))
    for _, r in info_df.iterrows()
}


def _get_test_ids_and_sample_df():
    """
    Change (score/correctness): always derive test ids from sample_submission.csv to guarantee
    ordering/alignment for scoring; directory listing can be incomplete/mismatched.
    """
    if "id" in df_sample.columns and "predicted" in df_sample.columns:
        ids = df_sample["id"].astype(str).tolist()
        return ids, df_sample.copy(), "id", "predicted"

    if "img" in df_sample.columns:
        ids = df_sample["img"].astype(str).tolist()
        return ids, df_sample.copy(), "img", "pixels"

    id_col = df_sample.columns[0]
    pred_col = df_sample.columns[1] if len(df_sample.columns) > 1 else "predicted"
    ids = df_sample[id_col].astype(str).tolist()
    return ids, df_sample.copy(), id_col, pred_col


TEST_IDS, SAMPLE_DF_NORM, SUB_ID_COL, SUB_PRED_COL = _get_test_ids_and_sample_df()
print("Using submission columns:", SUB_ID_COL, SUB_PRED_COL)
print("N test ids:", len(TEST_IDS))




## === cell 2
def enc2mask(encs, shape_hw):
    h, w = shape_hw
    img = np.zeros(h * w, dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, (float, np.floating)) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start : start + length] = 1 + m
    return img.reshape((w, h)).T


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()  # column-major
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
    pixels = img.T.flatten()  # column-major, competition convention
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40  # saturation blanking threshold

p_th_px = 1000 * (sz // 256) ** 2
p_th_sum = int(
    p_th_px * 30
)  # ~avg intensity 10 over 3 channels per pixel; conservative, avoids overskipping


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return Array of size (N,4), where N - number of tiles,
    2nd axis represents slices: x1,x2,y1,y2.

    shape is (H,W).
    """
    h, w = shape
    step = max(window - min_overlap, 1)

    if h <= window:
        x1 = np.array([0], dtype=np.int64)
    else:
        x1 = np.arange(0, h - window + 1, step, dtype=np.int64)
        if x1[-1] != h - window:
            x1 = np.concatenate([x1, np.array([h - window], dtype=np.int64)])
    x2 = (x1 + window).clip(0, h)

    if w <= window:
        y1 = np.array([0], dtype=np.int64)
    else:
        y1 = np.arange(0, w - window + 1, step, dtype=np.int64)
        if y1[-1] != w - window:
            y1 = np.concatenate([y1, np.array([w - window], dtype=np.int64)])
    y2 = (y1 + window).clip(0, w)

    slices = np.zeros((len(x1), len(y1), 4), dtype=np.int64)
    for i in range(len(x1)):
        for j in range(len(y1)):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(len(x1) * len(y1), 4)


def open_tiff_memmap(path):
    try:
        arr = tiff.memmap(path)  # may raise ValueError
    except Exception:
        arr = tiff.imread(path)

    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    return arr




## === cell 4
def _safe_load_models():
    if smp is None:
        return None, f"segmentation_models_pytorch import failed: {SMP_IMPORT_ERROR}"
    existing = [p for p in MODELS if os.path.exists(p)]
    if len(existing) == 0:
        return None, "No model checkpoint files found at expected MODELS paths."
    models_local = []
    last_err = None
    for path in existing:
        try:
            state_dict = torch.load(path, map_location=torch.device("cpu"))
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict, strict=True)
            model.float()
            model.eval()
            model.to(device)
            models_local.append(model)
        except Exception as e:
            last_err = e
    if len(models_local) == 0:
        return None, f"All model loads failed. Last error: {repr(last_err)}"
    return models_local, f"Loaded {len(models_local)} models."


def _get_hw_for_id(idx: str):
    if idx in ID2HW:
        return ID2HW[idx]
    tiff_path = os.path.join(DATA, f"{idx}.tiff")
    if os.path.exists(tiff_path):
        arr = open_tiff_memmap(tiff_path)
        return int(arr.shape[0]), int(arr.shape[1])
    return 1, 1


def _find_tiff_for_id(idx: str):
    exts = [".tiff", ".tif"]
    dirs = [DATA] + [d for d in DATA_CANDIDATES if d != DATA]
    for d in dirs:
        for ext in exts:
            p = os.path.join(d, f"{idx}{ext}")
            if os.path.exists(p):
                return p
    return None


models, model_status = _safe_load_models()
print("Model status:", model_status)

names, preds = [], []

has_any_tiff = any(_find_tiff_for_id(idx) is not None for idx in TEST_IDS)
print("Found any TIFFs for TEST_IDS:", has_any_tiff)

if (models is not None) and shift and has_any_tiff:

    class HuBMAPDataset(Dataset):
        def __init__(self, tiff_path, sz=sz, reduce=reduce):
            self.data = open_tiff_memmap(tiff_path)
            self.shape = (self.data.shape[0], self.data.shape[1])  # H, W
            self.reduce = reduce
            self.sz = reduce * sz  # window size on original resolution
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            patch = self.data[x1:x2, y1:y2]
            if patch.shape[-1] >= 3:
                img_rgb = patch[..., :3].astype(np.uint8, copy=False)
                img = img_rgb[..., ::-1].copy()  # BGR for OpenCV
            else:
                img = np.repeat(patch.astype(np.uint8, copy=False), 3, axis=-1)

            if self.reduce != 1:
                img = cv2.resize(
                    img,
                    (self.sz // reduce, self.sz // reduce),
                    interpolation=cv2.INTER_AREA,
                )

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            _, s, _ = cv2.split(hsv)

            vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

            img_sum = int(img.astype(np.uint32, copy=False).sum())
            sat_px = int((s > s_th).sum())

            if (sat_px <= p_th_px) and (img_sum <= p_th_sum):
                y_out = -1
            else:
                y_out = idx

            return img2tensor((img / 255.0 - mean) / std), vertices, y_out

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
                        n_eval = 0

                        for model in self.models:
                            p = model(x)
                            p = torch.sigmoid(p).detach()
                            py = p if py is None else (py + p)
                            n_eval += 1

                        if self.tta:
                            flips = [[-1], [-2], [-2, -1]]
                            for f in flips:
                                xf = torch.flip(x, f)
                                for model in self.models:
                                    p = model(xf)
                                    p = torch.flip(p, f)
                                    p = torch.sigmoid(p).detach()
                                    py = p if py is None else (py + p)
                                    n_eval += 1

                        py = py / max(n_eval, 1)

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

    VOTE_TH = 2

    for idx in TEST_IDS:
        tiff_path = _find_tiff_for_id(idx)
        if tiff_path is None:
            names.append(idx)
            preds.append("")
            continue

        ds = HuBMAPDataset(tiff_path)
        dl = DataLoader(
            ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
        )
        mp = Model_pred(models, dl, tta=True)

        mask = np.zeros(ds.shape, dtype=np.uint16)
        for pred, vert, _i in iter(mp):
            x1, x2, y1, y2 = vert
            mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint16)

        mask = (mask >= VOTE_TH).astype(np.uint8)
        rle = rle_encode_less_memory(mask)

        names.append(idx)
        preds.append(rle)

        del mask, ds, dl
        gc.collect()

else:
    for idx in TEST_IDS:
        names.append(idx)
        preds.append("")



## === cell 5
sub = pd.DataFrame({SUB_ID_COL: names, SUB_PRED_COL: preds})
sub = SAMPLE_DF_NORM[[SUB_ID_COL]].merge(sub, on=SUB_ID_COL, how="left")
sub[SUB_PRED_COL] = sub[SUB_PRED_COL].fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
print("Unique non-empty masks:", (sub[SUB_PRED_COL].astype(str).str.len() > 0).sum())
print("DATA dir used:", DATA)
print("EXTRACTED_TEST_DIR:", EXTRACTED_TEST_DIR)
