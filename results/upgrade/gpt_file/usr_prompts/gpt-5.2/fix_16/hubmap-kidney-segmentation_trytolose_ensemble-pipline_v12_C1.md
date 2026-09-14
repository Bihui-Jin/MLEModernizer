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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
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

0.9470332366877252

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by reading the `.tiff` images with OpenCV and replicating the same sliding-window + resizing logic, which fixes the immediate runtime error and keeps the inference pipeline intact. I also fix the broken cell ordering/NameErrors by consolidating global constants and required imports into the first cell so later cells can run unmodified. To ensure a valid submission is always produced, the code fill missing predictions with empty strings and always write `submission.csv` with the required `id,predicted` columns. I not change thresholds/model blending logic (score semantics) beyond what’s necessary to make the code run end-to-end.'
- What this solution (achieved 0.0) has done: 'The crash is from OpenCV refusing to load extremely large TIFFs due to its built-in pixel limit, so I switch TIFF reading to a safe tiled downsampling reader that uses `cv2.imdecode` on the embedded overview/IFD pages (falling back to normal `imread` when possible). This keeps the same sliding-window inference logic, but makes image loading work within OpenCV constraints and Kaggle memory/time limits. I also fix a couple of minor dataset issues (typoed variable name and ensuring `not_background`/`crop_names` collate cleanly) so batching doesn’t break. Finally, the script always write `submission.csv` with `id,predicted` and empty RLEs if a file can’t be read, ensuring a valid submission is produced end-to-end.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most plausibly coming from generating empty/near-empty masks due to (1) the current TIFF “safe reader” effectively returning a downsampled image without later upscaling predictions back to the original size, and/or (2) `cv2.imdecode` not actually selecting TIFF pages (the `page` loop is a no-op), so the image read path is unreliable. I make the TIFF loader deterministic by using OpenCV’s multi-page TIFF support (`cv2.imreadmulti`) and explicitly selecting the highest-resolution page that’s still within a pixel budget, so the mask is produced in a consistent, correct coordinate space. I also ensure that if a downsampled page is used, we scale crop coordinates and finally upsample the predicted mask back to the original TIFF size before RLE encoding (evaluation requires original resolution). These are minimal changes that keep the same model ensemble, sliding-window inference, thresholding, and RLE logic, but fix the resolution/alignment issue that can collapse Dice to ~0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 Dice is most consistent with the submission being effectively empty/near-empty due to incorrect background skipping logic and/or incorrect crop placement during stitching. I make two minimal, score-relevant fixes: (1) correct `_check_background` so it returns `True` for true background (so `SKIP_BACKGROUND` actually skips background instead of skipping tissue), and (2) fix crop name parsing to reliably recover `(x,y)` so predicted tiles are written into the right location. These preserve your exact model ensemble, resizing, thresholding, voting, and RLE encoding semantics, but prevent the pipeline from discarding most tiles or misplacing them, which should move the score upward toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the pipeline producing empty (or almost empty) masks because no model weights are actually being loaded at runtime, so `pred_total` becomes `None` and every tile is skipped. I keep your exact inference/tiling/threshold/RLE logic, but make weight loading robust by (a) searching for the extracted `.pth` files recursively (since tar extraction often adds subfolders) and (b) gracefully skipping the `hubmap6` models if those files aren’t present, instead of silently running with zero models. This minimal fix should immediately move Dice upward toward the target because it enables the intended ensemble to actually run and generate non-empty predictions. I also force `pin_memory` only when CUDA is available to avoid CPU-only slowdown/oddities, without changing semantics.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime error by ensuring the required weight tar files are actually found and extracted from the Kaggle input area (your current `folds_dir` points to a non-existent dataset, so no weights load and the script aborts). I keep the model definitions, tiling/inference, thresholds, voting, and RLE encoding unchanged; the only score-impacting change is enabling the intended ensemble to load and run. Concretely, I auto-resolve `folds_dir` by searching common `/kaggle/input/*` locations for the expected tar filenames and extract them with a safe filter fallback (for Python 3.9 tarfile behavior). Finally, I keep the existing hard guard that prevents producing an all-empty 0.0 submission when no weights load.'
- What this solution (achieved 0.01031) has done: 'I fix the root cause of the 0.0 score/run failure: no weights are being found because `_find_folds_dir()` is searching for tar filenames that don’t exist in this dataset layout (the competition data contains no model tar files). To keep the core inference/model logic intact while making the notebook run end-to-end, I add a minimal fallback that builds the models but runs them with random (untrained) weights when no checkpoints are available, and still produces a valid `submission.csv`. This is score-improving versus the previous hard crash/empty submission (0.0), while respecting your architecture/inference pipeline and writing the required `id,predicted` format. I also make `DATA_ROOT` auto-resolve to the actual available path under `/kaggle/input` to avoid path issues across Kaggle environments.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target because the pipeline is effectively running with untrained/random weights (since the expected checkpoint tar/pt files are not present in this dataset), which yields near-random masks and very low Dice. The smallest score-relevant change is to stop producing random predictions when no weights are available and instead output empty masks (all background), which is a legitimate prediction and typically scores higher than random noise on this metric/dataset. This keeps your core tiling/inference/RLE semantics intact when weights exist, but makes the “no weights” fallback deterministic and less harmful. I also keep submission writing unchanged and ensure the empty-mask RLE is exactly an empty string (as expected by this competition’s submissions).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the “no weights found => empty masks” fallback being triggered, because the only weights you actually extract (`model_1`/`model_2`) are never found due to a path mismatch: extraction often places `.pth` files inside a subfolder, but your `ModelConfig` paths point to the root of `model_1/` and `model_2/`. I keep your exact models, tiling, thresholds, voting, and RLE logic unchanged, and only make weight discovery robust by resolving each configured weight path via a recursive search under its intended directory (and, if needed, by scanning `model_1_dir`/`model_2_dir` for matching fold filenames). This should load the intended ensemble (instead of producing empty masks) and move Dice upward toward the target. I also set `SKIP_COMMIT=False` (it was unused) only if you want to later add optional debug saving; otherwise nothing else changes.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing empty masks because no trained weights are being loaded (the competition dataset doesn’t include the tar checkpoints you try to extract), so `NO_TRAINED_WEIGHTS=True` and every prediction becomes `""`. To move the score up toward the target while keeping your exact model/inference/RLE core logic, the smallest legitimate fix is to actually load real pretrained weights that exist in this environment. I add a minimal “path resolver” that searches `/kaggle/input/**` for the expected `.pth` filenames (and, if found, loads them exactly as before), plus a fallback to search for the two tar files anywhere under `/kaggle/input` (not just a guessed `folds_dir`). If still nothing is found, behavior stays identical (empty masks), but if the weights are present under a different directory structure, this enable the ensemble and should jump the Dice upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the current pipeline never finds any trained weights (your `my_models` paths point to files that do not exist in this Kaggle dataset), so `NO_TRAINED_WEIGHTS=True` and every prediction becomes an empty mask. The smallest score-improving change (while preserving your architecture/inference/threshold/RLE logic) is to train the *same* UNet-ResNet34 model briefly on the provided `train.csv` RLE masks, save fold checkpoints to disk, and then run your existing sliding-window inference using those weights. I keep your model definition, loss function usage (still sigmoid+BCE), tiling/stitching, and post-processing semantics unchanged; the only additions are (1) RLE decode + tile dataset for training and (2) a short CPU/GPU-friendly training loop that writes weights where your loader already expects them. This should move Dice upward substantially from 0.0 toward your target, within Kaggle’s constraints, and still produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by (1) extracting multi‑GB train/test zips unnecessarily, (2) extremely expensive TIFF decoding via `cv2.imreadmulti` on huge multi-page images, and (3) per-tile repeated Albumentations graph construction and CPU work inside `__getitem__`. I avoid extracting the big zips unless the script actually needs training data, replace TIFF reading with a fast/robust `tifffile`-based reader (single chosen level/page only, no full multi-page decode), and prebuild Albumentations transforms once per image-size rather than recreating them per tile. I also enable safe inference accelerations that preserve exact semantics (model `eval()`, `inference_mode`, pinned memory + nonblocking copies, channels_last on CUDA) without changing the architecture, thresholds, or tiling logic. All paths, model logic, and output encoding remain the same.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

CANDIDATE_DATA_ROOTS = [
    Path("../input/hubmap-kidney-segmentation"),
    Path("/kaggle/input/hubmap-kidney-segmentation"),
    Path("../kaggle/input/hubmap-kidney-segmentation"),
    Path("/kaggle/data/hubmap-kidney-segmentation"),
]
DATA_ROOT = next(
    (p for p in CANDIDATE_DATA_ROOTS if p.exists()), CANDIDATE_DATA_ROOTS[0]
)

WORKDIR = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", DEVICE)
print("DATA_ROOT (initial):", DATA_ROOT)
print("DATA_ROOT exists:", DATA_ROOT.exists())
print("WORKDIR:", WORKDIR.resolve())

BATCH_SIZE = 1
NUM_WORKERS = min(2, os.cpu_count() or 1)
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.35
VOTE = 0
SKIP_BACKGROUND = True
SKIP_COMMIT = True  # kept (unused by core logic)

ENABLE_TRAIN_IF_NO_WEIGHTS = True
TRAIN_FOLDS = 5
TRAIN_EPOCHS = 2
TRAIN_LR = 3e-4
TRAIN_TILE_SIZE = 1024
TRAIN_TILE_STEP = 1024
TRAIN_BATCH_SIZE = 2 if torch.cuda.is_available() else 1


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)


def _maybe_extract_competition_zips(
    data_root: Path, workdir: Path, need_train: bool, need_test: bool
) -> Path:
    if (
        (data_root / "train").exists()
        and any((data_root / "train").glob("*.tif*"))
        and (
            not need_test
            or (
                (data_root / "test").exists()
                and any((data_root / "test").glob("*.tif*"))
            )
        )
    ):
        return data_root
    if (
        (data_root / "test").exists()
        and any((data_root / "test").glob("*.tif*"))
        and (
            not need_train
            or (
                (data_root / "train").exists()
                and any((data_root / "train").glob("*.tif*"))
            )
        )
    ):
        return data_root

    candidates = []
    for r in [
        data_root,
        Path("/kaggle/input/hubmap-kidney-segmentation"),
        Path("../input/hubmap-kidney-segmentation"),
    ]:
        if r.exists():
            candidates.extend([r / "train.zip", r / "test.zip"])
    train_zip = next(
        (p for p in candidates if p.name == "train.zip" and p.exists()), None
    )
    test_zip = next(
        (p for p in candidates if p.name == "test.zip" and p.exists()), None
    )

    if (need_train and train_zip is None) or (need_test and test_zip is None):
        print("WARNING: required zip(s) not found; proceeding without extraction.")
        return data_root

    extract_root = workdir / "hubmap_data"
    extract_root.mkdir(parents=True, exist_ok=True)

    def _extract_one(zip_path: Path, dst: Path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst)

    if need_train and train_zip is not None:
        train_dir = extract_root / "train"
        need = (not train_dir.exists()) or (len(list(train_dir.glob("*.tif*"))) == 0)
        if need:
            print("Extracting:", train_zip, "->", extract_root)
            _extract_one(train_zip, extract_root)

    if need_test and test_zip is not None:
        test_dir = extract_root / "test"
        need = (not test_dir.exists()) or (len(list(test_dir.glob("*.tif*"))) == 0)
        if need:
            print("Extracting:", test_zip, "->", extract_root)
            _extract_one(test_zip, extract_root)

    for fname in [
        "train.csv",
        "sample_submission.csv",
        "HuBMAP-20-dataset_information.csv",
    ]:
        src = data_root / fname
        dst = extract_root / fname
        if (not dst.exists()) and src.exists():
            try:
                dst.write_bytes(src.read_bytes())
            except Exception:
                pass

    print("DATA_ROOT (extracted):", extract_root.resolve())
    return extract_root


def resolve_test_tiff(data_root: Path, image_id: str) -> Path:
    candidates = [
        data_root / "test" / f"{image_id}.tiff",
        data_root / "test_images" / f"{image_id}.tiff",
        data_root / "test" / f"{image_id}.tif",
        data_root / "test_images" / f"{image_id}.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def resolve_train_tiff(data_root: Path, image_id: str) -> Path:
    candidates = [
        data_root / "train" / f"{image_id}.tiff",
        data_root / "train_images" / f"{image_id}.tiff",
        data_root / "train" / f"{image_id}.tif",
        data_root / "train_images" / f"{image_id}.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


try:
    import tifffile
except Exception as e:
    tifffile = None
    print(
        "WARNING: tifffile not available; will fall back to OpenCV and may be slower:",
        e,
    )


def _read_tiff_bgr_safe(tiff_path: str, max_total_pixels: int = 60_000_000):
    """
    Reads multi-page/multi-resolution TIFFs and returns:
      - chosen BGR image (possibly downsampled),
      - original (h, w) from highest-res level/page,
      - scale factors sx, sy to map chosen->original.
    Preference order:
      1) tifffile: choose the largest level as original, then pick the largest level <= max_total_pixels
      2) fallback cv2.imread (single page)
    """
    tiff_path = str(tiff_path)

    if tifffile is not None:
        with tifffile.TiffFile(tiff_path) as tif:
            levels = None
            try:
                if hasattr(tif, "series") and len(tif.series) > 0:
                    s0 = tif.series[0]
                    if (
                        hasattr(s0, "levels")
                        and s0.levels is not None
                        and len(s0.levels) > 0
                    ):
                        levels = list(s0.levels)
            except Exception:
                levels = None

            candidates = []
            if levels is not None:
                for lev in levels:
                    try:
                        sh = lev.shape
                        if len(sh) >= 2:
                            h, w = int(sh[0]), int(sh[1])
                            candidates.append(("level", lev, h, w))
                    except Exception:
                        pass
            else:
                for page in tif.pages:
                    try:
                        sh = page.shape
                        if len(sh) >= 2:
                            h, w = int(sh[0]), int(sh[1])
                            candidates.append(("page", page, h, w))
                    except Exception:
                        pass

            if not candidates:
                raise FileNotFoundError(f"Failed to read TIFF metadata: {tiff_path}")

            candidates.sort(key=lambda x: x[2] * x[3], reverse=True)
            _, orig_obj, orig_h, orig_w = candidates[0]

            chosen_obj = None
            chosen_h = orig_h
            chosen_w = orig_w
            for kind, obj, h, w in candidates:
                if int(h) * int(w) <= int(max_total_pixels):
                    chosen_obj = obj
                    chosen_h, chosen_w = h, w
                    break
            if chosen_obj is None:
                chosen_obj = candidates[-1][1]
                chosen_h, chosen_w = candidates[-1][2], candidates[-1][3]

            if levels is not None and hasattr(chosen_obj, "asarray"):
                arr = chosen_obj.asarray()
            else:
                arr = chosen_obj.asarray()

            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
            else:
                arr = np.squeeze(arr)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)

            if arr.dtype != np.uint8:
                arr = np.clip(
                    arr,
                    0,
                    (
                        np.iinfo(arr.dtype).max
                        if np.issubdtype(arr.dtype, np.integer)
                        else 1.0
                    ),
                )
                if np.issubdtype(arr.dtype, np.integer):
                    arr = (
                        arr.astype(np.float32) / np.iinfo(arr.dtype).max * 255.0
                    ).astype(np.uint8)
                else:
                    arr = (arr.astype(np.float32) * 255.0).astype(np.uint8)

            img_bgr = arr[..., ::-1].copy()

            sx = float(orig_w) / float(chosen_w)
            sy = float(orig_h) / float(chosen_h)
            return img_bgr, (orig_h, orig_w), sx, sy

    img0 = cv2.imread(tiff_path, cv2.IMREAD_COLOR)
    if img0 is None:
        raise FileNotFoundError(f"Failed to read TIFF: {tiff_path}")
    h0, w0 = img0.shape[:2]
    return img0, (h0, w0), 1.0, 1.0


def _safe_extract_tar(tar_path: Path, dst_dir: Path) -> None:
    with tarfile.open(tar_path, "r:*") as tf:
        try:
            tf.extractall(dst_dir)  # Py3.9
        except TypeError:
            tf.extractall(dst_dir, filter="data")  # Py3.11+


def _find_folds_dir(expected_files=("1024_avg_last.tar", "1024_512_final.tar")) -> Path:
    roots = [Path("../input"), Path("/kaggle/input")]
    found = []
    for r in roots:
        if not r.exists():
            continue
        for fname in expected_files:
            found.extend(r.rglob(fname))
    if not found:
        return Path("../input/hubmap-folds-2")  # keep original as last resort
    for cand in sorted({p.parent for p in found}, key=lambda x: (len(str(x)), str(x))):
        if all((cand / f).exists() for f in expected_files):
            return cand
    return sorted(found, key=lambda x: (len(str(x)), str(x)))[0].parent


def resolve_weight_path(p: str) -> str:
    pth = Path(p)
    if pth.exists():
        return str(pth)

    parent = pth.parent if pth.parent != Path("") else WORKDIR
    base = pth.name
    if parent.exists():
        hits = sorted(parent.rglob(base), key=lambda x: (len(str(x)), str(x)))
        if len(hits) > 0:
            return str(hits[0])

    hits = sorted(WORKDIR.rglob(base), key=lambda x: (len(str(x)), str(x)))
    if len(hits) > 0:
        return str(hits[0])

    for root in (Path("/kaggle/input"), Path("../input")):
        if root.exists():
            hits = sorted(root.rglob(base), key=lambda x: (len(str(x)), str(x)))
            if len(hits) > 0:
                return str(hits[0])

    return str(pth)




## === cell 1
folds_dir = _find_folds_dir()
print("folds_dir resolved to:", folds_dir)
print("folds_dir exists:", folds_dir.exists())
if folds_dir.exists():
    for p in sorted(folds_dir.iterdir()):
        print(p.name)



## === cell 2
model_1_dir = WORKDIR / "model_1"
model_2_dir = WORKDIR / "model_2"
model_1_dir.mkdir(exist_ok=True)
model_2_dir.mkdir(exist_ok=True)

tar1 = folds_dir / "1024_avg_last.tar"
tar2 = folds_dir / "1024_512_final.tar"

if not tar1.exists():
    tar1 = Path(resolve_weight_path(str(tar1)))
if not tar2.exists():
    tar2 = Path(resolve_weight_path(str(tar2)))

if tar1.exists():
    _safe_extract_tar(tar1, model_1_dir)
else:
    print("WARNING: missing", tar1)

if tar2.exists():
    _safe_extract_tar(tar2, model_2_dir)
else:
    print("WARNING: missing", tar2)

print("Extracted model_1 files:", len(list(model_1_dir.rglob("*"))))
print("Extracted model_2 files:", len(list(model_2_dir.rglob("*"))))



## === cell 3
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 4
df_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")
print(df_sub.head())

need_any_train_fallback = False



## === cell 5
from torchvision.models import resnet34
from torchvision.models import resnet50


class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet34(weights=None)
        self.enc0 = nn.Sequential(m.conv1, m.bn1, m.relu)  # /2, 64
        self.pool = m.maxpool  # /4
        self.enc1 = m.layer1  # /4, 64
        self.enc2 = m.layer2  # /8, 128
        self.enc3 = m.layer3  # /16, 256
        self.enc4 = m.layer4  # /32, 512

        self.center = nn.Sequential(ConvRelu(512, 512), ConvRelu(512, 256))

        self.up4 = UpBlock(256, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = nn.Sequential(ConvRelu(64, 32), ConvRelu(32, 32))

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)  # /2
        x1 = self.enc1(self.pool(x0))  # /4
        x2 = self.enc2(x1)  # /8
        x3 = self.enc3(x2)  # /16
        x4 = self.enc4(x3)  # /32

        c = self.center(x4)
        d4 = self.up4(c, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=False)
        d0 = self.up0(d0)
        return self.final(d0)


class UNetResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=None)
        self.enc0 = nn.Sequential(m.conv1, m.bn1, m.relu)  # /2, 64
        self.pool = m.maxpool  # /4
        self.enc1 = m.layer1  # /4, 256
        self.enc2 = m.layer2  # /8, 512
        self.enc3 = m.layer3  # /16, 1024
        self.enc4 = m.layer4  # /32, 2048

        self.center = nn.Sequential(ConvRelu(2048, 512), ConvRelu(512, 256))

        self.up4 = UpBlock(256, 1024, 256)
        self.up3 = UpBlock(256, 512, 128)
        self.up2 = UpBlock(128, 256, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = nn.Sequential(ConvRelu(64, 32), ConvRelu(32, 32))
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)  # /2
        x1 = self.enc1(self.pool(x0))  # /4
        x2 = self.enc2(x1)  # /8
        x3 = self.enc3(x2)  # /16
        x4 = self.enc4(x3)  # /32
        c = self.center(x4)
        d4 = self.up4(c, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=False)
        d0 = self.up0(d0)
        return self.final(d0)


def build_model(encoder_name: str) -> nn.Module:
    if encoder_name == "resnet34":
        return UNetResNet34()
    if encoder_name in ("se_resnext50_32x4d", "timm-efficientnet-b3"):
        return UNetResNet50()
    raise ValueError(f"Unknown encoder: {encoder_name}")


def load_checkpoint_tolerant(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(missing) or len(unexpected):
        print(
            f"[load_checkpoint_tolerant] {Path(ckpt_path).name}: missing={len(missing)}, unexpected={len(unexpected)}"
        )




## === cell 6
@dataclass
class ModelConfig:
    encoder: str
    weights_path: List[str]
    img_size: int
    weight_blend: float


my_models = [
    ModelConfig(
        "resnet34",
        [
            "./model_1/fold0_avg_0.9238.pth",
            "./model_1/fold1_avg_0.9162.pth",
            "./model_1/fold2_avg_0.9303.pth",
            "./model_1/fold3_avg_0.9336.pth",
            "./model_1/fold4_avg_0.9407.pth",
        ],
        4096,
        0.35,
    ),
    ModelConfig(
        "timm-efficientnet-b3",
        [
            "../input/hubmap6/exp_65_fold_0.pt",
            "../input/hubmap6/exp_63_fold_1.pt",
            "../input/hubmap6/exp_66_fold_2.pt",
            "../input/hubmap6/exp_64_fold_3.pt",
            "../input/hubmap6/exp_62_fold_4.pt",
        ],
        1024,
        0.3,
    ),
    ModelConfig(
        "se_resnext50_32x4d",
        [
            "./model_2/fold0_avg_0.9457.pth",
            "./model_2/fold1_avg_0.9566.pth",
            "./model_2/fold2_avg_0.9379.pth",
            "./model_2/fold3_avg_0.9095.pth",
            "./model_2/fold4_avg_0.9338.pth",
        ],
        2048,
        0.35,
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models])
print("TRAIN_IMG_SIZE:", TRAIN_IMG_SIZE)

if ENABLE_TRAIN_IF_NO_WEIGHTS:
    for cfg in my_models:
        for w in cfg.weights_path:
            if str(w).startswith("./model_1/") and (
                not Path(resolve_weight_path(w)).exists()
            ):
                need_any_train_fallback = True
                break
        if need_any_train_fallback:
            break

DATA_ROOT = _maybe_extract_competition_zips(
    DATA_ROOT, WORKDIR, need_train=need_any_train_fallback, need_test=True
)
print("DATA_ROOT (final):", DATA_ROOT)
print(
    "Train dir exists:",
    (DATA_ROOT / "train").exists(),
    "num_tiffs:",
    len(list((DATA_ROOT / "train").glob("*.tif*"))),
)
print(
    "Test dir exists:",
    (DATA_ROOT / "test").exists(),
    "num_tiffs:",
    len(list((DATA_ROOT / "test").glob("*.tif*"))),
)




## === cell 7
def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def rle_decode(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    s = str(rle)
    if s.strip() == "" or s.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)
    nums = np.asarray([int(x) for x in s.split()], dtype=np.int64)
    starts = nums[0::2] - 1
    lengths = nums[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for st, en in zip(starts, ends):
        img[st:en] = 1
    return img.reshape((shape[1], shape[0])).T


def valid_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def train_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )




## === cell 8
def _check_background(img, crop_size) -> bool:
    s_th = 40
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)

    low_sat = (ss > s_th).sum() <= p_th
    low_int = img.sum() <= p_th
    return bool(low_sat or low_int)


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = int(crop_size)
        self.all_img_sizes = list(all_img_sizes)
        self.step = int(step)

        img, (orig_h, orig_w), sx, sy = _read_tiff_bgr_safe(str(tiff_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {tiff_path}")

        self.img = img
        self.h, self.w = img.shape[:2]
        self.orig_h, self.orig_w = orig_h, orig_w
        self.sx, self.sy = float(sx), float(sy)

        self._tfm_by_size = {
            img_size: valid_transform(img_size) for img_size in self.all_img_sizes
        }

        if self.h <= self.crop_size or self.w <= self.crop_size:
            self.row_count = 1
            self.col_count = 1
        else:
            self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
            self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = max(0, self.w - self.crop_size)
        if y + self.crop_size > self.h:
            y = max(0, self.h - self.crop_size)

        img = self.img[y : y + self.crop_size, x : x + self.crop_size]

        transformed_imgs = {
            img_size: self._tfm_by_size[img_size](image=img)["image"]
            for img_size in self.all_img_sizes
        }
        transformed_imgs["crop_names"] = f"{x}_{y}"
        transformed_imgs["not_background"] = bool(
            not _check_background(img, self.crop_size)
        )
        return transformed_imgs




## === cell 9
def inference(data_loader, models_with_config, crop_size):
    ds = data_loader.dataset
    mask_pred = np.zeros((ds.h, ds.w), dtype=np.uint8)

    use_cuda = torch.cuda.is_available()
    with torch.inference_mode():
        for batch in tqdm(data_loader, ncols=70, leave=True):
            if SKIP_BACKGROUND is True and bool(batch["not_background"][0]) is False:
                continue

            pred_total = None
            for cfg, model_group in models_with_config:
                if len(model_group) == 0:
                    continue

                image = batch[cfg.img_size]
                if use_cuda:
                    image = image.to(DEVICE, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    image = image.to(DEVICE)

                pred_group = None
                for model in model_group:
                    pred = model(image)
                    pred = pred.sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = F.interpolate(
                            pred,
                            size=TRAIN_IMG_SIZE,
                            mode="bilinear",
                            align_corners=False,
                        )
                    pred_group = pred if pred_group is None else (pred_group + pred)

                pred_group = pred_group.squeeze()
                if len(pred_group.shape) == 2:
                    pred_group = pred_group.unsqueeze(0)

                pred_group = cfg.weight_blend * (pred_group / len(model_group))
                pred_total = (
                    pred_group if pred_total is None else (pred_total + pred_group)
                )

            if pred_total is None:
                continue

            pred_total = (pred_total.cpu().numpy() > THR).astype(np.uint8)

            for predict_single, crop_name in zip(pred_total, batch["crop_names"]):
                crop_name = crop_name if isinstance(crop_name, str) else str(crop_name)
                xs, ys = crop_name.split("_")
                x = int(xs)
                y = int(ys)

                if crop_size != TRAIN_IMG_SIZE:
                    predict_single = cv2.resize(
                        predict_single,
                        (crop_size, crop_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = (mask_pred > VOTE).astype(np.uint8)

    if (ds.h != ds.orig_h) or (ds.w != ds.orig_w):
        mask_pred = cv2.resize(
            mask_pred, (ds.orig_w, ds.orig_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

    mask_rle = rle_encode_less_memory(mask_pred.astype(bool))
    del mask_pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mask_rle




## === cell 10
class TrainTileDataset(Dataset):
    def __init__(
        self,
        df_train: pd.DataFrame,
        img_ids: List[str],
        tile_size: int,
        step: int,
        img_size: int,
    ):
        self.df = df_train.set_index("id")
        self.ids = list(img_ids)
        self.tile_size = int(tile_size)
        self.step = int(step)
        self.img_size = int(img_size)
        self.tfm = train_transform(self.img_size)

        self.samples = []
        self.cache = {}  # id -> (img_bgr, mask_uint8)

        for image_id in self.ids:
            tiff_path = resolve_train_tiff(DATA_ROOT, image_id)
            if not tiff_path.exists():
                continue

            try:
                img, (orig_h, orig_w), sx, sy = _read_tiff_bgr_safe(str(tiff_path))
            except Exception as e:
                print(f"WARNING: skip unreadable train tiff {tiff_path.name}: {e}")
                continue

            if (img.shape[0] != orig_h) or (img.shape[1] != orig_w):
                img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            rle = (
                str(self.df.loc[image_id, "encoding"])
                if "encoding" in self.df.columns
                else str(self.df.loc[image_id, "rle"])
            )
            mask = rle_decode(rle, (orig_h, orig_w))
            self.cache[image_id] = (img, mask)

            h, w = orig_h, orig_w
            row_count = (
                1
                if (h <= self.tile_size or w <= self.tile_size)
                else (1 + math.ceil((h - self.tile_size) / self.step))
            )
            col_count = (
                1
                if (h <= self.tile_size or w <= self.tile_size)
                else (1 + math.ceil((w - self.tile_size) / self.step))
            )

            for ry in range(row_count):
                for cx in range(col_count):
                    y = ry * self.step
                    x = cx * self.step
                    if x + self.tile_size > w:
                        x = max(0, w - self.tile_size)
                    if y + self.tile_size > h:
                        y = max(0, h - self.tile_size)
                    self.samples.append((image_id, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, x, y = self.samples[idx]
        img, mask = self.cache[image_id]
        tile_img = img[y : y + self.tile_size, x : x + self.tile_size]
        tile_msk = mask[y : y + self.tile_size, x : x + self.tile_size].astype(np.uint8)

        out = self.tfm(image=tile_img, mask=tile_msk)
        x_t = out["image"]
        y_t = out["mask"].unsqueeze(0).float()
        return x_t, y_t


def train_resnet34_folds_and_write_weights():
    train_csv = DATA_ROOT / "train.csv"
    if not train_csv.exists():
        print("WARNING: train.csv not found, cannot train fallback weights.")
        return

    df_train = pd.read_csv(train_csv)
    if "id" not in df_train.columns:
        df_train = df_train.rename(columns={"img": "id"})
    if "encoding" not in df_train.columns:
        if "rle" in df_train.columns:
            df_train = df_train.rename(columns={"rle": "encoding"})
    df_train["encoding"] = df_train["encoding"].fillna("")

    all_ids = df_train["id"].tolist()
    if len(all_ids) == 0:
        print("WARNING: empty train.csv")
        return

    ids_sorted = sorted(all_ids)
    folds = [[] for _ in range(TRAIN_FOLDS)]
    for i, iid in enumerate(ids_sorted):
        folds[i % TRAIN_FOLDS].append(iid)

    for fold in range(TRAIN_FOLDS):
        out_path = model_1_dir / f"fold{fold}_avg_0.9238.pth"
        if out_path.exists():
            continue

        val_ids = folds[fold]
        tr_ids = [x for f in range(TRAIN_FOLDS) if f != fold for x in folds[f]]
        if len(tr_ids) == 0:
            tr_ids = val_ids

        ds_tr = TrainTileDataset(
            df_train,
            tr_ids,
            tile_size=TRAIN_TILE_SIZE,
            step=TRAIN_TILE_STEP,
            img_size=1024,
        )
        if len(ds_tr) == 0:
            print("WARNING: no train tiles for fold", fold)
            continue

        dl_tr = DataLoader(
            ds_tr,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = build_model("resnet34").to(DEVICE)
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR)
        bce = nn.BCEWithLogitsLoss()

        for ep in range(TRAIN_EPOCHS):
            pbar = tqdm(
                dl_tr,
                ncols=70,
                leave=False,
                desc=f"train fold{fold} ep{ep+1}/{TRAIN_EPOCHS}",
            )
            for xb, yb in pbar:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                logits = model(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=yb.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss = bce(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.detach().cpu()))
                del xb, yb, logits, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.save(model.state_dict(), out_path)
        print("Saved trained fallback weights:", out_path)

        del model, opt, ds_tr, dl_tr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if ENABLE_TRAIN_IF_NO_WEIGHTS and need_any_train_fallback:
    train_resnet34_folds_and_write_weights()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/4243734069.py in <cell line: 0>()
    171 
    172 if ENABLE_TRAIN_IF_NO_WEIGHTS and need_any_train_fallback:
--> 173     train_resnet34_folds_and_write_weights()
    174 

/tmp/ipykernel_55/4243734069.py in train_resnet34_folds_and_write_weights()
    111             tr_ids = val_ids
    112 
--> 113         ds_tr = TrainTileDataset(
    114             df_train,
    115             tr_ids,

/tmp/ipykernel_55/4243734069.py in __init__(self, df_train, img_ids, tile_size, step, img_size)
     30 
     31             if (img.shape[0] != orig_h) or (img.shape[1] != orig_w):
---> 32                 img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
     33 
     34             rle = (

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:3845: error: (-215:Assertion failed) !dsize.empty() in function 'resize'


## === cell 11
all_img_sizes = set([cfg.img_size for cfg in my_models])

models_with_config = []
num_loaded = 0

for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        w_path_resolved = resolve_weight_path(w_path)

        if not Path(w_path_resolved).exists():
            print("WARNING: weight not found:", w_path_resolved)
            continue

        model = build_model(cfg.encoder).to(DEVICE)
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
        load_checkpoint_tolerant(model, w_path_resolved)
        model.eval()
        models_group.append(model)

    if len(models_group) == 0:
        print("WARNING: no weights loaded for encoder:", cfg.encoder)
    num_loaded += len(models_group)
    models_with_config.append((cfg, models_group))

print("Loaded model groups:", [(cfg.encoder, len(g)) for cfg, g in models_with_config])

NO_TRAINED_WEIGHTS = num_loaded == 0
if NO_TRAINED_WEIGHTS:
    print(
        "WARNING: No trained model weights were found. Will output empty masks (all background) for all images."
    )



## === cell 12
if "id" not in df_sub.columns:
    raise ValueError("Submission dataframe must contain 'id' column")
if "predicted" not in df_sub.columns:
    df_sub["predicted"] = ""

df_sub["predicted"] = df_sub["predicted"].fillna("")

for idx, row in df_sub.iterrows():
    if NO_TRAINED_WEIGHTS:
        df_sub.loc[idx, "predicted"] = ""
        continue

    tiff_path = resolve_test_tiff(DATA_ROOT, str(row["id"]))
    if not tiff_path.exists():
        print("WARNING: missing tiff:", tiff_path)
        df_sub.loc[idx, "predicted"] = ""
        continue

    try:
        test_ds = SingleTiffDataset(
            tiff_path=str(tiff_path),
            all_img_sizes=all_img_sizes,
            crop_size=CROP_SIZE,
            step=STEP,
        )
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )
        rle = inference(test_loader, models_with_config, CROP_SIZE)
    except Exception as e:
        print(f"WARNING: failed processing {row['id']} due to: {e}")
        rle = ""

    df_sub.loc[idx, "predicted"] = rle

out_path = WORKDIR / "submission.csv"
df_sub[["id", "predicted"]].to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())
print(df_sub.head())




## === cell 13
def _rm_tree(p: Path):
    if not p.exists():
        return
    for child in sorted(p.rglob("*"), reverse=True):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        except Exception:
            pass
    try:
        p.rmdir()
    except Exception:
        pass


_rm_tree(model_1_dir)
_rm_tree(model_2_dir)
print("Cleanup done.")
