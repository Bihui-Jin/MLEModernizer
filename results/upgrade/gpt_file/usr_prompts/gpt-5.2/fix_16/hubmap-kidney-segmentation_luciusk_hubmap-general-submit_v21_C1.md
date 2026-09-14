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

0.923453214676238

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by replacing TIFF window reads with `tifffile.memmap`, which preserves the same tiling/inference logic and avoids loading full images into RAM. I also fix a few runtime issues: `tqdm.notebook` incompatibility, deprecated `np.float`, missing imports due to earlier cell failure, and `F.upsample` deprecation. Finally, I ensure the submission is always aligned to `sample_submission.csv` ids (even if something fails per-image) so the CSV has the correct number of rows and column names.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not available in this environment) by adding a small local UNet fallback that preserves the same “UNet-style” architecture and inference semantics, so the notebook runs end-to-end. I also fix the broken model weight paths by auto-discovering available `.pth` files under `../input/` and loading what exists; if none exist, the pipeline still run and produce a valid CSV (but score be poor). Finally, I ensure the submission is always aligned to `sample_submission.csv` and uses the exact required columns, while keeping the tiling + thresholding + RLE encoding logic unchanged.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime error in the weight-loading cell by only deleting `state` when it actually exists, so the notebook can proceed to inference and write a submission. I also make the model-path discovery more robust (but still minimal) to ensure that if the intended weights are present anywhere under `../input/`, they get picked up, which should move the score up from 0.0 toward the target. Finally, I add lightweight exception logging during per-image inference so failures don’t silently produce empty masks (which yields 0.0), while still guaranteeing a correctly formatted `submission.csv`. No changes are made to the tiling/inference logic, thresholding, or RLE encoding semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being effectively empty/invalid for the metric (most commonly: wrong submission column name, or writing blank RLEs due to per-image failures). I make two minimal fixes: (1) write the prediction column using the exact sample_submission schema (`predicted` vs `pixels` mismatch is a common cause, so we align to the provided sample), and (2) ensure we are actually reading the `.tiff` images (your current folder points to JSON-only paths in this dataset layout) by switching `DATA` to the correct test image directory and auto-falling-back if needed. I also add a tiny guard so that if an image can’t be processed, we still emit a valid empty RLE (`np.nan`) rather than `""`, matching Kaggle expectations and avoiding scoring quirks. These changes preserve your tiling/inference/threshold/RLE logic and are aimed purely at moving from 0.0 toward the target by producing a valid, non-empty submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle treating the submission as empty/invalid, even though a CSV was written. The smallest high-impact fix is to ensure the submission column name exactly matches the competition’s required schema (`predicted`) and that truly-empty masks are encoded as an empty string `""` (not `np.nan`), because many RLE evaluators interpret NaN as “missing prediction” and score it as zero. I keep your tiling, model inference, thresholding, and RLE logic unchanged, but adjust RLE encoding to return `""` for empty masks and force the output DataFrame to use the sample submission’s columns exactly and in the same order. This should move the score up from 0.0 toward your target by making the submission valid and scorable.'
- What this solution (achieved 0.0) has done: 'Your 0.0 strongly suggests the model is never producing meaningful masks (likely because no real pretrained weights are being loaded, so you’re effectively running random/untrained inference). The smallest change that should move the score upward (without changing your tiling/inference/RLE semantics) is to (1) look for the *actual* HuBMAP kidney weights by explicitly checking common weight filenames/locations under `../input/`, and (2) make the weight loader robust to the common key-prefix patterns so the correct tensors actually land in `cnn_model.*`. I also add a one-time sanity print of the mean predicted probability on the first image’s first batch to confirm the pipeline is not degenerate (near-0 everywhere), without changing any thresholds or post-processing. Everything else (tile size, reduce factor, threshold, stitching, RLE) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model is not using the intended pretrained SEResNeXt UNet at all: `segmentation_models_pytorch` is unavailable, so you fall back to a random UNet and/or load incompatible weights non-strictly, producing near-empty masks. To move the score up toward the target while keeping your tiling/inference/RLE pipeline unchanged, I (1) force the same SEResNeXt50-UNet *architecture locally* (minimal backbone+UNet wrapper) so the discovered competition weights can actually load, and (2) make weight loading verify that a meaningful fraction of keys matched; if not, we skip that weight file instead of ensembling garbage. This preserves the same core inference semantics (sigmoid → threshold → stitch → RLE) but fixes the “random model / wrong weights” failure mode that yields 0.0. The submission formatting stays aligned to `sample_submission.csv` (`predicted` column) and still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The immediate crash comes from using a torchvision API (`torchvision.models.se_resnext50_32x4d`) that doesn’t exist in torchvision 0.21, so the model never instantiates and all downstream inference/submission logic can’t run. I fix this by switching the backbone creation to `timm`-style if available, but since extra packages aren’t guaranteed, the minimal safe fix is to use `torchvision.models.resnext50_32x4d` as the closest available drop-in encoder and keep the same UNet-style decoder/head and inference/RLE logic unchanged. I also make the weight-loader slightly more permissive to common key names for resnext/seresnext checkpoints (still using the same matching logic) so that if compatible weights exist, they actually load—moving the score up from 0.0 toward your target. Everything else (tiling, thresholding, stitching, submission column name alignment) is preserved.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model producing almost-all-empty masks because the intended pretrained weights are not being loaded into the actual network (key-prefix mismatch between checkpoints and `HuBMAP.cnn_model.*`), so inference degenerates. I make the smallest weight-loading fix: try loading into both the wrapper (`HuBMAP`) and the inner module (`HuBMAP.cnn_model`) and prefer the option with the higher key/shape match ratio, without changing inference/tiling/threshold/RLE logic. I also slightly raise the minimum match ratio so we don’t accidentally include nearly-random incompatible checkpoints in the ensemble (which can keep predictions near-zero). Everything else (tile reading, TTA disabled, threshold=0.25, submission schema aligned to sample) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “valid CSV but effectively empty/near-empty masks,” which usually comes from (a) reading the wrong test files (no `.tiff` found, so per-image exceptions → empty RLEs), or (b) almost-all tiles being discarded by the “blank tile” filter due to a BGR/RGB mismatch when computing HSV. I keep your tiling/inference/threshold/RLE logic identical, but make two minimal, score-relevant fixes: (1) robustly locate the actual test `.tiff` directory across the provided dataset layouts, and (2) correct the HSV conversion to match the actual channel order from TIFF (RGB) so the blank-tile filtering doesn’t mistakenly drop tissue tiles. These are targeted to turn “mostly empty predictions” into real masks, pushing the score upward toward your target without changing the model, threshold, or stitching semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the submission being effectively empty because you’re not actually loading compatible pretrained weights: this code uses a ResNeXt50-based UNet fallback, but the competition weights you’re trying to discover are typically for a *SE-ResNeXt50* encoder (or SMP’s `se_resnext50_32x4d`), so the key/shape match is near-zero and you end up predicting near-random/near-empty masks. To move the score upward toward your target without changing the tiling/inference/threshold/RLE semantics, I (1) add an in-notebook implementation of SE-ResNeXt50_32x4d (minimal, using torchvision’s ResNeXt blocks plus SE modules) so the intended checkpoints can load, and (2) tighten weight selection to prefer checkpoints that match a meaningful fraction of parameters so we don’t ensemble incompatible weights that keep outputs near-empty. Everything else (tile reading via `tifffile.memmap`, blank-tile filtering, sigmoid→threshold at TH=0.25, stitching, RLE, submission column name) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission that is effectively empty because the test TIFFs aren’t being found/used (so every image hits the exception path) and/or because the RLE encoding is incorrect for this competition’s required pixel order (top-to-bottom then left-to-right). I make two minimal, score-relevant fixes: (1) make test TIFF discovery robust to the actual HuBMAP dataset layout by searching specifically for `test/*.tif*` under the known root and using the directory that contains those TIFFs, and (2) fix RLE encoding/decoding to follow the competition’s pixel numbering (column-major flatten without extra transpose), keeping your mask creation and thresholding logic unchanged. These changes preserve your model/inference/tiling pipeline, but turn “all-empty/invalid” predictions into properly-scored masks, moving the score upward toward your target. The submission still be aligned to `sample_submission.csv` and written as `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from an RLE pixel-order mismatch (the competition expects pixels numbered top-to-bottom then left-to-right, i.e., column-major/Fortran order), plus an off-by-two bug in the current RLE encoder that can shift runs. I make the smallest possible change by fixing `rle_encode_less_memory()` to use the correct flattening order and standard run indexing, while leaving your tiling, thresholding (TH=0.25), stitching, and model inference unchanged. I also add a tiny safety cast to ensure the mask is strictly binary `0/1` before encoding (which matches the metric’s requirement). These are tightly scoped changes that should move the score up from 0.0 toward your target without altering core modeling logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the submission being *scorable but mis-encoded*, i.e., RLE pixel order is wrong for this competition, which can collapse Dice to ~0 even when the mask “looks” reasonable. I make the smallest score-relevant change by fixing the RLE encoder to follow the competition’s pixel numbering (top-to-bottom then left-to-right → flatten in C-order on the transposed mask) while keeping your model, tiling, thresholding, stitching, and submission schema unchanged. I also add a tiny binary cast guard right before encoding to ensure the mask is strictly 0/1 as required (no semantic change, just safety). Everything else remains the same so runtime and core logic are preserved.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with a scorable CSV that nevertheless encodes masks in the wrong pixel order for this competition, which can drive Dice to ~0 even when the predicted mask is non-empty. I make the smallest score-relevant fix by correcting the RLE encoder to follow the competition’s required numbering (top-to-bottom then left-to-right), while keeping your tiling, thresholding (TH=0.25), stitching, and model inference unchanged. I also keep a tiny safety cast to ensure the mask is strictly binary before encoding, which avoids edge-case encoding issues but doesn’t change semantics. Everything else (paths, model loading, submission schema aligned to `sample_submission.csv`) stays the same.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torchvision

try:
    sys.path.append("../input/segmentation-models-pytorch-install")
except Exception:
    pass

try:
    import segmentation_models_pytorch as smp  # noqa: F401

    HAS_SMP = True
except Exception:
    HAS_SMP = False

print("HAS_SMP:", HAS_SMP)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())




## === cell 1
sz = 256  # tile size at model input
reduce = 4  # downsample factor when reading tiles
TH = 0.25  # threshold for positive predictions

DATA_CANDIDATES = [
    "../input/hubmap-kidney-segmentation/test/",
    "../input/test/",
    "/kaggle/input/hubmap-kidney-segmentation/test/",
    "/kaggle/data/hubmap-kidney-segmentation/test/",
    "/kaggle/data/test/",
]


def pick_test_image_dir(cands):
    best = None
    best_n = -1
    for d in cands:
        if os.path.isdir(d):
            tiffs = glob.glob(os.path.join(d, "*.tif")) + glob.glob(
                os.path.join(d, "*.tiff")
            )
            if len(tiffs) > best_n:
                best = d
                best_n = len(tiffs)
    if best is not None and best_n > 0:
        return best

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        tiffs = glob.glob(os.path.join(root, "**", "test", "*.tif"), recursive=True)
        tiffs += glob.glob(os.path.join(root, "**", "test", "*.tiff"), recursive=True)
        if len(tiffs) > 0:
            return os.path.dirname(tiffs[0])

    return cands[0]


DATA = pick_test_image_dir(DATA_CANDIDATES)
tiff_list = glob.glob(os.path.join(DATA, "*.tif")) + glob.glob(
    os.path.join(DATA, "*.tiff")
)
print("Using DATA:", DATA, "tiff_count:", len(tiff_list))


def discover_model_paths():
    candidates = glob.glob("../input/**/*.pth", recursive=True)
    candidates = sorted(set(candidates))
    return candidates


MODELS = [
    f"../input/normalize-seresnext/result/se_resnext50_32x4d-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("sample rows:", len(df_sample), "ids:", df_sample["id"].tolist())
print("Default MODELS[0]:", MODELS[0])




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if (
            enc is None
            or (isinstance(enc, float) and np.isnan(enc))
            or (isinstance(enc, str) and enc.strip() == "")
        ):
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start : start + length] = 1 + m
    return img.reshape((shape[1], shape[0])).T


def mask2enc(mask, n=1):
    pixels = mask.reshape((-1,), order="F")
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append("")
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(" ".join(str(x) for x in runs))
    return encs


def rle_encode_less_memory(img):
    if img is None:
        return ""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)

    img = (img > 0).astype(np.uint8, copy=False)
    if img.sum() == 0:
        return ""

    pixels = img.T.reshape(-1, order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # minimum number of pixels threshold


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    """
    rasterio isn't available in this Kaggle environment.
    Replace rasterio window reading with tifffile.memmap-based window slicing.
    Core tiling logic (padding, reduced read, HSV blank filtering) is preserved.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path_tiff = os.path.join(DATA, idx + ".tiff")
        self.path_tif = os.path.join(DATA, idx + ".tif")

        if os.path.exists(self.path_tiff):
            self.path = self.path_tiff
        elif os.path.exists(self.path_tif):
            self.path = self.path_tif
        else:
            raise FileNotFoundError(
                f"Missing TIFF for id={idx}: tried {self.path_tiff} and {self.path_tif}"
            )

        self.data = tiff.memmap(self.path)

        if self.data.ndim == 3 and self.data.shape[-1] == 3:
            self.is_chw = False
            self.shape = (self.data.shape[0], self.data.shape[1])  # H, W
        elif self.data.ndim == 3 and self.data.shape[0] == 3:
            self.is_chw = True
            self.shape = (self.data.shape[1], self.data.shape[2])  # H, W
        else:
            raise ValueError(
                f"Unexpected TIFF shape for {self.path}: {self.data.shape}"
            )

        self.reduce = reduce
        self.sz = reduce * sz

        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def _read_window_hwc(self, p00, p01, p10, p11):
        if not self.is_chw:
            return np.asarray(self.data[p00:p01, p10:p11, :], dtype=np.uint8)
        patch = np.asarray(self.data[:, p00:p01, p10:p11], dtype=np.uint8)
        return np.moveaxis(patch, 0, -1)

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        if (p01 > p00) and (p11 > p10):
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = (
                self._read_window_hwc(p00, p01, p10, p11)
            )

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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
                if (y >= 0).sum() > 0:  # exclude empty tiles
                    x = x[y >= 0].to(device, non_blocking=True)
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


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.avg_pool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=32,
        base_width=4,
        reduction=16,
    ):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.se = SEModule(planes * self.expansion, reduction=reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SEResNeXt50Encoder(nn.Module):
    def __init__(self, layers=(3, 4, 6, 3), groups=32, width_per_group=4):
        super().__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=layers[3], stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        outplanes = planes * SEBottleneck.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(
            SEBottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
            )
        )
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(
                SEBottleneck(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64,  H/2
        x1 = self.maxpool(x0)  # H/4
        x1 = self.layer1(x1)  # 256, H/4
        x2 = self.layer2(x1)  # 512, H/8
        x3 = self.layer3(x2)  # 1024,H/16
        x4 = self.layer4(x3)  # 2048,H/32
        return x0, x1, x2, x3, x4


class SeResNeXt50_UNet(nn.Module):
    """
    Local SE-ResNeXt50_32x4d UNet so that 'se_resnext50_32x4d' HuBMAP checkpoints can load.
    Decoder/head forward semantics are unchanged vs previous UNet-style model.
    """

    def __init__(self, classes=1):
        super().__init__()

        self.encoder = SEResNeXt50Encoder(
            layers=(3, 4, 6, 3), groups=32, width_per_group=4
        )

        self.center = nn.Sequential(ConvRelu(2048, 512), ConvRelu(512, 512))
        self.dec4 = DecoderBlock(512, 1024, 256)
        self.dec3 = DecoderBlock(256, 512, 128)
        self.dec2 = DecoderBlock(128, 256, 64)
        self.dec1 = DecoderBlock(64, 64, 32)  # skip from stem

        self.head = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        x = self.center(x4)  # C=512, H/32
        x = self.dec4(x, x3)  # -> 256, H/16
        x = self.dec3(x, x2)  # -> 128, H/8
        x = self.dec2(x, x1)  # -> 64,  H/4
        x = self.dec1(x, x0)  # -> 32,  H/2

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetFallback(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 8)
        self.up1 = Up(base * 16, base * 4)
        self.up2 = Up(base * 8, base * 2)
        self.up3 = Up(base * 4, base)
        self.up4 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

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
        if HAS_SMP:
            import segmentation_models_pytorch as smp  # local import to avoid hard dependency

            self.cnn_model = smp.Unet(
                "se_resnext50_32x4d", encoder_weights=None, classes=1
            )
        else:
            self.cnn_model = SeResNeXt50_UNet(classes=1)

    def forward(self, imgs):
        return self.cnn_model(imgs)




## === cell 6
def prioritize_hubmap_weights(paths):
    def score(p):
        pl = p.lower()
        s = 0
        if "hubmap" in pl or "hubmap-kidney" in pl:
            s += 50
        if "kidney" in pl:
            s += 30
        if "seresnext" in pl or "se_resnext" in pl:
            s += 40
        elif "resnext" in pl:
            s += 10
        if "unet" in pl:
            s += 10
        if "fold" in pl:
            s += 10
        if "best" in pl:
            s += 5
        if pl.endswith(".pth"):
            s += 1
        return s

    return sorted(paths, key=score, reverse=True)


def extract_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        return state["state_dict"]
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        return state["model"]
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        return state
    return None


def normalize_keys_for_wrapper(sd):
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]

        if not nk.startswith("cnn_model.") and (
            nk.startswith("encoder.")
            or nk.startswith("decoder.")
            or nk.startswith("segmentation_head.")
            or nk.startswith("head.")
            or nk.startswith("stem.")
            or nk.startswith("layer")
            or nk.startswith("center.")
            or nk.startswith("dec")
            or nk.startswith("inc.")
            or nk.startswith("down")
            or nk.startswith("up")
            or nk.startswith("outc.")
        ):
            nk = "cnn_model." + nk

        new_sd[nk] = v
    return new_sd


def normalize_keys_for_inner(sd):
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        new_sd[nk] = v
    return new_sd


def match_ratio(target_sd, candidate_sd):
    matched = 0
    for k, v in candidate_sd.items():
        if (
            k in target_sd
            and isinstance(v, torch.Tensor)
            and target_sd[k].shape == v.shape
        ):
            matched += 1
    return matched / max(1, len(target_sd))


def load_best_of_wrapper_or_inner(model: HuBMAP, raw_sd: dict, min_match_ratio=0.30):
    wrapper_sd = model.state_dict()
    inner_sd = model.cnn_model.state_dict()

    cand_wrapper = normalize_keys_for_wrapper(raw_sd)
    cand_inner = normalize_keys_for_inner(raw_sd)

    r_wrapper = match_ratio(wrapper_sd, cand_wrapper)
    r_inner = match_ratio(inner_sd, cand_inner)

    if max(r_wrapper, r_inner) < min_match_ratio:
        return False, (r_wrapper, r_inner), "none"

    if r_inner >= r_wrapper:
        model.cnn_model.load_state_dict(cand_inner, strict=False)
        return True, (r_wrapper, r_inner), "inner"
    else:
        model.load_state_dict(cand_wrapper, strict=False)
        return True, (r_wrapper, r_inner), "wrapper"


models = []

existing = [p for p in MODELS if os.path.exists(p)]
if not existing:
    discovered = discover_model_paths()
    discovered = prioritize_hubmap_weights(discovered)
    existing = discovered[:30]

print("Found weight files (candidates):", len(existing))
for p in existing[:10]:
    print(" ", p)

state = None

if len(existing) == 0:
    model = HuBMAP().to(device).eval()
    models = [model]
    print("No weights found; using randomly initialized model (will score poorly).")
else:
    for path in existing:
        try:
            state = torch.load(path, map_location=torch.device("cpu"))
        except Exception as e:
            print("Failed to torch.load:", path, "err:", repr(e))
            continue

        model = HuBMAP()
        state_dict = extract_state_dict(state)
        if state_dict is None:
            print("Skipping (no state_dict):", os.path.basename(path))
            continue

        ok, (r_wrap, r_in), loaded_into = load_best_of_wrapper_or_inner(
            model, state_dict, min_match_ratio=0.30
        )
        if not ok:
            print(
                f"Skipping {os.path.basename(path)} (low match ratios wrapper={r_wrap:.3f}, inner={r_in:.3f})"
            )
            continue

        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        print(
            f"Loaded {os.path.basename(path)} (match wrapper={r_wrap:.3f}, inner={r_in:.3f}, loaded_into={loaded_into})"
        )

        if len(models) >= 5:
            break

if len(models) == 0:
    model = HuBMAP().to(device).eval()
    models = [model]
    print(
        "No compatible weights loaded; using randomly initialized model (will score poorly)."
    )

if state is not None:
    del state
gc.collect()

print("Ensemble size:", len(models))




## === cell 7
names, preds = [], []
debug_errors = []
printed_sanity = False

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    img_id = row["id"]
    names.append(img_id)

    try:
        ds = HuBMAPDataset(img_id)
        dl = DataLoader(
            ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
        )
        mp = Model_pred(models, dl)

        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.uint8)

        if not printed_sanity:
            with torch.no_grad():
                for x0, y0 in iter(dl):
                    if (y0 >= 0).sum() > 0:
                        x0 = x0[y0 >= 0].to(device)
                        p0 = models[0](x0)
                        p0s = torch.sigmoid(p0)
                        print(
                            "Sanity: first-batch prob mean/std/min/max =",
                            float(p0s.mean().detach().cpu()),
                            float(p0s.std().detach().cpu()),
                            float(p0s.min().detach().cpu()),
                            float(p0s.max().detach().cpu()),
                        )
                        printed_sanity = True
                        break

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
        preds.append(rle if isinstance(rle, str) else "")

        del mask, ds, dl
        gc.collect()

    except Exception as e:
        debug_errors.append((img_id, repr(e)))
        preds.append("")
        gc.collect()

if len(debug_errors) > 0:
    print("Per-image errors (showing up to 5):")
    for x in debug_errors[:5]:
        print(" ", x)




## === cell 8
assert "id" in df_sample.columns, "sample_submission must have 'id' column"
pred_col = [c for c in df_sample.columns if c != "id"][0]

df = pd.DataFrame({"id": names, pred_col: preds})
df = df.set_index("id").reindex(df_sample["id"]).reset_index()

df[pred_col] = df[pred_col].fillna("").astype(str)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with rows:", len(df), "pred_col:", pred_col)
print("Non-empty predictions:", (df[pred_col].str.len() > 0).sum())
print("Empty predictions:", (df[pred_col].str.len() == 0).sum())
