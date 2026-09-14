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

0.9334098149324968

# 6. Current score

0.04734

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (it’s not installed) by switching TIFF reading to `tifffile` while keeping the same tiling/shift inference logic and model usage. I also fix two runtime issues that would crash even after that: `np.float` (deprecated/removed) and use of `torch` before it’s imported due to the earlier failing cell. Finally, I make the window/tiling robust at image edges (avoid variable tile shapes) and ensure a valid `submission.csv` is always written with the required `id,predicted` columns.'
- What this solution (achieved 0.02936) has done: 'I remove the missing `segmentation_models_pytorch` dependency by defining the same core UNet architecture locally (a small U-Net that matches the inference semantics: 3-channel input, 1-channel output with sigmoid). I also fix the test image loading failure by switching from TIFF reading (which requires unavailable `imagecodecs` for JPEG-compressed TIFFs) to rendering the provided glomerulus JSON polygons into a binary mask directly, which still produces a valid RLE submission. Finally, I keep the existing tiling/inference loop intact but make it gracefully fall back to this JSON-based mask when model weights or TIFFs are unavailable, ensuring an end-to-end run and a non-empty submission (improving score from 0.0 toward the target). All paths and the submission format (`id,predicted`) remain unchanged.'
- What this solution (achieved 0.04734) has done: 'Your current pipeline is scoring very low mainly because it always prefers the test JSON glomerulus polygons (which are not the ground-truth target for the test set) and never actually uses the image/model for prediction; we remove that JSON shortcut so inference runs on the real test TIFFs. To make that work end-to-end in this environment (where `tifffile` may fail on JPEG-compressed TIFFs without `imagecodecs`), we add a minimal OpenCV-based TIFF reader fallback that can decode these files reliably. We also fix a critical axis mix-up in tiling: your dataset treats `(H,W)` as `(x,y)` which transposes tiles and damages the assembled mask; correcting this preserves the same tiling logic but aligns it with the competition coordinate convention. These are minimal, directly score-relevant fixes that should move Dice substantially upward toward the target while keeping the same model/inference semantics and submission format.'
- What this solution (achieved 0.0) has done: 'Your current score gap is large (0.04734 vs target 0.9334), so the smallest meaningful improvement is to ensure you actually use the intended pretrained EfficientNet-B2 UNet weights instead of the fallback TinyUNet (which silently fails to load those checkpoints due to architecture mismatch). I keep your tiling/inference logic, thresholding, and RLE exactly the same, but replace only the model definition/loading with a minimal EfficientNet-B2 encoder + UNet-style decoder that matches the checkpoint keys, so the provided 5 folds can be loaded and ensembled as intended. This directly improves Dice while preserving the same evaluation semantics and staying within Kaggle constraints. I also keep the robust TIFF reader fallback and your corrected (H,W) tiling grid as-is.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash by correcting the EfficientNet-B2 channel/skip definitions in your UNet wrapper so the decoder expects the actual feature map channel sizes produced by `torchvision`’s `efficientnet_b2`. This keeps your core inference logic (tiling, preprocessing, sigmoid+threshold, accumulation, RLE) unchanged while making the provided fold checkpoints loadable/executable, which should move the Dice score up substantially from 0.0 toward the target. I also make checkpoint loading more robust by allowing the common case where the saved weights have a different `head` output name (e.g., `logits`) while still preferring strict loading when possible. The rest of the pipeline and output submission format (`id,predicted`) stays the same, and it write `submission.csv` end-to-end.'
- What this solution (achieved 0.04734) has done: 'We fix the broadcast error by making the model output resolution match the dataset’s expected tile size before stitching back into the full mask. The root cause is that your `EffB2_UNet` currently outputs at a lower spatial stride than assumed, so after the `scale_factor=reduce` upsample you still get 512×512 predictions for a 1024×1024 full-res tile. To preserve your core tiling/inference logic, we minimally upsample the network head output by ×2 inside `EffB2_UNet.forward()` (bringing it in line with the original pipeline expectation) and keep the rest (thresholding, accumulation, RLE) unchanged. This should both unblock end-to-end execution and restore intended prediction alignment, improving Dice from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so the smallest high-impact improvements are to (1) ensure we actually read the real test TIFFs from the correct directory (your `DATA` path currently points to `.../test/` which in this dataset contains JSON, while the TIFFs are under `.../test/test/`), and (2) fix the TTA averaging bug (you add 3 flip predictions but divide only by 4 instead of by `1 + 3*#models`, which systematically inflates probabilities and hurts Dice at a fixed threshold). These changes preserve your model, tiling, thresholding, and RLE semantics, but should substantially increase Dice by restoring proper image-based inference and correct probability calibration. I also enable your existing TTA flag (already implemented) since it’s a minimal, directly score-relevant inference-only change and does not alter the training approach. The rest of the pipeline and submission format stays unchanged and still writes `submission.csv`.'
- What this solution (achieved 0.04734) has done: 'I fix the root cause of the crash: your `DATA` directory points to a nested `test/test/` that does not contain the expected `.tiff`/`.json` files for all ids, so file reads fail. I add a minimal, robust path resolver that auto-detects whether files live in `.../test/` or `.../test/test/` and uses the correct base directory without changing the inference/model logic. I also make the JSON fallback search both possible locations so the pipeline always completes and writes `submission.csv`. These changes are score-positive (they make the model actually run on real test TIFFs rather than failing) while preserving your architecture, tiling, TTA, thresholding, and RLE semantics.'
- What this solution (achieved 0.04734) has done: 'Your current score is extremely far from the target, so the most likely “minimal but high-impact” fix is that your prediction stitching is binarizing each tile before aggregation, which destroys probability information at overlaps and typically tanks Dice. I keep the same model, same tiling, same TTA, same threshold, and same RLE, but change stitching to accumulate *probabilities* (not thresholded tiles) and then apply the same threshold once at the end. To preserve your existing “count-based” aggregation semantics, I also accumulate a weight map (number of contributions per pixel) so the final probability is an average over overlapping tiles. This is a localized inference-only change and is directly score-relevant while keeping the rest of your pipeline intact.'
- What this solution (achieved 0.04734) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly increase Dice without changing the model/tiling core logic. The biggest score-lever here is post-processing: a single global threshold (0.35) is often badly calibrated, so we add a tiny, deterministic threshold calibration on the provided training RLE masks (using the same inference pipeline on the train images) and then use that single calibrated threshold for test. To avoid slowing down too much, we calibrate on at most a few training images (available in this dataset) and keep everything else (model, TTA, probability-stitching, RLE) the same. This should move the score substantially upward toward the target while staying within constraints and still producing a valid `submission.csv`.'
- What this solution (achieved 0.04734) has done: 'I fix the calibration crash by making the ground-truth mask decoding use the same (H,W) orientation as the predicted probability map, eliminating the current transpose mismatch that causes broadcasting errors. Concretely, I adjust `enc2mask` (and keep `mask2enc` consistent) to return masks shaped `(H,W)` without a final `.T`, which matches how tiles are stitched (`prob_sum[y1:y2, x1:x2]`). This is a minimal, score-relevant correctness fix: it unblocks threshold calibration and makes the calibrated threshold meaningful rather than computed on misaligned masks. The rest of the model, tiling, TTA, probability stitching, and submission writing remain unchanged.'
- What this solution (achieved 0.04734) has done: 'I make two minimal, score-relevant fixes that preserve your model, tiling, TTA, and loss/architecture semantics. First, I correct the RLE encoding function: your current `rle_encode_less_memory` forcibly zeroes the first/last pixel, which systematically corrupts masks (especially for large objects) and can heavily depress Dice. Second, I ensure `rle_encode_less_memory` always operates on a copy (so it never mutates the original mask view) and uses the standard Kaggle RLE convention without edge-clobbering, improving correctness without changing prediction logic. Everything else (model loading, probability stitching, threshold calibration, submission schema/paths) stays unchanged.'
- What this solution (achieved 0.04734) has done: 'Your current score (0.04734) is far below the target (0.9334), so the most likely minimal, high-impact move is fixing a remaining correctness issue that can silently destroy Dice: the EfficientNet-B2 skip-channel sizes in `EffB2_UNet` are hardcoded and very likely wrong for `torchvision`’s `efficientnet_b2`, so even if checkpoints load “loosely”, the model’s feature fusion is not the intended one. I keep the same backbone, UNet-style decoder idea, tiling, TTA, probability stitching, threshold calibration, and RLE exactly as-is, but I make the skip-channel sizes be inferred from the actual encoder outputs using a one-time dummy forward pass (so the decoder matches the real feature map shapes). This preserves the same core architecture semantics (EffNet encoder + UNet decoder) while making checkpoint loading and inference behavior consistent with the trained model, which should move Dice substantially upward. I also keep submission writing unchanged and ensure it still produces a valid `submission.csv`.'
- What this solution (achieved 0.04734) has done: 'Your current score is far below the target, so we should focus on small, correctness-level fixes that can significantly raise Dice without changing the modeling/tiling approach. The biggest likely remaining issue is that `_calibrate_threshold()` is reading the wrong column from `train.csv` (your competition uses `rle`, but the code often fails to find it and silently falls back to the default TH=0.35), so we robustly detect the RLE column and properly handle multiple RLE rows per image by joining them into one binary mask. Second, we ensure the calibration uses the exact same test-time prediction pipeline but on the true train directory (without accidentally pointing to test dirs), so the calibrated threshold is meaningful and should move the score upward. Everything else (EffB2_UNet, tiling grid, probability stitching, TTA, and final RLE submission format) stays the same.'
- What this solution (achieved 0.04734) has done: 'Your current score is far below the target, so the most likely minimal, high-impact improvement is to ensure predictions are aligned with the competition’s RLE convention at every step (decode/encode and any sanity checks), because a single transpose/flatten mismatch can collapse Dice to near-zero even if the model is good. I make `enc2mask`, `mask2enc`, and `rle_encode_less_memory` strictly consistent with the stated “top-to-bottom then left-to-right” indexing by using Fortran-order reshaping/flattening (no `.T` tricks), which is both minimal and directly score-relevant. I also remove the remaining “shape mismatch fallback transpose” during threshold calibration (it can silently accept a wrong orientation and pick a bad threshold), keeping the same calibration approach but making it deterministic and correct. Everything else (model, tiling, TTA, probability stitching, thresholding concept, submission schema/paths) stays the same.'

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)



## === cell 1
sz = 256  # tile size at model input resolution
reduce = 4  # downscale factor applied before model (then upsampled back)
TH = 0.35  # default threshold (will be calibrated if possible)


def _resolve_test_dir():
    candidates = [
        "../input/hubmap-kidney-segmentation/test/test/",
        "../input/hubmap-kidney-segmentation/test/",
        "/kaggle/input/hubmap-kidney-segmentation/test/test/",
        "/kaggle/input/hubmap-kidney-segmentation/test/",
    ]
    for d in candidates:
        if os.path.isdir(d):
            try:
                files = os.listdir(d)
            except Exception:
                continue
            has_expected = any(f.endswith(".tiff") for f in files) or any(
                f.endswith(".json") for f in files
            )
            if has_expected:
                return d
    for d in candidates:
        if os.path.isdir(d):
            return d
    return candidates[0]


DATA = _resolve_test_dir()

MODELS = [
    f"../input/b2256shiftbcefreeze/efficientnet-b2-256-FOLD-{i}-model.pth"
    for i in range(5)
]
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b2"
shift = True
minoverlap = 300

print("Resolved DATA directory:", DATA)
print("Sample ids:", df_sample["id"].tolist()[:3])




## === cell 2
def enc2mask(encs, shape_hw):
    """
    FIX (score-relevant): use a single, explicit convention consistent with the competition:
    pixels are numbered top-to-bottom, then left-to-right -> this corresponds to flattening
    the (H,W) mask in Fortran order ('F').

    Therefore:
      - encoding uses mask.flatten(order='F')
      - decoding uses reshape((H,W), order='F')
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    img = np.zeros(H * W, dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        if enc is None or (isinstance(enc, str) and len(enc.strip()) == 0):
            continue
        s = str(enc).split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start : start + length] = 1  # binary union is enough for this comp
    return img.reshape((H, W), order="F")


def mask2enc(mask, n=1):
    """
    Encode mask of shape (H,W) to RLE using the same explicit Fortran-order convention.
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
    FIX (score-relevant): standard Kaggle RLE with the same Fortran-order flattening,
    and no clobbering of first/last pixel (which corrupts masks and hurts Dice).
    """
    pixels = img.flatten(order="F").astype(np.uint8, copy=True)
    if pixels.size == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # min number of pixels threshold


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape_hw, window=256, min_overlap=32):
    """
    Return array (N,4) with y1,y2,x1,x2 in image coordinates for shape (H,W).
    """
    H, W = shape_hw
    ny = H // (window - min_overlap) + 1
    y1 = np.linspace(0, H, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = max(0, H - window)
    y2 = (y1 + window).clip(0, H)

    nx = W // (window - min_overlap) + 1
    x1 = np.linspace(0, W, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = max(0, W - window)
    x2 = (x1 + window).clip(0, W)

    slices = np.zeros((ny, nx, 4), dtype=np.int64)
    for i in range(ny):
        for j in range(nx):
            slices[i, j] = y1[i], y2[i], x1[j], x2[j]
    return slices.reshape(ny * nx, 4)


def _read_tiff_rgb(path):
    """
    Read TIFF as RGB uint8.
    Add cv2.imdecode fallback for JPEG-compressed TIFFs when tifffile fails.
    """
    try:
        img = tiff.imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[0] in (3, 4) and img.shape[0] < img.shape[-1]:
                img = np.moveaxis(img, 0, -1)
            if img.shape[-1] > 3:
                img = img[..., :3]
        else:
            raise ValueError(f"Unsupported TIFF shape {img.shape} for {path}")

        if img.dtype != np.uint8:
            imin = float(np.min(img))
            imax = float(np.max(img))
            if imax > imin:
                img = (img.astype(np.float32) - imin) / (imax - imin)
            else:
                img = np.zeros_like(img, dtype=np.float32)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        return img
    except Exception:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def _load_json_polygons(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    polys = []
    for feat in data:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = coords[0]
        if ring and isinstance(ring[0], list) and len(ring[0]) >= 2:
            polys.append(np.array(ring, dtype=np.float32))
    return polys


def _candidate_data_dirs():
    dirs = [DATA]
    if DATA.rstrip("/").endswith("/test/test"):
        alt = DATA.rstrip("/").rsplit("/test/test", 1)[0] + "/test/"
        dirs.append(alt)
    elif DATA.rstrip("/").endswith("/test"):
        alt = DATA.rstrip("/").rsplit("/test", 1)[0] + "/test/test/"
        dirs.append(alt)
    out = []
    for d in dirs:
        if d and d not in out and os.path.isdir(d):
            out.append(d if d.endswith("/") else d + "/")
    return out


def _resolve_train_dir():
    candidates = [
        "../input/hubmap-kidney-segmentation/train/train/",
        "../input/hubmap-kidney-segmentation/train/",
        "/kaggle/input/hubmap-kidney-segmentation/train/train/",
        "/kaggle/input/hubmap-kidney-segmentation/train/",
    ]
    for d in candidates:
        if os.path.isdir(d):
            try:
                files = os.listdir(d)
            except Exception:
                continue
            if any(f.endswith(".tiff") for f in files):
                return d if d.endswith("/") else d + "/"
    for d in candidates:
        if os.path.isdir(d):
            return d if d.endswith("/") else d + "/"
    return candidates[0]


TRAIN_DIR = _resolve_train_dir()


def _json_to_mask(idx, shape_hw=None):
    json_path = None
    for d in _candidate_data_dirs():
        p = os.path.join(d, f"{idx}.json")
        if os.path.exists(p):
            json_path = p
            break
    if json_path is None:
        if shape_hw is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return np.zeros(shape_hw, dtype=np.uint8)

    polys = _load_json_polygons(json_path)

    if shape_hw is None:
        if len(polys) == 0:
            shape_hw = (1, 1)
        else:
            all_xy = np.concatenate(polys, axis=0)
            max_x = int(np.ceil(np.max(all_xy[:, 0])))
            max_y = int(np.ceil(np.max(all_xy[:, 1])))
            shape_hw = (max_y + 1, max_x + 1)  # (H,W)

    mask = np.zeros(shape_hw, dtype=np.uint8)
    for poly in polys:
        pts = np.round(poly).astype(np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, mask.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, mask.shape[0] - 1)
        cv2.fillPoly(mask, [pts], 1)
    return mask




## === cell 4
import torchvision


class ConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnAct(in_ch + skip_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = ConvBnAct(out_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x


class EffB2_UNet(nn.Module):
    def __init__(self, out_ch=1):
        super().__init__()
        weights = None  # no internet; avoid downloading pretrained weights
        self.encoder = torchvision.models.efficientnet_b2(weights=weights).features

        self._tap_ids = [1, 2, 3, 5, 7]  # feature indices for skips
        enc_out_ch = 1408  # final features channels for effnet-b2

        self._skip_ch = self._infer_skip_channels()

        self.center = nn.Sequential(
            ConvBnAct(enc_out_ch, 256, k=3, s=1, p=1),
            ConvBnAct(256, 256, k=3, s=1, p=1),
        )

        s1c, s2c, s3c, s4c, s5c = self._skip_ch
        self.dec5 = DecoderBlock(256, s5c, 192)
        self.dec4 = DecoderBlock(192, s4c, 128)
        self.dec3 = DecoderBlock(128, s3c, 96)
        self.dec2 = DecoderBlock(96, s2c, 64)
        self.dec1 = DecoderBlock(64, s1c, 32)

        self.head = nn.Conv2d(32, out_ch, kernel_size=1)

    def _infer_skip_channels(self):
        x = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        skips = []
        with torch.no_grad():
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i in self._tap_ids:
                    skips.append(x)
        if len(skips) != 5:
            return [16, 24, 48, 120, 352]
        return [int(t.shape[1]) for t in skips]

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self._tap_ids:
                skips.append(x)

        x = self.center(x)
        s1, s2, s3, s4, s5 = skips
        x = self.dec5(x, s5)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.head(x)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return x


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce, base_dir=None):
        self.idx = idx
        self.base_dir = base_dir if base_dir is not None else DATA

        tiff_path = None
        search_dirs = [self.base_dir]
        if self.base_dir == DATA:
            search_dirs = _candidate_data_dirs()
        for d in search_dirs:
            p = os.path.join(d, idx + ".tiff")
            if os.path.exists(p):
                tiff_path = p
                break
        self.path = (
            tiff_path
            if tiff_path is not None
            else os.path.join(self.base_dir, idx + ".tiff")
        )

        self._use_dummy = False
        try:
            self.img_full = _read_tiff_rgb(self.path)  # (H,W,3) uint8
            self.shape = self.img_full.shape[:2]  # (H,W)
        except Exception:
            self.shape = _json_to_mask(idx).shape
            self.img_full = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
            self._use_dummy = True

        self.reduce = reduce
        self.sz = reduce * sz  # tile size at full-res before downscale
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, i):
        y1, y2, x1, x2 = self.mask_grid[i]

        tile = self.img_full[y1:y2, x1:x2]
        if tile.shape[0] != self.sz or tile.shape[1] != self.sz:
            pad_h = self.sz - tile.shape[0]
            pad_w = self.sz - tile.shape[1]
            tile = np.pad(
                tile,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if self.reduce != 1:
            tile = cv2.resize(
                tile,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        _, s, _ = cv2.split(hsv)

        vertices = torch.tensor([y1, y2, x1, x2], dtype=torch.int64)

        if (s > s_th).sum() <= p_th or tile.sum() <= p_th:
            return img2tensor((tile / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((tile / 255.0 - mean) / std), vertices, i


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
                        denom = float(len(self.models) * (1 + len(flips)))
                        py = py / denom
                    else:
                        py = py / max(1, len(self.models))

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


def _clean_state_dict(state_dict):
    if (
        isinstance(state_dict, dict)
        and "state_dict" in state_dict
        and isinstance(state_dict["state_dict"], dict)
    ):
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("model.", "module.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v
    return cleaned


models = []
existing = [p for p in MODELS if os.path.exists(p)]
if len(existing) > 0:
    for path in existing:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        cleaned = _clean_state_dict(state_dict)

        model = EffB2_UNet(out_ch=1)

        try:
            model.load_state_dict(cleaned, strict=True)
        except Exception:
            try:
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                if len(missing) > 20:
                    continue
            except Exception:
                continue

        model.float().eval().to(device)
        models.append(model)
        del state_dict, cleaned
else:
    model = EffB2_UNet(out_ch=1).to(device).eval().float()
    models = [model]

gc.collect()
print(f"Loaded {len(models)} model(s)")




## === cell 5
def _dice_coef(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    inter = np.logical_and(pred, true).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + true.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float((2.0 * inter) / denom)


def _predict_prob_for_id(idx: str, base_dir: str, tta: bool = True) -> np.ndarray:
    ds = HuBMAPDataset(idx, base_dir=base_dir)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl, tta=tta)

    prob_sum = np.zeros(ds.shape, dtype=np.float32)
    prob_w = np.zeros(ds.shape, dtype=np.float32)

    for pred, vert, _ in iter(mp):
        y1, y2, x1, x2 = vert
        h = y2 - y1
        w = x2 - x1
        tile_pred = pred[:h, :w].astype(np.float32, copy=False)
        prob_sum[y1:y2, x1:x2] += tile_pred
        prob_w[y1:y2, x1:x2] += 1.0

    prob = prob_sum / np.maximum(prob_w, 1.0)
    del prob_sum, prob_w, ds, dl
    gc.collect()
    return prob


def _find_rle_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for key in ("rle", "encoding", "mask", "pixels"):
        if key in lower:
            return lower[key]
    for c in cols:
        cl = c.lower()
        if "rle" in cl or "enc" in cl:
            return c
    return None


def _get_train_image_path(tid: str) -> str:
    p = os.path.join(TRAIN_DIR, tid + ".tiff")
    if os.path.exists(p):
        return p
    alt1 = (
        TRAIN_DIR.rstrip("/").rsplit("/train/train", 1)[0] + "/train/" + tid + ".tiff"
    )
    if os.path.exists(alt1):
        return alt1
    alt2 = (
        TRAIN_DIR.rstrip("/").rsplit("/train", 1)[0] + "/train/train/" + tid + ".tiff"
    )
    if os.path.exists(alt2):
        return alt2
    return p


def _calibrate_threshold(max_train_images: int = 3) -> float:
    train_csv_candidates = [
        "../input/hubmap-kidney-segmentation/train.csv",
        "/kaggle/input/hubmap-kidney-segmentation/train.csv",
        "../input/train.csv",
        "/kaggle/input/train.csv",
    ]
    train_csv = None
    for p in train_csv_candidates:
        if os.path.exists(p):
            train_csv = p
            break
    if train_csv is None:
        print("No train.csv found; using default TH =", TH)
        return TH

    df_train = pd.read_csv(train_csv)
    if "id" not in df_train.columns:
        print("train.csv missing 'id' column; using default TH =", TH)
        return TH

    enc_col = _find_rle_column(df_train)
    if enc_col is None:
        print("Could not locate RLE column in train.csv; using default TH =", TH)
        return TH

    id_list = df_train["id"].astype(str).unique().tolist()[:max_train_images]

    thresholds = np.linspace(0.20, 0.60, 17, dtype=np.float32)
    dice_sums = np.zeros_like(thresholds, dtype=np.float64)
    used = 0

    for tid in id_list:
        tiff_path = _get_train_image_path(tid)
        if not os.path.exists(tiff_path):
            continue

        try:
            img = _read_tiff_rgb(tiff_path)
        except Exception:
            continue
        shape_hw = img.shape[:2]

        rles = df_train.loc[
            df_train["id"].astype(str) == str(tid), enc_col
        ].values.tolist()
        if len(rles) == 0:
            continue
        true = enc2mask(rles, shape_hw)
        true = (true > 0).astype(np.uint8)

        train_base_dir = os.path.dirname(tiff_path) + "/"
        prob = _predict_prob_for_id(tid, base_dir=train_base_dir, tta=True)

        if prob.shape != true.shape:
            continue

        for j, th in enumerate(thresholds):
            pred = (prob > float(th)).astype(np.uint8)
            dice_sums[j] += _dice_coef(pred, true)

        used += 1
        del img, prob, true
        gc.collect()

    if used == 0:
        print("Could not use any train images for calibration; using default TH =", TH)
        return TH

    avg_dice = dice_sums / used
    best_idx = int(np.argmax(avg_dice))
    best_th = float(thresholds[best_idx])
    print(
        f"Calibrated TH on {used} train image(s): best_th={best_th:.3f}, avg_dice={avg_dice[best_idx]:.4f} (rle_col='{enc_col}')"
    )
    return best_th


TH = _calibrate_threshold(max_train_images=3)



## === cell 6
names, preds = [], []

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]

    ds = HuBMAPDataset(idx)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)

    mp = Model_pred(models, dl, tta=True)

    prob_sum = np.zeros(ds.shape, dtype=np.float32)
    prob_w = np.zeros(ds.shape, dtype=np.float32)

    for pred, vert, i in iter(mp):
        y1, y2, x1, x2 = vert
        h = y2 - y1
        w = x2 - x1
        tile_pred = pred[:h, :w].astype(np.float32, copy=False)
        prob_sum[y1:y2, x1:x2] += tile_pred
        prob_w[y1:y2, x1:x2] += 1.0

    prob = prob_sum / np.maximum(prob_w, 1.0)
    mask = (prob > TH).astype(np.uint8)

    rle = rle_encode_less_memory(mask)
    names.append(idx)
    preds.append(rle)

    del mask, prob, prob_sum, prob_w, ds, dl
    gc.collect()



## === cell 7
df = pd.DataFrame({"id": names, "predicted": preds})

df = df.set_index("id").reindex(df_sample["id"]).reset_index()
df["predicted"] = df["predicted"].fillna("")

df.to_csv("submission.csv", index=False)
print(df.head())
print("Using TH =", TH)
print("Wrote submission.csv with shape:", df.shape)
