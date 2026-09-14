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

0.9263369580879148

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the unavailable `segmentation_models_pytorch` dependency and replacing it with a minimal PyTorch UNet implementation that matches the expected `smp.Unet(..., classes=1)` call signature so the rest of the pipeline stays unchanged. I also fix the cell ordering/variable-scope issue by ensuring `TEST_DIR`/`shift` are defined before use (your script started at cell 0, which broke later cells). Finally, I add a safe fallback: if the provided model weight paths don’t exist in this environment, the code still generate a valid `submission.csv` (all-empty masks) instead of crashing, so you can submit and obtain a score.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an “all-empty mask” submission caused by no model weights being loaded (your `MODELS` paths point to a dataset that isn’t present in this environment), so the smallest improvement is to load weights from an actually available input path. I keep the same inference/core tiling + UNet stub logic, but change `MODELS` to point to `.pth` files found under `/kaggle/input` (including nested folders) and only fall back to empty predictions if none exist. I also make RLE encoding return an empty string (not `"0"`/invalid) when the mask is empty, and force output order to exactly match `sample_submission.csv` ids to avoid silent misalignment. These changes should move the score upward toward your target by enabling non-empty predictions rather than a guaranteed 0.0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because no real trained weights are being loaded, so the model outputs are effectively random/empty after thresholding; the smallest meaningful improvement is to (1) load *any* `.pth` weights that actually match this UNet stub (not just filenames containing “model”), and (2) handle common checkpoint formats (`{"state_dict":...}`, `{"model":...}`) plus `module.` prefixes so weights don’t silently fail to load. I keep your inference/tiling/thresholding logic identical, but broaden weight discovery to all `.pth` files under `/kaggle/input` and add a robust state-dict extractor + key-fixer so more checkpoints successfully load. Finally, I add a tiny debug print of which checkpoints loaded so you can confirm you’re no longer submitting all-empty masks (which guarantees ~0 Dice). These changes should increase the score toward your target by enabling non-empty, learned predictions rather than empty masks.'
- What this solution (achieved 0.0) has done: 'Your 0.0 Dice is still most consistent with “no real weights loaded” (so you submit mostly-empty masks) or with “weights exist but don’t load due to key mismatches/strict loading”, so the smallest score-moving change is to make weight loading robust without changing the model/inference logic. I keep the exact same UNet stub, tiling, thresholding, and RLE pipeline, but (1) prefer checkpoints whose tensor shapes actually match this UNet, (2) support common key prefixes like `model.` / `net.` in addition to `module.`, and (3) allow `strict=False` (only if shape-compatible) so partial key mismatches don’t cause all checkpoints to be skipped. This should increase the chance that at least one usable checkpoint loads (non-empty predictions), moving the score upward toward your target, while keeping evaluation semantics identical.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still consistent with producing (almost) all-empty masks because the post-processing is overly strict for this metric (high threshold and a second hard vote threshold), even if some weights load. I keep the exact same model stub, tiling, and inference flow, but make two minimal, metric-aligned calibration tweaks: lower `TH` to a safer default for sigmoid outputs and fix the shifted-tiling mask merge to use averaging (coverage-normalized) instead of a binary “any tile positive” rule that can collapse to empty when predictions are weak. I also ensure the weight loader doesn’t silently accept nearly-empty partial matches by requiring a minimal number of matched keys, increasing the chance you’re using a meaningful checkpoint (still without changing architecture). These are small changes that should increase Dice away from 0.0 toward your target by producing non-empty, better-calibrated masks, while keeping the core logic intact and still writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the current pipeline is still producing mostly/all-empty masks, most likely because the tile “tissue filter” is rejecting too many tiles on this dataset and/or because the fixed threshold is slightly too strict for whatever weights (if any) actually load. To move the score upward with minimal risk and without changing the model/inference core, I (1) relax the background/tissue rejection gate so we don’t skip most tiles, and (2) add a tiny, safe threshold fallback: if an image ends up completely empty after thresholding, retry once with a slightly lower threshold to avoid guaranteed-zero Dice. I also broaden prefix-stripping for checkpoints (`encoder.` / `decoder.` etc.) so more available `.pth` files successfully load into the same UNet stub (still using filtered, shape-matching keys only). These are small, score-relevant changes aimed at producing non-empty masks rather than optimizing for best score.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with generating mostly-empty masks because (a) no compatible checkpoints are actually being loaded into the current UNet stub and/or (b) the tile “tissue filter” skips too many regions, so even a loaded model has little chance to contribute. I keep the exact same model/inference/tiling core, but make two minimal, score-relevant changes: (1) make the checkpoint loader accept more common key-prefix patterns (so weights don’t get rejected unnecessarily) while still shape-filtering, and (2) relax the tile skip-gate so we don’t throw away most tiles (which otherwise drives the submission toward empty masks and Dice≈0). Finally, I add a small “soft” empty-mask retry that’s already in your shift branch, but make it slightly more likely to trigger by computing a per-image empty check after merging tiles (no change to metric semantics, just avoids a guaranteed-zero output when the logits are weak). These changes are the smallest ones likely to move the score upward toward your target by producing non-empty, learned masks.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the pipeline producing (almost) all-empty masks, and the most common cause here is the aggressive “tissue tile skip” gate that can reject most tiles on HuBMAP, leaving no pixels to segment even when models are loaded. I keep your exact UNet stub, tiling/inference, and thresholding approach, but make the skip gate less destructive by only skipping tiles that are extremely blank (very low brightness), instead of using saturation-based filtering that can misfire. I also enable lightweight TTA (the code already supports it) to stabilize predictions toward non-empty masks without changing the model or training. Finally, I add a deterministic, minimal postprocess safety: if an image is still empty, apply a very small morphological closing to connect fragmented positives (only in the “empty/fallback” situation), which typically increases Dice from 0.0 while staying within the same evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
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

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


class _DoubleConv(nn.Module):
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


class _UNetSmall(nn.Module):
    def __init__(self, in_channels=3, classes=1, base=32):
        super().__init__()
        self.inc = _DoubleConv(in_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(base * 2, base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(base * 4, base * 8))

        self.up1 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv1 = _DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = _DoubleConv(base * 4, base * 2)

        self.up3 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv3 = _DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        if x.shape[-2:] != x3.shape[-2:]:
            x = F.interpolate(
                x, size=x3.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.conv1(torch.cat([x3, x], dim=1))

        x = self.up2(x)
        if x.shape[-2:] != x2.shape[-2:]:
            x = F.interpolate(
                x, size=x2.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.conv2(torch.cat([x2, x], dim=1))

        x = self.up3(x)
        if x.shape[-2:] != x1.shape[-2:]:
            x = F.interpolate(
                x, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.conv3(torch.cat([x1, x], dim=1))

        return self.outc(x)


class smp:
    class Unet(nn.Module):
        def __init__(self, encoder_name, encoder_weights=None, classes=1):
            super().__init__()
            self.model = _UNetSmall(in_channels=3, classes=classes, base=32)

        def forward(self, x):
            return self.model(x)


KAGGLE_INPUT = "/kaggle/input"
DEFAULT_DATA_ROOT = os.path.join(KAGGLE_INPUT, "hubmap-kidney-segmentation")
if os.path.isdir(DEFAULT_DATA_ROOT):
    DATA_ROOT = DEFAULT_DATA_ROOT
else:
    DATA_ROOT = "../input/hubmap-kidney-segmentation"

TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")



## === cell 1
sz = 256  # tile size at reduced resolution
reduce = 4  # downsample factor

TH = 0.18
TH_FALLBACK_IF_EMPTY = 0.12

DATA = TEST_DIR


def _discover_model_paths(search_root="/kaggle/input"):
    pths = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            if fn.lower().endswith(".pth"):
                pths.append(os.path.join(root, fn))
    return sorted(pths)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return None


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    prefixes = (
        "module.",
        "model.",
        "net.",
        "segmentation_model.",
        "unet.",
        "backbone.",
        "encoder.",
        "decoder.",
        "segmentation_head.",
        "head.",
    )
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _state_dict_match_score(model, state_dict):
    msd = model.state_dict()
    ok = 0
    total = 0
    for k, mv in msd.items():
        total += 1
        sv = state_dict.get(k, None)
        if sv is None:
            continue
        if isinstance(sv, torch.Tensor) and sv.shape == mv.shape:
            ok += 1
    return ok, total


MODELS = _discover_model_paths(KAGGLE_INPUT)

df_sample = pd.read_csv(SAMPLE_SUB_PATH)

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b2"
shift = True
minoverlap = 300

TTA = True

print("Discovered", len(MODELS), "candidate .pth files under", KAGGLE_INPUT)




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
    if img is None:
        return ""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.sum() == 0:
        return ""
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

BLANK_MEAN_TH = 8.0  # on uint8 grayscale mean
BLANK_STD_TH = 3.0  # on uint8 grayscale std

s_th = 10
p_th = 80 * (sz // 256) ** 2


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
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


def _read_tiff_memmap(path):
    arr = tiff.memmap(path)
    if arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[1] > 64 and arr.shape[2] > 64:
            arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path}")
    if arr.shape[-1] != 3:
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        else:
            pad = 3 - arr.shape[-1]
            arr = np.concatenate([arr] + [arr[..., -1:]] * pad, axis=-1)
    return arr




## === cell 4
if not shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.idx = idx
            self.data = _read_tiff_memmap(os.path.join(DATA, idx + ".tiff"))
            self.shape = self.data.shape[:2]  # (H,W)
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
            patch = self.data[p00:p01, p10:p11]
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = patch.astype(
                np.uint8, copy=False
            )

            if self.reduce != 1:
                img = cv2.resize(
                    img,
                    (self.sz // reduce, self.sz // reduce),
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.mean() <= BLANK_MEAN_TH and gray.std() <= BLANK_STD_TH:
                return img2tensor((img / 255.0 - mean) / std), -1
            else:
                return img2tensor((img / 255.0 - mean) / std), idx

    class Model_pred:
        def __init__(self, models, dl, tta: bool = False, half: bool = False):
            self.models = models
            self.dl = dl
            self.tta = tta
            self.half = half

        def __iter__(self):
            with torch.no_grad():
                for x, y in iter(self.dl):
                    if (y >= 0).sum() > 0:
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
                            py,
                            scale_factor=reduce,
                            mode="bilinear",
                            align_corners=False,
                        )
                        py = py.permute(0, 2, 3, 1).float().cpu()

                        for i in range(len(py)):
                            yield py[i], y[i]

        def __len__(self):
            return len(self.dl.dataset)

    models = []
    loaded_paths = []
    candidate_infos = []

    probe_model = smp.Unet(model_name, encoder_weights=None, classes=1)

    for path in MODELS:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location=torch.device("cpu"))
        except Exception:
            continue
        state_dict = _extract_state_dict(ckpt)
        if state_dict is None:
            continue
        state_dict = _strip_known_prefixes(state_dict)

        ok, total = _state_dict_match_score(probe_model, state_dict)
        if ok == 0:
            continue
        candidate_infos.append((ok, total, path, state_dict))

    candidate_infos.sort(key=lambda x: (-x[0], x[2]))

    for ok, total, path, state_dict in candidate_infos:
        if ok < max(50, int(0.30 * total)):
            continue

        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_sd
            and isinstance(v, torch.Tensor)
            and v.shape == model_sd[k].shape
        }
        try:
            model.load_state_dict(filtered, strict=False)
        except Exception:
            continue
        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        loaded_paths.append(f"{path} (matched_keys={len(filtered)}/{total})")
        if len(models) >= 5:
            break

    del probe_model
    if "ckpt" in locals():
        del ckpt
    if "state_dict" in locals():
        del state_dict

    print("Loaded", len(models), "models:")
    for p in loaded_paths[:10]:
        print(" -", p)

    names, preds = [], []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = row["id"]
        if len(models) == 0:
            names.append(idx)
            preds.append("")
            continue

        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
        )
        mp = Model_pred(models, dl, tta=TTA)

        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for p, i in iter(mp):
            mask[i.item()] = p.squeeze(-1) > TH

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

        rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
        names.append(idx)
        preds.append(rle)
        del mask, ds, dl
        gc.collect()



## === cell 5
if shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.idx = idx
            self.data = _read_tiff_memmap(os.path.join(DATA, idx + ".tiff"))
            self.shape = self.data.shape[:2]  # (H,W)
            self.reduce = reduce
            self.sz = reduce * sz
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            img = self.data[x1:x2, y1:y2].astype(np.uint8, copy=False)

            if self.reduce != 1:
                img = cv2.resize(
                    img,
                    (self.sz // reduce, self.sz // reduce),
                    interpolation=cv2.INTER_AREA,
                )

            vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.mean() <= BLANK_MEAN_TH and gray.std() <= BLANK_STD_TH:
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
    loaded_paths = []
    candidate_infos = []

    probe_model = smp.Unet(model_name, encoder_weights=None, classes=1)

    for path in MODELS:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location=torch.device("cpu"))
        except Exception:
            continue
        state_dict = _extract_state_dict(ckpt)
        if state_dict is None:
            continue
        state_dict = _strip_known_prefixes(state_dict)

        ok, total = _state_dict_match_score(probe_model, state_dict)
        if ok == 0:
            continue
        candidate_infos.append((ok, total, path, state_dict))

    candidate_infos.sort(key=lambda x: (-x[0], x[2]))

    for ok, total, path, state_dict in candidate_infos:
        if ok < max(50, int(0.30 * total)):
            continue

        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_sd
            and isinstance(v, torch.Tensor)
            and v.shape == model_sd[k].shape
        }
        try:
            model.load_state_dict(filtered, strict=False)
        except Exception:
            continue
        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        loaded_paths.append(f"{path} (matched_keys={len(filtered)}/{total})")
        if len(models) >= 5:
            break

    del probe_model
    if "ckpt" in locals():
        del ckpt
    if "state_dict" in locals():
        del state_dict

    print("Loaded", len(models), "models:")
    for p in loaded_paths[:10]:
        print(" -", p)

    names, preds = [], []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = row["id"]
        if len(models) == 0:
            names.append(idx)
            preds.append("")
            continue

        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
        )
        mp = Model_pred(models, dl, tta=TTA)

        prob_sum = np.zeros(ds.shape, dtype=np.float32)
        cov = np.zeros(ds.shape, dtype=np.float32)

        for pred, vert, _i in iter(mp):
            x1, x2, y1, y2 = vert
            prob_sum[x1:x2, y1:y2] += pred.astype(np.float32)
            cov[x1:x2, y1:y2] += 1.0

        cov = np.maximum(cov, 1.0)
        prob = prob_sum / cov

        mask = (prob > TH).astype(np.uint8)

        if (
            mask.sum() == 0
            and TH_FALLBACK_IF_EMPTY is not None
            and TH_FALLBACK_IF_EMPTY < TH
        ):
            mask = (prob > TH_FALLBACK_IF_EMPTY).astype(np.uint8)

            if mask.sum() > 0:
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        rle = rle_encode_less_memory(mask)
        names.append(idx)
        preds.append(rle)
        del prob_sum, cov, prob, mask, ds, dl
        gc.collect()



## === cell 6
if "names" not in locals() or "preds" not in locals() or len(names) == 0:
    names = df_sample["id"].tolist()
    preds = [""] * len(names)

pred_map = dict(zip(names, preds))
out_ids = df_sample["id"].tolist()
out_preds = [pred_map.get(i, "") for i in out_ids]

df = pd.DataFrame({"id": out_ids, "predicted": out_preds})
df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with", len(df), "rows")
print("Models found/loaded:", len(models) if "models" in locals() else 0)
print(
    "Non-empty predictions:",
    sum(1 for p in out_preds if isinstance(p, str) and len(p) > 0),
)
