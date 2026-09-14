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

0.9498136762796549

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the unavailable `segmentation_models_pytorch` dependency and replacing it with a tiny compatibility wrapper that can load the provided checkpoints by instantiating the correct U-Net from `torchvision` (if available) or falling back to a safe, minimal UNet implementation. I also make the notebook cells self-contained (so missing imports like `pd`, `Dataset`, `dataclass` don’t break later cells) while keeping your inference logic, tiling, thresholds, voting, and RLE encoding unchanged. Finally, I ensure the submission is always written to `submission.csv` with the exact `id,predicted` columns from `sample_submission.csv`, even when `SKIP_COMMIT=True` (so you always get a valid CSV to submit). These changes are aimed at correctness/stability first so you can obtain a score; once it runs, the score should be in the right ballpark given the pretrained weights.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed in your environment) by switching the TIFF tile reading to `tifffile`, which is available in Kaggle and keeps the same tiling/inference logic. I also fix model weight loading so at least one model group is always available: instead of asserting, we gracefully fall back to a deterministic “empty mask” predictor when weights are missing, ensuring a valid `submission.csv` is always produced (this is necessary to avoid the current 0.0 due to no valid submission). Finally, I correct the submission writing to always fill every row (your current `len(df_sub)>count_thr` gate can skip all predictions and leave blanks), while keeping your thresholding, voting, and RLE encoding semantics unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from predicting empty masks because none of the provided checkpoints can load into the current `_TinyUNet` fallback (shape/key mismatch → effectively random/empty after thresholding), so the submission is valid but has near-all blanks. To move the score up toward the target with minimal disruption, I (1) replace the fallback “compat” model with a standard U-Net implementation that matches the common `segmentation_models_pytorch.Unet` key structure (`encoder.*`, `decoder.*`, `segmentation_head.*`) so your existing `.pth` weights can actually load, and (2) make weight loading slightly more robust by unwrapping common checkpoint dict formats and stripping prefixes without changing inference/tiling/RLE logic. Everything else (tiling, transforms, THR/VOTE/RAMKA, ensembling, and RLE encoding) stays the same so evaluation semantics are preserved. The code still always write a valid `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with “models didn’t actually load / wrong architecture for the checkpoints”, so the pipeline runs but predicts essentially empty masks. With minimal disruption, I (1) stop trying to instantiate unsupported encoder names (efficientnet/se_resnext) and instead infer the correct ResNet encoder from each checkpoint’s tensor shapes, (2) make state-dict key remapping a bit more SMP-like (handle `encoder.` vs `backbone.`/etc and unwrap common checkpoint dict formats), and (3) ensure at least the ResNet-based groups reliably load so inference produces non-empty masks. This keeps your tiling, thresholds, RAMKA, voting, and RLE encoding unchanged, and still writes a valid `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with “valid CSV but effectively empty masks” caused by a mismatch between your fallback SMP-like U-Net and the non‑ResNet checkpoints (efficientnet / se_resnext), plus some checkpoints likely storing logits in a slightly different head name so most weights don’t actually land in the model. I keep your tiling/inference/voting/RLE logic unchanged, but make weight loading actually usable by (1) skipping clearly incompatible model groups rather than forcing them into a ResNet encoder, and (2) adding a tiny state-dict key remap that covers common SMP head names (`classification_head`, `final_conv`, `mask_conv`, `segmentation_head.0.*`) so the segmentation head loads instead of staying random. This should produce non-empty, meaningful masks from the resnet34 fold models you do have, lifting score toward your target without changing evaluation semantics. The script still always write a valid `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is consistent with “valid CSV but effectively empty/near-empty masks” caused by a subtle but critical RLE bug and a too-aggressive background skip. I (1) fix the RLE encoder to correctly add sentinel zeros without overwriting real pixels (current code can erase true positives at the first/last pixel, often collapsing masks to empty), and (2) make the background check consistent by using RGB→HSV (your image is RGB but you used BGR conversion), which can incorrectly label tissue tiles as background and skip nearly all inference. These are minimal changes that preserve your model/tiling/voting logic while making the predicted masks non-degenerate, moving the Dice score upward toward the target. The script still runs end-to-end and writes `submission.csv` in the required `id,predicted` format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing a “valid-looking” submission that is effectively all-empty masks, which in your current pipeline can happen if the background-skip logic is inverted and it ends up skipping almost every tile. I make one minimal, score-critical fix: rename the dataset flag to `is_background` and correct the skip condition so we only skip true background tiles (not tissue), keeping your tiling/inference/voting/RLE logic unchanged. I also harden the background check to return a plain Python `bool` to avoid `.item()` edge cases, but won’t change thresholds or model behavior. Everything still runs end-to-end and writes `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing almost-all empty masks due to weight-loading incompatibility: your current SMP-compat U-Net only supports ResNet encoders, yet you still keep configs for EfficientNet/SE-ResNeXt and then skip those groups, leaving fewer/zero effective models. With minimal changes (no change to tiling/inference/voting/RLE semantics), I (1) make model-group activation explicit by filtering `my_models` down to only groups that can actually be instantiated/loaded in this environment, and (2) harden state-dict remapping slightly to better match common SMP ResNet34 checkpoints (`decoder.*`, `segmentation_head.*`), increasing the chance you get non-empty, meaningful predictions and thus a Dice score moving toward the target. I also keep the submission writing unconditional and schema-correct as you already do.'
- What this solution (achieved 0.0) has done: 'I make two minimal, score-critical fixes that are very likely preventing any meaningful Dice score: first, ensure your inference actually runs on the provided test images by reading `.tif` (this dataset uses `.tif`, not `.tiff`), and second, output the submission in the competition-required column names (`id,predicted`) rather than renaming to (`img,pixels`). These changes preserve your full model/tiling/voting/threshold/RLE logic and only correct I/O/schema issues that would otherwise yield a 0.0/invalid score. I also make RLE return an empty string for empty masks (standard in this competition) to avoid edge-case formatting problems. Everything still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 (with a valid CSV) is most consistent with producing essentially all-empty masks, and in this pipeline the highest-probability cause is that the model ensemble is effectively disabled by the aggressive filtering to only `resnet*` encoders (dropping two strong model groups) plus potentially failing to locate/extract weights. To move the Dice score upward toward your target with minimal disruption, I keep your tiling/inference/voting/RLE logic unchanged, but (1) stop dropping non-resnet configs and instead load them via the same SMP-compat U-Net fallback (so they at least produce meaningful non-empty predictions when weights are present), and (2) make weight discovery more robust by trying the extracted folders first and then searching common extracted paths when a listed weight is missing. These changes do not alter the evaluation semantics; they only ensure the intended ensemble actually runs and produces non-degenerate masks, which should increase the score substantially from 0.0 toward the target.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import tifffile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


class _ConvBNReLU(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _SCSEModule(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cSE = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, max(channels // reduction, 1), 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(max(channels // reduction, 1), channels, 1),
            torch.nn.Sigmoid(),
        )
        self.sSE = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 1, 1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class _DecoderBlock(torch.nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_scse=True):
        super().__init__()
        self.conv1 = _ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = _ConvBNReLU(out_ch, out_ch)
        self.attn = _SCSEModule(out_ch) if use_scse else torch.nn.Identity()

    def forward(self, x, skip):
        x = torch.nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn(x)
        return x


class _SMPResNetEncoder(torch.nn.Module):
    """
    ResNet-like encoder with feature names aligned to common SMP checkpoints:
    encoder.conv1, encoder.bn1, encoder.layer1..layer4, etc.
    """

    def __init__(self, name="resnet34", in_ch=3):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50

        if name == "resnet18":
            m = resnet18(weights=None)
            chs = [64, 64, 128, 256, 512]
        elif name == "resnet50":
            m = resnet50(weights=None)
            chs = [64, 256, 512, 1024, 2048]
        else:  # default resnet34
            m = resnet34(weights=None)
            chs = [64, 64, 128, 256, 512]

        if in_ch != 3:
            m.conv1 = torch.nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.out_channels = chs

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # /2
        x1 = self.layer1(self.maxpool(x0))  # /4
        x2 = self.layer2(x1)  # /8
        x3 = self.layer3(x2)  # /16
        x4 = self.layer4(x3)  # /32
        return [x0, x1, x2, x3, x4]


class _SMPUnetLike(torch.nn.Module):
    """
    Minimal SMP-compatible Unet:
    - module names: encoder.*, decoder.blocks.*, segmentation_head.*
    - output: (B,1,H,W) logits
    """

    def __init__(self, encoder_name="resnet34", in_ch=3, classes=1):
        super().__init__()
        enc_name = encoder_name
        if not str(enc_name).startswith("resnet"):
            enc_name = "resnet34"

        self.encoder = _SMPResNetEncoder(enc_name, in_ch=in_ch)
        e = self.encoder.out_channels  # [c0,c1,c2,c3,c4]

        self.decoder = torch.nn.Module()
        self.decoder.blocks = torch.nn.ModuleList(
            [
                _DecoderBlock(in_ch=e[4], skip_ch=e[3], out_ch=256, use_scse=True),
                _DecoderBlock(in_ch=256, skip_ch=e[2], out_ch=128, use_scse=True),
                _DecoderBlock(in_ch=128, skip_ch=e[1], out_ch=64, use_scse=True),
                _DecoderBlock(in_ch=64, skip_ch=e[0], out_ch=32, use_scse=True),
            ]
        )

        self.segmentation_head = torch.nn.Sequential(
            torch.nn.Conv2d(32, classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        d = self.decoder.blocks[0](x4, x3)
        d = self.decoder.blocks[1](d, x2)
        d = self.decoder.blocks[2](d, x1)
        d = self.decoder.blocks[3](d, x0)
        d = torch.nn.functional.interpolate(
            d, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.segmentation_head(d)


class _SMPCompat:
    @staticmethod
    def Unet(encoder_name: str, encoder_weights=None):
        return _SMPUnetLike(encoder_name=encoder_name, in_ch=3, classes=1)


smp = _SMPCompat()



## === cell 1
import subprocess


def _run(cmd: str):
    try:
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        print(f"[WARN] Command failed (continuing): {cmd}\n{e}")


_run("ls -l ../input/hubmap-folds-2 || true")
_run("mkdir -p model_1")
_run("tar -xvf ../input/hubmap-folds-2/1024_avg_last.tar -C model_1 || true")
_run("mkdir -p model_2")
_run("tar -xvf ../input/hubmap-folds-2/1024_512_pseudo_v1_avg.tar -C model_2 || true")



## === cell 2
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.5
VOTE = 0
SKIP_BACKGROUND = True
SKIP_COMMIT = True
RAMKA = 256

df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
if "predicted" not in df_sub.columns and "pixels" in df_sub.columns:
    df_sub = df_sub.rename(columns={"pixels": "predicted"})
assert "id" in df_sub.columns and "predicted" in df_sub.columns




## === cell 3
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
        0.4,
    ),
    ModelConfig(
        "timm-efficientnet-b3",
        [
            f"../input/hubmap5/exp_25_fold_0.pt",
            f"../input/hubmap5/exp_24_fold_1.pt",
            f"../input/hubmap5/exp_23_fold_2.pt",
            f"../input/hubmap5/exp_22_fold_3.pt",
            f"../input/hubmap5/exp_21_fold_4.pt",
        ],
        1024,
        0.2,
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
        0.4,
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models]) if len(my_models) else 4096
TRAIN_IMG_SIZE




## === cell 4
def rle_encode_less_memory(img):
    if img is None or np.max(img) == 0:
        return ""
    pixels = img.T.flatten().astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def valid_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def _ensure_hwc_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            maxv = (
                np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 255
            )
            img = (np.clip(img.astype(np.float32) / float(maxv), 0, 1) * 255.0).astype(
                np.uint8
            )
    return img


def _read_tiff_window(arr_hwc: np.ndarray, x: int, y: int, crop: int) -> np.ndarray:
    return arr_hwc[y : y + crop, x : x + crop, :]




## === cell 5
def _check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blancking threshold
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, ss, _ = cv2.split(hsv)

    is_background = ((ss > s_th).sum() <= p_th) or (img.sum() <= p_th)
    return bool(is_background)


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step

        arr = tifffile.imread(tiff_path)
        arr = _ensure_hwc_uint8(arr)
        self.img = arr  # HWC uint8 (RGB)
        self.h, self.w = self.img.shape[:2]

        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size

        img = _read_tiff_window(self.img, x=x, y=y, crop=self.crop_size)

        transormed_imgs = {
            img_size: valid_transform(img_size)(image=img)["image"]
            for img_size in self.all_img_sizes
        }
        transormed_imgs["crop_names"] = f"{x}_{y}"
        transormed_imgs["is_background"] = _check_background(img, self.crop_size)
        return transormed_imgs




## === cell 6
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)
    mask_pred = np.zeros(img_size, dtype=np.uint8)

    for batch in tqdm(data_loader, ncols=70, leave=True):
        if SKIP_BACKGROUND is True and bool(batch["is_background"][0]):
            continue

        pred_total = None  # accumulation for averaged by fold all models predictions
        for cfg, model_group in models_with_config:
            image = batch[cfg.img_size].to(device, non_blocking=True)
            pred_group = None  # accumulation for all folds

            for model in model_group:
                pred = model(image)
                pred = pred.sigmoid()
                if cfg.img_size != TRAIN_IMG_SIZE:
                    pred = torch.nn.functional.interpolate(
                        pred, size=TRAIN_IMG_SIZE, mode="bilinear", align_corners=False
                    )
                pred_group = pred if pred_group is None else (pred_group + pred)

            pred_group = pred_group.squeeze()
            if len(pred_group.shape) == 2:
                pred_group = pred_group.unsqueeze(0)

            pred_group = cfg.weight_blend * (pred_group / len(model_group))
            pred_total = pred_group if pred_total is None else (pred_total + pred_group)

            del image, pred_group

        if pred_total is None:
            continue

        pred_total = (pred_total.detach().cpu().numpy() > THR).astype(np.uint8)
        for predict_single, crop_name in zip(pred_total, batch["crop_names"]):
            x = int(crop_name.split("_")[-2])
            y = int(crop_name.split("_")[-1])

            if crop_size != TRAIN_IMG_SIZE:
                predict_single = cv2.resize(
                    predict_single,
                    (crop_size, crop_size),
                    interpolation=cv2.INTER_NEAREST,
                )

            predict_single[0:RAMKA] = 0
            predict_single[-RAMKA:] = 0
            predict_single[:, 0:RAMKA] = 0
            predict_single[:, -RAMKA:] = 0

            mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

        del pred_total
        gc.collect()

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred)
    del mask_pred
    gc.collect()
    return mask_rle




## === cell 7
all_img_sizes = set([cfg.img_size for cfg in my_models])
models_with_config = []


def _load_state_dict(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    return obj


def _strip_prefix(sd, prefixes=("module.", "model.", "net.", "generator.")):
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _remap_common_encoder_keys(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k

        if nk.startswith("backbone."):
            nk = "encoder." + nk[len("backbone.") :]
        if nk.startswith("enc."):
            nk = "encoder." + nk[len("enc.") :]
        if nk.startswith("encoder_q."):
            nk = "encoder." + nk[len("encoder_q.") :]

        if nk.startswith("decoder."):
            nk = "decoder." + nk[len("decoder.") :]

        if nk.startswith("final."):
            nk = "segmentation_head." + nk[len("final.") :]
        if nk.startswith("seg_head."):
            nk = "segmentation_head." + nk[len("seg_head.") :]
        if nk.startswith("final_conv."):
            nk = "segmentation_head.0." + nk[len("final_conv.") :]
        if nk.startswith("mask_conv."):
            nk = "segmentation_head.0." + nk[len("mask_conv.") :]
        if nk.startswith("classification_head."):
            nk = "segmentation_head.0." + nk[len("classification_head.") :]

        out[nk] = v
    return out


def _infer_resnet_encoder_from_sd(sd: dict, default="resnet34") -> str:
    if not isinstance(sd, dict):
        return default
    candidates = [
        "encoder.layer1.0.conv1.weight",
        "layer1.0.conv1.weight",
        "backbone.layer1.0.conv1.weight",
    ]
    w = None
    for k in candidates:
        if k in sd and hasattr(sd[k], "shape"):
            w = sd[k]
            break
    if w is None:
        return default
    out_ch = int(w.shape[0])
    if out_ch != 64:
        return default
    k3 = None
    for kk in [
        "encoder.layer1.0.conv3.weight",
        "layer1.0.conv3.weight",
        "backbone.layer1.0.conv3.weight",
    ]:
        if kk in sd and hasattr(sd[kk], "shape"):
            k3 = sd[kk]
            break
    if k3 is not None:
        if int(k3.shape[0]) == 256:
            return "resnet50"
    return "resnet34"


def _resolve_weight_path(p: str) -> str:
    pp = Path(p)
    if pp.exists():
        return str(pp)

    base = pp.name
    search_roots = [Path("./model_1"), Path("./model_2")]
    for root in search_roots:
        if root.exists():
            hits = list(root.rglob(base))
            if len(hits) > 0:
                return str(hits[0])

    return str(pp)  # will be handled as missing


for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        w_path2 = _resolve_weight_path(w_path)
        if not Path(w_path2).exists():
            print(
                f"[WARN] Missing weights: {w_path} (resolved to {w_path2}) (skipping this fold)"
            )
            continue

        sd = _load_state_dict(w_path2)
        sd = _strip_prefix(sd)
        sd = _remap_common_encoder_keys(sd)

        encoder_name = _infer_resnet_encoder_from_sd(sd, default=cfg.encoder)

        model = smp.Unet(encoder_name, encoder_weights=None).to(device)

        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"[INFO] Loaded {w_path2} as encoder={encoder_name} with strict=False; missing={len(missing)} unexpected={len(unexpected)}"
                )
            model.eval()
            models_group.append(model)
        except Exception as e:
            print(f"[WARN] Failed to load weights {w_path2}: {e} (skipping this fold)")
            del model
            gc.collect()
            continue

    if len(models_group) == 0:
        print(
            f"[WARN] No weights loaded for encoder={cfg.encoder}; skipping this model group."
        )
        continue
    models_with_config.append((cfg, models_group))

if len(models_with_config) == 0:
    print("[WARN] No models were loaded; predictions will be empty masks (RLE='').")



## === cell 8
for idx, row in df_sub.iterrows():
    tif_path = f"../input/hubmap-kidney-segmentation/test/{row['id']}.tif"
    tiff_path = f"../input/hubmap-kidney-segmentation/test/{row['id']}.tiff"

    if Path(tif_path).exists():
        img_path = tif_path
    elif Path(tiff_path).exists():
        img_path = tiff_path
    else:
        print(
            f"[WARN] Missing test image: {tif_path} (and .tiff fallback); leaving prediction empty."
        )
        df_sub.loc[idx, "predicted"] = ""
        continue

    if len(models_with_config) == 0:
        df_sub.loc[idx, "predicted"] = ""
        continue

    test_ds = SingleTiffDataset(
        tiff_path=img_path,
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
    )

    rle = inference(test_loader, models_with_config, CROP_SIZE)
    df_sub.loc[idx, "predicted"] = "" if rle is None else str(rle)



## === cell 9
import shutil

for d in ["model_1", "model_2"]:
    if Path(d).exists():
        shutil.rmtree(d, ignore_errors=True)



## === cell 10
df_out = df_sub[["id", "predicted"]].copy()
df_out["predicted"] = df_out["predicted"].fillna("").astype(str)

df_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out.shape)
print(df_out.head())
