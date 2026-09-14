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

0.944107616229138

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed) and replace it with a tiny compatible `smp.Unet` stub so the existing model-loading code and forward calls still work. I also make the model checkpoint discovery robust: it search `/kaggle/input` for the expected `efficientnet-b4 ... FOLD-*-model.pth` files and, if none are found, fall back to a safe “empty mask” submission rather than crashing. Finally, I ensure `names`/`preds` are always defined and that a valid `submission.csv` with `id,predicted` is written end-to-end in all cases.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from never actually using the intended trained UNet checkpoints: when `segmentation_models_pytorch` isn’t installed, the tiny UNet stub loads almost none of the checkpoint weights (strict=False), producing essentially meaningless masks. To move the Dice score up toward the target with minimal semantic change, I (1) implement a lightweight “real” EfficientNet-B4 encoder + UNet-style decoder in pure `torchvision` that matches the saved checkpoint structure closely enough to load weights (still the same Unet + EfficientNet-B4 core idea), and (2) require that checkpoints load with high coverage (otherwise safely fall back to empty masks for that model) to avoid garbage predictions. I also enable TTA (already present in your code path) because it’s a minimal inference-only change that typically improves Dice without altering training or architecture semantics. Finally, I keep the submission formatting identical and guaranteed to write `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely because you’re predicting almost-all-empty (or severely misaligned) masks due to a subtle but critical axis mix-up: your tiling grid is built using (H,W) but `make_grid()` treats the input as (x,y) and your windows are then applied as `mask[x1:x2, y1:y2]`, so many tiles don’t correspond to the intended image regions. I minimally fix the grid builder to explicitly use (H,W) semantics (still returning x1,x2,y1,y2) and ensure `min_overlap` is never larger than the window size (which otherwise makes the stride negative/invalid). This keeps your model, checkpoints, TTA, thresholding, and RLE encoding unchanged, but makes the sliding-window inference actually cover the image correctly, which should move Dice substantially upward toward your target. I also keep submission writing identical and guaranteed.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing near-empty (or effectively wrong) masks because no compatible checkpoints are being loaded and/or the discovered checkpoint paths don’t exist in this dataset snapshot. To move Dice up toward the target with minimal semantic change, I (1) relax checkpoint discovery to also look for any `*FOLD*model.pth` under the competition input folder, (2) make state-dict loading robust to common wrapper prefixes like `module.` / `model.` so more weights actually load into the same Unet-style model, and (3) if checkpoints still can’t be used, fall back to a non-empty but conservative baseline by rasterizing the provided **anatomical-structure polygons** into a binary mask and submitting that (this is legitimate use of test metadata and should score > 0 compared to empty). Submission formatting and RLE encoding remain unchanged, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by `tifffile.memmap()` failing on non‑mmap’able TIFFs by adding a safe fallback to `tifffile.imread()` (still returning the same HWC uint8 RGB windows). This is a correctness/stability fix that unblocks end‑to‑end execution and ensures `submission.csv` is always written. I keep the inference logic unchanged, but make sure the TIFF reader consistently handles channel ordering and dtype for both memmap and non‑memmap cases. This should also move your score up from 0.0 (which currently comes from crashing before writing a usable submission).'
- What this solution (achieved 0.0) has done: 'I fix the TIFF loading crash by avoiding full `tifffile.imread()` on JPEG-compressed TIFFs (which requires the unavailable `imagecodecs`) and instead reading the test `.tiff` files via OpenCV’s `imreadmulti` (which can decode these in Kaggle) while keeping the same HWC uint8 RGB contract. This unblocks end-to-end execution so a valid `submission.csv` is always written, and it should raise the score from 0.0 because inference (or at least the anatomical-structure fallback) can now run instead of erroring. I also correct a small TTA averaging bug (it currently divides twice), which is an inference-only calibration fix that should improve Dice without changing the core model or training logic. All paths, thresholding, tiling, and RLE formatting remain unchanged.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF reading crash that prevents end-to-end execution by making `TiffReader` robust to JPEG-compressed TIFFs without `imagecodecs`, preferring OpenCV and then using `tifffile` only when it can decode. I also correct a small TTA averaging logic bug (it currently divides twice, shrinking probabilities) which is inference-only and should move Dice upward toward your target without changing the model/training semantics. Finally, I keep the original inference/tiling/RLE logic intact and ensure `submission.csv` is always written with the required `id,predicted` columns.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF loading failure that currently prevents any submission from being generated by adding a robust reader path that can decode the competition’s JPEG‑compressed TIFFs without needing `imagecodecs`, using OpenCV’s `imdecode` on the raw bytes (and falling back to other methods when possible). This is the minimal change that unblocks end‑to‑end inference and should move the score up from 0.0 because your pipeline actually produce masks instead of crashing. I keep your model/inference/tiling/TTA/thresholding/RLE logic the same, only improving I/O robustness and ensuring the script always writes a valid `submission.csv`. No training, architecture, or metric semantics are changed.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF decoding so it can reliably read the competition’s JPEG-compressed `.tiff` files without `imagecodecs`, which is currently the runtime blocker preventing any submission and causing the 0.0 score. The minimal safe approach is to decode via Pillow first (it can handle these TIFFs in this environment), then fall back to OpenCV and finally to tifffile only for non-JPEG cases. This preserves your tiling/inference/RLE logic exactly, but unblocks end-to-end prediction so the model (or anatomical-structure fallback) can actually produce non-empty masks and move the Dice score upward toward the target. I also keep submission writing unchanged and guaranteed.'
- What this solution (achieved 0.0) has done: 'I fix the runtime TIFF decoding failure that prevents any submission from being generated by adding a robust, JPEG-TIFF-safe reader path that uses OpenCV’s `imreadmulti(..., IMREAD_UNCHANGED)` to load the multi-page TIFF and then converts it to an HWC 3-channel uint8 image. This avoids the `tifffile` + missing `imagecodecs` issue while keeping the rest of your inference/tiling/TTA/thresholding/RLE logic unchanged. I also make the fallback order explicit: OpenCV multi-page first (most reliable here), then Pillow, then finally `tifffile` only for non-JPEG cases. These changes are score-positive only insofar as they unblock actual inference (instead of crashing), moving you up from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the runtime failure in `TiffReader` by adding a JPEG-TIFF-safe decoding path that does not require `imagecodecs`: it first try OpenCV, then Pillow, and then a final OpenCV `imdecode` fallback on the raw bytes (which is the missing piece causing your current crash). This is a minimal, targeted I/O robustness change that unblocks end-to-end inference and ensures `submission.csv` is always written. I keep the model/inference/tiling/TTA/RLE logic unchanged, so any score increase comes purely from successfully producing real predictions instead of crashing/empty output. I also make the exception message include which decoder paths were attempted for easier debugging if any file still fails.'
- What this solution (achieved 0.0) has done: 'I fix the runtime TIFF decoding failure by making the `TiffReader` handle JPEG-compressed TIFFs via OpenCV’s `imdecode` using `IMREAD_COLOR`, which is the most reliable path here and avoids the `tifffile`+`imagecodecs` requirement. I also make the OpenCV multi-page path request color decoding, and ensure we always end up with a 3-channel uint8 BGR image so downstream tiling/inference stays unchanged. These are targeted I/O robustness fixes only; model, tiling, TTA, thresholding, and RLE formatting remain the same, but the pipeline now run end-to-end and produce a non-empty (or at least valid) `submission.csv`, improving the score from 0.0 by actually generating predictions.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import glob
import json
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
    import segmentation_models_pytorch as smp  # type: ignore

    _HAS_SMP = True
except Exception:
    _HAS_SMP = False

    try:
        import torchvision
        from torchvision.models import efficientnet_b4
    except Exception as e:
        raise RuntimeError(
            "torchvision is required for the fallback EfficientNet-B4 UNet implementation."
        ) from e

    class ConvBNAct(nn.Module):
        def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch, 3, 1, 1)
            self.conv2 = ConvBNAct(out_ch, out_ch, 3, 1, 1)

        def forward(self, x, skip):
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
            x = self.conv2(self.conv1(x))
            return x

    class EfficientNetB4UNet(nn.Module):
        """
        A compact UNet-like decoder on top of torchvision EfficientNet-B4.
        Intended to be checkpoint-compatible with common smp-efficientnet-b4-unet training runs.
        """

        def __init__(
            self, encoder_name="efficientnet-b4", encoder_weights=None, classes=1
        ):
            super().__init__()
            if encoder_name not in (
                "efficientnet-b4",
                "tf_efficientnet_b4_ns",
                "efficientnet_b4",
            ):
                raise ValueError(
                    f"Unsupported encoder_name for fallback: {encoder_name}"
                )

            self.encoder = efficientnet_b4(weights=None)
            self.features = self.encoder.features

            with torch.no_grad():
                self._probe_channels()

            self.center = nn.Sequential(
                ConvBNAct(self.c5, self.c5, 3, 1, 1),
                ConvBNAct(self.c5, self.c5, 3, 1, 1),
            )
            self.up4 = UpBlock(self.c5, self.c4, 256)
            self.up3 = UpBlock(256, self.c3, 128)
            self.up2 = UpBlock(128, self.c2, 64)
            self.up1 = UpBlock(64, self.c1, 32)
            self.head = nn.Conv2d(32, classes, kernel_size=1)

        def _collect_skips(self, x):
            skips = []
            for layer in self.features:
                x = layer(x)
                skips.append(x)
            return skips

        def _probe_channels(self):
            x = torch.zeros(1, 3, 256, 256)
            feats = self._collect_skips(x)

            by_res = {}
            for f in feats:
                by_res[f.shape[-2:]] = f
            res_sorted = sorted(by_res.keys(), key=lambda r: r[0] * r[1], reverse=True)
            picked = res_sorted[:5]
            f1, f2, f3, f4, f5 = [by_res[r] for r in picked]

            self._picked_res = picked
            self.c1 = f1.shape[1]
            self.c2 = f2.shape[1]
            self.c3 = f3.shape[1]
            self.c4 = f4.shape[1]
            self.c5 = f5.shape[1]

        def forward(self, x):
            feats = self._collect_skips(x)
            by_res = {}
            for f in feats:
                by_res[f.shape[-2:]] = f
            f1, f2, f3, f4, f5 = [by_res[r] for r in self._picked_res]

            x = self.center(f5)
            x = self.up4(x, f4)
            x = self.up3(x, f3)
            x = self.up2(x, f2)
            x = self.up1(x, f1)
            x = self.head(x)
            return x

    import types

    smp = types.SimpleNamespace(Unet=EfficientNetB4UNet)




## === cell 1
def _resolve_comp_root():
    candidates = [
        "../input/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    base = "/kaggle/input"
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            if "sample_submission.csv" in files:
                return root
    raise FileNotFoundError(
        "Could not locate competition dataset directory containing sample_submission.csv"
    )


COMP_ROOT = _resolve_comp_root()
TEST_DIR = os.path.join(COMP_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(COMP_ROOT, "sample_submission.csv")

sz = 256  # tile size (model input)
reduce = 4  # downscale factor relative to original TIFF
TH = 0.5  # threshold for positive predictions

DATA = TEST_DIR + "/"  # preserve original semantics (string concatenations)


def _discover_model_paths():
    patterns = [
        "/kaggle/input/**/efficientnet-b4-unet-BCELoss-256-FOLD-*-model.pth",
        "../input/**/efficientnet-b4-unet-BCELoss-256-FOLD-*-model.pth",
        "/kaggle/input/**/*.pth",
        "../input/**/*.pth",
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    found = sorted(set(found))
    likely = []
    for p in found:
        bn = os.path.basename(p).lower()
        if ("fold" in bn and "model" in bn) or ("unet" in bn and "efficientnet" in bn):
            likely.append(p)
    return likely if len(likely) > 0 else found


MODELS = _discover_model_paths()

df_sample = pd.read_csv(SAMPLE_SUB_PATH)

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
shift = True
minoverlap = 300

TTA = True

print("COMP_ROOT:", COMP_ROOT)
print(
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    "num_test_json:",
    len(glob.glob(os.path.join(TEST_DIR, "*.json"))),
)
print("Discovered model checkpoints:", len(MODELS))
if len(MODELS) > 0:
    print("First checkpoint:", MODELS[0])
print("Using segmentation_models_pytorch:", _HAS_SMP)




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None:
            continue
        if isinstance(enc, (float, np.floating)) and np.isnan(enc):
            continue
        if isinstance(enc, str) and len(enc.strip()) == 0:
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

s_th = 40
p_th = 1000 * (sz // 256) ** 2


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape_hw, window=256, min_overlap=32):
    H, W = int(shape_hw[0]), int(shape_hw[1])
    window = int(window)
    min_overlap = int(min_overlap)

    min_overlap = max(0, min(min_overlap, window - 1))
    stride = window - min_overlap

    nx = max(1, (H - window) // stride + 1)
    x1 = np.arange(nx, dtype=np.int64) * stride
    x1[-1] = max(0, H - window)
    x2 = (x1 + window).clip(0, H)

    ny = max(1, (W - window) // stride + 1)
    y1 = np.arange(ny, dtype=np.int64) * stride
    y1[-1] = max(0, W - window)
    y2 = (y1 + window).clip(0, W)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


class TiffReader:
    """
    Bugfix: reliably decode competition JPEG-compressed TIFFs without 'imagecodecs'.

    Key change vs previous failing version:
      - Ensure OpenCV decoders are asked for COLOR output (IMREAD_COLOR).
        This makes cv2 handle JPEG-compressed TIFFs in this Kaggle environment more reliably.
      - Keep output contract unchanged: cache full image as HWC uint8, BGR order.
    """

    def __init__(self, path):
        self.path = path
        img = None
        tried = []

        def _ensure_bgr_uint8(arr):
            if arr is None:
                return None
            a = arr
            if a.ndim == 2:
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            elif a.ndim == 3 and a.shape[2] == 4:
                a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
            elif a.ndim == 3 and a.shape[2] >= 3:
                a = a[:, :, :3]
            else:
                return None

            if a.dtype != np.uint8:
                if np.issubdtype(a.dtype, np.integer):
                    maxv = np.iinfo(a.dtype).max
                    a = (
                        (a.astype(np.float32) * (255.0 / maxv))
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                else:
                    a = (a.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
            return a

        try:
            tried.append("cv2.imreadmulti(IMREAD_COLOR)")
            ok, pages = cv2.imreadmulti(path, flags=cv2.IMREAD_COLOR)
            if ok and pages is not None and len(pages) > 0 and pages[0] is not None:
                img = _ensure_bgr_uint8(pages[0])
        except Exception:
            img = None

        if img is None:
            try:
                tried.append("PIL.Image.open")
                from PIL import Image

                with Image.open(path) as im:
                    im = im.convert("RGB")
                    rgb = np.asarray(im, dtype=np.uint8)
                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                img = None

        if img is None:
            try:
                tried.append("cv2.imdecode(raw_bytes, IMREAD_COLOR)")
                with open(path, "rb") as f:
                    raw = f.read()
                buf = np.frombuffer(raw, dtype=np.uint8)
                dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                img = _ensure_bgr_uint8(dec)
            except Exception:
                img = None

        if img is None:
            tried.append("tifffile.memmap/imread")
            try:
                try:
                    arr = tiff.memmap(path)
                except Exception:
                    arr = tiff.imread(path)
                a = arr
                if a.ndim == 2:
                    a = a[..., None]
                elif a.ndim == 3:
                    if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
                        a = np.moveaxis(a, 0, -1)
                else:
                    raise ValueError(f"Unexpected TIFF ndim={a.ndim} for {path}")

                if a.shape[-1] < 3:
                    pad = 3 - a.shape[-1]
                    a = np.concatenate([a] + [a[..., -1:]] * pad, axis=-1)
                a = a[..., :3]

                if a.dtype != np.uint8:
                    if np.issubdtype(a.dtype, np.integer):
                        maxv = np.iinfo(a.dtype).max
                        a = (
                            (a.astype(np.float32) * (255.0 / maxv))
                            .clip(0, 255)
                            .astype(np.uint8)
                        )
                    else:
                        a = (a.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)

                img = a
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read TIFF {path}. Tried: {', '.join(tried)}. "
                    f"Last error: {repr(e)}"
                ) from e

        if img is None:
            raise RuntimeError(
                f"Failed to read TIFF {path}. Tried: {', '.join(tried)}."
            )

        self.a = img
        self.shape = self.a.shape[:2]  # (H, W)

    def read_window(self, x1, x2, y1, y2):
        return self.a[x1:x2, y1:y2, :]


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _poly_to_mask(polys, shape_hw):
    H, W = int(shape_hw[0]), int(shape_hw[1])
    m = np.zeros((H, W), dtype=np.uint8)
    if polys is None:
        return m
    for feat in polys:
        try:
            coords = feat["geometry"]["coordinates"]
        except Exception:
            continue
        for ring in coords:
            if not ring:
                continue
            pts = np.asarray(ring, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue
            pts = np.round(pts[:, :2]).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1)
    return m




## === cell 4
names, preds = [], []


def _unwrap_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
        and isinstance(state["model_state_dict"], dict)
    ):
        state = state["model_state_dict"]
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    if not isinstance(state, dict):
        return state

    def strip_prefix(sd, pref):
        out = {}
        for k, v in sd.items():
            if k.startswith(pref):
                out[k[len(pref) :]] = v
            else:
                out[k] = v
        return out

    state2 = state
    for pref in ("module.", "model.", "net."):
        if any(k.startswith(pref) for k in state2.keys()):
            state2 = strip_prefix(state2, pref)
    return state2


if len(MODELS) == 0:
    for _, row in df_sample.iterrows():
        img_id = row["id"]
        tiff_path = os.path.join(DATA, img_id + ".tiff")
        reader = TiffReader(tiff_path)
        anat_path = os.path.join(TEST_DIR, img_id + "-anatomical-structure.json")
        if os.path.exists(anat_path):
            anat = _load_json(anat_path)
            mask = _poly_to_mask(anat, reader.shape)
            rle = rle_encode_less_memory(mask.astype(np.uint8))
        else:
            rle = ""
        names.append(img_id)
        preds.append(rle)
else:
    if shift:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.reader = TiffReader(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.reader.shape  # (H, W)
                self.reduce = reduce
                self.sz = reduce * sz
                self.mask_grid = make_grid(
                    self.shape, window=self.sz, min_overlap=minoverlap
                )

            def __len__(self):
                return len(self.mask_grid)

            def __getitem__(self, idx):
                x1, x2, y1, y2 = self.mask_grid[idx]
                img = self.reader.read_window(x1, x2, y1, y2)

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
                            base = 0
                            for model in self.models:
                                p = model(x)
                                p = torch.sigmoid(p).detach()
                                py = p if py is None else (py + p)
                                base += 1

                            if self.tta:
                                flips = [[-1], [-2], [-2, -1]]
                                for f in flips:
                                    xf = torch.flip(x, f)
                                    for model in self.models:
                                        p = model(xf)
                                        p = torch.flip(p, f)
                                        py += torch.sigmoid(p).detach()
                                        base += 1

                            py /= float(base)

                            py = F.interpolate(
                                py,
                                scale_factor=reduce,
                                mode="bilinear",
                                align_corners=False,
                            )
                            py = py.permute(0, 2, 3, 1).float().cpu()

                            py = py.squeeze(-1).numpy()
                            z = z.numpy()

                            batch_size = len(py)
                            for i in range(batch_size):
                                yield py[i], z[i], y[i]

            def __len__(self):
                return len(self.dl.dataset)

        def _load_model_checkpoint(path: str):
            state = torch.load(path, map_location=torch.device("cpu"))
            state = _unwrap_state_dict(state)

            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            missing, unexpected = model.load_state_dict(state, strict=False)
            total_params = sum(1 for _ in model.state_dict().keys())
            loaded_params = total_params - len(missing)
            load_ratio = loaded_params / max(1, total_params)

            if load_ratio < 0.60:
                print(
                    f"Skipping checkpoint (low compatibility {load_ratio:.2%}): {os.path.basename(path)} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
                return None

            if len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"Checkpoint loaded ({load_ratio:.2%}) for {os.path.basename(path)}; "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

            model.float()
            model.eval()
            model.to(device)
            return model

        models = []
        for path in MODELS:
            try:
                m = _load_model_checkpoint(path)
            except Exception as e:
                print("Failed to load checkpoint:", path, "err:", repr(e))
                m = None
            if m is not None:
                models.append(m)

        if len(models) == 0:
            for _, row in df_sample.iterrows():
                img_id = row["id"]
                tiff_path = os.path.join(DATA, img_id + ".tiff")
                reader = TiffReader(tiff_path)
                anat_path = os.path.join(
                    TEST_DIR, img_id + "-anatomical-structure.json"
                )
                if os.path.exists(anat_path):
                    anat = _load_json(anat_path)
                    mask = _poly_to_mask(anat, reader.shape)
                    rle = rle_encode_less_memory(mask.astype(np.uint8))
                else:
                    rle = ""
                names.append(img_id)
                preds.append(rle)
        else:
            for _, row in df_sample.iterrows():
                img_id = row["id"]
                ds = HuBMAPDataset(img_id)
                dl = DataLoader(
                    ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
                )
                mp = Model_pred(models, dl, tta=TTA)

                mask = np.zeros(ds.shape, dtype=np.uint8)
                for pred, vert, i in iter(mp):
                    x1, x2, y1, y2 = vert
                    mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

                mask = (mask > 0.5).astype(np.uint8)
                rle = rle_encode_less_memory(mask)

                names.append(img_id)
                preds.append(rle)

                del mask, ds, dl
                gc.collect()

    else:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.reader = TiffReader(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.reader.shape
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
                win = self.reader.read_window(p00, p01, p10, p11)
                img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = win

                if self.reduce != 1:
                    img = cv2.resize(
                        img,
                        (self.sz // reduce, self.sz // reduce),
                        interpolation=cv2.INTER_AREA,
                    )

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                _, s, _ = cv2.split(hsv)
                if (s > s_th).sum() <= p_th or img.sum() <= p_th:
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
                            base = 0
                            for model in self.models:
                                p = model(x)
                                p = torch.sigmoid(p).detach()
                                py = p if py is None else (py + p)
                                base += 1

                            if self.tta:
                                flips = [[-1], [-2], [-2, -1]]
                                for f in flips:
                                    xf = torch.flip(x, f)
                                    for model in self.models:
                                        p = model(xf)
                                        p = torch.flip(p, f)
                                        py += torch.sigmoid(p).detach()
                                        base += 1

                            py /= float(base)

                            py = F.interpolate(
                                py,
                                scale_factor=reduce,
                                mode="bilinear",
                                align_corners=False,
                            )
                            py = py.permute(0, 2, 3, 1).float().cpu()

                            batch_size = len(py)
                            for i in range(batch_size):
                                yield py[i], y[i]

            def __len__(self):
                return len(self.dl.dataset)

        def _load_model_checkpoint(path: str):
            state = torch.load(path, map_location=torch.device("cpu"))
            state = _unwrap_state_dict(state)

            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            missing, unexpected = model.load_state_dict(state, strict=False)

            total_params = sum(1 for _ in model.state_dict().keys())
            loaded_params = total_params - len(missing)
            load_ratio = loaded_params / max(1, total_params)
            if load_ratio < 0.60:
                print(
                    f"Skipping checkpoint (low compatibility {load_ratio:.2%}): {os.path.basename(path)} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
                return None

            if len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"Checkpoint loaded ({load_ratio:.2%}) for {os.path.basename(path)}; "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

            model.float()
            model.eval()
            model.to(device)
            return model

        models = []
        for path in MODELS:
            try:
                m = _load_model_checkpoint(path)
            except Exception as e:
                print("Failed to load checkpoint:", path, "err:", repr(e))
                m = None
            if m is not None:
                models.append(m)

        if len(models) == 0:
            for _, row in df_sample.iterrows():
                img_id = row["id"]
                tiff_path = os.path.join(DATA, img_id + ".tiff")
                reader = TiffReader(tiff_path)
                anat_path = os.path.join(
                    TEST_DIR, img_id + "-anatomical-structure.json"
                )
                if os.path.exists(anat_path):
                    anat = _load_json(anat_path)
                    mask = _poly_to_mask(anat, reader.shape)
                    rle = rle_encode_less_memory(mask.astype(np.uint8))
                else:
                    rle = ""
                names.append(img_id)
                preds.append(rle)
        else:
            for _, row in df_sample.iterrows():
                img_id = row["id"]
                ds = HuBMAPDataset(img_id)
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
                    // 2 : (
                        -(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz
                    ),
                    ds.pad1
                    // 2 : (
                        -(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz
                    ),
                ]

                rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
                names.append(img_id)
                preds.append(rle)

                del mask, ds, dl
                gc.collect()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4098636057.py in __init__(self, path)
    118                 try:
--> 119                     arr = tiff.memmap(path)
    120                 except Exception:

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in memmap(filename, shape, dtype, page, series, level, mode, **kwargs)
   1533                 if tiffseries.dataoffset is None:
-> 1534                     raise ValueError('image data are not memory-mappable')
   1535                 shape = tiffseries.shape

ValueError: image data are not memory-mappable

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4098636057.py in __init__(self, path)
    120                 except Exception:
--> 121                     arr = tiff.imread(path)
    122                 a = arr

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in imread(files, selection, aszarr, key, series, level, squeeze, maxworkers, buffersize, mode, name, offset, size, pattern, axesorder, categories, imread, imreadargs, sort, container, chunkshape, chunkdtype, axestiled, ioworkers, chunkmode, fillvalue, zattrs, multiscales, omexml, out, out_inplace, _multifile, _useframes, **kwargs)
   1237                     return zarr_selection(store, selection, out=out)
-> 1238                 return tif.asarray(
   1239                     key=key,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, key, series, level, squeeze, out, maxworkers, buffersize)
   4541                 raise ValueError('page is None')
-> 4542             result = page0.asarray(
   4543                 out=out, maxworkers=maxworkers, buffersize=buffersize

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, out, squeeze, lock, maxworkers, buffersize)
   8884 
-> 8885             for _ in self.segments(
   8886                 func=func,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in segments(self, lock, maxworkers, func, sort, buffersize, _fullsize)
   8698                 ):
-> 8699                     yield from executor.map(decode, segments)
   8700 

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    316         try:
--> 317             return fut.result(timeout)
    318         finally:

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    400             try:
--> 401                 raise self._exception
    402             finally:

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode(args, decodeargs, decode)
   8671             def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
-> 8672                 return func(decode(*args, **decodeargs))
   8673 

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode_raise_compression(exc, *args, **kwargs)
   8093             def decode_raise_compression(*args, exc=str(exc)[1:-1], **kwargs):
-> 8094                 raise ValueError(f'{exc}')
   8095 

ValueError: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3368994013.py in <cell line: 0>()
     45         img_id = row["id"]
     46         tiff_path = os.path.join(DATA, img_id + ".tiff")
---> 47         reader = TiffReader(tiff_path)
     48         anat_path = os.path.join(TEST_DIR, img_id + "-anatomical-structure.json")
     49         if os.path.exists(anat_path):

/tmp/ipykernel_55/4098636057.py in __init__(self, path)
    147                 img = a
    148             except Exception as e:
--> 149                 raise RuntimeError(
    150                     f"Failed to read TIFF {path}. Tried: {', '.join(tried)}. "
    151                     f"Last error: {repr(e)}"

RuntimeError: Failed to read TIFF ../input/hubmap-kidney-segmentation/test/8242609fa.tiff. Tried: cv2.imreadmulti(IMREAD_COLOR), PIL.Image.open, cv2.imdecode(raw_bytes, IMREAD_COLOR), tifffile.memmap/imread. Last error: ValueError("<COMPRESSION.JPEG: 7> requires the 'imagecodecs' package")

## === cell 5
df = pd.DataFrame({"id": names, "predicted": preds})

df = df_sample[["id"]].merge(df, on="id", how="left")
df["predicted"] = df["predicted"].fillna("")

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows")
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)
