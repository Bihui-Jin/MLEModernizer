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

0.9451351679481818

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The main blocker is TIFF reading: OpenCV refuses huge images and `tifffile` needs `imagecodecs` for JPEG-compressed TIFFs, so the current pipeline crashes before producing any rows. I fix this by switching to `PIL.Image` for TIFF decoding (it can read these Kaggle TIFFs without `imagecodecs`) and by reading tiles on-demand instead of loading the whole slide into RAM. I also fix the submission format to match the competition (`id,predicted`) and ensure we always write a non-empty `submission.csv`. These changes are execution-critical and score-positive/neutral because they preserve the same tiling, model inference, thresholding, and RLE encoding semantics—just with a working image IO path.'
- What this solution (achieved 0.0) has done: 'I fix the crash by making the TIFF reader robust: first try `tifffile.imread` (which can handle many TIFF variants), and if that fails fall back to PIL, and if both fail return an “all-empty” dataset for that image so the pipeline still produces a valid (blank) RLE. I also correct a small but important TTA averaging bug (it currently overcounts models when TTA is enabled), which is score-positive while keeping the same inference semantics. Finally, I replace deprecated `F.upsample` with `F.interpolate` to avoid runtime issues across PyTorch versions and keep output shapes consistent. These changes preserve the tiling/inference/RLE core logic while ensuring the notebook runs end-to-end and always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a “valid CSV but essentially empty/incorrect masks” failure mode, which here is likely caused by a coordinate/order mismatch when writing tile predictions back into the full-resolution canvas and then encoding (this can silently produce near-empty masks). I make the smallest fix: ensure all geometry is consistently treated as (H, W) with (row=x, col=y) everywhere, and fix the TIFF-to-RGB color convention so the HSV saturation blank-tile filter doesn’t incorrectly drop most informative tiles. These changes preserve your exact tiling, model inference, thresholding, and RLE encoding logic, but make the reconstructed mask align to the original image coordinate system so Dice increases toward your target. I also keep the robust TIFF loading and submission formatting intact.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with “predictions are effectively garbage/empty” because the checkpoint files aren’t present, so you silently run an untrained UNet and then threshold it—this score ~0 even if the submission CSV is valid. The smallest score-positive change (without altering your architecture, tiling, loss, or post-processing semantics) is to load fold checkpoints from a known-available source by globbing common Kaggle input locations and verifying state_dict compatibility, instead of hardcoding a missing `../input/skfoldalldata/` path. I also fix a small TTA averaging bug (currently divides twice when TTA is enabled) but keep TTA disabled by default so behavior is unchanged unless you turn it on. Finally, I make the “missing model” case explicit (warn + still produce a valid CSV) so you don’t unknowingly submit random weights again.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing almost-all-empty masks due to a tile-indexing mismatch: in `Model_pred` you filter out blank tiles, but you still yield the *filtered* `y` (0..k-1 within that filtered batch) instead of the original dataset tile index, so the caller can’t reliably map predictions back (and this silently tends to reconstruct poor/empty masks). I make the smallest fix by yielding the original tile index (and returning it from the dataset as a stable tile id), while keeping the exact same tiling, model inference, thresholding, and RLE encoding logic. I also fix a small TTA averaging bug (even though TTA is currently off) in a way that preserves semantics when `tta=False` and prevents unintended scaling if you enable it later. These changes are execution-safe and should move Dice up substantially toward your target without changing the model architecture or training approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with generating almost-all-empty/incorrect masks despite producing a valid CSV. The smallest score-positive change that preserves your core tiling/inference/RLE logic is to fix the TTA averaging math bug (it currently divides wrong and then divides again), and to make tile “blank filtering” less destructive by using a more reliable blank criterion while keeping the same intent (skip truly empty tiles). I also ensure we read the correct test folder (`../input/hubmap-kidney-segmentation/test/` is JSON-only; the TIFFs are typically in `../input/hubmap-kidney-segmentation/test/` at competition time, but in your environment they may be under the top-level `../input/test/`), by auto-detecting where the `.tiff` files actually are without changing downstream logic. These are minimal execution/quality fixes that should move Dice upward toward your target without changing the model architecture or post-processing semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with predicting essentially-empty masks because the code is not actually loading the competition’s trained checkpoints (so inference runs with random/untrained weights) even though it produces a valid CSV. I make the smallest execution-safe change that increases score: load the known available “best public” pretrained UNet weights shipped in `../input/hubmap-kidney-segmentation/` (the Kaggle baseline artifact) while keeping the exact same UNet inference/tiling/threshold/RLE pipeline. I also fix one metric-critical detail in RLE encoding: don’t forcibly zero out the first/last pixels in-place (that can delete true positives at borders); instead use the standard pad-based RLE that preserves all pixels while remaining memory-light. These changes preserve your core logic and should move Dice upward toward your target.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F

sys.path.append("../input/segmentation-models-pytorch-install")

try:
    import segmentation_models_pytorch as smp  # type: ignore

    _HAS_SMP = True
except Exception:
    smp = None
    _HAS_SMP = False

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

    class _Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

        def forward(self, x):
            return self.net(x)

    class _Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = _DoubleConv(in_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class SimpleUNet(nn.Module):
        def __init__(self, in_channels=3, classes=1, base=32):
            super().__init__()
            self.inc = _DoubleConv(in_channels, base)
            self.down1 = _Down(base, base * 2)
            self.down2 = _Down(base * 2, base * 4)
            self.down3 = _Down(base * 4, base * 8)
            self.down4 = _Down(base * 8, base * 8)

            self.up1 = _Up(base * 16, base * 4)
            self.up2 = _Up(base * 8, base * 2)
            self.up3 = _Up(base * 4, base)
            self.up4 = _Up(base * 2, base)
            self.outc = nn.Conv2d(base, classes, kernel_size=1)

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

    class _SMPShim:
        @staticmethod
        def Unet(encoder_name=None, encoder_weights=None, classes=1):
            return SimpleUNet(in_channels=3, classes=classes, base=32)

    smp = _SMPShim()



## === cell 1
sz = 256  # the size of tiles (at reduced scale)
reduce = 4  # reduce the original images by 4 times
TH = 0.55  # threshold for positive predictions

_CAND_TEST_DIRS = [
    "../input/hubmap-kidney-segmentation/test/",
    "../input/test/",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test/",
]
DATA = None
for d in _CAND_TEST_DIRS:
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.lower().endswith((".tif", ".tiff")):
                DATA = d
                break
    if DATA is not None:
        break
if DATA is None:
    DATA = "../input/hubmap-kidney-segmentation/test/"

MODELS = [
    f"../input/skfoldalldata/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"  # efficientnet-b4, se_resnext50_32x4d

shift = True
minoverlap = 300

print(f"Using TEST TIFF directory: {DATA}")




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None or (isinstance(enc, float) and np.isnan(enc)):
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


def rle_encode_less_memory(img: np.ndarray) -> str:
    pixels = img.T.flatten().astype(np.uint8, copy=False)
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
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return Array of size (N,4), where N - number of tiles,
    2nd axis represent slices: x1,x2,y1,y2

    NOTE: shape is (H, W). We keep x as row (0..H) and y as col (0..W).
    """
    x, y = shape  # x=H, y=W
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




## === cell 4
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None  # allow very large slides


class TiffReader:
    def __init__(self, path):
        self.path = path
        self._backend = None

        try:
            arr = tiff.imread(path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3:
                if arr.shape[-1] in (3, 4):
                    arr = arr[..., :3]
                elif arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                    arr = arr[..., :3]
                else:
                    arr = np.asarray(arr)
            else:
                arr = np.asarray(arr)

            if arr.dtype != np.uint8:
                mx = float(np.max(arr)) if arr.size else 0.0
                if mx > 0:
                    arr = (arr.astype(np.float32) / mx * 255.0).astype(
                        np.uint8, copy=False
                    )
                else:
                    arr = arr.astype(np.uint8, copy=False)

            self.arr = arr
            self.shape = (int(self.arr.shape[0]), int(self.arr.shape[1]))  # (H, W)
            self._backend = "tifffile"
            return
        except Exception:
            self.arr = None

        try:
            im = Image.open(path)
            self.im = im.convert("RGB")  # ensure 3-channel RGB
            self.shape = (int(self.im.height), int(self.im.width))  # (H, W)
            self._backend = "pil"
        except (UnidentifiedImageError, OSError) as e:
            raise e

    def read_window(self, x1, x2, y1, y2):
        if self._backend == "tifffile":
            tile = self.arr[int(x1) : int(x2), int(y1) : int(y2)]
            return tile
        tile = self.im.crop((int(y1), int(x1), int(y2), int(x2)))
        arr = np.asarray(tile)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return arr


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")
        if not os.path.exists(self.path):
            alt = os.path.join(DATA, idx + ".tif")
            if os.path.exists(alt):
                self.path = alt

        self.reduce = reduce
        self.sz = reduce * sz  # tile size at full resolution

        try:
            self.data = TiffReader(self.path)
            self.shape = self.data.shape  # (H, W)
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )
            self._ok = True
        except Exception as e:
            self.data = None
            self.shape = (1, 1)
            self.mask_grid = np.zeros((0, 4), dtype=np.int64)
            self._ok = False
            self._err = str(e)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, tile_i):
        x1, x2, y1, y2 = self.mask_grid[tile_i]
        img = self.data.read_window(int(x1), int(x2), int(y1), int(y2))

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, s, _ = cv2.split(hsv)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

        sat_count = int((s > s_th).sum())
        nz_count = int((gray > 10).sum())
        is_blank = (sat_count <= p_th) and (nz_count <= p_th)

        return (
            img2tensor((img / 255.0 - mean) / std),
            vertices,
            int(tile_i),
            int(is_blank),
        )


class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, z, tile_i, is_blank in iter(self.dl):
                keep = is_blank == 0
                if keep.sum() == 0:
                    continue

                x = x[keep].to(device)
                z = z[keep]
                tile_i = tile_i[keep]

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
                    py /= 1 + len(flips)  # average over TTA views per model-sum

                py /= len(self.models)  # average over models

                py = F.interpolate(
                    py, scale_factor=reduce, mode="bilinear", align_corners=False
                )
                py = py.permute(0, 2, 3, 1).float().cpu()

                py = py.squeeze(-1).numpy()
                z = z.numpy()
                tile_i = tile_i.numpy()

                batch_size = len(py)
                for i in range(batch_size):
                    yield py[i], z[i], int(tile_i[i])

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
import glob


def _discover_model_paths():
    candidates = []

    candidates.extend([p for p in MODELS if os.path.exists(p)])

    baseline_roots = [
        "../input/hubmap-kidney-segmentation/",
        "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/",
    ]
    for root in baseline_roots:
        if os.path.isdir(root):
            for p in glob.glob(os.path.join(root, "**", "*.pth"), recursive=True):
                candidates.append(p)

    if len(candidates) == 0:
        patterns = [
            "../input/**/**/*.pth",
            "../input/**/*.pth",
        ]
        for pat in patterns:
            for p in glob.glob(pat, recursive=True):
                candidates.append(p)

    seen = set()
    out = []
    for p in candidates:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return sorted(out)


def _clean_state_dict(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_sd[nk] = v
        return new_sd
    return sd


model_paths = _discover_model_paths()

models = []
loaded_any = False
for path in model_paths:
    try:
        state = torch.load(path, map_location=torch.device("cpu"))
        state = _clean_state_dict(state)

        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        missing, unexpected = model.load_state_dict(state, strict=False)
        total_keys = len(model.state_dict().keys())
        ok = (len(unexpected) == 0) and (len(missing) / max(total_keys, 1) < 0.05)
        if not ok:
            continue

        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        loaded_any = True
        if len(models) >= 5:
            break
    except Exception:
        continue

if not loaded_any:
    print(
        "WARNING: No compatible .pth checkpoints found under ../input. Inference will use untrained weights => very low score."
    )
    model = smp.Unet(model_name, encoder_weights=None, classes=1)
    model.float()
    model.eval()
    model.to(device)
    models = [model]
else:
    print(f"Loaded {len(models)} checkpoint(s) for inference.")
    for i, p in enumerate(model_paths[: len(models)]):
        print(f"  model[{i}]: {p}")

gc.collect()



## === cell 6
names, preds = [], []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)
    if not getattr(ds, "_ok", True) or len(ds) == 0:
        names.append(idx)
        preds.append("")
        del ds
        gc.collect()
        continue

    dl = DataLoader(
        ds,
        batch_size=bs,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=0,
    )

    mp = Model_pred(models, dl, tta=False)

    mask_votes = np.zeros((ds.shape[0], ds.shape[1]), dtype=np.uint8)

    for pred, vert, tile_i in iter(mp):
        x1, x2, y1, y2 = map(int, vert)

        x1 = max(0, min(x1, ds.shape[0]))
        x2 = max(0, min(x2, ds.shape[0]))
        y1 = max(0, min(y1, ds.shape[1]))
        y2 = max(0, min(y2, ds.shape[1]))

        if x2 <= x1 or y2 <= y1:
            continue

        ph, pw = int(x2 - x1), int(y2 - y1)
        if pred.shape[0] != ph or pred.shape[1] != pw:
            pred = cv2.resize(pred, (pw, ph), interpolation=cv2.INTER_LINEAR)

        mask_votes[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

    mask = (mask_votes > 0).astype(np.uint8)
    rle = rle_encode_less_memory(mask)

    names.append(idx)
    preds.append(rle)

    del ds, dl, mask_votes, mask
    gc.collect()



## === cell 7
sub = pd.DataFrame({"id": names, "predicted": preds})

sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")  # safety: never NaN strings

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(f"Wrote submission.csv with {len(sub)} rows")
assert len(sub) == len(
    df_sample
), f"Row mismatch: submission {len(sub)} vs sample {len(df_sample)}"
assert list(sub.columns) == ["id", "predicted"]
assert os.path.exists("submission.csv")
