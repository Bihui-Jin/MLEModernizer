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

0.9328032830223908

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on missing external model weight files (the immediate cause of the crash) by falling back to a deterministic, valid “all-empty mask” submission when weights are unavailable; this guarantees an end-to-end run and a correctly formatted `submission.csv`. I also fix the submission column name to match the competition’s `sample_submission.csv` (`id,predicted`) and ensure `names/preds` are always defined so the final cell can’t crash. These changes are score-neutral to slightly worse than a real model, but they satisfy the requirement to produce a valid submission file reliably in this Kaggle environment where the referenced weights do not exist. If you later add the missing weights into `/kaggle/input/...`, the original inference path automatically run without further code changes.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing an all-empty mask submission because the referenced weight files are missing, so the smallest way to move toward the target is to actually load valid weights that exist in this environment and then run the exact same inference pipeline. I only change the model-weights path resolution: search `/kaggle/input` for the expected `*-model.pth` files (or any `.pth` fallback) and use them if found, otherwise keep your current empty-mask fallback to guarantee a valid `submission.csv`. This preserves your model architecture, tiling/shift logic, thresholding, and RLE encoding, while turning the solution from “always empty” into “real predictions” when weights are present. I also make the found-weights selection deterministic (sorted) to avoid run-to-run variance.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is because the pipeline almost certainly falls back to an all-empty submission when it can’t find the external `.pth` weights, so the smallest improvement toward the 0.9328 target is to make weight discovery actually succeed in this environment. I keep your exact model/inference/RLE logic, but expand `_find_model_weights` to also search `/kaggle/data/input` and `/kaggle/data` (your file tree shows data lives there), and I only accept candidate `.pth` files whose tensor shapes match this `_TinyUNet` to avoid silently loading incompatible weights. If no compatible weights exist, it still produce the same valid empty-mask `submission.csv` as before. This should move the score up from 0.0 whenever any compatible weights are present, without changing evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with always falling back to an empty-mask submission because no compatible `.pth` weights exist in this environment, so the only legitimate way to move toward the 0.9328 target is to ensure the script can use real weights when they are present. I keep your model, tiling, thresholding, and RLE logic identical, but make weight discovery both broader and more robust by (1) also searching for `.pt` files and (2) accepting checkpoints that store weights under common keys even when the top-level dict is not itself a state_dict. I also add a strict sanity-check that aborts model usage if loading results in too few parameters matched (to prevent silent “junk predictions” that can score even worse than empty). If no compatible weights are found, it still reliably produce a valid `submission.csv` with the correct `id,predicted` format.'

# 9. Code solution

## === cell 0
import os, sys, gc, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


class _TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(
                d3, size=e3.shape[-2:], mode="bilinear", align_corners=False
            )
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


class _SMPStub:
    def Unet(self, encoder_name, encoder_weights=None, classes=1):
        return _TinyUNet(in_ch=3, out_ch=classes)


smp = _SMPStub()


KAGGLE_INPUT_ROOT = "/kaggle/input"
ALT_INPUT_ROOT = "/kaggle/data/input"


def resolve_input_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("../input/"):
        p2 = os.path.join(KAGGLE_INPUT_ROOT, p[len("../input/") :])
        if os.path.exists(p2):
            return p2
        p3 = os.path.join(ALT_INPUT_ROOT, p[len("../input/") :])
        if os.path.exists(p3):
            return p3
    return p  # last resort


def _clean_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    if isinstance(sd, dict) and any(key.startswith("module.") for key in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _is_state_dict_compatible_with_tinyunet(sd: dict) -> bool:
    """
    Prevent "found weights but incompatible" from silently producing junk.
    Only accept weights whose tensor shapes match our model.
    """
    try:
        sd = _clean_state_dict(sd)
        if not isinstance(sd, dict) or len(sd) == 0:
            return False
        m = _TinyUNet(in_ch=3, out_ch=1)
        msd = m.state_dict()
        overlap = 0
        for k, v in sd.items():
            if k in msd and torch.is_tensor(v) and torch.is_tensor(msd[k]):
                overlap += 1
                if tuple(v.shape) != tuple(msd[k].shape):
                    return False
        return overlap >= 10  # conservative minimum to avoid random pth files
    except Exception:
        return False


def _find_model_weights(expected_paths):
    search_roots = [KAGGLE_INPUT_ROOT, ALT_INPUT_ROOT, "/kaggle/data"]

    existing = [p for p in expected_paths if os.path.exists(p)]
    if len(existing) == len(expected_paths) and len(existing) > 0:
        return expected_paths

    expected_basenames = set(os.path.basename(p) for p in expected_paths)
    found = []
    for root0 in search_roots:
        if not os.path.isdir(root0):
            continue
        for root, _dirs, files in os.walk(root0):
            for fn in files:
                if fn in expected_basenames:
                    found.append(os.path.join(root, fn))
    found = sorted(set(found))

    candidates = found
    if len(candidates) == 0:
        any_ckpt = []
        for root0 in search_roots:
            if not os.path.isdir(root0):
                continue
            for root, _dirs, files in os.walk(root0):
                for fn in files:
                    lfn = fn.lower()
                    if lfn.endswith(".pth") or lfn.endswith(".pt"):
                        any_ckpt.append(os.path.join(root, fn))
        candidates = sorted(set(any_ckpt))

    compatible = []
    for p in candidates:
        try:
            ckpt = torch.load(p, map_location=torch.device("cpu"))
            if _is_state_dict_compatible_with_tinyunet(ckpt):
                compatible.append(p)
        except Exception:
            continue

    if len(compatible) > 0:
        compatible = sorted(compatible)
        return compatible[: min(5, len(compatible))]

    return expected_paths


sz = 256  # the size of tiles (model input)
reduce = 4  # reduce the original images by 4 times
TH = 0.35  # threshold for positive predictions

DATA = resolve_input_path("../input/hubmap-kidney-segmentation/test/")
MODELS = [
    resolve_input_path(
        f"../input/b4256shiftfreezebce/efficientnet-b4-256-FOLD-{i}-model.pth"
    )
    for i in range(5)
]
MODELS = _find_model_weights(MODELS)

df_sample = pd.read_csv(
    resolve_input_path("../input/hubmap-kidney-segmentation/sample_submission.csv")
)

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
shift = True
minoverlap = 300

if not os.path.isdir(DATA):
    raise FileNotFoundError(f"Test folder not found: {DATA}")

missing_models = [p for p in MODELS if not os.path.exists(p)]
USE_MODEL = len(missing_models) == 0 and len(MODELS) > 0
if not USE_MODEL:
    print(
        "[WARN] Model weights not found/compatible in input. Falling back to empty-mask submission."
    )
    print("[WARN] Missing model weights (showing up to 5):", missing_models[:5])
else:
    print(f"[INFO] Using {len(MODELS)} model weight file(s). Example:", MODELS[0])

names, preds = [], []




## === cell 1
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
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40
p_th = 1000 * (sz // 256) ** 2


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array of size (N,4), where N is number of tiles.
    2nd axis represents slices: x1,x2,y1,y2
    """
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




## === cell 3
class TiffReader:
    def __init__(self, path: str):
        self.path = path
        arr = tiff.imread(path)
        if arr.ndim == 3:
            if arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
                arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
        elif arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        else:
            raise ValueError(f"Unsupported TIFF shape {arr.shape} for {path}")

        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.shape[2] != 3:
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            else:
                raise ValueError(f"Expected 3 channels, got {arr.shape[2]} for {path}")

        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            mn = arr_f.min()
            mx = arr_f.max()
            if mx > mn:
                arr_f = (arr_f - mn) / (mx - mn) * 255.0
            else:
                arr_f = np.zeros_like(arr_f)
            arr = np.clip(arr_f, 0, 255).astype(np.uint8)

        self.arr = arr
        self.shape = (arr.shape[0], arr.shape[1])  # (H,W)

    def read_window(self, x1, x2, y1, y2):
        return self.arr[x1:x2, y1:y2, :]




## === cell 4
if not USE_MODEL:
    names = df_sample["id"].tolist()
    preds = [""] * len(names)
else:
    if not shift:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.data = TiffReader(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.data.shape
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
                tile = self.data.read_window(p00, p01, p10, p11)
                img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = tile

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
        for path in MODELS:
            ckpt = torch.load(path, map_location=torch.device("cpu"))
            state_dict = _clean_state_dict(ckpt)
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            matched = len(model.state_dict()) - len(missing)
            if matched < 10:
                print(
                    f"[WARN] Too few keys matched when loading {os.path.basename(path)} (matched={matched}). Disabling model usage."
                )
                USE_MODEL = False
                models = []
                break

            if len(unexpected) > 0:
                print(
                    f"[WARN] Unexpected keys when loading {os.path.basename(path)}: {unexpected[:5]}"
                )
            if len(missing) > 0:
                print(
                    f"[WARN] Missing keys when loading {os.path.basename(path)}: {missing[:5]}"
                )
            model.float().eval().to(device)
            models.append(model)
        del ckpt, state_dict

        if not USE_MODEL:
            names = df_sample["id"].tolist()
            preds = [""] * len(names)
        else:
            names, preds = [], []
            for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
                idx = row["id"]
                ds = HuBMAPDataset(idx)
                dl = DataLoader(
                    ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
                )
                mp = Model_pred(models, dl)

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
                names.append(idx)
                preds.append(rle)

                del mask, ds, dl
                gc.collect()

    if shift and USE_MODEL:

        class HuBMAPDataset(Dataset):
            def __init__(self, idx, sz=sz, reduce=reduce):
                self.data = TiffReader(os.path.join(DATA, idx + ".tiff"))
                self.shape = self.data.shape
                self.reduce = reduce
                self.sz = reduce * sz
                self.mask_grid = make_grid(
                    self.shape, window=self.sz, min_overlap=minoverlap
                )

            def __len__(self):
                return len(self.mask_grid)

            def __getitem__(self, idx):
                x1, x2, y1, y2 = self.mask_grid[idx]
                img = self.data.read_window(x1, x2, y1, y2)

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
        for path in MODELS:
            ckpt = torch.load(path, map_location=torch.device("cpu"))
            state_dict = _clean_state_dict(ckpt)
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            matched = len(model.state_dict()) - len(missing)
            if matched < 10:
                print(
                    f"[WARN] Too few keys matched when loading {os.path.basename(path)} (matched={matched}). Disabling model usage."
                )
                USE_MODEL = False
                models = []
                break

            if len(unexpected) > 0:
                print(
                    f"[WARN] Unexpected keys when loading {os.path.basename(path)}: {unexpected[:5]}"
                )
            if len(missing) > 0:
                print(
                    f"[WARN] Missing keys when loading {os.path.basename(path)}: {missing[:5]}"
                )
            model.float().eval().to(device)
            models.append(model)
        del ckpt, state_dict

        if not USE_MODEL:
            names = df_sample["id"].tolist()
            preds = [""] * len(names)
        else:
            names, preds = [], []
            for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
                idx = row["id"]
                ds = HuBMAPDataset(idx)
                dl = DataLoader(
                    ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
                )
                mp = Model_pred(models, dl)

                mask = np.zeros(ds.shape, dtype=np.uint8)
                for pred, vert, _i in iter(mp):
                    x1, x2, y1, y2 = vert
                    mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

                mask = (mask > 0.5).astype(np.uint8)

                rle = rle_encode_less_memory(mask)
                names.append(idx)
                preds.append(rle)

                del mask, ds, dl
                gc.collect()




## === cell 5
sub = pd.DataFrame({"id": names, "predicted": preds})
sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("").astype(str)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print(
    f"Wrote submission.csv with {len(sub)} rows to {os.path.abspath('submission.csv')}"
)
